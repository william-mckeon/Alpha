"""Isolated actual-model HTTP and joint-update diagnostic. Never promotes weights."""
import argparse,json,sys,tempfile,threading,uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baby_arcus.shared_experience import capture
from baby_arcus.shared_checkpoint import load,save,digest,restore_optimizer
from baby_arcus.shared_learning import update
from baby_arcus.services.shared_worker import Worker
from baby_arcus.services.playroom import PlayroomApplication
from baby_arcus.transport import Client,RemoteError,serve

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/baby_arcus/shared.json');args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text());root=Path(cfg['root']);manifest=json.loads((root/'candidate.json').read_text())
    torch.set_num_threads(2);worker=Worker(args.config,manifest=manifest)
    app=PlayroomApplication();app.world.environment.add_toys();past=capture(app);app.advance();row=capture(app);temporal_target={}
    if worker.model.version>=9:
        from baby_arcus.shared_memory import sensory_features
        row['memory']=[{'id':past['id'],'features':sensory_features(past)}]
    if worker.model.version>=8:
        from baby_arcus.shared_temporal import targets
        action={'kind':'gaze','yaw':.25,'pitch':0}
        status,result=app('POST','/v1/action',{'request_id':'temporal-diagnostic','source':'human','action':action})
        immediate={'experience_id':row['id'],'action':action,'executed':status==200,'result':result,'after':capture(app)}
        for _ in range(3):app.advance()
        temporal_target=targets(row,immediate,capture(app));row['executed_action']=action
    app.close()
    if worker.model.version>=3:row['objects']=[];row['object_source']='none'
    row['hearing']=[{'text':'Please look around.'}]
    token=uuid.uuid4().hex;server=serve('127.0.0.1',0,worker,token)
    threading.Thread(target=server.serve_forever,daemon=True).start();client=Client(f'http://127.0.0.1:{server.server_port}',token,timeout=60,attempts=1)
    try:
        auth=False
        try:Client(client.base_url,attempts=1).request('GET','/ready')
        except RemoteError as exc:auth=exc.status==401
        reply=client.request('POST','/v1/shared/observe',row)
        from copy import deepcopy
        changed=deepcopy(row);changed['hearing']=[{'text':'Different sound.'}];changed['internal'][0]=1.
        other=client.request('POST','/v1/shared/observe',changed)
        initial_body=worker.model.body([row['senses']]).detach()
        shared_body=worker.model([row],worker.tokenizer)['body'].detach()
        from baby_arcus.body_vocabulary import mask
        allowed=torch.tensor([mask(row['senses'])],device=initial_body.device)
        initial_parity=torch.allclose(initial_body[allowed],shared_body[allowed],atol=1e-5) and int(initial_body.argmax())==int(shared_body.argmax())
        source_hash=digest(cfg['body_checkpoint'])
        # All updates target an isolated candidate, never the immutable inference worker.
        device='cuda' if torch.cuda.is_available() else 'cpu'
        model,data=load(root,manifest,device);model.requires_grad_(True)
        optimizer=restore_optimizer(model,data,cfg['learning_rate'])
        from baby_arcus.language_stream import inventory,documents
        corpus_cfg=json.loads(Path(cfg['dataset_config']).read_text(encoding='utf-8'))
        corpus=inventory(corpus_cfg['dataset_root'],corpus_cfg['source_patterns'])
        text=None;document=None;source=Path(corpus['root'])/corpus['files'][0]['path']
        for number,value in documents(source):
            if number%10 and len(worker.tokenizer.encode(value))>=8:text=value;document=number;break
        if text is None:raise ValueError('No eligible corpus passage')
        ids=worker.tokenizer.encode(text)[:64];training=deepcopy(row)
        training['language_prefix_ids']=ids[:-1];training['text_source']='dataset'
        training['hearing']=[];training['eligibility']['training']=True
        from baby_arcus.rest_environment import rewards
        from baby_arcus.curiosity_environment import utility
        target={'body':int(initial_body[0].argmax()),'text':ids[-1], 'rest':rewards(training['internal']),
            'curiosity':[1.],'activity':0,'language_choice':1,'objects':[utility(o['features']) for o in row['objects']]}
        if model.version>=2:target['gaze_choice']=1
        target.update(temporal_target)
        losses=[]
        for _ in range(3):losses.append(update(model,optimizer,[training],worker.tokenizer,[target]))
        from arcus.model import ArcusMoDE
        one_core=sum(isinstance(m,ArcusMoDE) for m in model.modules())==1
        gradients={name:bool(p.grad is not None and p.grad.abs().sum()>0) for name,p in (
            ('rgb',model.visual_input.weight),('body_sensations',model.body_sensation_input.weight),
            ('text',model.language.embedding.weight),('internal',model.internal_input.weight))}
        if model.version>=3:gradients['pixels']=any(p.grad is not None and p.grad.abs().sum()>0 for p in model.perception.parameters())
        else:gradients['objects']=bool(model.object_input.weight.grad is not None and model.object_input.weight.grad.abs().sum()>0)
        if model.version>=2:gradients['gaze']=bool(model.gaze_input.weight.grad is not None and model.gaze_choice.weight.grad.abs().sum()>0)
        if model.version>=8:
            for key in ('future_body','future_rgb','action_quality'):
                parameter=model.causal_predictors[0][-1].weight if model.version>=9 and key!='action_quality' else getattr(model,key).weight
                gradient=parameter.grad
                if model.version>=9 and key!='action_quality' and gradient is not None:gradient=gradient[:20] if key=='future_body' else gradient[20:]
                gradients[key]=bool(gradient is not None and gradient.abs().sum()>0)
        if model.version>=9:gradients['memory']=bool(model.memory_input.weight.grad is not None and model.memory_input.weight.grad.abs().sum()>0)
        progress=data['progress'];progress['updates']+=3
        progress['receipts'].append({'experience_id':row['id'],'accepted_updates':3,'language_targets':3,
            'dataset_file':corpus['files'][0]['path'],'document':document,'fingerprint':corpus['fingerprint']})
        trained=save(root,model,optimizer,progress)
        report={'integration':bool(auth and one_core and initial_parity and all(gradients.values()) and reply['generation']==manifest['generation']),
            'retention':False,'cross_modal':False,'live':False,'sha256':manifest['sha256'],
            'diagnostics':{'authenticated_http':auth,'one_core':one_core,'gradients':gradients,
                'context_changes_scores':reply['scores']['rest']!=other['scores']['rest'],
                'initial_standing_logits_preserved':initial_parity,'joint_update_losses':losses,
                'original_body_unchanged':source_hash==digest(cfg['body_checkpoint'])},
            'candidate':manifest,'diagnostic_child':trained,'promoted':False,
            'remaining':'200-episode retention, controlled transfer, all action/language/recovery live gates; diagnostic input dependence is not learning transfer'}
        from baby_arcus.shared_qualification import source_snapshot
        report['runtime_sources']=source_snapshot()
        (root/'integration.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        (root/'trained-candidate.json').write_text(json.dumps(trained,indent=2),encoding='utf-8')
        print(json.dumps(report),flush=True)
        if not report['integration']:raise SystemExit(1)
    finally:server.shutdown();server.server_close()

if __name__=='__main__':main()
