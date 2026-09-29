"""Sequential, matched 64-update local comparison. No automatic promotion."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse, gc, hashlib, json, math, resource, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from arcus3.config import read,authorize,validate_specialization,validate_depth,deadline,check_live
from arcus3.checkpoint import digest,save as dense_save,restore as dense_restore,verify as verify_dense
from arcus3.expanded_checkpoint import save as expanded_save,restore as expanded_restore,verify as verify_expanded
from arcus3.donor import load,verify
from arcus3.adapters import attach,attach_expanded
from arcus3.model import enable_full_depth
from arcus3.data import encode
from arcus3.training import train
from scripts.qualify_arcus3_training import frozen_digest
from baby_arcus.language_stream import atomic_json
from baby_arcus.gpu_job_control import gpu_job

def review_required(baseline,current,limit=.2):
    return not math.isfinite(current) or current>baseline+limit

def resume_generation(root,arm,cfg,config_sha):
    root=Path(root);folder=root/arm/'checkpoints';pointer=read(folder/'latest.json')
    name=pointer['generation']
    if Path(name).name!=name:raise ValueError('Unsafe resume generation')
    checkpoint=folder/name
    if digest(checkpoint/'manifest.json')!=pointer['manifest_sha256']:raise ValueError('Resume pointer tamper')
    m=verify_dense(checkpoint) if arm=='dense' else verify_expanded(checkpoint,cfg['parent_sha256'],cfg['data_sha256'],config_sha)
    if m['data_sha256']!=cfg['data_sha256'] or m['config_sha256']!=config_sha:raise ValueError('Resume lineage differs')
    return checkpoint

def main(args):
    if os.environ.get('ARCUS3_CONTROLLED_DOCKER')!='1' or not Path('/.dockerenv').exists():raise RuntimeError('Docker CUDA required')
    authorize(read('/app/configs/arcus3/project.json'),'specialization')
    cfg=validate_specialization(read('/app/configs/arcus3/specialization.json'))
    arch=validate_depth(read('/app/configs/arcus3/architecture-phase7.json'))
    if arch['depth_capacity']!=1 or arch['depth_trainable']:raise ValueError('Full frozen depth required')
    out=Path(args.output);data=Path(args.data);end=deadline(args.deadline)
    if digest(data/'manifest.json')!=cfg['data_sha256'] or digest(Path(args.converted)/'manifest.json')!=cfg['parent_sha256']:raise ValueError('Pinned inputs differ')
    manifest=read(data/'manifest.json')
    if manifest['source']!=read('/app/configs/arcus3/data_sources.json'):raise ValueError('Source pin differs')
    for n,h in manifest['files'].items():
        if Path(n).name!=n or digest(data/n)!=h:raise ValueError('Data tamper')
    verify(args.donor)
    import torch
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    report={'schema':'arcus3-specialization-report-v1','config':cfg,'architecture':arch,'arms':{},'complete':False}
    previous=read(Path(args.resume)/'campaign-report.json') if args.resume else None
    if previous and (previous['config']!=cfg or previous['architecture']!=arch):raise ValueError('Resume protocol differs')
    if previous and previous.get('review_required'):raise ValueError('Regression requires review, not operational resume')
    atomic_json(out/'campaign-report.json',report)
    def live():
        check_live(end,out)
        if (out/'pause-training').exists():raise RuntimeError('Paused')
    with gpu_job():
        torch.cuda.set_per_process_memory_fraction(.7)
        for arm in ('dense','expanded'):
            config_sha=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
            old=previous.get('arms',{}).get(arm) if previous else None
            resume_path=resume_generation(args.resume,arm,cfg,config_sha) if previous and (Path(args.resume)/arm/'checkpoints/latest.json').exists() else None
            if old and old.get('frozen_unchanged') and old['milestones'][-1]['state']['updates']==64:
                if resume_path is None:raise ValueError('Completed arm missing checkpoint')
                report['arms'][arm]=old;atomic_json(out/'campaign-report.json',report);continue
            live();started=time.monotonic();torch.manual_seed(cfg['seed']);torch.cuda.reset_peak_memory_stats()
            model,tok=load(args.donor,converted=args.converted if arm=='expanded' else None)
            rows={split:[r for item in (json.loads(l) for l in (data/(split+'.jsonl')).read_text().splitlines()) if (r:=encode(tok,item['messages'],cfg['max_length'])) is not None] for split in ('train','test')}
            x=torch.tensor([rows['test'][0]['input_ids']],device='cuda')
            with torch.no_grad():before=model(x,use_cache=False).logits.clone()
            if arm=='expanded':
                enable_full_depth(model,arch['layers']);count=attach_expanded(model,cfg)
            else:model,count=attach(model,cfg)
            model.eval()
            with torch.no_grad():parity=torch.equal(before,model(x,use_cache=False).logits)
            if not parity:raise RuntimeError('Initial full-depth/adapter parity failed')
            del before,x
            frozen=frozen_digest(model)
            opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'])
            state={'updates':0,'cursor':0,'target_tokens':0,'parent_sha256':cfg['parent_sha256'],'data_sha256':cfg['data_sha256'],'config_sha256':hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),'campaign':'matched-64-v1'}
            def heldout():
                model.eval();total=0;tokens=0
                with torch.no_grad():
                    for row in rows['test']:
                        live();x=torch.tensor([row['input_ids']],device='cuda');y=torch.tensor([row['labels']],device='cuda')
                        total+=float(model(x,labels=y,use_cache=False).loss)*row['target_tokens'];tokens+=row['target_tokens']
                return {'nll':total/tokens,'perplexity':math.exp(total/tokens),'target_tokens':tokens}
            baseline=heldout();track={'initial_parity':parity,'trainable_parameters':count,'parameters':sum(p.numel() for p in model.parameters()),'baseline':baseline,'milestones':[],'rows':{k:len(v) for k,v in rows.items()}}
            if resume_path:
                state=expanded_restore(resume_path,model,opt,cfg['parent_sha256'],cfg['data_sha256'],config_sha) if arm=='expanded' else dense_restore(resume_path,model,opt,cfg['data_sha256'],config_sha)
                if old:
                    if old['baseline']!=baseline:raise ValueError('Resume baseline differs')
                    track['milestones']=[m for m in old['milestones'] if m['heldout'] and m['state']['updates']<=state['updates']]
                track['resumed_from']=str(resume_path)
            report['arms'][arm]=track
            for milestone in cfg['milestones']:
                if any(m['state']['updates']==milestone for m in track['milestones']):continue
                if milestone<state['updates']:raise ValueError('Missing earlier evaluated milestone')
                local={**cfg,'max_updates':milestone,'router_aux_coefficient':cfg['router_aux_coefficient'] if arm=='expanded' else 0}
                result=train(model,opt,rows['train'],local,state,out/arm,live,save_fn=expanded_save if arm=='expanded' else dense_save)
                record={**result,'state':dict(state),'heldout':None};track['milestones'].append(record)
                atomic_json(out/'campaign-report.json',report)
                if result['reason']!='update_budget':return
                record['heldout']=heldout()
                if review_required(baseline['nll'],record['heldout']['nll'],cfg['nll_regression_limit']):
                    report['review_required']=arm;atomic_json(out/'campaign-report.json',report);(out/'pause-training').write_text('NLL regression review');return
                atomic_json(out/'campaign-report.json',report)
            track.update(frozen_unchanged=frozen_digest(model)==frozen,seconds=time.monotonic()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated(),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
            if arm=='expanded':
                gates=[m.depth_gate for m in model.modules() if hasattr(m,'depth_gate')]
                track['depth']={'capacity':1.0,'parameters':sum(p.numel() for g in gates for p in g.parameters()),'trainable':False,'gradients_present':any(p.grad is not None for g in gates for p in g.parameters()),'snapshots':[g.last_observation for g in gates]}
                del gates
            if not track['frozen_unchanged']:raise RuntimeError('Frozen parameters changed')
            del model,opt,tok;gc.collect();torch.cuda.empty_cache()
            atomic_json(out/'campaign-report.json',report)
        a,b=[report['arms'][k]['milestones'][-1]['state'] for k in ('dense','expanded')]
        report['matched_exposure']=all(a[k]==b[k] for k in ('updates','cursor','target_tokens'))
        report['complete']=report['matched_exposure'];atomic_json(out/'campaign-report.json',report)
        if not report['complete']:raise RuntimeError('Unmatched exposure')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--donor',default='/donor');p.add_argument('--data',default='/data');p.add_argument('--converted',required=True);p.add_argument('--output',default='/output');p.add_argument('--deadline',required=True);p.add_argument('--resume');p.add_argument('--max-new-tokens',type=int,default=128);main(p.parse_args())
