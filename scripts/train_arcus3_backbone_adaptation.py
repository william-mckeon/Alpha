"""Phase 8 full-expert qualification and bounded token stage, with durable pauses."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,hashlib,json,sys,time,gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from arcus3.config import read,deadline,check_live,safe_child
from arcus3.checkpoint import digest
from arcus3.campaign import validate,due,in_window,accept_evaluation
from arcus3.corpus_stream import CorpusStream
from arcus3.donor import load,verify
from arcus3.adapters import train_added_experts
from arcus3.backbone_adaptation import update
from arcus3.expanded_checkpoint import save,restore,verify as verify_delta
from scripts.qualify_arcus3_training import frozen_digest
from baby_arcus.language_stream import atomic_json
from baby_arcus.gpu_job_control import gpu_job

def main(a):
    if os.environ.get('ARCUS3_CONTROLLED_DOCKER')!='1' or not Path('/.dockerenv').exists():raise RuntimeError('Docker CUDA only')
    import torch
    from safetensors.torch import load_file
    cfg=validate(read(a.config));out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if a.qualification:end=deadline(a.deadline)
    else:
        from datetime import datetime,timezone
        end=datetime.fromisoformat(a.deadline.replace('Z','+00:00'))
        if end.tzinfo is None or not 0<(end-datetime.now(timezone.utc)).total_seconds()<=86400:raise ValueError('Bounded session deadline required')
    if not a.qualification:
        if not cfg['campaign_enabled'] or not in_window(read(a.windows)):raise ValueError('Campaign or window disabled')
        qualified=read(a.qualification_report)
        if not qualified.get('qualified') or qualified.get('parent_sha256')!=cfg['parent_sha256'] or qualified.get('trainability')!=cfg['trainability']:
            raise ValueError('Matching passed qualification required')
        qualification_sha=hashlib.sha256(json.dumps({**cfg,'campaign_enabled':False},sort_keys=True).encode()).hexdigest()
        if qualified.get('config_sha256')!=qualification_sha:raise ValueError('Requalify changed training objective/runtime policy')
    teacher=read(Path(a.teacher)/'manifest.json');stream=CorpusStream(a.data)
    if not a.qualification and stream.manifest['qualification_only']:raise ValueError('Combined corpus not ready')
    if not a.qualification and len(teacher['files'])<stream.manifest['records']:raise ValueError('Teacher cache incomplete for this prepared stage')
    if teacher['data_sha256']!=stream.sha or teacher['top_k']!=cfg['teacher_top_k']:raise ValueError('Teacher/data mismatch')
    if digest(Path(a.converted)/'manifest.json')!=cfg['parent_sha256']:raise ValueError('Parent mismatch')
    verify(a.donor)
    if digest(Path(a.donor)/'manifest.json')!=teacher['donor_manifest_sha256']:raise ValueError('Teacher lineage mismatch')
    config_sha=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    teacher_sha=digest(Path(a.teacher)/'manifest.json')
    schedule=read('/app/configs/arcus3/phase8_evaluation.json')
    state={'campaign':'backbone-adaptation-v1','updates':0,'input_tokens':0,'target_tokens':0,'cursor':0,
           'parent_sha256':cfg['parent_sha256'],'data_sha256':stream.sha,'config_sha256':config_sha,
           'teacher_sha256':teacher_sha,'stream':stream.snapshot(),'config':cfg,'evaluation_pending':[],'evaluation_completed':[],
           'scheduler':'constant-lr','scaler':None,'accumulation_position':0}
    report={'schema':'arcus3-phase8-report-v1','qualification':a.qualification,'qualified':False,'complete':False,'records':[]}
    atomic_json(out/'report.json',report)
    def live():
        check_live(end,out)
        if (out/'pause-training').exists():raise RuntimeError('Manual training pause')
        if not a.qualification and not in_window(read(a.windows)):raise RuntimeError('Outside training window')
    def target(row):
        name=row['sha256']+'.safetensors';path=safe_child(a.teacher,name)
        if name not in teacher['files'] or digest(path)!=teacher['files'][name]:raise ValueError('Teacher cache missing or changed')
        return load_file(str(path))
    torch.set_num_threads(2);torch.manual_seed(cfg['seed']);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    with gpu_job():
        torch.cuda.set_per_process_memory_fraction(.7);torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        model,_=load(a.donor,converted=a.converted);count=train_added_experts(model)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'],foreach=False)
        frozen=frozen_digest(model);report.update(trainable_parameters=count,total_parameters=sum(p.numel() for p in model.parameters()))
        if a.resume:
            state=restore(a.resume,model,optimizer,cfg['parent_sha256'],stream.sha,config_sha)
            if state['teacher_sha256']!=teacher_sha:raise ValueError('Teacher changed on resume')
            stream=CorpusStream(a.data,state['stream'])
        if a.evaluation_root:
            if not a.resume or not state['evaluation_pending']:raise ValueError('No pending checkpoint evaluation')
            result=read(Path(a.evaluation_root)/'scores.json')
            from arcus3.evaluation import sha
            state=accept_evaluation(state,result,digest(Path(a.resume)/'manifest.json'),sha('/app/evaluation/arcus3/baseline-v1.json'),sha('/app/configs/arcus3/evaluation.json'),cfg['nll_regression_limit'])
            state['evaluation_completed'][-1]['scores_sha256']=digest(Path(a.evaluation_root)/'scores.json')
        # Full evaluations are explicit durable work items; do not silently train past them.
        if not a.qualification and not state['evaluation_completed']:
            state['evaluation_pending']=['baseline-full'];report['reason']='evaluation_required'
        else:
            while state['input_tokens']<cfg['stage_input_tokens']:
                try:live()
                except RuntimeError as e:report['reason']=str(e);break
                if a.qualification and state['updates']>=cfg['qualification_max_updates']:report['reason']='qualification_update_limit';break
                import shutil
                required=sum(p.numel()*p.element_size() for p in model.parameters() if p.requires_grad)*3+64*1024*1024
                # Reserve one final save before accepting another optimizer update.
                if shutil.disk_usage(out).free<2*required+2*1024**3:report['reason']='checkpoint_storage_pause';break
                if state['evaluation_pending']:report['reason']='evaluation_required';break
                previous=stream.snapshot();row=stream.next()
                if len(row['input_ids'])>cfg['max_length'] or (not a.qualification and len(row['input_ids'])>qualified['max_input_tokens_tested']):
                    raise ValueError('Sequence exceeds measured qualification; requalify before increasing length')
                if state['input_tokens']+len(row['input_ids'])>cfg['stage_input_tokens']:
                    stream=CorpusStream(a.data,previous);report['reason']='stage_boundary_no_overshoot'
                    if not state.get('stage_end_pending'):
                        state['stage_end_pending']=True;state['evaluation_pending']=['full']
                    elif not state['evaluation_pending']:report['complete']=True
                    break
                metrics=update(model,optimizer,row,target(row),cfg)
                state['updates']+=1;state['cursor']+=1;state['input_tokens']+=metrics['input_tokens'];state['target_tokens']+=metrics['target_tokens'];state['stream']=stream.snapshot()
                state['teacher_target_positions']=state.get('teacher_target_positions',0)+len(row['input_ids'])-1
                report['records'].append({'update':state['updates'],**metrics})
                with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(report['records'][-1])+'\n')
                print(json.dumps(report['records'][-1]),flush=True)
                report['records']=report['records'][-64:]
                if not a.qualification:state['evaluation_pending']=due(state['updates'],schedule)
                if state['updates']%cfg['save_every']==0 or state['updates']==1:
                    cp=save(out/'checkpoints',model,optimizer,state);report['checkpoint']=str(cp)
                atomic_json(out/'report.json',report)
        if not a.qualification and state['input_tokens']==cfg['stage_input_tokens']:
            if not state.get('stage_end_pending'):state['stage_end_pending']=True;state['evaluation_pending']=['full']
            elif not state['evaluation_pending']:report['complete']=True
        cp=save(out/'checkpoints',model,optimizer,state);report['checkpoint']=str(cp)
        report['frozen_unchanged']=frozen_digest(model)==frozen
        if not report['frozen_unchanged']:raise RuntimeError('Frozen backbone changed')
        if a.qualification and state['updates']==cfg['qualification_max_updates'] and len(report['records'])>=2:
            first=read(out/'checkpoints/latest.json') # final checkpoint remains authoritative
            initial=next(p for p in (out/'checkpoints').iterdir() if p.is_dir() and p.name.startswith('step-1-'))
            # Fingerprint final trainables without retaining a second model copy.
            def trained_hash():
                h=hashlib.sha256()
                for n,p in model.named_parameters():
                    if p.requires_grad:h.update(n.encode());h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
                return h.hexdigest()
            expected=trained_hash();expected_optimizer=[]
            for v in optimizer.state.values():
                expected_optimizer.append({k:digest_tensor(t) for k,t in v.items()})
            restored=restore(initial,model,optimizer,cfg['parent_sha256'],stream.sha,config_sha)
            replay=CorpusStream(a.data,restored['stream']);row=replay.next();update(model,optimizer,row,target(row),cfg)
            actual_optimizer=[{k:digest_tensor(t) for k,t in v.items()} for v in optimizer.state.values()]
            report['exact_replay']=trained_hash()==expected and expected_optimizer==actual_optimizer and replay.snapshot()==state['stream']
            report['qualified']=report['exact_replay'] and all(sum(r['gradient_groups'][k] for r in report['records'])>0 for k in ('expert','router','gate'))
        report.update(state=state,seconds=time.monotonic()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(),checkpoint_manifest_sha256=digest(cp/'manifest.json'))
        atomic_json(out/'report.json',report)
        if a.qualification and state['updates']==cfg['qualification_max_updates'] and not report['qualified']:raise RuntimeError('Qualification failed')

def digest_tensor(t):
    import torch
    return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest() if torch.is_tensor(t) else t

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name,default in [('donor','/donor'),('data','/data'),('converted','/converted'),('teacher','/teacher'),('output','/output'),('config','/app/configs/arcus3/backbone_adaptation.json'),('windows','/app/configs/arcus3/training_windows.json')]:p.add_argument('--'+name,default=default)
    p.add_argument('--deadline',required=True);p.add_argument('--qualification',action='store_true');p.add_argument('--qualification-report');p.add_argument('--resume');p.add_argument('--evaluation-root');p.add_argument('--max-new-tokens',type=int,default=128);a=p.parse_args()
    try:main(a)
    except Exception as e:
        atomic_json(Path(a.output)/'error.json',{'type':type(e).__name__,'message':str(e)});raise
