"""Phase 8 full-expert qualification and bounded token stage, with durable pauses."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,hashlib,json,sys,time,gc,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from arcus3.config import read,deadline,check_live,safe_child
from arcus3.checkpoint import digest
from arcus3.campaign import validate,due,in_window,accept_evaluation
from arcus3.corpus_stream import CorpusStream
from arcus3.donor import load,verify
from arcus3.adapters import train_added_experts
from arcus3.backbone_adaptation import update
from arcus3.expanded_checkpoint import save,restore,verify as verify_delta,CheckpointRetentionError
from arcus3.learning_rate import build_optimizer,initial_state,validate_state,apply_for_update
from baby_arcus.language_stream import atomic_json
from baby_arcus.gpu_job_control import gpu_job

def main(a):
    if os.environ.get('ARCUS3_CONTROLLED_DOCKER')!='1' or not Path('/.dockerenv').exists():raise RuntimeError('Docker CUDA only')
    import torch
    from scripts.qualify_arcus3_training import frozen_digest
    from safetensors.torch import load_file
    cfg=validate(read(a.config));out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    initialize_only=bool(getattr(a,'initialize_only',False))
    if initialize_only:
        if (a.qualification or a.resume or getattr(a,'transition',None) or a.evaluation_root
                or getattr(a,'production_policy',None) or not a.checkpoint_root):
            raise ValueError('Fresh initialization must be an isolated zero-update checkpoint operation')
        selection=cfg.get('learning_rate_schedule',{}).get('selection',{})
        if (cfg.get('model_label')!='alpha3.2.2' or not cfg.get('campaign_enabled')
                or selection.get('status')!='qualified' or not a.calibration_receipt):
            raise ValueError('Fresh initialization requires a selected Alpha 3.2.2 schedule')
        receipt=read(a.calibration_receipt)
        if (digest(a.calibration_receipt)!=selection.get('receipt_sha256')
                or receipt.get('schema')!='arcus3-alpha322-schedule-selection-v1'
                or receipt.get('lineage_id')!=cfg['lineage']['id']
                or receipt.get('selected_warmup_input_tokens')!=cfg['learning_rate_schedule']['warmup_input_tokens']
                or receipt.get('campaign_updates')!=0):
            raise ValueError('Selected Alpha 3.2.2 calibration receipt mismatch')
    checkpoint_root=Path(a.checkpoint_root) if a.checkpoint_root else out/'checkpoints'
    checkpoint_root.mkdir(parents=True,exist_ok=True)
    if a.qualification:end=deadline(a.deadline)
    else:
        from datetime import datetime,timezone
        end=datetime.fromisoformat(a.deadline.replace('Z','+00:00'))
        if end.tzinfo is None or not 0<(end-datetime.now(timezone.utc)).total_seconds()<=86400:raise ValueError('Bounded session deadline required')
    if not a.qualification:
        if not cfg['campaign_enabled'] or (not initialize_only and not in_window(read(a.windows))):raise ValueError('Campaign or window disabled')
        qualified=read(a.qualification_report)
        if not qualified.get('qualified') or qualified.get('parent_sha256')!=cfg['parent_sha256'] or qualified.get('trainability')!=cfg['trainability']:
            raise ValueError('Matching passed qualification required')
        qualification_sha=hashlib.sha256(json.dumps({**cfg,'campaign_enabled':False},sort_keys=True).encode()).hexdigest()
        if qualified.get('config_sha256')!=qualification_sha:raise ValueError('Requalify changed training objective/runtime policy')
    from arcus3.production import validate_policy,identity,transition,token_due,thresholds
    production=validate_policy(read(a.production_policy)) if getattr(a,'production_policy',None) else None
    teacher=read(Path(a.teacher)/'manifest.json');stream=CorpusStream(a.data,repeat=not bool(production))
    from arcus3.tokenizer_contract import validate_data
    token_contract=validate_data(a.donor,stream.manifest)
    if production and cfg['max_length']!=token_contract['context_tokens']:raise ValueError('Production context must exactly match donor')
    if not a.qualification and stream.manifest['qualification_only']:raise ValueError('Combined corpus not ready')
    if not a.qualification and teacher.get('teacher_input_tokens')!=stream.manifest['input_tokens_per_pass']:raise ValueError('Teacher cache incomplete for this prepared stage')
    if teacher['data_sha256']!=stream.sha or teacher['top_k']!=cfg['teacher_top_k']:raise ValueError('Teacher/data mismatch')
    if digest(Path(a.converted)/'manifest.json')!=cfg['parent_sha256']:raise ValueError('Parent mismatch')
    verify(a.donor)
    if digest(Path(a.donor)/'manifest.json')!=teacher['donor_manifest_sha256']:raise ValueError('Teacher lineage mismatch')
    config_sha=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    if not a.qualification:
        from arcus3.checkpoint_retention import preflight
        preflight(checkpoint_root,2 if production else None,cfg['parent_sha256'],config_sha)
    teacher_sha=digest(Path(a.teacher)/'manifest.json')
    schedule=read('/app/configs/arcus3/phase8_evaluation.json')
    state={'campaign':'backbone-adaptation-v1','updates':0,'input_tokens':0,'target_tokens':0,'cursor':0,
           'parent_sha256':cfg['parent_sha256'],'data_sha256':stream.sha,'config_sha256':config_sha,
           'teacher_sha256':teacher_sha,'stream':stream.snapshot(),'config':cfg,'evaluation_pending':[],'evaluation_completed':[],
           'scheduler':initial_state(cfg),'scaler':None,'accumulation_position':0,
           'retention_policy':None if a.qualification else 'latest-two-plus-major-evaluations-v1'}
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
    flash=cfg.get('attention_backend','math')=='flash'
    torch.backends.cuda.enable_flash_sdp(flash);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(not flash)
    with gpu_job():
        torch.cuda.set_per_process_memory_fraction(.7);torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        model,_=load(a.donor,converted=a.converted);count=train_added_experts(model)
        from arcus3.routing_objectives import configure
        configure(model,cfg,initialize=not bool(a.resume))
        optimizer=build_optimizer(model,cfg)
        frozen=frozen_digest(model);report.update(trainable_parameters=count,total_parameters=sum(p.numel() for p in model.parameters()))
        def save_progress():
            report['frozen_unchanged']=frozen_digest(model)==frozen
            if not report['frozen_unchanged']:raise RuntimeError('Frozen backbone changed')
            try:
                checkpoint=save(checkpoint_root,model,optimizer,state)
            except CheckpointRetentionError as error:
                checkpoint=error.checkpoint
                report.update(reason='checkpoint_retention_failure',complete=False,state=copy.deepcopy(state),checkpoint=str(checkpoint),
                              checkpoint_manifest_sha256=digest(checkpoint/'manifest.json'),error=str(error))
                atomic_json(out/'report.json',report)
                raise
            report.update(state=copy.deepcopy(state),checkpoint=str(checkpoint),checkpoint_manifest_sha256=digest(checkpoint/'manifest.json'))
            atomic_json(out/'report.json',report)
            return checkpoint
        if a.resume:
            old_manifest=read(Path(a.resume)/'manifest.json')
            migration=getattr(a,'transition',None)
            state=restore(a.resume,model,optimizer,cfg['parent_sha256'],old_manifest['data_sha256'] if migration else stream.sha,config_sha)
            validate_state(state.get('scheduler'),cfg,state['input_tokens'])
            if not migration and state['teacher_sha256']!=teacher_sha:raise ValueError('Teacher changed on resume')
            if production and not migration and state.get('production',{}).get('policy_sha256')!=identity(production):raise ValueError('Production migration required')
        if a.evaluation_root:
            if not a.resume or not state['evaluation_pending']:raise ValueError('No pending checkpoint evaluation')
            result=read(Path(a.evaluation_root)/'scores.json')
            from arcus3.evaluation import sha
            state=accept_evaluation(state,result,digest(Path(a.resume)/'manifest.json'),sha('/app/evaluation/arcus3/baseline-v1.json'),sha('/app/configs/arcus3/evaluation.json'),cfg['nll_regression_limit'])
            state['evaluation_completed'][-1]['scores_sha256']=digest(Path(a.evaluation_root)/'scores.json')
            if production:
                from arcus3.production import accept_donor_receipt
                accepted=accept_donor_receipt(read(Path(a.evaluation_root)/'donor-scores.json'),digest(Path(a.resume)/'manifest.json'),result.get('tier','full'),production)
                state.setdefault('donor_evaluations',[]).append(accepted)
        if a.resume and migration:
            if not production:raise ValueError('Transition requires production policy')
            receipt=read(migration)
            if not state.get('production'):
                if production.get('defer_startup_evaluation'):
                    if receipt.get('baseline_deferred_to_input_tokens')!=production['joint_evaluation_input_tokens']:
                        raise ValueError('Deferred startup evaluation receipt mismatch')
                else:
                    from arcus3.production import accept_donor_receipt
                    accept_donor_receipt(receipt['baseline_donor'],'donor','light',production)
                    accept_donor_receipt(receipt['baseline_arcus'],digest(Path(a.resume)/'manifest.json'),'light',production)
            state=transition(state,receipt,digest(Path(a.resume)/'manifest.json'),stream.sha,teacher_sha,production,
                             same_data=old_manifest['data_sha256']==stream.sha)
        if a.resume:stream=CorpusStream(a.data,state['stream'],repeat=not bool(production))
        # Full evaluations are explicit durable work items; do not silently train past them.
        if initialize_only:
            deferred=cfg.get('evaluation_deferred_until_input_tokens')
            if deferred:
                state['evaluation_pending']=[]
                state['evaluation_deferred_until_input_tokens']=deferred
            else:
                state['evaluation_pending']=['baseline-full']
            report['reason']='fresh_initialization_created'
        elif (not a.qualification and not state['evaluation_completed']
              and not cfg.get('evaluation_deferred_until_input_tokens')):
            state['evaluation_pending']=['baseline-full'];report['reason']='evaluation_required'
        else:
            limit=production['review_input_tokens'] if production else cfg['stage_input_tokens']
            while state['input_tokens']<limit:
                try:live()
                except RuntimeError as e:report['reason']=str(e);break
                if a.qualification and state['updates']>=cfg['qualification_max_updates']:report['reason']='qualification_update_limit';break
                import shutil
                required=sum(p.numel()*p.element_size() for p in model.parameters() if p.requires_grad)*3+64*1024*1024
                # Reserve one final save before accepting another optimizer update.
                if shutil.disk_usage(checkpoint_root).free<required+2*1024**3:report['reason']='checkpoint_storage_pause';break
                if state['evaluation_pending']:report['reason']='evaluation_required';break
                previous=stream.snapshot()
                try:row=stream.next()
                except StopIteration:
                    state['stream']=stream.snapshot();state['batch_complete']=True;report['reason']='batch_complete';break
                if len(row['input_ids'])>cfg['max_length'] or (not a.qualification and len(row['input_ids'])>qualified['max_input_tokens_tested']):
                    raise ValueError('Sequence exceeds measured qualification; requalify before increasing length')
                if state['input_tokens']+len(row['input_ids'])>limit:
                    stream=CorpusStream(a.data,previous,repeat=not bool(production));report['reason']='stage_boundary_no_overshoot'
                    if not state.get('stage_end_pending'):
                        state['stage_end_pending']=True;state['evaluation_pending']=['full']
                    elif not state['evaluation_pending']:report['complete']=True
                    break
                before=state['input_tokens']
                scheduler=apply_for_update(optimizer,cfg,state.get('scheduler'),before,len(row['input_ids']))
                metrics=update(model,optimizer,row,target(row),cfg)
                state['updates']+=1;state['cursor']+=1;state['input_tokens']+=metrics['input_tokens'];state['target_tokens']+=metrics['target_tokens'];state['stream']=stream.snapshot()
                if scheduler is not None:state['scheduler']=scheduler
                state['teacher_target_positions']=state.get('teacher_target_positions',0)+len(row['input_ids'])-1
                report['records'].append({'update':state['updates'],**metrics})
                with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(report['records'][-1])+'\n')
                print(json.dumps(report['records'][-1]),flush=True)
                report['records']=report['records'][-64:]
                if not a.qualification:
                    state['evaluation_pending']=token_due(before,state['input_tokens'],production['evaluation']) if production else due(state['updates'],schedule)
                    if production:
                        state['production']['next_evaluation']=thresholds(state['input_tokens'],production['evaluation'])
                        category=row.get('source','unknown');exposure=state.setdefault('source_exposure',{})
                        exposure[category]=exposure.get(category,0)+metrics['input_tokens']
                if state['updates']%cfg['save_every']==0 or state['updates']==1:
                    cp=save_progress()
                atomic_json(out/'report.json',report)
        if not a.qualification and state['input_tokens']==(production['review_input_tokens'] if production else cfg['stage_input_tokens']):
            if not state.get('stage_end_pending'):state['stage_end_pending']=True;state['evaluation_pending']=['full']
            elif not state['evaluation_pending']:report['complete']=True
        cp=save_progress()
        report['frozen_unchanged']=frozen_digest(model)==frozen
        if not report['frozen_unchanged']:raise RuntimeError('Frozen backbone changed')
        if a.qualification and state['updates']==cfg['qualification_max_updates'] and len(report['records'])>=2:
            first=read(checkpoint_root/'latest.json') # final checkpoint remains authoritative
            initial=next(p for p in (checkpoint_root).iterdir() if p.is_dir() and p.name.startswith('step-1-'))
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
            validate_state(restored.get('scheduler'),cfg,restored['input_tokens'])
            replay=CorpusStream(a.data,restored['stream']);row=replay.next()
            apply_for_update(optimizer,cfg,restored.get('scheduler'),restored['input_tokens'],len(row['input_ids']))
            update(model,optimizer,row,target(row),cfg)
            actual_optimizer=[{k:digest_tensor(t) for k,t in v.items()} for v in optimizer.state.values()]
            report['exact_replay']=trained_hash()==expected and expected_optimizer==actual_optimizer and replay.snapshot()==state['stream']
            full_context=(cfg.get('model_label')!='alpha3.2.2'
                          or max(r['input_tokens'] for r in report['records'])==cfg['max_length'])
            report['qualified']=(report['exact_replay'] and full_context
                                 and all(sum(r['gradient_groups'][k] for r in report['records'])>0
                                         for k in ('expert','router','gate')))
        if initialize_only:
            verify_delta(cp,cfg['parent_sha256'],stream.sha,config_sha)
            from arcus3.checkpoint_retention import protect_initialization
            report['initialization_retention']=protect_initialization(checkpoint_root,cp,cfg['parent_sha256'],config_sha)
            report['initialization_payload_hashes_verified']=True
            report['calibration_receipt_sha256']=digest(a.calibration_receipt)
        report.update(state=state,seconds=time.monotonic()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(),checkpoint_manifest_sha256=digest(cp/'manifest.json'),
                      tokenizer_contract=token_contract,parent_sha256=cfg['parent_sha256'],config_sha256=config_sha,
                      trainability=cfg['trainability'],max_input_tokens_tested=max((r['input_tokens'] for r in report['records']),default=0),
                      learning_rate_schedule=cfg.get('learning_rate_schedule'),optimizer_policy=cfg.get('optimizer'))
        atomic_json(out/'report.json',report)
        if a.qualification and state['updates']==cfg['qualification_max_updates'] and not report['qualified']:raise RuntimeError('Qualification failed')

def digest_tensor(t):
    import torch
    return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest() if torch.is_tensor(t) else t

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name,default in [('donor','/donor'),('data','/data'),('converted','/converted'),('teacher','/teacher'),('output','/output'),('config','/app/configs/arcus3/backbone_adaptation.json'),('windows','/app/configs/arcus3/training_windows.json')]:p.add_argument('--'+name,default=default)
    p.add_argument('--production-policy');p.add_argument('--transition');p.add_argument('--calibration-receipt')
    p.add_argument('--initialize-only',action='store_true')
    p.add_argument('--checkpoint-root');p.add_argument('--deadline',required=True);p.add_argument('--qualification',action='store_true');p.add_argument('--qualification-report');p.add_argument('--resume');p.add_argument('--evaluation-root');p.add_argument('--max-new-tokens',type=int,default=128);a=p.parse_args()
    try:main(a)
    except Exception as e:
        atomic_json(Path(a.output)/'error.json',{'type':type(e).__name__,'message':str(e)});raise
