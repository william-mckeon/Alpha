"""Bounded single-GPU Nanotron-engine training on a new from-random lineage."""
import argparse
import copy
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def run(cfg, sources, corpus, root, max_updates, deadline, resume=False, pilot=True, eval_settings=None):
    import torch
    import torch.distributed as dist
    from arcus.tokenizer import get_tokenizer
    from baby_arcus.runtime_contract import require_gpu
    from baby_arcus.gpu_job_control import gpu_job
    from baby_arcus.foundation_schedule import accumulation, learning_rate, stop_reason, estimate
    from baby_arcus.foundation_data import PackedStream, audit
    from baby_arcus.foundation_checkpoint import save, load, config_hash
    from baby_arcus.nanotron_adapter import create_model, ArcusForTraining, train_microbatches
    from baby_arcus.routing_trace import RoutingTrace, observe_gradients
    from baby_arcus.language_stream import atomic_json
    from baby_arcus.foundation_evaluation import evaluate
    require_gpu()
    if not 1 <= max_updates <= cfg['updates'] or cfg['world_size']!=1 or cfg['micro_batch']!=1:
        raise ValueError('Invalid single-GPU update request')
    if pilot and max_updates>2:raise ValueError('Pilot capped at two updates')
    if not deadline:raise ValueError('Explicit future deadline required')
    root=Path(root)
    if 'foundation' not in root.parts:raise ValueError('Use an isolated foundation run directory')
    if stop_reason(root,deadline):return {'status':'paused','reason':stop_reason(root,deadline),'updates':None}
    if root.exists() and any(p.name!='runtime.json' for p in root.iterdir()) and not resume:raise ValueError('Refusing to overwrite an existing run')
    corpus_audit=audit(corpus,[s['name'] for s in sources])
    if not corpus_audit['integrity'] or not corpus_audit['provenance_present'] or not corpus_audit['all_splits_covered']:
        raise ValueError('Corpus failed required audit')
    # Dataset identity participates in immutable resume identity.
    cfg={**cfg,'corpus_sha256':corpus_audit['sha256'],'source_mixture':sources}
    root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.70)
    random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    with gpu_job():
        rendezvous=tempfile.mktemp(prefix='arcus-dist-')
        dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=0,world_size=1)
        stream=None
        try:
            model=create_model(cfg)
            optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],betas=tuple(cfg['betas']),eps=cfg['epsilon'],weight_decay=cfg['weight_decay'],foreach=False)
            stream=PackedStream(corpus,sources,cfg['sequence_length'],cfg['seed'])
            progress={'initialization':'random','lineage':cfg['lineage'],'updates':0,'target_tokens':0,'microsteps':0}
            pointer=None
            if resume:progress,pointer=load(root,model,optimizer,stream,cfg)
            else:
                atomic_json(root/'config.json',cfg)
                pointer=save(root,model,optimizer,progress,stream,cfg)
            adapter=ArcusForTraining(model,cfg['loss_chunk_size']);adapter.train()
            count=accumulation(cfg['sequence_length'],1,1,cfg['tokens_per_update'])
            initial_updates=progress['updates'];target=min(cfg['updates'],initial_updates+max_updates)
            start=time.perf_counter();durations=[];reason=None
            while progress['updates']<target:
                reason=stop_reason(root,deadline)
                if reason:break
                # A stopped partial update is discarded on resume from the last durable pointer.
                step=progress['updates']+1
                for group in optimizer.param_groups:group['lr']=learning_rate(step,cfg)
                optimizer.zero_grad(set_to_none=True)
                def batches():
                    for micro in range(count):
                        if stop_reason(root,deadline):raise InterruptedError('Pause/deadline during accumulation')
                        x,y=stream.next_window()
                        x=torch.tensor([x],device='cuda');y=torch.tensor([y],device='cuda')
                        atomic_json(root/'status.json',{'status':'training','durable_updates':pointer['updates'],'next_update':step,'microstep':micro+1,'accumulation':count})
                        yield {'input_ids':x,'input_mask':torch.ones_like(x,dtype=torch.bool),'label_ids':y,'label_mask':torch.ones_like(y,dtype=torch.bool)}
                trace=RoutingTrace(model,max_events=128,max_positions=8) if step%cfg['trace_every']==0 else None
                from contextlib import nullcontext
                torch.cuda.synchronize();started=time.perf_counter()
                try:
                    with trace if trace else nullcontext():
                        with torch.autocast('cuda',dtype=torch.bfloat16):outputs=train_microbatches(adapter,batches(),count,dist.group.WORLD)
                        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip_grad'],error_if_nonfinite=True)
                        observe_gradients(model)
                    optimizer.step()
                except InterruptedError:
                    reason='partial_update_discarded';break
                torch.cuda.synchronize();seconds=time.perf_counter()-started;durations.append(seconds)
                progress['updates']=step;progress['target_tokens']+=cfg['tokens_per_update'];progress['microsteps']+=count
                progress['exposures']=copy.deepcopy(stream.exposures)
                metrics={'updates':step,'seconds':seconds,'nll':sum(float(o['nll']) for o in outputs)/count,
                         'router_loss':sum(float(o['router_loss']) for o in outputs)/count,'gradient_norm':float(norm),'learning_rate':optimizer.param_groups[0]['lr']}
                atomic_json(root/f'update-{step}.json',metrics)
                print(json.dumps(metrics),flush=True)
                if trace:atomic_json(root/f'routing-{step}.json',trace.report())
                if step%cfg['checkpoint_every']==0 or step==target or step%cfg['evaluate_every']==0:
                    pointer=save(root,model,optimizer,progress,stream,cfg)
                if eval_settings and step%cfg['evaluate_every']==0 and not stop_reason(root,deadline):
                    report=evaluate(model,get_tokenizer(cfg['encoding']),corpus,sources,eval_settings)
                    atomic_json(root/f'evaluation-{step}.json',{'candidate':pointer,'config_hash':config_hash(cfg),
                        'lineage':cfg['lineage'],'checkpoint_unchanged':True,
                        'evaluation_identity':config_hash({'settings':eval_settings,'corpus':cfg['corpus_sha256'],'encoding':cfg['encoding']}),**report})
            if reason and reason!='partial_update_discarded' and progress['updates']!=pointer['updates']:
                pointer=save(root,model,optimizer,progress,stream,cfg)
            if reason:(root/'pause-training').touch()
            report={'status':'paused' if reason else 'bounded_run_complete','reason':reason,'candidate':pointer,
                    'completed_updates_this_run':pointer['updates']-initial_updates,'parameters':sum(p.numel() for p in model.parameters()),
                    'step_seconds':durations,'elapsed_seconds':time.perf_counter()-start,
                    'peak_cuda_allocated':torch.cuda.max_memory_allocated(),'peak_cuda_reserved':torch.cuda.max_memory_reserved(),
                    'pilot':pilot,'campaign_complete':pointer['updates']==cfg['updates']}
            if durations:report['projection']=estimate(sum(durations),len(durations)*cfg['tokens_per_update'])
            atomic_json(root/'status.json',report);return report
        finally:
            if stream:stream.close()
            dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/baby_arcus/arcus_128m_smollm2_pretrain.yaml');p.add_argument('--sources',default='configs/baby_arcus/arcus_128m_smollm2_sources.json');p.add_argument('--corpus',required=True);p.add_argument('--root',required=True);p.add_argument('--max-updates',type=int,default=1);p.add_argument('--deadline',required=True);p.add_argument('--resume',action='store_true');p.add_argument('--campaign-authorization');p.add_argument('--evaluation',default='configs/baby_arcus/arcus_128m_smollm2_evaluation.json');a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());pilot=not a.campaign_authorization
    if not pilot:
        authorization=json.loads(Path(a.campaign_authorization).read_text())
        from baby_arcus.foundation_checkpoint import config_hash
        if authorization.get('config_hash')!=config_hash(cfg) or authorization.get('deadline')!=a.deadline or authorization.get('authorized') is not True:
            raise ValueError('Campaign authorization does not match config and deadline')
    print(json.dumps(run(cfg,json.loads(Path(a.sources).read_text())['sources'],a.corpus,a.root,a.max_updates,a.deadline,a.resume,pilot,json.loads(Path(a.evaluation).read_text())),indent=2))
