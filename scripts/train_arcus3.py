"""Explicit, bounded dense LoRA control or disposable preflight. Docker CUDA only."""
import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from arcus3.config import read,authorize,deadline,check_live,validate_control
from arcus3.donor import verify,load
from arcus3.adapters import attach
from arcus3.data import encode
from arcus3.checkpoint import digest,restore
from arcus3.training import train
from baby_arcus.gpu_job_control import gpu_job

def main(args):
    if os.environ.get('ARCUS3_CONTROLLED_DOCKER')!='1' or not Path('/.dockerenv').exists():raise RuntimeError('Docker CUDA required')
    authorize(read('/app/configs/arcus3/project.json'),'training')
    cfg=validate_control(read('/app/configs/arcus3/dense_control.json'))
    if args.preflight:cfg={**cfg,'max_updates':2,'max_train_seconds':120,'max_target_tokens':2048}
    elif not args.preflight_report:raise ValueError('Measured preflight required')
    config_sha=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    end=deadline(args.deadline);out=Path(args.output)
    data=Path(args.data);manifest=read(data/'manifest.json')
    if manifest['source']!=read('/app/configs/arcus3/data_sources.json'):
        raise ValueError('Unreviewed source manifest')
    for name,h in manifest['files'].items():
        if Path(name).name!=name or digest(data/name)!=h:raise ValueError('Data tamper')
    data_sha=digest(data/'manifest.json')
    if not args.preflight:
        receipt=read(args.preflight_report)
        if not receipt['preflight'] or receipt['state']['updates']!=2 or receipt['state']['data_sha256']!=data_sha or not receipt['initial_parity']:
            raise ValueError('Invalid preflight receipt')
        for key in ('rank','alpha','targets','max_length','accumulation','learning_rate','seed'):
            if receipt['effective_config'][key]!=cfg[key]:raise ValueError('Preflight configuration mismatch: '+key)
    verify(args.donor)
    import torch
    from arcus3.evaluation import aggregate
    torch.set_num_threads(2)
    def live():
        check_live(end,out)
        if (out/'pause-training').exists():raise RuntimeError('Training paused')
    with gpu_job():
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        torch.cuda.set_per_process_memory_fraction(.7);torch.manual_seed(cfg['seed'])
        model,tokenizer=load(args.donor)
        rows={};rejected={}
        for split in ('train','test'):
            source=[json.loads(line) for line in (data/(split+'.jsonl')).read_text().splitlines()]
            rows[split]=[encoded for row in source if (encoded:=encode(tokenizer,row['messages'],cfg['max_length'])) is not None]
            rejected[split]=len(source)-len(rows[split])
            if not rows[split]:raise ValueError('No tokenized rows: '+split)
        x=torch.tensor([rows['test'][0]['input_ids']],device='cuda')
        with torch.inference_mode():before=model(x,use_cache=False).logits.float().clone()
        model,trainable=attach(model,cfg);model.eval()
        with torch.inference_mode():after=model(x,use_cache=False).logits.float()
        error=float((before-after).abs().max());del before,after,x
        if error!=0:raise RuntimeError('Initial adapter parity failed')
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'])
        state={'updates':0,'cursor':0,'target_tokens':0,'data_sha256':data_sha,'config_sha256':config_sha}
        if args.resume:state=restore(args.resume,model,optimizer,data_sha,config_sha)
        def heldout():
            model.eval();scores=[]
            with torch.inference_mode():
                for row in rows['test']:
                    live();x=torch.tensor([row['input_ids']],device='cuda');y=torch.tensor([row['labels']],device='cuda')
                    loss=model(input_ids=x,labels=y,use_cache=False).loss.float().item()
                    scores.append({'nll_sum':loss*row['target_tokens'],'target_tokens':row['target_tokens']})
            return aggregate(scores)
        baseline=heldout()
        result=train(model,optimizer,rows['train'],cfg,state,out,live)
        result.update(preflight=args.preflight,initial_parity=error==0,parity_max_abs=error,trainable_parameters=trainable,
                      effective_config=cfg,heldout_before=baseline,heldout_after=None,
                      selected_rows={k:len(v) for k,v in rows.items()},length_rejections=rejected,
                      peak_cuda_bytes=torch.cuda.max_memory_allocated(),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        from baby_arcus.language_stream import atomic_json
        atomic_json(out/'training-report.json',result)
        if result['reason']!='pause_or_deadline':
            try:result['heldout_after']=heldout()
            except RuntimeError:
                try:live()
                except RuntimeError:result['evaluation_paused']=True
                else:raise
        atomic_json(out/'training-report.json',result)
        print(json.dumps({k:result[k] for k in ('reason','state','heldout_before','heldout_after','peak_cuda_bytes')}),flush=True)

def parser():
    p=argparse.ArgumentParser();p.add_argument('--donor',default='/donor');p.add_argument('--output',default='/output');p.add_argument('--data',default='/data')
    p.add_argument('--deadline',required=True);p.add_argument('--preflight',action='store_true');p.add_argument('--preflight-report');p.add_argument('--resume')
    p.add_argument('--max-new-tokens',type=int,default=128) # shared launcher, unused by training
    return p

if __name__=='__main__':main(parser().parse_args())
