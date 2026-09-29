"""Bounded expanded preflight, including an actual checkpoint replay."""
import argparse
import hashlib
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import resource
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from arcus3.config import read,authorize,validate_expanded,deadline,check_live
from arcus3.checkpoint import digest
from arcus3.donor import verify,load
from arcus3.data import encode
from arcus3.adapters import attach_expanded
from arcus3.training import train
from arcus3.expanded_checkpoint import save,restore
from baby_arcus.language_stream import atomic_json
from baby_arcus.gpu_job_control import gpu_job

def frozen_digest(model):
    import torch
    h=hashlib.sha256()
    for n,p in model.named_parameters():
        if not p.requires_grad:h.update(n.encode());h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def main(args):
    if os.environ.get('ARCUS3_CONTROLLED_DOCKER')!='1' or not Path('/.dockerenv').exists():raise RuntimeError('Docker CUDA required')
    authorize(read('/app/configs/arcus3/project.json'),'expanded_preflight')
    cfg=validate_expanded(read('/app/configs/arcus3/expanded_preflight.json'))
    end=deadline(args.deadline);out=Path(args.output);data=Path(args.data)
    manifest=read(data/'manifest.json')
    if manifest['source']!=read('/app/configs/arcus3/data_sources.json'):raise ValueError('Data source mismatch')
    for n,h in manifest['files'].items():
        if Path(n).name!=n or digest(data/n)!=h:raise ValueError('Data tamper')
    verify(args.donor)
    import torch
    torch.set_num_threads(2);torch.manual_seed(cfg['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    def live():
        check_live(end,out)
        if (out/'pause-training').exists():raise RuntimeError('Paused')
    with gpu_job():
        torch.cuda.set_per_process_memory_fraction(.7)
        model,tok=load(args.donor,converted=args.converted)
        rows={split:[r for item in (json.loads(l) for l in (data/(split+'.jsonl')).read_text().splitlines())
                     if (r:=encode(tok,item['messages'],cfg['max_length'])) is not None] for split in ('train','test')}
        x=torch.tensor([rows['test'][0]['input_ids']],device='cuda')
        with torch.no_grad():before=model(x,use_cache=False).logits.clone()
        count=attach_expanded(model,cfg);model.eval()
        with torch.no_grad():assert torch.equal(before,model(x,use_cache=False).logits),'Adapter parity'
        del before,x
        frozen=frozen_digest(model)
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'])
        state={'updates':0,'cursor':0,'target_tokens':0,'parent_sha256':digest(Path(args.converted)/'manifest.json'),
               'data_sha256':digest(data/'manifest.json'),'config_sha256':hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()}
        def heldout():
            model.eval();total=0;tokens=0
            with torch.no_grad():
                for row in rows['test']:
                    live();x=torch.tensor([row['input_ids']],device='cuda');y=torch.tensor([row['labels']],device='cuda')
                    total+=float(model(x,labels=y,use_cache=False).loss)*row['target_tokens'];tokens+=row['target_tokens']
            import math
            return {'nll':total/tokens,'perplexity':math.exp(total/tokens),'target_tokens':tokens}
        baseline=heldout()
        result=train(model,opt,rows['train'],cfg,state,out,live,save_fn=save)
        result.update(trainable_parameters=count,config=cfg,heldout_before=baseline,heldout_after=None,replay_exact=None,
                      parent_sha256=state['parent_sha256'],frozen_unchanged=frozen_digest(model)==frozen)
        atomic_json(out/'qualification-report.json',result)
        if result['reason']!='update_budget':return
        expected={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        checkpoint=next((out/'checkpoints').glob('step-2-*'))
        resumed=restore(checkpoint,model,opt,state['parent_sha256'],state['data_sha256'],state['config_sha256'])
        replay=train(model,opt,rows['train'],cfg,resumed,out/'replay',live,save_fn=save)
        result['replay_exact']=all(torch.equal(expected[n],p) for n,p in model.named_parameters() if p.requires_grad)
        result['replay_max_abs']=max(float((expected[n]-p.detach()).abs().max()) for n,p in model.named_parameters() if p.requires_grad)
        result['deterministic_attention']='math-sdpa';result['deterministic_algorithms']=True
        result['replay_updates']=len(replay['records']);result['physical_optimizer_steps']=len(result['records'])+len(replay['records'])
        result['heldout_after']=heldout();result['frozen_unchanged']=result['frozen_unchanged'] and frozen_digest(model)==frozen
        result.update(peak_cuda_bytes=torch.cuda.max_memory_allocated(),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        atomic_json(out/'qualification-report.json',result)
        if not result['replay_exact'] or not result['frozen_unchanged']:raise RuntimeError('Qualification failed')
        print(json.dumps({k:result[k] for k in ('replay_exact','frozen_unchanged','heldout_before','heldout_after','trainable_parameters','peak_cuda_bytes')}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--donor',default='/donor');p.add_argument('--converted',required=True)
    p.add_argument('--data',default='/data');p.add_argument('--output',default='/output');p.add_argument('--deadline',required=True)
    p.add_argument('--max-new-tokens',type=int,default=128);main(p.parse_args())
