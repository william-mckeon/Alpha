"""Read-only checkpoint inputs; repeat one disposable update to verify restoration."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from arcus3.config import read,deadline,check_live,safe_child
from arcus3.checkpoint import digest
from arcus3.expanded_checkpoint import restore
from arcus3.donor import load,verify
from arcus3.adapters import train_added_experts
from arcus3.corpus_stream import CorpusStream
from arcus3.backbone_adaptation import update
from scripts.train_arcus3_backbone_adaptation import digest_tensor
from scripts.qualify_arcus3_training import frozen_digest
from baby_arcus.gpu_job_control import gpu_job
from baby_arcus.language_stream import atomic_json

def main(a):
    if not Path('/.dockerenv').exists() or os.environ.get('ARCUS3_CONTROLLED_DOCKER')!='1':raise RuntimeError('Controlled Docker required')
    import torch
    from safetensors.torch import load_file
    end=deadline(a.deadline);cfg=read('/app/configs/arcus3/backbone_adaptation.json');config_sha=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    data_sha=digest(Path(a.data)/'manifest.json');verify(a.donor)
    roots=Path(a.previous)/'checkpoints';first=next(p for p in roots.iterdir() if p.is_dir() and p.name.startswith('step-1-'))
    pointer=read(roots/'latest.json');last=safe_child(roots,pointer['generation'])
    if digest(last/'manifest.json')!=pointer['manifest_sha256']:raise ValueError('Checkpoint pointer mismatch')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    with gpu_job():
        torch.cuda.set_per_process_memory_fraction(.7);model,_=load(a.donor,converted=a.converted);train_added_experts(model)
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'],foreach=False)
        final=restore(last,model,opt,cfg['parent_sha256'],data_sha,config_sha)
        def weights():return {n:digest_tensor(p) for n,p in model.named_parameters() if p.requires_grad}
        def optim():return [{k:digest_tensor(t) for k,t in v.items()} for v in opt.state.values()]
        expected=weights();expected_opt=optim();before=frozen_digest(model)
        saved=restore(first,model,opt,cfg['parent_sha256'],data_sha,config_sha);stream=CorpusStream(a.data,saved['stream']);row=stream.next()
        teacher=read(Path(a.teacher)/'manifest.json');name=row['sha256']+'.safetensors'
        if saved['teacher_sha256']!=digest(Path(a.teacher)/'manifest.json') or digest(Path(a.teacher)/name)!=teacher['files'][name]:raise ValueError('Teacher mismatch')
        check_live(end,a.output);metrics=update(model,opt,row,load_file(str(Path(a.teacher)/name)),cfg)
        result={'schema':'arcus3-phase8-replay-v1','exact_weights':weights()==expected,'exact_optimizer':optim()==expected_opt,
                'exact_data_cursor':stream.snapshot()==final['stream'],'frozen_unchanged':frozen_digest(model)==before,
                'checkpoint_manifest_sha256':digest(last/'manifest.json'),'updates':final['updates'],'input_tokens':final['input_tokens'],
                'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'replayed_metrics':metrics,'production_update_added':False}
        result.update(config_sha256=config_sha,parent_sha256=cfg['parent_sha256'],
                      max_input_tokens_tested=max(len(json.loads(l)['input_ids']) for s in stream.manifest['shards'] for l in (Path(a.data)/s['path']).open()),
                      trainability=cfg['trainability'],cuda_fraction=.7)
        result['qualified']=all(result[k] for k in ('exact_weights','exact_optimizer','exact_data_cursor','frozen_unchanged'))
        atomic_json(Path(a.output)/'replay-report.json',result)
        if not result['qualified']:raise RuntimeError('Production replay mismatch')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('donor','converted','data','teacher','previous','output'):p.add_argument('--'+name,default='/'+name)
    p.add_argument('--deadline',required=True);main(p.parse_args())
