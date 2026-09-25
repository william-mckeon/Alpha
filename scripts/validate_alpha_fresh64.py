"""Read-only fresh checkpoint validation; one disposable short-window update."""
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from arcus.tokenizer import get_tokenizer
    from baby_arcus.shared_checkpoint import load,digest,restore_optimizer
    from baby_arcus.shared_curriculum import example
    from baby_arcus.shared_objectives import step
    from baby_arcus.runtime_contract import require_gpu
    from baby_arcus.gpu_job_control import gpu_job
    require_gpu();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.75)
    root=Path('/parent');manifest=json.loads((root/'candidate.json').read_text())
    assert manifest['updates']==0 and manifest['depth_capacity']==1.
    report={'candidate':manifest,'full_64k_training_qualified':False,'complete':False}
    with gpu_job():
        model,data=load(root,manifest,'cuda');assert model.core.cfg.max_seq_len==65536
        assert data['progress']['initialization']=='random' and not data['progress']['sources']
        del data;model.eval();torch.manual_seed(2101)
        torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        with torch.inference_mode(),sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            ids=torch.randint(0,model.language.embedding.num_embeddings,(1,65536),device='cuda')
            result=model.language(model.core,ids,last_only=True);assert bool(result.isfinite().all())
        torch.cuda.synchronize()
        report['forward_64k']={'seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
        del ids,result;torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5)
        row,_,_=example(1,'training','commands');row['hearing']=[]
        target={'tokens':list(range(513))};model.core.gradient_checkpointing=True
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            report['disposable_512_token_training']=step(model,optimizer,get_tokenizer('o200k_base'),row,target)
        report['training_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        report['checkpoint_unchanged']=digest(root/(manifest['generation']+'.pt'))==manifest['sha256']
        report['complete']=report['checkpoint_unchanged']
        Path('/evidence/validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
