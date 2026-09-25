"""Read-only CUDA comparisons on the retained 39k model, with fixed inputs."""
import json
import statistics
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baby_arcus.shared_checkpoint import load,digest
from baby_arcus.gpu_job_control import gpu_job
from baby_arcus.runtime_contract import require_gpu
from scripts.audit_alpha_efficiency import inventory


def main():
    require_gpu();torch.set_num_threads(2);torch.manual_seed(2101)
    root=Path('/parent');manifest=json.loads((root/'candidate.json').read_text())
    if manifest['updates']!=39000:raise ValueError('Requires the 39k baseline')
    report={'candidate':manifest,'measurements':{},'complete':False,'training':False}
    def write():Path('/evidence/benchmark.json').write_text(json.dumps(report,indent=2))
    def measure(name,fn):
        for _ in range(3):fn()
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        times=[]
        for _ in range(5):
            start=time.perf_counter()
            for _ in range(10):fn()
            torch.cuda.synchronize();times.append((time.perf_counter()-start)/10)
        report['measurements'][name]={'median_seconds':statistics.median(times),'samples_seconds':times,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
        write()
    with gpu_job(),torch.inference_mode():
        start=time.perf_counter();model,data=load(root,manifest,'cuda');del data
        torch.cuda.synchronize();report['load_seconds']=time.perf_counter()-start
        model.eval();report['inventory']=inventory(model)
        ids=torch.randint(0,model.language.embedding.num_embeddings,(1,512),device='cuda')
        reference=model.language(model.core,ids,last_only=True).clone()
        measure('language_padded',lambda:model.language(model.core,ids,last_only=True))
        for block in model.core.blocks:block.moe.dispatch_mode='compact'
        actual=model.language(model.core,ids,last_only=True)
        torch.testing.assert_close(actual,reference,rtol=5e-4,atol=5e-4)
        report['compact_logits_max_abs_error']=float((actual-reference).abs().max())
        measure('language_compact',lambda:model.language(model.core,ids,last_only=True))
        # Default stays reference until the benchmark justifies a change.
        for block in model.core.blocks:block.moe.dispatch_mode='padded'
        from arcus.tokenizer import get_tokenizer
        from baby_arcus.shared_curriculum import example
        row,_,_=example(1,'training','commands');tokenizer=get_tokenizer('o200k_base')
        full=model([row],tokenizer)
        hidden=model([row],tokenizer,requested=('hidden',))
        torch.testing.assert_close(hidden['hidden'],full['hidden'])
        measure('shared_all_outputs',lambda:model([row],tokenizer))
        measure('shared_hidden_only',lambda:model([row],tokenizer,requested=('hidden',)))
        from arcus.kv_cache import KVCache
        tokens=ids[:,:96]
        def cached():
            cache=KVCache(128)
            value=model.language.cached_logits(model.core,tokens[:,:64],cache)
            for index in range(64,96):value=model.language.cached_logits(model.core,tokens[:,index:index+1],cache)
            return value
        def uncached():
            value=None
            for end in range(64,97):value=model.language(model.core,tokens[:,:end],last_only=True)[:,-1]
            return value
        torch.testing.assert_close(cached(),uncached(),rtol=5e-4,atol=5e-4)
        # Decoding samples use five repetitions rather than 50 long generations.
        for name,fn in (('decode_cached',cached),('decode_uncached',uncached)):
            torch.cuda.reset_peak_memory_stats();times=[]
            for _ in range(5):
                start=time.perf_counter();fn();torch.cuda.synchronize();times.append(time.perf_counter()-start)
            report['measurements'][name]={'median_seconds':statistics.median(times),'samples_seconds':times,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated()};write()
        report['checkpoint_unchanged']=digest(root/(manifest['generation']+'.pt'))==manifest['sha256']
        report['complete']=report['checkpoint_unchanged'];write()
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
