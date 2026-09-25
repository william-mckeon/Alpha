"""Bounded read-only 39k precision, GQA and pooling comparisons."""
import json
import statistics
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baby_arcus.shared_checkpoint import load,digest
from baby_arcus.runtime_contract import require_gpu
from baby_arcus.gpu_job_control import gpu_job


def main():
    require_gpu();torch.set_num_threads(2);torch.manual_seed(2101)
    root=Path('/parent');manifest=json.loads((root/'candidate.json').read_text())
    assert manifest['updates']==39000
    report={'candidate':manifest,'measurements':{},'complete':False}
    def measure(name,fn):
        for _ in range(2):fn()
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
        for _ in range(7):
            start=time.perf_counter();fn();torch.cuda.synchronize();times.append(time.perf_counter()-start)
        report['measurements'][name]={'median_seconds':statistics.median(times),'samples':times,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    with gpu_job(),torch.no_grad():
        model,data=load(root,manifest,'cuda');del data
        model.eval();ids=torch.randint(0,model.language.embedding.num_embeddings,(1,512),device='cuda')
        fn=lambda:model.language(model.core,ids,last_only=True)
        reference=fn().clone();measure('fp32_repeated_gqa',fn)
        for block in model.core.blocks:block.attn.native_gqa=True
        actual=fn();torch.testing.assert_close(actual,reference,rtol=5e-4,atol=5e-4)
        report['gqa_max_abs_error']=float((actual-reference).abs().max());measure('fp32_native_gqa',fn)
        for block in model.core.blocks:block.attn.native_gqa=False
        if torch.cuda.is_bf16_supported():
            try:
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    actual=fn();measure('bf16_autocast',fn)
                report['bf16']={'finite':bool(actual.isfinite().all()),'max_abs_error':float((actual-reference).abs().max()),
                    'top_token_matches':bool(actual.argmax(-1).eq(reference.argmax(-1)).all()),'qualified_for_training':False}
            except RuntimeError as exc:report['bf16']={'supported_execution':False,'reason':str(exc)}
        from baby_arcus.shared_pooling import separable_pool
        x=torch.randn(1,16,97,95,device='cuda')
        expected=torch.nn.functional.adaptive_avg_pool2d(x.cpu(),(4,4)).cuda()
        actual=separable_pool(x,(4,4));torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-6)
        measure('pool_cpu_fallback',lambda:torch.nn.functional.adaptive_avg_pool2d(x.cpu(),(4,4)).cuda())
        measure('pool_separable_gpu',lambda:separable_pool(x,(4,4)))
        report['pool_max_abs_error']=float((actual-expected).abs().max())
        report['checkpoint_unchanged']=digest(root/(manifest['generation']+'.pt'))==manifest['sha256']
        report['complete']=report['checkpoint_unchanged']
    Path('/evidence/profile.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
