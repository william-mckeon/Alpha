"""Read-only CUDA context profiling; synthetic inputs are not capability scores."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.gpu_job_control import serialized


@serialized
def profile(pointer,output,lengths):
    import torch
    from baby_arcus.runtime_contract import require_gpu
    from baby_arcus.shared_checkpoint import load,digest
    from baby_arcus.context_contract import extend_core,context_tokens
    require_gpu()
    for length in lengths: context_tokens({'context_tokens':length})
    path=Path(output)
    if path.exists(): raise ValueError('Use a new report path')
    path.parent.mkdir(parents=True,exist_ok=True)
    pointer=Path(pointer); manifest=json.loads(pointer.read_text())
    model,data=load(pointer.parent,manifest,'cuda'); del data
    model.eval(); report={'checkpoint':manifest,'training':False,'capability_established':False,'stages':[]}
    torch.set_num_threads(2)
    try:
        for length in lengths:
            free,_=torch.cuda.mem_get_info()
            if free<3*1024**3: raise RuntimeError('Insufficient measured free GPU headroom')
            extend_core(model,length)
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            started=time.monotonic()
            with torch.no_grad():
                ids=torch.zeros((1,length),device='cuda',dtype=torch.long)
                result=model.language(model.core,ids,last_only=True)
                finite=bool(torch.isfinite(result).all())
            torch.cuda.synchronize()
            report['stages'].append({'tokens':length,'finite':finite,'seconds':time.monotonic()-started,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
            del ids,result
            if not finite: raise RuntimeError('Nonfinite output')
            path.write_text(json.dumps(report,indent=2),encoding='utf-8')
        report['checkpoint_unchanged']=digest(pointer.parent/(manifest['generation']+'.pt'))==manifest['sha256']
        report['complete']=True
    except Exception as exc:
        report.update(complete=False,error=str(exc)); raise
    finally:
        path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--candidate',required=True)
    p.add_argument('--output',required=True); p.add_argument('--lengths',type=int,nargs='+',default=[512,2048])
    a=p.parse_args(); print(json.dumps(profile(a.candidate,a.output,a.lengths),indent=2))
