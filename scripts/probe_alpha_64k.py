"""Isolated 64K FP32 inference probe; does not qualify a production context stage."""
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from arcus.backbone import build_rope_cache
    from baby_arcus.runtime_contract import require_gpu
    from baby_arcus.gpu_job_control import gpu_job
    from baby_arcus.shared_checkpoint import load,digest
    require_gpu();torch.set_num_threads(2)
    # Allocator ceiling is additional to the external host/GPU watchdog.
    torch.cuda.set_per_process_memory_fraction(.75)
    root=Path('/parent');output=Path('/evidence/probe.json')
    manifest=json.loads((root/'candidate.json').read_text())
    assert manifest['updates']==39000 and manifest['depth_capacity']==1.
    report={'candidate':manifest,'training':False,'precision':'float32','batch_size':1,
            'dispatch':'padded','attention':'efficient SDPA only; math fallback disabled',
            'allocator_fraction':.75,'requested_tokens':65536,'stages':[],
            'complete':False,'capability_established':False}
    def write():output.write_text(json.dumps(report,indent=2))
    write()
    with gpu_job():
        try:
            model,data=load(root,manifest,'cuda');del data
            model.eval();torch.manual_seed(2101)
            for length in (8192,16384,32768,65536):
                free,total=torch.cuda.mem_get_info()
                if free<max(3*1024**3,total*.2):raise RuntimeError('GPU headroom exhausted')
                report['active_tokens']=length;write()
                # Diagnostic-only RoPE extension, never saved as a checkpoint.
                cos,sin=build_rope_cache(model.core.cfg.head_dim,length,model.core.cfg.rope_theta,device='cuda')
                model.core.rope_cos=cos;model.core.rope_sin=sin
                model.core.cfg.max_seq_len=length
                torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
                started=time.perf_counter()
                with torch.inference_mode(),sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                    ids=torch.randint(0,model.language.embedding.num_embeddings,(1,length),device='cuda')
                    value=model.language(model.core,ids,last_only=True)
                    finite=bool(value.isfinite().all())
                torch.cuda.synchronize()
                report['stages'].append({'tokens':length,'seconds':time.perf_counter()-started,
                    'finite':finite,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
                del ids,value
                write()
                if not finite:raise RuntimeError('Nonfinite output')
            report['complete']=True
        except Exception as exc:
            report.update(error_type=type(exc).__name__,error=str(exc))
        finally:
            report['checkpoint_unchanged']=digest(root/(manifest['generation']+'.pt'))==manifest['sha256']
            write()
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
