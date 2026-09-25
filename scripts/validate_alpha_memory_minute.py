"""Fixed read-only depth-1 CUDA workload for before/after memory comparison."""
import json,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def run():
    import torch
    from baby_arcus.runtime_contract import require_gpu
    from baby_arcus.gpu_job_control import gpu_job
    from baby_arcus.shared_checkpoint import load
    out=Path('/evidence'); pointer=Path('/parent/candidate.json')
    manifest=json.loads(pointer.read_text())
    if manifest.get('updates')!=39000:
        raise ValueError('This comparison requires the verified 39000-update checkpoint')
    report={'workload':'depth1-resident-language-prefill-512-v1','checkpoint':manifest,
            'duration_target_seconds':60,'training':False,'complete':False,'iterations':0}
    def write(): (out/'model-report.json').write_text(json.dumps(report,indent=2))
    write(); require_gpu(); torch.set_num_threads(2)
    with gpu_job():
        model,data=load(pointer.parent,manifest,'cuda'); del data
        if model.body.cfg.capacity!=1. or any(b.capacity!=1. for b in model.core.blocks):
            raise ValueError('Depth must be exactly one')
        model.eval(); torch.manual_seed(2101)
        ids=torch.randint(0,model.language.embedding.num_embeddings,(1,512),device='cuda')
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        start=time.monotonic(); last=0
        (out/'started.json').write_text(json.dumps({'started_at':time.time()}))
        try:
            with torch.inference_mode():
                while time.monotonic()-start<60:
                    free,total=torch.cuda.mem_get_info()
                    memory=int(Path('/sys/fs/cgroup/memory.current').read_text())
                    if free<max(3*1024**3,total*.2) or memory>6*1024**3:
                        report['stop_reason']='memory_guard'; break
                    value=model.language(model.core,ids,last_only=True)
                    if not bool(torch.isfinite(value).all()):
                        report['stop_reason']='nonfinite_output'; break
                    del value
                    report['iterations']+=1
                    elapsed=time.monotonic()-start
                    if elapsed-last>=1:
                        report.update(elapsed_seconds=elapsed,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            peak_reserved_bytes=torch.cuda.max_memory_reserved(),container_memory_bytes=memory)
                        write(); last=elapsed
                else:
                    report.update(complete=True,stop_reason='one_minute_complete')
        except Exception as exc:
            report['stop_reason']=type(exc).__name__; report['error']=str(exc); raise
        finally:
            torch.cuda.synchronize()
            report.update(elapsed_seconds=time.monotonic()-start,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            write()
    print(json.dumps(report),flush=True)

if __name__=='__main__': run()
