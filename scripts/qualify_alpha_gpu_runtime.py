"""Bounded disposable GPU smoke check; no checkpoint or training data is loaded."""
import argparse
import json
from pathlib import Path
import signal
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baby_arcus.runtime_contract import require_evidence, scope


def qualify(output):
    output = Path(output)
    if output.exists(): raise ValueError('Use a new output file')
    # Only host clearance is required here: requiring a GPU receipt would be circular.
    require_evidence('ALPHA_HOST_QUALIFICATION', 'alpha-host-qualification-v1')
    report = {'schema':'alpha-gpu-qualification-v1','created_at':time.time(),
              'scope':scope(),'complete':False,'checks':{},'model_loaded':False}
    def timeout(*_): raise TimeoutError('GPU smoke check exceeded 45 seconds')
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(45)
    try:
        from baby_arcus.gpu_job_control import gpu_job
        with gpu_job():
            import torch
            torch.set_num_threads(1)
            if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
            report['torch'] = torch.__version__
            report['cuda'] = torch.version.cuda
            report['device'] = torch.cuda.get_device_name(0)
            torch.cuda.reset_peak_memory_stats()
            x = torch.arange(4096, dtype=torch.float32).reshape(64,64) / 4096
            expected = x @ x.T
            actual = x.to('cuda')
            for _ in range(3):
                result = actual @ actual.T
                torch.cuda.synchronize()
                if not torch.allclose(result.cpu(), expected, rtol=1e-4, atol=1e-5):
                    raise RuntimeError('GPU numerical mismatch')
            report['checks'] = {'numerics':True,'synchronized':True,'bounded_allocation':
                                torch.cuda.max_memory_allocated() < 64*1024*1024}
            report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            report['complete'] = all(report['checks'].values())
    except Exception as exc:
        report['error'] = str(exc)
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous)
        report['seconds'] = time.time()-report['created_at']
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    print(json.dumps(qualify(parser.parse_args().output)))
