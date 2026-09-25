"""User-authorized, read-only Docker diagnostics; never training clearance.

This separate entry point permits a bounded smoke/model diagnostic before host
stability is established. Production training guards remain unchanged. It cannot
save weights, restore an optimizer, execute source data, or serve training jobs.
"""
import argparse
import faulthandler
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baby_arcus.runtime_contract import require_container, scope

RELEASE = '9e6f8e21bc4b5e70d94d92cfcdff2872980e28c6585ebcaabad11a0fbb810520'


def validate_request(request, runtime_scope, now):
    capacity = request.get('capacity', 1.0)
    if type(capacity) not in (int, float) or not math.isfinite(capacity) or not 0 < capacity <= 1:
        raise ValueError('Invalid diagnostic capacity')
    if (request.get('schema') != 'alpha-readonly-diagnostic-v1'
            or request.get('user_authorized_readonly') is not True
            or request.get('scope') != runtime_scope
            or request.get('stage') not in ('smoke','model')
            or request.get('checkpoint_sha256') != RELEASE
            or request.get('training_updates') != 0
            or type(request.get('created_at')) not in (int,float)
            or not 0 <= now-request['created_at'] <= 900):
        raise ValueError('Invalid or expired read-only diagnostic request')
    return request


def validate_heads(heads, row):
    import torch
    from baby_arcus.body_vocabulary import mask
    masked = {}
    for name, value in heads.items():
        if name in ('body','lying','sitting'):
            allowed = torch.tensor([mask(row['senses'])], device=value.device)
            if (value.shape != allowed.shape or not bool(torch.isfinite(value[allowed]).all())
                    or not bool(torch.isneginf(value[~allowed]).all())):
                raise RuntimeError('Invalid finite logits or joint mask: '+name)
            masked[name] = int((~allowed).sum())
        elif not bool(torch.isfinite(value).all()):
            raise RuntimeError('Nonfinite integrated model output: '+name)
    return masked


def run(directory):
    require_container()
    directory = Path(directory)
    request = validate_request(json.loads((directory/'request.json').read_text()), scope(), time.time())
    if (directory/'events.jsonl').exists():
        raise ValueError('Fresh diagnostic output required')
    started = time.monotonic()
    def event(stage, **extra):
        record = {'time': time.time(), 'elapsed_seconds': time.monotonic()-started,
                  'stage': stage, 'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, **extra}
        with (directory/'events.jsonl').open('a', encoding='utf-8') as output:
            output.write(json.dumps(record)+'\n'); output.flush(); os.fsync(output.fileno())
        print(json.dumps(record), flush=True)
    fault = (directory/'faults.log').open('w')
    faulthandler.enable(fault, all_threads=True)
    def timeout(*_): raise TimeoutError('Read-only diagnostic time budget reached')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(90 if request['stage']=='smoke' else 240)
    report = {'schema':'alpha-readonly-diagnostic-result-v1','scope':scope(),'stage':request['stage'],
              'training_updates':0,'complete':False,'host_stability_established':False,
              'production_training_authorized':False}
    try:
        event('before_torch_import')
        import torch
        from baby_arcus.gpu_job_control import gpu_job
        torch.set_num_threads(1)
        event('torch_imported', torch_version=torch.__version__, cuda_version=torch.version.cuda)
        with gpu_job():
            event('before_cuda_initialization')
            if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable inside container')
            torch.cuda.init()
            # PyTorch allocator cap, not a claim of a whole-device VRAM limit.
            torch.cuda.set_per_process_memory_fraction(.25,0)
            torch.cuda.reset_peak_memory_stats()
            event('cuda_initialized', device=torch.cuda.get_device_name(0))
            if request['stage']=='smoke':
                cpu = torch.arange(4096,dtype=torch.float32).reshape(64,64)/4096
                expected = cpu@cpu.T
                event('before_small_tensor_transfer')
                x = cpu.to('cuda'); torch.cuda.synchronize()
                for index in range(3):
                    event('before_small_matmul', index=index)
                    actual = x@x.T; torch.cuda.synchronize()
                    if not torch.allclose(actual.cpu(),expected,rtol=1e-4,atol=1e-5):
                        raise RuntimeError('GPU numerical mismatch')
                    event('small_matmul_passed', index=index)
                report['numerical_checks'] = 3
            else:
                prior = json.loads((directory/'smoke.json').read_text())
                if (prior.get('complete') is not True or prior.get('stage')!='smoke'
                        or prior.get('scope')!=scope() or not 0<=time.time()-prior.get('finished_at',0)<=900):
                    raise ValueError('Fresh matching smoke result required before model diagnostic')
                from baby_arcus.shared_checkpoint import load, digest
                root = Path('/release')
                candidate = json.loads((root/'candidate.json').read_text())
                if candidate['sha256'] != RELEASE or candidate['updates'] != 37000:
                    raise ValueError('Diagnostic requires the unchanged release')
                event('before_release_hash_check')
                if digest(root/(candidate['generation']+'.pt')) != RELEASE:
                    raise ValueError('Release hash mismatch')
                event('before_cpu_model_load')
                # Explicit CPU load validates the hash/container without granting
                # general production CUDA access. Only this bounded diagnostic
                # transfers the resulting immutable model, and never an optimizer.
                model, data = load(root,candidate,'cpu')
                parameters = sum(p.numel() for p in model.parameters())
                del data
                model.eval().requires_grad_(False)
                from arcus.model import MoDEBlock
                capacity = request.get('capacity', 1.0)
                model.body.cfg.capacity = capacity
                model.core.cfg.capacity = capacity
                blocks = [module for module in model.modules() if isinstance(module, MoDEBlock)]
                if not blocks:
                    raise RuntimeError('No routing blocks found')
                # The immutable release retains its original training capacity.
                # Replace only this diagnostic instance's fixed-budget hooks.
                def diagnostic_budget_guard(block, args):
                    if block.capacity != capacity:
                        raise ValueError('Diagnostic routing budget changed')
                for block in blocks:
                    if hasattr(block, '_fixed_depth_guard'):
                        block._fixed_depth_guard.remove()
                    block.capacity = capacity
                    block._fixed_depth_guard = block.register_forward_pre_hook(diagnostic_budget_guard)
                report.update(requested_capacity=capacity, routing_blocks=len(blocks))
                event('cpu_model_loaded', parameters=parameters)
                event('before_model_transfer')
                model = model.to('cuda'); torch.cuda.synchronize()
                event('model_on_gpu')
                from arcus.tokenizer import get_tokenizer
                from baby_arcus.shared_curriculum import example
                tokenizer=get_tokenizer('o200k_base')
                observations=[]
                with torch.inference_mode():
                    for index,family in enumerate(('commands','color_reference','rest')):
                        row,_,_=example(index,'validation',family)
                        event('before_integrated_inference', family=family)
                        heads=model([row],tokenizer,requested=('activity','body','lying','sitting','rest','approach','text'))
                        torch.cuda.synchronize()
                        masked = validate_heads(heads, row)
                        sample={'family':family,'activity':int(heads['activity'][0].argmax()),
                                'routing_fraction':float(model.core.last_compute_fraction),
                                'correctly_masked_joint_logits':masked}
                        observations.append(sample); event('integrated_inference_passed', **sample)
                report.update(parameters=parameters,observations=observations,checkpoint_sha256=RELEASE)
                event('before_final_release_hash_check')
                if digest(root/(candidate['generation']+'.pt')) != RELEASE:
                    raise ValueError('Release changed during diagnostic')
                report['checkpoint_unchanged']=True
            report['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
            report['peak_cuda_reserved_bytes']=torch.cuda.max_memory_reserved()
            report['complete']=True
            event('diagnostic_passed', peak_cuda_allocated_bytes=report['peak_cuda_allocated_bytes'])
    except BaseException as exc:
        report['error']=str(exc)
        event('diagnostic_failed', error=str(exc))
        traceback.print_exc(file=fault); fault.flush(); os.fsync(fault.fileno())
        raise
    finally:
        signal.alarm(0)
        report['finished_at']=time.time(); report['seconds']=time.monotonic()-started
        with (directory/'report.json').open('w',encoding='utf-8') as output:
            json.dump(report,output,indent=2); output.flush(); os.fsync(output.fileno())
        faulthandler.disable(); fault.close()
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',default='/diagnostics')
    print(json.dumps(run(parser.parse_args().directory)))
