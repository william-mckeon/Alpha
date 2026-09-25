"""Bounded fresh-run supervisor. Stop exactly at 40K; never promote or prune."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def next_count(current):
    if type(current) is not int or not 0<=current<=40000: raise ValueError('Unexpected update count')
    if current==40000: return 0
    milestones=(3,64,1000,5000,10000,20000,30000,40000)
    return min(256,min(n for n in milestones if n>current)-current)


def supervise(config):
    from baby_arcus.shared_factory import read_config
    from baby_arcus.shared_checkpoint import digest
    from baby_arcus.language_stream import atomic_json
    from baby_arcus.process_lock import ProcessLock
    cfg=read_config(config); root=Path(cfg['root']); out=root/'run-40000';out.mkdir(exist_ok=True)
    lock=ProcessLock(root/'fresh-supervisor.lock')
    state={'target_total_updates':40000,'automatic_promotion':False}
    def update(**values):
        state.update(values,updated_at=time.time()); atomic_json(out/'status.json',state)
    def candidate(): return json.loads((root/'candidate.json').read_text())
    def evaluate(stage, initial=False):
        from scripts.evaluate_alpha_fresh import evaluation_identity,reusable
        identity=evaluation_identity(config)
        report=out/stage/'evaluation.json'
        if report.exists():
            try: saved=json.loads(report.read_text())
            except (ValueError,OSError): saved={}
            expected=json.loads((root/'initial.json').read_text()) if initial else candidate()
            if reusable(saved,expected,identity):
                return saved
            # Preserve incomplete/stale evidence; retry in a fresh directory.
        if report.parent.exists():
            report.parent.rename(out/(stage+'-previous-'+str(time.time_ns())))
        update(state='evaluation',stage=stage,candidate=candidate())
        command=[sys.executable,'scripts/evaluate_alpha_fresh.py','--config',config,'--output',str(report),'--coding']
        if initial:command.append('--initial')
        with (out/(stage+'-'+str(time.time_ns())+'.log')).open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Evaluation {stage} failed')
        saved=json.loads(report.read_text())
        if not saved.get('complete') or not saved.get('checkpoint_unchanged'):raise RuntimeError('Incomplete evaluation evidence')
        return saved
    try:
        if (root/'pause-training').exists():update(state='paused',candidate=candidate());return
        evaluate('initial', initial=True)
        if candidate()['updates'] in (3,64,1000,5000,10000,20000,30000):
            evaluate('evaluation-'+str(candidate()['updates']))
        while True:
            pointer=candidate(); current=pointer['updates']; count=next_count(current)
            if (root/'pause-training').exists(): update(state='paused',candidate=pointer); return
            if count==0:
                final=evaluate('evaluation-40000')
                initial=json.loads((out/'initial/evaluation.json').read_text())
                from scripts.report_alpha_fresh_run import build
                atomic_json(out/'report.json',build(root,initial,final))
                (root/'pause-training').touch(); update(state='complete',candidate=pointer,report=str(out/'report.json')); return
            update(state='training',stage=f'updates-{current}-{current+count}',candidate=pointer,saved_updates=current,next_updates=count)
            with (out/f'updates-{current}-{current+count}-{time.time_ns()}.log').open('w') as log:
                result=subprocess.run([sys.executable,__file__,'--config',config,'--worker','--updates',str(count)],stdout=log,stderr=subprocess.STDOUT)
            if result.returncode: raise RuntimeError(f'Training worker exited {result.returncode}')
            now=candidate()
            if digest(root/(now['generation']+'.pt'))!=now['sha256']: raise RuntimeError('Checkpoint checksum mismatch')
            if (root/'pause-training').exists():
                update(state='paused',candidate=now,saved_updates=now['updates']); return
            if not current<now['updates']<=current+count: raise RuntimeError('Worker made no progress or exceeded its budget')
            if digest(root/(now['generation']+'.pt'))!=now['sha256']: raise RuntimeError('Checkpoint checksum mismatch')
            update(state='checkpoint_verified',candidate=now)
            if now['updates'] in (3,64,1000,5000,10000,20000,30000):
                evaluate('evaluation-'+str(now['updates']))
    except Exception as exc:
        (root/'pause-training').touch(); update(state='failed',error=str(exc)); raise
    finally: lock.close()


if __name__=='__main__':
    from baby_arcus.runtime_contract import require_gpu
    require_gpu()
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--config',required=True)
    parser.add_argument('--worker',action='store_true'); parser.add_argument('--updates',type=int)
    args=parser.parse_args()
    if args.worker:
        import torch
        torch.set_num_threads(2)
        from baby_arcus.three_stage_training import train
        print(json.dumps(train(args.config,args.updates,checkpoint_every=args.updates)),flush=True)
    else: supervise(args.config)
