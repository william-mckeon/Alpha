"""Durable, bounded GPU continuation to exactly 39,000 total updates; no promotion."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.language_stream import atomic_json

CONFIG='configs/baby_arcus/alpha_three_stage_learner.container.json'
ROOT=Path('runs/test2/alpha-three-stage')
OUT=ROOT/'phase2-39000'
TARGET=39000
MILESTONES=(37013,37130,37650,38300,39000)


def next_count(current):
    if type(current) is not int or not 37000 <= current <= TARGET:
        raise ValueError('Unexpected update count; refusing to overrun the authorized target')
    if current==TARGET:return 0
    return min(64,min(x for x in MILESTONES if x>current)-current)


def worker(operation, count, stage):
    from baby_arcus.runtime_contract import require_gpu
    require_gpu()
    import torch
    torch.set_num_threads(2)
    if operation=='train':
        from baby_arcus.three_stage_training import train
        print(json.dumps(train(CONFIG,count,checkpoint_every=count)),flush=True)
    else:
        from baby_arcus.gpu_job_control import gpu_job
        with gpu_job():
            if operation=='evaluate':
                from scripts.evaluate_arcus_baseline_parity import evaluate
                result=evaluate(CONFIG,'validation',2,12,report_directory=OUT/stage)
                print(json.dumps(result),flush=True)
            else:
                from scripts.evaluate_alpha_three_stage import main
                sys.argv=['evaluate_alpha_three_stage.py','--config',CONFIG,'--output',str(OUT/'final-comparison'),'--resume']
                main()


def supervise():
    from baby_arcus.process_lock import ProcessLock
    from baby_arcus.shared_checkpoint import digest
    OUT.mkdir(parents=True,exist_ok=True)
    lock=ProcessLock(ROOT/'phase2-supervisor.lock')
    def candidate():return json.loads((ROOT/'candidate.json').read_text())
    state={'target_updates':TARGET,'started_at':time.time(),'state':'starting','automatic_promotion':False}
    path=OUT/'status.json'
    def status(**values):
        state.update(values,updated_at=time.time(),candidate=candidate())
        atomic_json(path,state)
    def run(operation,count=0,stage=''):
        log=OUT/(stage+'.log')
        status(state=operation,stage=stage,log=str(log))
        with log.open('a',buffering=1) as stream:
            result=subprocess.run([sys.executable,__file__,'--operation',operation,'--count',str(count),'--stage',stage],stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{stage} exited {result.returncode}; inspect {log}')
    try:
        current=candidate()['updates']; next_count(current)
        if not (OUT/'initial-evaluation-complete.json').exists():
            if current!=37000:raise ValueError('Missing frozen initial evaluation')
            run('evaluate',stage='initial-37000')
            atomic_json(OUT/'initial-evaluation-complete.json',candidate())
        while current<TARGET:
            if (ROOT/'pause-training').exists():
                status(state='paused',reason='Explicit pause flag');return
            count=next_count(current)
            run('train',count,stage=f'train-{current}-{current+count}')
            now=candidate()
            if not current<=now['updates']<=current+count:raise ValueError('Unexpected checkpoint advancement')
            if digest(ROOT/(now['generation']+'.pt'))!=now['sha256']:raise ValueError('Saved checkpoint hash mismatch')
            if now['updates']==current:raise RuntimeError('Training made no progress; no retry loop')
            current=now['updates']; status(state='checkpoint_verified')
            if current in (37130,37650,38300,39000):run('evaluate',stage=f'evaluation-{current}')
        (ROOT/'pause-training').touch()
        status(state='final_evaluation')
        run('compare',stage='final-comparison')
        status(state='complete',report=str(OUT/'final-comparison/report.json'))
    except Exception as exc:
        (ROOT/'pause-training').touch()
        status(state='failed',error=str(exc));raise
    finally:lock.close()


if __name__=='__main__':
    from baby_arcus.runtime_contract import require_container
    require_container()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operation',choices=['supervise','train','evaluate','compare'],default='supervise')
    parser.add_argument('--count',type=int,default=0);parser.add_argument('--stage',default='')
    args=parser.parse_args()
    if args.operation=='supervise':supervise()
    else:worker(args.operation,args.count,args.stage)
