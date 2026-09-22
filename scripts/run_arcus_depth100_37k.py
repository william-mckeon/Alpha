"""Continue the existing full-depth learner to exactly 37,000 total updates."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baby_arcus.language_stream import atomic_json
from baby_arcus.process_lock import ProcessLock

ROOT = Path('runs/test2/depth100-seed-2101')
OUTPUT = Path('runs/test2/depth100-37000')
CONFIG = 'configs/baby_arcus/test2.depth100-sustained.json'
TARGET = 37000


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = ProcessLock(OUTPUT/'supervisor.lock')
    def status(stage, **extra):
        atomic_json(OUTPUT/'status.json', dict(stage=stage, pid=os.getpid(), target=TARGET,
                                              updated_at=time.time(), **extra))
    def run(script, args, name):
        with (OUTPUT/(name+'.log')).open('a', encoding='utf-8') as log:
            subprocess.run([sys.executable, 'scripts/'+script, *args],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    try:
        while True:
            candidate = json.loads((ROOT/'candidate.json').read_text())
            count = candidate['updates']
            if candidate['depth_capacity'] != 1.0 or count > TARGET:
                raise ValueError('Unexpected checkpoint capacity or update count')
            if count == TARGET:
                break
            if (ROOT/'pause-training').exists():
                status('paused', candidate=candidate)
                return
            status('training', candidate=candidate)
            run('train_arcus_to_baseline.py', ['--config',CONFIG,'--updates',str(min(4096,TARGET-count))],
                'training-after-'+str(count))
            if json.loads((ROOT/'candidate.json').read_text())['updates'] <= count:
                raise RuntimeError('Training made no committed progress; check pause flag and logs')
        status('evaluating', candidate=candidate)
        report = ROOT/f'baseline-validation-{candidate["generation"]}.json'
        if not report.exists():
            run('evaluate_arcus_baseline_parity.py', ['--config',CONFIG], 'evaluation')
        result = json.loads(report.read_text())
        if not result.get('complete') or not result.get('checkpoint_unchanged'):
            raise RuntimeError('Evaluation incomplete or checkpoint integrity failed')
        atomic_json(OUTPUT/'report.json', result)
        status('report_ready', candidate=candidate, report=str(report))
    except Exception as exc:
        status('error', error=str(exc))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
