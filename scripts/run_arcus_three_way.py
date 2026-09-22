"""Finish .25 evaluation, then matched 1.0 training, then a three-way report."""
import json
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baby_arcus.language_stream import atomic_json

ROOT = Path('runs/test2/three-way-8204')
QUARTER = Path('runs/test2/depth025-control-seed-2101')
FULL = Path('runs/test2/depth100-seed-2101')

def main(wait_pid):
    import psutil
    ROOT.mkdir(parents=True, exist_ok=True)
    def stage(name):
        atomic_json(ROOT/'status.json', {'stage': name, 'updated_at': time.time(), 'pid': __import__('os').getpid()})
    def run(name, script, args):
        stage(name)
        with (ROOT/(name+'.log')).open('a') as log:
            subprocess.run([sys.executable, 'scripts/'+script, *args], stdout=log, stderr=subprocess.STDOUT, check=True)
    def evaluate(name, root, config, extra=()):
        manifest=json.loads((root/('active.json' if extra else 'candidate.json')).read_text())
        destination=ROOT/'original' if extra else root
        path=destination/f'baseline-validation-{manifest["generation"]}.json'
        if not path.exists():
            run(name, 'evaluate_arcus_baseline_parity.py', ['--config',config,*extra])
        report=json.loads(path.read_text())
        if not report.get('complete') or not report.get('checkpoint_unchanged'):
            raise RuntimeError('Incomplete evaluation: '+str(path))
        return report
    try:
        stage('waiting_for_quarter_training')
        try: psutil.Process(wait_pid).wait()
        except psutil.NoSuchProcess: pass
        if json.loads((QUARTER/'candidate.json').read_text())['updates'] != 8204:
            raise RuntimeError('Quarter model did not finish at the agreed 8204 updates')
        quarter=evaluate('evaluate-quarter',QUARTER,'configs/baby_arcus/test2.depth025-sustained.json')
        while True:
            count=json.loads((FULL/'candidate.json').read_text())['updates']
            if count == 8204: break
            if count > 8204: raise RuntimeError('Full-depth update count exceeded comparison target')
            run('train-full-after-'+str(count),'train_arcus_to_baseline.py',
                ['--config','configs/baby_arcus/test2.depth100-sustained.json','--updates',str(min(4096,8204-count))])
            if json.loads((FULL/'candidate.json').read_text())['updates'] <= count:
                raise RuntimeError('Full-depth training made no committed progress; check pause-training before resuming')
        full=evaluate('evaluate-full',FULL,'configs/baby_arcus/test2.depth100-sustained.json')
        original=evaluate('evaluate-original',Path('runs/arcus_shared_continuity025_v4'),'configs/baby_arcus/shared.json',
                          ('--baseline-output',str(ROOT/'original')))
        atomic_json(ROOT/'report.json',{'original':original,'fresh025':quarter,'fresh100':full,
            'limitations':['Single seed; preliminary validation cohorts, not original-sized confirmation or mastery.',
                          'Fresh arms share update counts and schedule; sampled motor experiences differ.',
                          'Original model has different training history; it is a reference, not an equal-compute control.']})
        stage('report_ready')
    except Exception as exc:
        atomic_json(ROOT/'status.json',{'stage':'error','error':str(exc),'updated_at':time.time()})
        raise

if __name__ == '__main__':
    main(int(sys.argv[1]))
