"""User-started stage controller; retain bounded workers and stop on any failure."""
import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arcus3.config import read
from baby_arcus.language_stream import atomic_json
from scripts.pause_arcus3_training import request

def may_continue(root, session, result):
    return (not (root/'pause-training').exists()
            and not (session/'pause-training').exists()
            and result['reason'] in ('window_or_user_pause', 'insufficient_deadline_margin')
            and datetime.now(timezone.utc) >= datetime.fromisoformat(read(session/'session-policy.json')['stop_at'])-timedelta(minutes=10))

def run(a):
    root=Path(a.root)
    root.mkdir(exist_ok=False, parents=True)
    for index in range(10000):
        if (root/'pause-training').exists(): break
        session=Path('runs/arcus3')/f'adaptation-continuous-{root.name}-{index:04d}'
        end=datetime.now(timezone.utc)+timedelta(hours=23)
        atomic_json(root/'session.json', {'active_run':str(session), 'mode':'controller', 'deadline':None})
        cmd=[sys.executable,'scripts/run_arcus3_phase8_session.py','--root',str(session),'--stop-at',end.isoformat()]
        for name in ('data','teacher','converted','qualification_report'):
            cmd += ['--'+name.replace('_','-'),getattr(a,name)]
        process=subprocess.Popen(cmd)
        while process.poll() is None:
            if (root/'pause-training').exists() and (session/'session.json').exists() and not (session/'pause-training').exists():
                request(session)
            time.sleep(2)
        if process.returncode:
            atomic_json(root/'session-result.json',{'stopped':True,'reason':'worker_failure','returncode':process.returncode})
            return
        result=read(session/'session-result.json')
        if not may_continue(root,session,result):
            atomic_json(root/'session-result.json',result)
            return

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','data','teacher','converted','qualification-report'):p.add_argument('--'+name,required=True)
    run(p.parse_args())
