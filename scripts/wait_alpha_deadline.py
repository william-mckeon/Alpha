"""Deadline-only controller: no RAM or GPU memory termination rules."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import time


def deadline_timestamp(value):
    value=datetime.fromisoformat(value)
    if value.tzinfo is None:raise ValueError('Deadline requires an explicit UTC offset')
    return value.timestamp()


def wait(name, root, folder, deadline):
    if not re.fullmatch(r'alpha-tool-correction-[a-z0-9-]+',name):raise ValueError('Invalid container')
    root,folder=Path(root),Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    pause=root/'pause-training'
    requested=False
    while True:
        now=time.time()
        # Give the current chunk time to save before the absolute cutoff.
        if now>=deadline-300 and not requested:
            pause.touch();requested=True
        if now>=deadline:
            pause.touch()
            result=subprocess.run(['docker','kill',name],capture_output=True,text=True,timeout=20)
            (folder/'deadline.json').write_text(json.dumps({'reason':'deadline_reached',
                'deadline':deadline,'pause_requested':True,'kill_returncode':result.returncode}))
            return
        try:
            state=json.loads(subprocess.check_output(['docker','inspect',name,'--format','{{json .State}}'],
                                                    text=True,timeout=10))
            if not state['Running']:
                (folder/'deadline.json').write_text(json.dumps({'reason':'container_exited',
                    'deadline':deadline,'pause_requested':requested,'container_state':state}))
                return
        except (subprocess.SubprocessError,ValueError):
            # Keep the deadline active across transient Docker API failures.
            pass
        time.sleep(min(2,max(.1,deadline-time.time())))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('root');p.add_argument('folder');p.add_argument('deadline')
    a=p.parse_args();wait(a.name,a.root,a.folder,deadline_timestamp(a.deadline))
