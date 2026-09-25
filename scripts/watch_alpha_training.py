"""Host-side long-run Docker watchdog with the user-selected host RAM cutoff."""
import argparse
import ctypes
import json
import re
import subprocess
import time
from pathlib import Path
from watch_alpha_memory_minute import Memory

# Explicitly authorized September 25 for this fresh run only.
HOST_MINIMUM_FREE_BYTES = 512 * 1024**2


def watch(name, folder, max_seconds):
    if not re.fullmatch(r'alpha-fresh128m-[a-z0-9-]+',name): raise ValueError('Unexpected training container')
    if not 1<=max_seconds<=7*86400: raise ValueError('Invalid monitoring duration')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();reason=None;state={}
    def command(args): return subprocess.check_output(args,text=True,timeout=10).strip()
    try:
        with (folder/'host-samples.jsonl').open('a',buffering=1) as log:
            while True:
                state=json.loads(command(['docker','inspect',name,'--format','{{json .State}}']))
                if not state['Running']: reason='container_exited';break
                memory=Memory();memory.length=ctypes.sizeof(memory)
                if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)): raise RuntimeError('Host memory unavailable')
                total,used,free=map(int,command(['nvidia-smi','--query-gpu=memory.total,memory.used,memory.free','--format=csv,noheader,nounits']).splitlines()[0].split(','))
                log.write(json.dumps({'time':time.time(),'host_free_bytes':memory.free,'host_minimum_free_bytes':HOST_MINIMUM_FREE_BYTES,'gpu_used_mib':used,'gpu_free_mib':free})+'\n')
                if memory.free<HOST_MINIMUM_FREE_BYTES: reason='host_memory_guard';break
                if free<max(3072,total*.2): reason='gpu_memory_guard';break
                if time.monotonic()-start>max_seconds: reason='watchdog_deadline';break
                time.sleep(2)
    except Exception as exc: reason='watchdog_error:'+str(exc)
    finally:
        if reason!='container_exited': subprocess.run(['docker','kill',name],capture_output=True,timeout=15)
        try:
            state=json.loads(command(['docker','inspect',name,'--format','{{json .State}}']))
        except Exception as exc: state={'inspection_error':str(exc)}
        (folder/'watchdog.json').write_text(json.dumps({'reason':reason,'elapsed':time.monotonic()-start,'container_state':state},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('name');parser.add_argument('folder')
    parser.add_argument('--max-seconds',type=int,default=7*86400)
    args=parser.parse_args();watch(args.name,args.folder,args.max_seconds)
