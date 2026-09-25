"""Host watchdog; kills only its explicitly named diagnostic container."""
import ctypes,json,subprocess,time,argparse,re
from pathlib import Path

HOST_MINIMUM_FREE_BYTES = 2 * 1024**3

class Memory(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in
        ('total','free','page_total','page_free','virtual_total','virtual_free','extended')]

def watch(name,folder):
    if name!='alpha-memory-before' and not re.fullmatch(r'alpha-efficiency-[a-z0-9-]+',name):
        raise ValueError('Unexpected diagnostic container')
    folder=Path(folder); start=time.monotonic(); reason=None; samples=[]; exitcode=None
    def command(args,timeout=5):
        return subprocess.check_output(args,text=True,timeout=timeout).strip()
    try:
        while True:
            state=json.loads(command(['docker','inspect',name,'--format','{{json .State}}']))
            if not state['Running']:
                exitcode=state['ExitCode']; reason='container_exited'; break
            mem=Memory(); mem.length=ctypes.sizeof(mem)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)): raise RuntimeError('Host memory unavailable')
            rows=command(['nvidia-smi','--query-gpu=memory.total,memory.used,memory.free','--format=csv,noheader,nounits'])
            total,used,free=map(int,rows.splitlines()[0].split(','))
            sample={'elapsed':time.monotonic()-start,'gpu_total_mib':total,'gpu_used_mib':used,
                    'gpu_free_mib':free,'host_free_bytes':mem.free,'host_total_bytes':mem.total}
            samples.append(sample)
            with (folder/'host-samples.jsonl').open('a') as f:f.write(json.dumps(sample)+'\n')
            if free<max(3072,total*.2):
                reason='gpu_memory_guard'; break
            if mem.free<HOST_MINIMUM_FREE_BYTES:
                reason='host_memory_guard'; break
            active=folder/'started.json'
            if active.exists() and time.time()-json.loads(active.read_text())['started_at']>65:
                reason='active_time_guard'; break
            if time.monotonic()-start>360:
                reason='overall_timeout'; break
            time.sleep(.5)
    except Exception as exc:
        reason='watchdog_error:'+str(exc)
    finally:
        if reason!='container_exited':
            subprocess.run(['docker','kill',name],capture_output=True,timeout=10)
        result={'reason':reason,'container_exit_code':exitcode,'samples':len(samples),
                'host_minimum_free_bytes':HOST_MINIMUM_FREE_BYTES,
                'peak_gpu_used_mib':max((r['gpu_used_mib'] for r in samples),default=None),
                'minimum_host_free_bytes':min((r['host_free_bytes'] for r in samples),default=None)}
        (folder/'watchdog.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('folder');a=p.parse_args();watch(a.name,a.folder)
