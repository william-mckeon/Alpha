"""Trusted, restricted Docker broker. Never imports a model or accepts shell commands."""
import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import threading
import time
import uuid
from baby_arcus.coding_curriculum import task
from baby_arcus.transport import serve

IMAGE = 'python@sha256:db3ff2e1800a8581e2c48a27c3995339d47bdf046da21c7627accd3d51053a93'
HARNESS = """import json,sys
data=json.loads(sys.stdin.read())
scope={}
exec(compile(data['source'],'solution.py','exec'),scope)
print(json.dumps([scope['solve'](case) for case in data['cases']]))
"""


def execute(task_name, source):
    from baby_arcus.developmental_tasks import TASKS
    lesson = TASKS[task_name] if task_name in TASKS else task(task_name)
    if not isinstance(source,str) or len(source.encode()) > 12000:
        raise ValueError('Source exceeds sandbox budget')
    name = 'alpha-coding-' + uuid.uuid4().hex
    command = ['docker','run','--name',name,'--rm','-i','--network','none','--log-driver','none',
               '--read-only','--cap-drop','ALL','--security-opt','no-new-privileges',
               '--pids-limit','32','--memory','256m','--cpus','0.5','--user','10002:10002',
               '--tmpfs','/tmp:rw,noexec,nosuid,size=16m','--entrypoint','python3',IMAGE,'-I','-B','-c',('import json,sys; exec(compile(json.loads(sys.stdin.read())["source"],"solution.py","exec"),{})' if lesson.get('stdout') else HARNESS)]
    payload=json.dumps({'source':source,'cases':lesson['cases']}).encode()
    with tempfile.TemporaryFile() as stdin, tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        stdin.write(payload); stdin.seek(0)
        process=subprocess.Popen(command,stdin=stdin,stdout=stdout,stderr=stderr)
        reason=None
        try:
            deadline=time.monotonic()+45
            while process.poll() is None:
                if time.monotonic() >= deadline or os.fstat(stdout.fileno()).st_size+os.fstat(stderr.fileno()).st_size > 1048576:
                    reason='timeout_or_output_limit'; break
                time.sleep(.05)
        finally:
            subprocess.run(['docker','rm','-f',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=15)
            if process.poll() is None: process.kill()
            process.wait(timeout=5)
        stdout.seek(0); stderr.seek(0)
        output=stdout.read(1048576).decode(errors='replace'); error=stderr.read(1048576).decode(errors='replace')
    try: actual=json.loads(output)
    except ValueError: actual=None
    if lesson.get('stdout'):
        actual=[int(output.strip()==lesson['expected'])];lesson={**lesson,'expected':[1]}
    return {'passed':reason is None and process.returncode==0 and isinstance(actual,list)
            and all(type(value) is int for value in actual) and actual==lesson['expected'],
            'returncode':process.returncode,'output':output[-2000:],'error':reason or error[-2000:],
            'solution_sha256':hashlib.sha256(source.encode()).hexdigest(),'executor_image':IMAGE}


class Application:
    def __init__(self): self.lock=threading.Lock()
    def __call__(self, method, path, body):
        if method=='GET' and path=='/health': return 200, {'ready':True,'image':IMAGE}
        if method!='POST' or path!='/execute': raise KeyError(path)
        if not isinstance(body,dict) or set(body)!={'task','source'}: raise ValueError('Only task and source accepted')
        if not self.lock.acquire(blocking=False): return 409, {'error':'Executor busy'}
        try: return 200, execute(body['task'],body['source'])
        finally: self.lock.release()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='127.0.0.1'); parser.add_argument('--port',type=int,default=8933)
    args=parser.parse_args(); token=os.environ['ALPHA_EXECUTOR_TOKEN']
    if len(token)<24: raise ValueError('Strong executor credential required')
    server=serve(args.host,args.port,Application(),token)
    try: server.serve_forever()
    finally: server.server_close()


if __name__=='__main__': main()
