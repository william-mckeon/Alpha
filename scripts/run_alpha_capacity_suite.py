"""Host metadata-only launcher for a bounded Docker GPU evaluation; never trains."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import uuid
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.inspect_alpha_resources import inspect


def docker(*args):
    result=subprocess.run(['docker',*args],capture_output=True,text=True,timeout=90)
    if result.returncode:raise RuntimeError(result.stderr[-1000:])
    return result.stdout.strip()


def main():
    p=argparse.ArgumentParser();p.add_argument('--host-report',required=True);p.add_argument('--capacity',type=float,default=.8)
    a=p.parse_args()
    if not 0<a.capacity<=1:raise ValueError('Invalid capacity')
    project=Path(__file__).resolve().parents[1]
    name='alpha-eval-cap'+str(a.capacity).replace('.','')+'-'+uuid.uuid4().hex[:8]
    out=project/'runs/test2'/name;out.mkdir()
    def save(name,value):(out/name).write_text(json.dumps(value,indent=2))
    preflight=inspect(mode='job',service='learner');save('resource-preflight.json',preflight)
    if not preflight['complete']:raise RuntimeError('Resource preflight failed')
    compose=json.loads(docker('compose','--env-file','.env','-f','docker/baby-arcus/compose.alpha-three-stage.yaml','config','--format','json'))
    learner=compose['services']['learner'];env=learner['environment']
    dataset=next(v['source'] for v in learner['volumes'] if v['target']=='/dataset')
    image=docker('image','inspect','arcus-alpha-three-stage:phase2-39000','--format','{{.Id}}')
    host=json.loads(Path(a.host_report).read_text(encoding='utf-8-sig'))
    # Start only the CPU executor, never learner/playroom or training services.
    docker('compose','--env-file','.env','-f','docker/baby-arcus/compose.alpha-three-stage.yaml','up','-d','--no-deps','executor')
    networks=json.loads(docker('inspect','arcus-alpha-three-stage-executor-1','--format','{{json .NetworkSettings.Networks}}'))
    network=next(iter(networks))
    for _ in range(20):
        health=docker('inspect','arcus-alpha-three-stage-executor-1','--format','{{.State.Health.Status}}')
        if health=='healthy':break
        time.sleep(1)
    if health!='healthy':raise RuntimeError('Coding executor unhealthy')
    target='/app/runs/test2/'+name
    cmd=['create','--name',name,'--restart','no','--gpus','all','--network',network,
         '--memory','8g','--memory-swap','8g','--cpus','2','--pids-limit','128',
         '--read-only','--cap-drop','ALL','--security-opt','no-new-privileges',
         '--tmpfs','/tmp:rw,nosuid,size=512m,uid=10002,gid=10002',
         '--log-driver','json-file','--log-opt','max-size=10m','--log-opt','max-file=3']
    settings={'ALPHA_GPU_MODE':'controlled-docker','ALPHA_RUNTIME_IMAGE':image,
              'ALPHA_HOST_FINGERPRINT':host['host_fingerprint'],'ALPHA_JOB_CONTROL':'/control',
              'ALPHA_EXECUTOR_URL':'http://executor:8933','ALPHA_EXECUTOR_TOKEN':env['ALPHA_EXECUTOR_TOKEN'],
              'PYTHONUNBUFFERED':'1','PYTHONFAULTHANDLER':'1','PYTHONDONTWRITEBYTECODE':'1',
              'CUDA_LAUNCH_BLOCKING':'1','OMP_NUM_THREADS':'2'}
    for k,v in settings.items():cmd+=['-e',k+'='+v]
    cmd+=['--mount','type=volume,source=arcus-alpha-job-control,target=/control']
    mounts=[(str(out),target,False),(str(project/'runs/test2/alpha-idle-fresh-release-20260923'),'/app/runs/test2/alpha-idle-fresh-release-20260923',True),(dataset,'/dataset',True)]
    sources=['scripts/alpha_evaluation_capacity.py','scripts/evaluate_alpha_capacity_suite.py',
             'scripts/evaluate_arcus_baseline_parity.py','scripts/evaluate_alpha_coding.py',
             'scripts/evaluate_arcus_shared_curiosity.py','baby_arcus/shared_causal.py']
    import hashlib
    save('source-hashes.json',{s:hashlib.sha256((project/s).read_bytes()).hexdigest() for s in sources})
    for s in sources:mounts.append((str(project/s),'/app/'+s,True))
    for source,dest,readonly in mounts:cmd+=['--mount',f'type=bind,source={source},target={dest}'+(',readonly' if readonly else '')]
    cmd += [image,'scripts/evaluate_alpha_capacity_suite.py','--output',target,'--capacity',str(a.capacity)]
    identifier=docker(*cmd)
    save('supervisor.json',{'container':name,'id':identifier,'image':image,'capacity':a.capacity,'state':'created','training_updates':0,'deadline_seconds':7200})
    print(json.dumps({'output':str(out),'container':name}),flush=True)
    with (out/'docker.log').open('w') as log:
        docker('start',name)
        follower=subprocess.Popen(['docker','logs','--timestamps','--follow',name],stdout=log,stderr=subprocess.STDOUT)
        begin=time.monotonic()
        try:
            while True:
                state=json.loads(docker('inspect',name,'--format','{{json .State}}'))
                if not state['Running']:break
                if time.monotonic()-begin>7200:
                    docker('stop','-t','5',name);raise TimeoutError('Evaluation deadline reached')
                time.sleep(5)
        finally:
            state=json.loads(docker('inspect',name,'--format','{{json .State}}'))
            if state['Running']:docker('stop','-t','5',name)
            follower.wait(timeout=15)
            save('container-state.json',json.loads(docker('inspect',name,'--format','{{json .State}}')))
    print(json.dumps({'output':str(out),'exit_code':state['ExitCode'],'oom_killed':state['OOMKilled']}),flush=True)
    summary=json.loads((out/'supervisor.json').read_text())
    summary.update(state='complete' if state['ExitCode']==0 else 'failed',exit_code=state['ExitCode'],oom_killed=state['OOMKilled'])
    save('supervisor.json',summary)
    if state['ExitCode']!=0:raise SystemExit(1)


if __name__=='__main__':main()
