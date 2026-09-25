"""Host-only Docker supervisor with durable logs and no automatic restart/reboot."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.inspect_alpha_resources import inspect

RELEASE='9e6f8e21bc4b5e70d94d92cfcdff2872980e28c6585ebcaabad11a0fbb810520'


def docker(*args):
    result=subprocess.run(['docker',*args],capture_output=True,text=True,timeout=45)
    if result.returncode: raise RuntimeError(result.stderr[-2000:])
    return result.stdout.strip()


def save(path, data):
    with path.open('w',encoding='utf-8') as output:
        json.dump(data,output,indent=2); output.flush(); os.fsync(output.fileno())


def run(image,stage,host_report,smoke=None,requested_capacity=1.0):
    project=Path(__file__).resolve().parents[1]
    identifier='alpha-readonly-'+stage+'-'+uuid.uuid4().hex[:10]
    output=project/'runs'/'diagnostics'/identifier; output.mkdir(parents=True)
    facts=json.loads(Path(host_report).read_text(encoding='utf-8-sig'))
    capacity=inspect(mode='job',service='gpu-probe' if stage=='smoke' else 'learner')
    save(output/'resource-preflight.json',capacity)
    if not capacity['complete']: raise RuntimeError('Resource or competing GPU check failed')
    image_id=docker('image','inspect',image,'--format','{{.Id}}')
    request={'schema':'alpha-readonly-diagnostic-v1','user_authorized_readonly':True,
             'created_at':time.time(),'scope':{'image_id':image_id,'host_fingerprint':facts['host_fingerprint']},
             'stage':stage,'training_updates':0,'checkpoint_sha256':RELEASE,'capacity':requested_capacity}
    save(output/'request.json',request)
    if stage=='model':
        if not smoke: raise ValueError('Smoke result required')
        prior=json.loads(Path(smoke).read_text())
        if (not prior.get('complete') or prior.get('scope')!=request['scope']
                or prior.get('stage')!='smoke' or not 0<=time.time()-prior.get('finished_at',0)<=900):
            raise ValueError('Fresh, successful matching smoke report required')
        shutil.copyfile(smoke,output/'smoke.json')
    memory='2g' if stage=='smoke' else '8g'
    args=['create','--name',identifier,'--restart','no','--gpus','all','--network','none',
          '--read-only','--cap-drop','ALL','--security-opt','no-new-privileges',
          '--memory',memory,'--memory-swap',memory,'--cpus','1','--pids-limit','128',
          '--log-driver','json-file','--log-opt','max-size=10m','--log-opt','max-file=3',
          '--tmpfs','/tmp:rw,nosuid,size=128m,uid=10002,gid=10002',
          '--mount',f'type=bind,source={output},target=/diagnostics',
          '--mount','type=volume,source=arcus-alpha-job-control,target=/control']
    for key,value in {'PYTHONUNBUFFERED':'1','PYTHONFAULTHANDLER':'1','PYTHONDONTWRITEBYTECODE':'1',
                      'CUDA_LAUNCH_BLOCKING':'1','TORCH_SHOW_CPP_STACKTRACES':'1','OMP_NUM_THREADS':'1',
                      'ALPHA_JOB_CONTROL':'/control','ALPHA_RUNTIME_IMAGE':image_id,
                      'ALPHA_HOST_FINGERPRINT':facts['host_fingerprint']}.items(): args+=['-e',key+'='+value]
    if stage=='model':
        args+=['--mount',f'type=bind,source={project / "runs/test2/alpha-idle-fresh-release-20260923"},target=/release,readonly']
    args += ['--mount',f'type=bind,source={project / "scripts/diagnose_alpha_container.py"},target=/app/scripts/diagnose_alpha_container.py,readonly']
    args += [image_id,'scripts/diagnose_alpha_container.py']
    container=docker(*args)
    save(output/'supervisor.json',{'container':container,'name':identifier,'stage':stage,'state':'created',
         'image_id':image_id,'restart_policy':'no','training_updates':0})
    process=None
    with (output/'docker.log').open('w',encoding='utf-8') as log:
        try:
            docker('start',container)
            process=subprocess.Popen(['docker','logs','--timestamps','--follow',container],stdout=log,stderr=subprocess.STDOUT)
            deadline=time.monotonic()+(110 if stage=='smoke' else 270)
            while True:
                state=json.loads(docker('inspect',container,'--format','{{json .State}}'))
                if not state['Running']: break
                if time.monotonic()>=deadline:
                    docker('stop','-t','3',container)
                    raise TimeoutError('Diagnostic supervisor deadline reached')
                time.sleep(1)
        finally:
            state=json.loads(docker('inspect',container,'--format','{{json .State}}'))
            if state['Running']:
                docker('stop','-t','3',container)
                state=json.loads(docker('inspect',container,'--format','{{json .State}}'))
            if process:
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)
            log.flush(); os.fsync(log.fileno())
            save(output/'container-state.json',state)
            limits=json.loads(docker('inspect',container,'--format','{{json .HostConfig}}'))
            save(output/'container-limits.json',{k:limits[k] for k in ('Memory','MemorySwap','NanoCpus','PidsLimit','ReadonlyRootfs','NetworkMode','RestartPolicy','LogConfig')})
    result={'directory':str(output),'container':identifier,'exit_code':state['ExitCode'],
            'oom_killed':state['OOMKilled'],'report':str(output/'report.json')}
    print(json.dumps(result),flush=True)
    if state['ExitCode']!=0: raise RuntimeError('Diagnostic failed; inspect the saved logs')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',required=True)
    parser.add_argument('--stage',choices=['smoke','model'],required=True)
    parser.add_argument('--host-report',required=True)
    parser.add_argument('--smoke-report')
    parser.add_argument('--capacity',type=float,default=1.0)
    args=parser.parse_args()
    run(args.image,args.stage,args.host_report,args.smoke_report,args.capacity)
