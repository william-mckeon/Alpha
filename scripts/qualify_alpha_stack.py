"""Host orchestration only: live four-container qualification with a tiny CPU learner."""
import argparse
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
import uuid
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.transport import Client, RemoteError


def docker(*args):
    result=subprocess.run(['docker',*args],capture_output=True,text=True,timeout=180)
    if result.returncode: raise RuntimeError(result.stderr[-3000:])
    return result.stdout.strip()


def save(path,value):
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def qualify(keep_ui=False,image='arcus-alpha-three-stage:qualification'):
    project=Path(__file__).resolve().parents[1]
    identifier='alpha-stack-'+uuid.uuid4().hex[:12]
    root=project/'runs'/'test2'/identifier; root.mkdir(parents=True)
    review=root/'review'; review.mkdir()
    target='/app/runs/test2/'+identifier
    executor='arcus-alpha-executor:qualification'
    names=[]; network=identifier; control=identifier+'-control'
    tokens={key:secrets.token_urlsafe(32) for key in ('learner','human','ingest','executor')}
    report={'fixture':True,'cpu_only':True,'root':str(root),'containers':names,'network':network,
            'control_volume':control,'images':{name:docker('image','inspect',name,'--format','{{.Id}}') for name in (image,executor)},
            'checks':{},'complete':False,'mastery_established':False}
    docker('network','create',network)
    try:
        docker('run','--rm','--network','none','--memory','2g','--cpus','1','-e','CUDA_VISIBLE_DEVICES=',
               '--mount',f'type=bind,source={root},target={target}',image,'scripts/prepare_alpha_stack_fixture.py',target)
        def start(role,command,port=None,environment=None):
            name=identifier+'-'+role; names.append(name)
            args=['run','-d','--name',name,'--network',network,'--network-alias',role,
                  '--memory','2g' if role in ('learner','playroom') else '1g','--cpus','1','--pids-limit','128']
            if port: args+=['-p',f'127.0.0.1:{port}:{port}']
            for key,value in (environment or {}).items(): args+=['-e',key+'='+value]
            if role=='executor':
                args+=['--mount','type=bind,source=/var/run/docker.sock,target=/var/run/docker.sock',executor]
            else:
                args+=['-e','CUDA_VISIBLE_DEVICES=','-e','ALPHA_JOB_CONTROL=/control',
                       '--mount',f'type=volume,source={control},target=/control',
                       '--mount',f'type=bind,source={root},target={target}',
                       '--mount',f'type=bind,source={review},target=/review'+(',readonly' if role=='learner' else ''),image]
            docker(*args,*command)
            return name
        start('executor',[],environment={'ALPHA_EXECUTOR_TOKEN':tokens['executor']})
        start('review',['-m','baby_arcus.services.data_review','--fixture','--host','0.0.0.0','--port','8932','--store','/review/staging.sqlite'],8932,
              {'ALPHA_REVIEW_TOKEN':tokens['human'],'ALPHA_INGEST_TOKEN':tokens['ingest']})
        common={'ARCUS_TEST2_TOKEN':tokens['learner'],'ALPHA_EXECUTOR_TOKEN':tokens['executor'],'ALPHA_EXECUTOR_URL':'http://executor:8933'}
        learner_name=start('learner',['-m','baby_arcus.services.shared_trainer','--config',target+'/config.json','--host','0.0.0.0','--port','8931'],8931,common)
        viewer_name=start('playroom',['-m','baby_arcus.services.test2_playroom','--config',target+'/config.json','--learner-url','http://learner:8931','--host','0.0.0.0','--port','8930'],8930,
                          dict(common,ALPHA_REVIEW_URL='http://review:8932',ALPHA_INGEST_TOKEN=tokens['ingest']))
        clients={'review':Client('http://127.0.0.1:8932',timeout=10,attempts=1),
                 'learner':Client('http://127.0.0.1:8931',tokens['learner'],timeout=120,attempts=1),
                 'viewer':Client('http://127.0.0.1:8930',timeout=30,attempts=1)}
        def wait_ready(client):
            client=Client(client.base_url,client.token,timeout=3,attempts=1)
            last=None
            deadline=time.monotonic()+60
            while time.monotonic()<deadline:
                try: return client.request('GET','/health')
                except Exception as exc: last=repr(exc); time.sleep(.25)
            raise TimeoutError('Service readiness failed: '+str(last))
        for role,client in clients.items(): report['checks'][role+'_ready']=bool(wait_ready(client)['ready'])
        snapshot=clients['viewer'].request('GET','/api/test2')
        body_id=snapshot['world']['arcus']['entity_id']
        report['checks']['training_initially_paused']=snapshot['idle_learning']['enabled'] is False
        records=json.loads((root/'source-lesson-records.json').read_text())
        if not records or any(not record['source'].startswith('fixture:') for record in records):
            raise ValueError('Only synthetic lesson qualification is allowed')
        review_client=clients['review']
        batch=review_client.request('POST','/stage',{'credential':tokens['ingest'],'records':records})['batch_id']
        report['checks']['source_lesson_staged']=len(records)==6
        decision={'credential':tokens['ingest'],'batch_id':batch,'decision':'approved','reviewer':'AUTOMATED FIXTURE'}
        try:
            review_client.request('POST','/review',decision)
            raise AssertionError('Machine credential approved data')
        except RemoteError as exc:
            if exc.status!=403: raise
        report['checks']['machine_approval_rejected']=True
        decision['credential']=tokens['human']; review_client.request('POST','/review',decision)
        source=review_client.request('POST','/sources',{'credential':tokens['ingest'],
                   'manifest':json.loads((root/'corpus-manifest.json').read_text())})['batch_id']
        review_client.request('POST','/review',dict(decision,batch_id=source))
        # Sandbox is exercised via a service-to-service call, not host execution.
        code="from baby_arcus.transport import Client; import os,json; c=Client('http://executor:8933',os.environ['ALPHA_EXECUTOR_TOKEN'],timeout=70,attempts=1); bad=c.request('POST','/execute',{'task':'positive_sum','source':'def solve(v): return sum(v)'}); good=c.request('POST','/execute',{'task':'positive_sum','source':'def solve(v): return sum(x for x in v if x>0)'}); assert not bad['passed'] and good['passed']; print(json.dumps({'bad':bad['passed'],'good':good['passed']}))"
        report['sandbox']=json.loads(docker('exec',learner_name,'python3','-c',code))
        report['checks']['executor_fail_fix_pass']=True
        plan=json.loads((root/'plan.json').read_text()); plan.update(training_enabled=True,approved_batches=[batch],approved_source_manifests=[source])
        save(root/'plan.json',plan)
        paused=clients['learner'].request('POST','/train',{'updates':3,'request_id':'fixture-paused'})
        if paused['updates_this_job']!=0: raise AssertionError('Pause ignored')
        (root/'learner'/'pause-training').unlink()
        trained=clients['learner'].request('POST','/train',{'updates':3,'request_id':'fixture-three-streams'})
        assert trained['candidate']['updates']==3
        repeated=clients['learner'].request('POST','/train',{'updates':3,'request_id':'fixture-three-streams'})
        assert repeated['already_trained'] is True
        report['checks']['three_stream_http_training']=True; report['checks']['retry_idempotent']=True
        clients['viewer'].request('POST','/api/test2/idle',{'action':'pause'})
        plan['training_enabled']=False; save(root/'plan.json',plan)
        clients['viewer'].request('POST','/api/test2/practice',{'task':'positive_sum'})
        clients['viewer'].request('POST','/api/test2/practice-stop',{})
        deadline=time.monotonic()+45
        while time.monotonic()<deadline:
            current=clients['viewer'].request('GET','/api/test2')
            if current['practice']['state']!='running': break
            time.sleep(.2)
        assert current['practice']['state']=='cancelled',current['practice']
        report['checks']['caregiver_practice_cancellation']=True
        docker('restart',viewer_name); wait_ready(clients['viewer'])
        restarted=clients['viewer'].request('GET','/api/test2')
        assert restarted['world']['arcus']['entity_id']==body_id
        assert restarted['idle_learning']['enabled'] is False
        report['checks']['body_identity_survives_restart']=True
        constraints={name:json.loads(docker('inspect',name,'--format','{{json .HostConfig}}')) for name in names}
        report['checks']['cpu_limits_and_no_gpu']=all(not c.get('DeviceRequests') and c['Memory']<=2*1024**3 and c['NanoCpus']<=1000000000 for c in constraints.values())
        report['complete']=all(report['checks'].values())
        report['updates']=3
        save(root/'ui-fixture-access.json',{'review_credential':tokens['human'],'batch_id':batch})
        return report
    finally:
        report['container_states']={}
        for name in names:
            try: report['container_states'][name]=json.loads(docker('inspect',name,'--format','{{json .State}}'))
            except Exception: pass
            try: (root/(name+'.log')).write_text(docker('logs',name),encoding='utf-8')
            except Exception: pass
        save(root/'report.json',report)
        if not keep_ui or not report['complete']:
            for name in reversed(names):
                try: docker('rm','-f',name)
                except Exception: pass
            try: docker('network','rm',network)
            except Exception: pass


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--keep-ui',action='store_true')
    parser.add_argument('--image',default='arcus-alpha-three-stage:qualification')
    args=parser.parse_args()
    print(json.dumps(qualify(args.keep_ui,args.image)))
