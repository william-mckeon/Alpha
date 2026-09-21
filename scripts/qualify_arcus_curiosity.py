"""Authenticated live GPU approach + symbolic learned toy selection."""
import json,sys,tempfile,threading,time,uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.services.playroom import PlayroomApplication
from baby_arcus.live_interaction_policy import LiveInteractionPolicy
from baby_arcus.body_controller_config import configuration
from baby_arcus.curiosity_runtime import CuriosityRuntime
from baby_arcus.transport import Client,serve,RemoteError

def main():
    project=Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as root:
        app=PlayroomApplication(root)
        app.policy=LiveInteractionPolicy(app,project/'.venv/Scripts/python.exe',workdir=project,**configuration(project))
        app.curiosity=CuriosityRuntime(app,project,require_live=False)
        token=uuid.uuid4().hex;server=serve('127.0.0.1',0,app,token)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        clock=threading.Thread(target=app.run_clock,daemon=True);clock.start()
        client=Client(f'http://127.0.0.1:{server.server_port}',token,attempts=1)
        def control(action):return client.request('POST','/v1/curiosity/control',{'request_id':uuid.uuid4().hex,'action':action})
        try:
            auth=False
            try:Client(client.base_url,attempts=1).request('GET','/v1/state')
            except RemoteError as exc:auth=exc.status==401
            control('add_toys');control('start')
            deadline=time.monotonic()+160
            while time.monotonic()<deadline:
                state=client.request('GET','/v1/state')
                if not state['curiosity']['enabled']:break
                time.sleep(.2)
            objects=state['environment']['objects'];status=state['curiosity']
            success=(status['status']=='completed' and status['discoveries']==3 and
                     all(o['visits']==1 and o['discovered'] for o in objects.values()))
            assert success,{'status':status,'objects':objects,'policy':state['policy']}
            control('start');time.sleep(.5)
            no_repeat=all(o['visits']==1 for o in app.world.environment.objects.values())
            # Reset fixture then test a real caregiver interruption during approach.
            client.request('POST','/v1/action',{'request_id':uuid.uuid4().hex,'source':'human','action':{'kind':'reset'}})
            control('add_toys');control('start');time.sleep(.3)
            client.request('POST','/v1/action',{'request_id':uuid.uuid4().hex,'source':'human','action':{'kind':'eyelids','openness':0}})
            interrupted=not app.curiosity.info['enabled'] and app.policy.info['status'] not in ('loading','running')
            report={'passed':success and auth and no_repeat and interrupted,'authentication':auth,
                    'all_three_discovered':success,'no_repeat_on_restart':no_repeat,'human_override':interrupted,
                    'checkpoint_sha256':status['checkpoint_sha256'],'objects':objects,'status':status,
                    'scope':'symbolic toy selection with existing learned body approach; not pixel recognition'}
            (project/'runs/arcus_curiosity_v1/live-qualification.json').write_text(json.dumps(report,indent=2))
            print(json.dumps(report),flush=True)
            if not report['passed']:raise SystemExit(1)
        finally:app.close();server.shutdown();server.server_close();clock.join(timeout=3)

if __name__=='__main__':main()
