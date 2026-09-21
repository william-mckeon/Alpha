"""Real GPU/HTTP regression for calls outside the camera crop, with closed eyes."""
import json,sys,tempfile,threading,time,uuid,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.services.playroom import PlayroomApplication
from baby_arcus.live_interaction_policy import LiveInteractionPolicy
from baby_arcus.body_controller_config import configuration
from baby_arcus.visual_runtime import VisualRuntime
from baby_arcus.transport import Client,serve

def main():
    project=Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as root:
        app=PlayroomApplication(root)
        app.policy=LiveInteractionPolicy(app,project/'.venv/Scripts/python.exe',workdir=project,**configuration(project))
        app.visual=VisualRuntime(app,project/'.venv/Scripts/python.exe',project)
        token=uuid.uuid4().hex;server=serve('127.0.0.1',0,app,token)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        clock=threading.Thread(target=app.run_clock,daemon=True);clock.start()
        client=Client(f'http://127.0.0.1:{server.server_port}',token,attempts=1)
        trials=[]
        try:
            for eyes in (1,0):
                with app.lock:
                    app.world.body.eyelid_openness=eyes
                    app.world.environment.placements[app.world.body.entity_id].update(x=6.28,y=3.5)
                    app.world.environment.human.update(x=1.801051636,y=1.543791582,present=True)
                command={'request_id':uuid.uuid4().hex,'source':'human','action':{'kind':'call'}}
                client.request('POST','/v1/action',command);client.request('POST','/v1/action',command)
                deadline=time.monotonic()+110
                while time.monotonic()<deadline:
                    state=client.request('GET','/v1/state')
                    if state['policy']['status'] not in ('loading','running'):break
                    time.sleep(.2)
                p=state['environment']['placements'][state['arcus']['entity_id']];h=state['environment']['human']
                distance=math.hypot(p['x']-h['x'],p['y']-h['y'])
                calls=[r for r in state['interactions'] if r['request_id']=='action-'+command['request_id']]
                passed=(state['policy']['status']=='completed' and distance<=1.1 and len(calls)==1
                        and calls[0].get('model_exposure') is True and not state['visual']['enabled'])
                trials.append({'eyes_open':bool(eyes),'distance':distance,'passed':passed,
                               'policy':state['policy'],'receipt':calls})
                assert passed,trials[-1]
            report={'passed':all(t['passed'] for t in trials),'trials':trials,
                    'hearing':'ideal symbolic localization, not waveform recognition'}
            (project/'runs/arcus_navigation_v5/hearing-qualification.json').write_text(json.dumps(report,indent=2))
            print(json.dumps(report),flush=True)
        finally:
            app.close();server.shutdown();server.server_close();clock.join(timeout=3)

if __name__=='__main__':main()
