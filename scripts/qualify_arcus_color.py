"""Live authenticated color lesson with a real worker, isolated from the saved dragon."""
import json,sys,tempfile,threading,time,uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.services.playroom import PlayroomApplication
from baby_arcus.visual_runtime import VisualRuntime
from baby_arcus.transport import Client,RemoteError,serve

def main():
    project=Path(__file__).resolve().parents[1]
    cfg=json.loads((project/'configs/baby_arcus/object_perception.json').read_text())
    qualified=json.loads((project/cfg['output']/'qualification.json').read_text())['passed']
    with tempfile.TemporaryDirectory() as directory:
        app=PlayroomApplication(directory);app.visual=VisualRuntime(app,project/'.venv/Scripts/python.exe',project)
        token=uuid.uuid4().hex;server=serve('127.0.0.1',0,app,token)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        clock=threading.Thread(target=app.run_clock,daemon=True);clock.start()
        client=Client(f'http://127.0.0.1:{server.server_port}',token,timeout=10,attempts=1)
        def action(value):return client.request('POST','/v1/action',{'request_id':uuid.uuid4().hex,'source':'human','action':value})
        try:
            auth=False
            try:Client(client.base_url,attempts=1).request('GET','/v1/state')
            except RemoteError as exc:auth=exc.status==401
            action({'kind':'color_lesson','colors':{'floor':'#aaaabb','wall':'#aabbcc','rug':'#bbbbcc'},'balls':['#cc3333','#3366cc']})
            before=client.request('GET','/v1/state');report={'authentication':auth,'offline_qualified':qualified}
            try:
                client.request('POST','/v1/visual/control',{'request_id':uuid.uuid4().hex,'action':'perceive'})
                if not qualified:raise AssertionError('Unqualified model was enabled')
                deadline=time.monotonic()+120
                while time.monotonic()<deadline:
                    state=client.request('GET','/v1/state')
                    if state['visual'].get('observation') or state['visual']['status']=='error':break
                    time.sleep(.2)
                assert state['visual'].get('observation'),state['visual']
                report['observation_received']=True
                report['no_movement']=before['environment']['placements']==state['environment']['placements']
                action({'kind':'eyelids','openness':0})
                report['eye_closure_stops']=not app.visual.info['enabled']
                report['passed']=all(report[k] for k in ('authentication','observation_received','no_movement','eye_closure_stops'))
            except RemoteError as exc:
                if qualified:raise
                report.update(candidate_rejected=exc.status in (400,409),passed=False,
                              engineering_gate_passed=auth and exc.status in (400,409))
            (project/cfg['output']/'live-service-qualification.json').write_text(json.dumps(report,indent=2))
            print(json.dumps(report),flush=True)
            if qualified and not report['passed']:raise SystemExit(1)
        finally:app.close();server.shutdown();server.server_close();clock.join(timeout=3)

if __name__=='__main__':main()
