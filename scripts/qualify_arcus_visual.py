"""Real HTTP controls and separate GPU worker on an isolated playpen identity."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import uuid
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.services.playroom import PlayroomApplication
from baby_arcus.visual_runtime import VisualRuntime
from baby_arcus.transport import Client,RemoteError,serve

def main():
    project=Path(__file__).resolve().parents[1]
    cfg=json.loads((project/'configs/baby_arcus/visual.json').read_text(encoding='utf-8'))
    with tempfile.TemporaryDirectory() as directory:
        app=PlayroomApplication(directory);token=uuid.uuid4().hex
        app.visual=VisualRuntime(app,project/'.venv/Scripts/python.exe',project)
        app.world.body.eyelid_openness=0
        app.world.environment.human.update(x=6.5,y=3.5)
        app.world.environment.placements[app.world.body.entity_id].update(x=1.,y=1.)
        server=serve('127.0.0.1',0,app,token)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        clock=threading.Thread(target=app.run_clock,daemon=True);clock.start()
        client=Client(f'http://127.0.0.1:{server.server_port}',token,timeout=5,attempts=1)
        try:
            unauthorized=False
            try:Client(client.base_url,attempts=1).request('GET','/v1/state')
            except RemoteError as exc:unauthorized=exc.status==401
            request={'request_id':uuid.uuid4().hex,'action':'start'}
            client.request('POST','/v1/visual/control',request)
            client.request('POST','/v1/visual/control',request)
            deadline=time.monotonic()+100
            while time.monotonic()<deadline:
                state=client.request('GET','/v1/state');visual=state['visual']
                if visual['status']=='error':raise AssertionError(visual['reason'])
                if visual['decisions']>=6:break
                time.sleep(.1)
            assert visual['decisions']>=6,'Visual worker did not make six decisions'
            opened=state['arcus']['eyelid_openness']>0;looked=state['arcus']['eye_yaw']>0
            assert opened,'Model did not open its eyes'
            assert looked,'Model did not orient toward the visible marker'
            client.request('POST','/v1/action',{'request_id':uuid.uuid4().hex,'source':'human',
                'action':{'kind':'eyelids','openness':0}})
            time.sleep(.3);stopped=client.request('GET','/v1/state')
            assert not stopped['visual']['enabled'] and stopped['arcus']['eyelid_openness']==0
            rows=[json.loads(line) for line in (Path(visual['session'])/'experiences.jsonl').read_text(encoding='utf-8').splitlines()]
            observations=[r for r in rows if 'observation' in r]
            assert observations and all(r['observation']['frame']['source']=='playpen' for r in observations)
            report={'passed':True,'http_authentication':unauthorized,'model_opened_eyes':opened,
                'model_looked_right':looked,'human_close_interrupted':True,'decisions':visual['decisions'],
                'identity':state['arcus']['entity_id'],'isolated_identity':True,'live_user_identity_changed':False,
                'session':visual['session'],'last_resources':visual['resources']}
            assert unauthorized
            path=project/cfg['output']/'live-qualification.json';path.write_text(json.dumps(report,indent=2),encoding='utf-8')
            print(json.dumps(report),flush=True)
        finally:
            app.close();server.shutdown();server.server_close();clock.join(timeout=3)

if __name__=='__main__':main()
