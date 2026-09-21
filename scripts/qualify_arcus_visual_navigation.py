"""Real authenticated HTTP host and separate GPU model; isolated saved identity."""
import json
from pathlib import Path
import sys,tempfile,threading,time,uuid
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.services.playroom import PlayroomApplication
from baby_arcus.visual_runtime import VisualRuntime
from baby_arcus.transport import Client,RemoteError,serve
from baby_arcus.visual_navigation_environment import distance
from baby_arcus.visual_replay import load,ReplayIndex
from baby_arcus.live_interaction_policy import LiveInteractionPolicy
from baby_arcus.body_controller_config import configuration
from baby_arcus.body_dynamics import pose

def main():
    project=Path(__file__).resolve().parents[1]
    cfg=json.loads((project/'configs/baby_arcus/visual_navigation.json').read_text())
    results=[]
    with tempfile.TemporaryDirectory() as directory:
        app=PlayroomApplication(directory);token=uuid.uuid4().hex
        app.policy=LiveInteractionPolicy(app,project/'.venv/Scripts/python.exe',workdir=project,**configuration(project))
        app.visual=VisualRuntime(app,project/'.venv/Scripts/python.exe',project,'configs/baby_arcus/visual_navigation.json')
        server=serve('127.0.0.1',0,app,token)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        clock=threading.Thread(target=app.run_clock,daemon=True);clock.start()
        client=Client(f'http://127.0.0.1:{server.server_port}',token,timeout=10,attempts=1)
        try:
            authenticated=False
            try:Client(client.base_url,attempts=1).request('GET','/v1/state')
            except RemoteError as exc:authenticated=exc.status==401
            for trial,(x,y) in enumerate(((7.,3.5),(3.,3.5),(5.,1.7),(5.,5.3))):
                with app.lock:
                    app.world.environment.placements[app.world.body.entity_id].update(x=5.,y=3.5)
                    app.world.environment.human.update(x=x,y=y,present=True,name='You' if trial%2==0 else 'Your wife')
                    app.world.body.eyelid_openness=0
                    app.world.view.epoch+=1
                request={'request_id':uuid.uuid4().hex,'action':'navigate'}
                client.request('POST','/v1/visual/control',request);client.request('POST','/v1/visual/control',request)
                deadline=time.monotonic()+100
                while time.monotonic()<deadline:
                    state=client.request('GET','/v1/state');visual=state['visual']
                    if visual['status'] not in ('running','loading'):break
                    time.sleep(.1)
                app.visual.thread.join(timeout=5)
                visual=app.visual.snapshot()
                assert visual['status']=='completed',visual
                rows=load([Path(visual['session'])/'experiences.jsonl'])
                success=visual.get('outcome')=='model_reports_arrived' and distance(app.world)<=1.4
                results.append({'target':[x,y],'distance':distance(app.world),'success':success,'decisions':visual['decisions'],
                                'session':visual['session'],'transitions':len(rows),'status':visual})
                assert rows and len(rows)==visual['decisions'],'Missing linked replay'
                indexed=app.visual.snapshot().get('replay',{})
                assert indexed.get('status')=='indexed',indexed
                replay=ReplayIndex(project/cfg['output']/'replay.sqlite3')
                ids={row['id'] for row in rows}
                assert ids.issubset({row['id'] for row in replay.rows()}),'Missing indexed transitions'
            negative=[]
            for present in (False,True):
                with app.lock:
                    app.world.environment.placements[app.world.body.entity_id].update(x=1.,y=1.)
                    app.world.environment.human.update(x=9.,y=6.,present=present)
                    app.world.body.eyelid_openness=1;app.world.view.epoch+=1
                client.request('POST','/v1/visual/control',{'request_id':uuid.uuid4().hex,'action':'navigate'})
                deadline=time.monotonic()+100
                while time.monotonic()<deadline:
                    visual=client.request('GET','/v1/state')['visual']
                    if visual['status'] not in ('running','loading'):break
                    time.sleep(.1)
                app.visual.thread.join(timeout=5)
                negative.append(visual.get('outcome')=='target_missing' and app.world.environment.placements[app.world.body.entity_id]=={'x':1.,'y':1.})
            # A fresh request is immediately cancelled by a human eyelid action.
            client.request('POST','/v1/visual/control',{'request_id':uuid.uuid4().hex,'action':'navigate'})
            client.request('POST','/v1/action',{'request_id':uuid.uuid4().hex,'source':'human','action':{'kind':'eyelids','openness':0}})
            app.visual.thread.join(timeout=10)
            interrupted=not app.visual.snapshot()['enabled'] and app.world.body.eyelid_openness==0
            handoffs=[]
            for posture in ('lying','sitting'):
                with app.lock:
                    joints=pose(0)
                    if posture=='sitting':joints={k:1 if k.startswith('front') else 0 for k in joints}
                    app.world.body.joint_positions=joints;app.world.body.previous_joints=dict(joints)
                    app.world.body.height=.25 if posture=='lying' else .625
                    app.world.body.motor_mode='independent';app.world.body.eyelid_openness=1
                    app.world.environment.placements[app.world.body.entity_id].update(x=5.,y=3.5)
                    app.world.environment.human.update(x=7.,y=3.5,present=True)
                    app.world.view.epoch+=1
                client.request('POST','/v1/visual/control',{'request_id':uuid.uuid4().hex,'action':'navigate'})
                deadline=time.monotonic()+140;stages=set()
                while time.monotonic()<deadline:
                    visual=client.request('GET','/v1/state')['visual'];stages.add(visual['status'])
                    if visual['status'] not in ('preparing','running','loading'):break
                    time.sleep(.1)
                if app.visual.thread:app.visual.thread.join(timeout=5)
                visual=app.visual.snapshot()
                success=(visual.get('outcome')=='model_reports_arrived' and distance(app.world)<=1.4
                         and 'preparing' in stages and app.policy.info['actions']>0)
                handoffs.append({'initial_posture':posture,'success':success,'stages':sorted(stages),
                                 'joint_actions':app.policy.info['actions'],'body':app.policy.snapshot(),'visual':visual})
                assert success,handoffs[-1]
            with app.lock:
                app.world.body.joint_positions=pose(0);app.world.body.previous_joints=pose(0)
                app.world.body.height=.25
            client.request('POST','/v1/visual/control',{'request_id':uuid.uuid4().hex,'action':'navigate'})
            client.request('POST','/v1/action',{'request_id':uuid.uuid4().hex,'source':'human','action':{'kind':'pause','value':True}})
            time.sleep(.5)
            preparation_cancelled=app.visual.preparation is None and not app.visual.info['enabled'] and app.policy.info['status']=='stopped'
            report={'passed':authenticated and interrupted and preparation_cancelled and all(negative) and all(r['success'] for r in results+handoffs),
                    'standing_handoffs':handoffs,'preparation_cancelled':preparation_cancelled,
                    'http_authentication':authenticated,'human_override':interrupted,'isolated_identity':True,
                    'durable_replay_verified':True,
                    'checkpoint_sha256':results[0]['status']['checkpoint_sha256'],
                    'absent_and_outside_targets_stopped':negative,'session':results[0]['session'],'results':results}
            (project/cfg['output']/'live-qualification.json').write_text(json.dumps(report,indent=2))
            print(json.dumps(report),flush=True)
            if not report['passed']:raise SystemExit(1)
        finally:app.close();server.shutdown();server.server_close();clock.join(timeout=3)

if __name__=='__main__':main()
