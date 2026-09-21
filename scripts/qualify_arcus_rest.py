"""Real HTTP rest choices, persistence and GPU learned-lying handoff."""
import json,sys,tempfile,threading,time,uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.services.playroom import PlayroomApplication
from baby_arcus.live_interaction_policy import LiveInteractionPolicy
from baby_arcus.body_controller_config import configuration
from baby_arcus.rest_runtime import RestRuntime
from baby_arcus.transport import Client,serve,RemoteError
from baby_arcus.body_dynamics import pose

def main():
    project=Path(__file__).resolve().parents[1];results={}
    with tempfile.TemporaryDirectory() as root:
        app=PlayroomApplication(root)
        app.policy=LiveInteractionPolicy(app,project/'.venv/Scripts/python.exe',workdir=project,**configuration(project))
        app.rest=RestRuntime(app,project,require_live=False)
        token=uuid.uuid4().hex;server=serve('127.0.0.1',0,app,token)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        clock=threading.Thread(target=app.run_clock,daemon=True);clock.start()
        client=Client(f'http://127.0.0.1:{server.server_port}',token,attempts=1)
        def wait_for(predicate,seconds=100):
            deadline=time.monotonic()+seconds
            while time.monotonic()<deadline:
                state=client.request('GET','/v1/state')
                if predicate(state):return state
                if state['rest']['status']=='stopped':raise AssertionError(state['rest'])
                time.sleep(.1)
            raise AssertionError('Rest scenario deadline')
        try:
            try:Client(client.base_url,attempts=1).request('GET','/v1/state')
            except RemoteError as exc:results['authentication']=exc.status==401
            with app.lock:app.world.body.rest_need=.95;app.world.body.stimulation=0
            command={'request_id':uuid.uuid4().hex,'action':'start'}
            client.request('POST','/v1/rest/control',command);client.request('POST','/v1/rest/control',command)
            state=wait_for(lambda s:s['arcus']['sleep_state']=='sleeping')
            results['learned_lying_then_sleep']=(state['arcus']['height']<=.30 and state['policy']['actions']>0
                and state['arcus']['motor_mode']=='independent')
            lying_actions=state['policy']['actions']
            # Quiet tired sleep should persist. This test does not accelerate the clock.
            time.sleep(1.2);results['quiet_sleep_retained']=app.world.body.sleep_state=='sleeping'
            client.request('POST','/v1/action',{'request_id':uuid.uuid4().hex,'source':'human','action':{'kind':'call'}})
            state=wait_for(lambda s:s['arcus']['sleep_state']=='awake',10)
            results['hearing_voluntary_wake']=(state['arcus']['height']<=.30 and state['arcus']['eyelid_openness']==0
                                             and state['policy']['status']!='running')
            with app.lock:
                app.world.body.rest_need=.1;app.world.body.stimulation=0
            state=wait_for(lambda s:s['arcus']['rest_mode']=='active',10)
            results['alert_while_lying']=state['arcus']['sleep_state']=='awake' and state['arcus']['height']<=.30
            with app.lock:
                # Separate held-out scenario: rested sleep, without a hearing cue.
                app.world.body.sleep_state='sleeping';app.world.body.rest_need=.05
                app.world.body.stimulation=0;app.world.body.rest_mode='resting'
            state=wait_for(lambda s:s['arcus']['sleep_state']=='awake',10)
            results['rested_voluntary_wake']=state['arcus']['height']<=.30
            with app.lock:
                q={k:1 if k.startswith('front') else 0 for k in pose(0)}
                app.world.body.joint_positions=q;app.world.body.previous_joints=dict(q)
                app.world.body.height=.625;app.world.body.rest_mode='resting';app.world.body.stimulation=1
            state=wait_for(lambda s:s['arcus']['rest_mode']=='active',10)
            results['alert_while_sitting']=state['arcus']['posture']=='sitting' and state['arcus']['sleep_state']=='awake'
            with app.lock:app.world.body.rest_need=.95;app.world.body.stimulation=0
            state=wait_for(lambda s:s['arcus']['sleep_state']=='sleeping')
            results['sitting_to_learned_lying_to_sleep']=state['policy']['actions']>0 and state['arcus']['height']<=.30
            client.request('POST','/v1/action',{'request_id':uuid.uuid4().hex,'source':'human','action':{'kind':'pause','value':True}})
            results['human_override']=not app.rest.info['enabled']
            identity=app.world.body.entity_id
            report={'passed':all(results.values()),'checks':results,'lying_joint_actions':lying_actions,
                    'checkpoint_sha256':app.rest.info['checkpoint_sha256'],'body_checkpoint_sha256':app.policy.expected_hash}
        finally:app.close();server.shutdown();server.server_close();clock.join(timeout=3)
        restored=PlayroomApplication(root);restored.rest=RestRuntime(restored,project,require_live=False)
        report['checks']['restart_preserves_identity_and_disables_controller']=(restored.world.body.entity_id==identity and not restored.rest.info['enabled'])
        restored.close();report['passed']=all(report['checks'].values())
        output=project/'runs/arcus_rest_v1/live-qualification.json';output.write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
        if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()
