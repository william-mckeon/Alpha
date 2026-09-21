"""Actual-model HTTP actions in an isolated playpen; never activates the desktop."""
import argparse,json,sys,threading,uuid,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.services.shared_worker import Worker
from baby_arcus.services.playroom import PlayroomApplication
from baby_arcus.transport import Client,serve
from baby_arcus.shared_experience import capture,current
from baby_arcus.body_dynamics import pose
from baby_arcus.body_senses import observe_body_senses
from baby_arcus.posture_goals import achieved
from baby_arcus.language_stream import atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());root=Path(cfg['root']);manifest=json.loads((root/'candidate.json').read_text())
    worker=Worker(a.config,manifest);token=uuid.uuid4().hex
    server=serve('127.0.0.1',0,worker,token);threading.Thread(target=server.serve_forever,daemon=True).start()
    client=Client(f'http://127.0.0.1:{server.server_port}',token,timeout=60,attempts=1)
    results={};trace=[]
    try:
        for goal,utterance in (('standing','Please stand up'),('lying','Please lie down'),('sitting','Please sit down'),('approach',None)):
            app=PlayroomApplication()
            try:
                body=app.world.body;body.motor_mode='independent';body.joint_positions=pose(0 if goal in ('standing','approach') else 1);body.previous_joints=dict(body.joint_positions)
                for _ in range(30):app.advance()
                if goal=='approach':app('POST','/v1/action',{'request_id':uuid.uuid4().hex,'source':'human','action':{'kind':'call'}})
                else:app('POST','/v1/messages',{'request_id':uuid.uuid4().hex,'sender':'you','text':utterance})
                hold=0;success=False
                for step in range(240):
                    row=capture(app);reply=client.request('POST','/v1/shared/observe',row)
                    if not current(app,row):raise ValueError('Stale live diagnostic response')
                    if reply['proposal']:
                        app('POST','/v1/action',{'request_id':uuid.uuid4().hex,'source':'policy','action':reply['proposal']})
                    app.advance()
                    if goal=='approach':
                        position=app.world.environment.placements[app.world.body.entity_id];human=app.world.environment.human
                        reached=math.hypot(position['x']-human['x'],position['y']-human['y'])<=.65
                    else:reached=achieved(observe_body_senses(app.world.body),goal)
                    hold=hold+1 if reached else 0
                    trace.append({'goal':goal,'step':step,'activity':reply['activity'],'proposal':reply['proposal'],'hold':hold})
                    if hold>=10:success=True;break
                results[goal]={'success':success,'steps':step+1}
                print(json.dumps({'goal':goal,**results[goal]}),flush=True)
            finally:app.close()
        app=PlayroomApplication()
        try:
            probe=capture(app);probe['hearing']=[{'text':'look left'}]
            before=client.request('POST','/v1/shared/observe',probe)
            reloaded=Worker(a.config,manifest);after=reloaded.predict(probe)
            recovery=before['proposal']==after['proposal'] and before['activity']==after['activity'] and before['sha256']==after['sha256']
        finally:app.close()
        report={'protocol':'current-body-http-v2','candidate':manifest,'postures':results,'recovery':recovery,
            'posture_http_passed':all(r['success'] for r in results.values()),'live':False,
            'training_updates':0,'desktop_changed':False,
            'remaining':'Full runtime journal/restart, hearing controls, expression and human-intervention qualification.'}
        atomic_json(root/'live-actions.json',report);atomic_json(root/'live-actions-trace.json',trace)
        print(json.dumps(report),flush=True)
    finally:server.shutdown();server.server_close()

if __name__=='__main__':main()
