"""Real-model isolated runtime qualification, without a fabricated active pointer."""
import argparse,json,os,sys,tempfile,threading,time,uuid
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.services.shared_worker import Worker
from baby_arcus.services.playroom import PlayroomApplication
from baby_arcus.shared_runtime import SharedRuntime
from baby_arcus.shared_replay import Replay
from baby_arcus.transport import serve
from baby_arcus.body_dynamics import pose
from baby_arcus.body_senses import observe_body_senses
from baby_arcus.posture_goals import achieved
from baby_arcus.language_stream import atomic_json

def until(predicate,seconds=60):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        if predicate():return True
        time.sleep(.05)
    return False

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    project=Path(__file__).resolve().parents[1];cfg=json.loads(Path(a.config).read_text());root=Path(cfg['root'])
    manifest=json.loads((root/'candidate.json').read_text());checks={};error=None
    with tempfile.TemporaryDirectory(prefix='arcus-runtime-') as folder:
        temporary=Path(folder);worker=Worker(a.config,manifest,hearing_state_path=temporary/'hearing.json')
        app=PlayroomApplication();runtime=SharedRuntime(app,sys.executable,project);app.shared=runtime
        runtime.cfg=dict(cfg,root=str(temporary/'runtime'),max_observations=1000,interval_seconds=.1)
        app.world.body.motor_mode='independent';app.world.body.joint_positions=pose(0);app.world.body.previous_joints=pose(0)
        for _ in range(30):app.advance()
        clock=threading.Thread(target=app.run_clock,daemon=True);clock.start()
        fault={'pickup':False}
        def endpoint(method,path,body):
            reply=worker(method,path,body)
            if fault['pickup'] and path=='/v1/shared/observe' and body.get('op')!='hearing':
                fault['pickup']=False;app.desktop_event({'request_id':uuid.uuid4().hex,'kind':'pickup'})
            return reply
        token=uuid.uuid4().hex;server=serve('127.0.0.1',0,endpoint,token)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        def say(text):
            app('POST','/v1/messages',{'request_id':uuid.uuid4().hex,'sender':'you','text':text})
        def start():
            # Qualification runs only this disposable app through the real _run loop.
            # It deliberately does not write active.json or claim production approval.
            runtime.cancel=threading.Event();runtime.info.update(enabled=True,status='loading',observations=0)
            runtime.heard_memory=None;runtime.intent_hold=0;runtime.history=[];runtime.previous_outcome=None
            runtime.thread=threading.Thread(target=runtime._run,args=(manifest,),daemon=True);runtime.thread.start()
        try:
            with patch.dict(os.environ,{'ARCUS_SHARED_URL':f'http://127.0.0.1:{server.server_port}','ARCUS_SHARED_TOKEN':token}):
                say('Please stand up');start()
                checks['learned_standing']=until(lambda:achieved(observe_body_senses(app.world.body),'standing'))
                print(json.dumps({'stage':'standing','passed':checks['learned_standing']}),flush=True)
                say('make a sound');checks['expression']=until(lambda:bool(runtime.info.get('expressions')),30)
                print(json.dumps({'stage':'expression','passed':checks['expression']}),flush=True)
                say('listen');checks['listening']=until(lambda:runtime.info.get('hearing_cursor',{}).get('exposures',0)>=64,30)
                say('pause listening');checks['pause']=until(lambda:runtime.info.get('hearing_cursor',{}).get('playing') is False,30)
                before=runtime.info.get('hearing_cursor',{}).get('exposures',0);time.sleep(.5)
                checks['paused_cursor_stable']=runtime.info.get('hearing_cursor',{}).get('exposures',0)==before
                say('resume listening');checks['resume']=until(lambda:runtime.info.get('hearing_cursor',{}).get('exposures',0)>before,30)
                runtime.stop('Qualification restart');runtime.thread.join(timeout=10)
                checks['shutdown']=not runtime.thread.is_alive()
                checks['depth_capacity']=worker.ready().get('depth_capacity')==.25
                queue=Replay(temporary/'runtime'/'replay.sqlite3')
                try:
                    counts=queue.counts();added=sum(queue.recover(path) for path in (temporary/'runtime'/'sessions').glob('*/experiences.jsonl'))
                    checks['replay_recovery']=added==0 and queue.counts()==counts
                    checks['language_replay']=queue.db.execute("SELECT COUNT(*) FROM samples WHERE task='language'").fetchone()[0]>0
                    checks['delayed_replay']=queue.db.execute("SELECT COUNT(*) FROM samples WHERE task='delayed_outcome'").fetchone()[0]>0
                    checks['sequence_replay']=queue.db.execute("SELECT COUNT(*) FROM samples WHERE task='sequence_outcome'").fetchone()[0]>0
                finally:queue.close()
                from baby_arcus.shared_memory import Memory
                remembered=Memory(temporary/'runtime'/'memory.sqlite3')
                try:
                    count=remembered.db.execute('SELECT COUNT(*) FROM memories').fetchone()[0]
                    recovered=sum(remembered.recover(path) for path in (temporary/'runtime'/'sessions').glob('*/experiences.jsonl'))
                    checks['memory_recovery']=count>0 and recovered==0 and remembered.db.execute('SELECT COUNT(*) FROM memories').fetchone()[0]==count
                finally:remembered.close()
                app.desktop_event({'request_id':uuid.uuid4().hex,'kind':'return'})
                fault['pickup']=True;start()
                checks['stale_response_rejected']=until(lambda:not runtime.info['enabled'],15) and runtime.info['observations']==0
                runtime.thread.join(timeout=10)
                checks['human_override']=app.world.view.held and runtime.info['status']=='stopped'
        except Exception as exc:error=f'{type(exc).__name__}: {exc}'
        finally:
            runtime.close();app.close();clock.join(timeout=5);server.shutdown();server.server_close();worker.close()
            import shutil
            evidence=root/'runtime-evidence';evidence.mkdir(exist_ok=True)
            for path in (temporary/'runtime'/'sessions').glob('*/experiences.jsonl'):
                shutil.copyfile(path,evidence/(path.parent.name+'.jsonl'))
        previous=json.loads((root/'live-actions.json').read_text()) if (root/'live-actions.json').exists() else {}
        actions=previous.get('candidate')==manifest and previous.get('posture_http_passed') is True
        hearing=json.loads((root/'hearing-live.json').read_text()) if (root/'hearing-live.json').exists() else {}
        hearing_passed=hearing.get('candidate')==manifest and hearing.get('hearing_passed') is True
        report={'candidate':manifest,'checks':checks,'error':error,'actions_passed':actions,
            'hearing_passed':hearing_passed,
            'live':bool(not error and actions and hearing_passed and len(checks)==15 and all(checks.values())),
            'recovery':checks.get('replay_recovery') is True and checks.get('shutdown') is True,
            'desktop_changed':False,'qualification_scope':'Disposable playpen, actual model and actual runtime over authenticated HTTP.'}
        from baby_arcus.shared_qualification import source_snapshot
        report['runtime_sources']=source_snapshot()
        atomic_json(root/'runtime-live.json',report);print(json.dumps(report),flush=True)
        if not report['live']:raise SystemExit(1)

if __name__=='__main__':main()
