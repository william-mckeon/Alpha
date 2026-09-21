"""Bounded asynchronous eye/gaze lesson bridge; native host remains Torch-free."""
from copy import deepcopy
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
import uuid
from baby_arcus.contracts import ContractError
from baby_arcus.visual_experience import observation,available,still_current,motor_action

class VisualRuntime:
    def __init__(self,app,python,project,config='configs/baby_arcus/visual.json'):
        self.app=app;self.python=Path(python);self.project=Path(project);self.config=config;self.default_config=config
        self.cfg=json.loads((self.project/config).read_text(encoding='utf-8'))
        self.thread=None;self.process=None;self.cancel=threading.Event()
        self.event_sequence=None
        self.preparation=None
        self.info={'status':'stopped','reason':'First visual lesson: open eyes and orient toward a visible marker',
                   'decisions':0,'training':False,'scope':'playpen','enabled':False}
    def snapshot(self):return deepcopy(self.info)
    def handles_calls(self):
        config=self.project/'configs/baby_arcus/visual_navigation.json'
        if not config.exists():return False
        try:
            cfg=json.loads(config.read_text(encoding='utf-8'))
            root=self.project/cfg['output']
            report=json.loads((root/'qualification.json').read_text(encoding='utf-8'))
            live=json.loads((root/'live-qualification.json').read_text(encoding='utf-8'))
            # Old reports put checkpoint identity on every trial, not at top level.
            identities={row['status']['checkpoint_sha256'] for row in live.get('results',[])}
            if live.get('checkpoint_sha256'):identities.add(live['checkpoint_sha256'])
            expected=report.get('checkpoint_sha256')
            return bool(report.get('passed') is True and live.get('passed') is True
                        and expected and identities=={expected})
        except (OSError,ValueError,KeyError,TypeError):
            return False
    def on_call(self,row):
        try:
            self.control('navigate');self.event_sequence=row['sequence']
            self.app.interactions.ack(row['sequence'],'delivered',reason='Visual navigation requested',model_exposure=False)
        except ContractError as exc:
            self.app.interactions.ack(row['sequence'],'interrupted',reason=str(exc))
            if not self.info.get('enabled'):self.info.update(status='stopped',reason=str(exc))
    def control(self,action):
        if action=='stop':self.stop();return self.snapshot()
        if action not in ('start','navigate','perceive'):raise ContractError('Unknown visual control')
        if self.thread and self.thread.is_alive():raise ContractError('Visual worker is running or stopping')
        if self.preparation:raise ContractError('Standing preparation is already running')
        self.config=('configs/baby_arcus/object_perception.json' if action=='perceive' else
                     'configs/baby_arcus/visual_navigation.json' if action=='navigate' else self.default_config)
        self.cfg=json.loads((self.project/self.config).read_text(encoding='utf-8'))
        navigation=self.cfg.get('lesson')=='navigation'
        if self.app.world.environment.objects and action!='perceive':
            raise ContractError('Visual model is qualified for an empty room; reset the room before this lesson')
        from baby_arcus.playroom import Playroom
        if action!='perceive' and self.app.world.environment.colors!=Playroom().colors:
            raise ContractError('Existing visual model requires default room colors')
        if not available(self.app.world):raise ContractError('Visual lesson needs an awake, unheld body in the active playpen')
        if action=='perceive' and self.app.world.body.eyelid_openness<=0:raise ContractError('Open eyes before perception')
        root=self.project/self.cfg['output'];report=root/'qualification.json'
        if not report.exists() or not json.loads(report.read_text(encoding='utf-8'))['passed']:
            raise ContractError('Visual adapter has not passed qualification')
        if navigation:
            from baby_arcus.body_senses import observe_body_senses
            if self.app.world.body.height<.99 or not observe_body_senses(self.app.world.body)['stable']:
                policy=self.app.policy
                if (not policy or not hasattr(policy,'revision') or 'standing' not in policy.goals
                        or not policy.expected_hash):
                    raise ContractError('Qualified standing controller required before navigation')
                policy.start('standing')
                self.preparation={'revision':policy.revision,'policy':policy,'started':time.monotonic(),
                                  'entity_id':self.app.world.body.entity_id,'scope_id':self.app.world.view.scope_id}
                self.info.update(status='preparing',enabled=True,decisions=0,lesson='navigation',
                                 last_action=None,resources=None,reason='Learned standing before visual approach')
                self.info.pop('replay',None);self.info.pop('outcome',None)
                return self.snapshot()
        if self.app.policy:self.app.policy.stop('Visual lesson selected')
        self.event_sequence=None
        self.info.pop('replay',None)
        self.info.pop('standing',None)
        self.cancel=threading.Event();self.info.pop('outcome',None)
        self.info.pop('observation',None);self.info.pop('frame',None)
        self.info.update(status='loading',enabled=True,decisions=0,last_action=None,resources=None,lesson=self.cfg.get('lesson','gaze'),reason='Loading visual model')
        self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start();return self.snapshot()
    def stop(self,reason='Stopped by you'):
        preparation=self.preparation;self.preparation=None
        if preparation and preparation['policy'].revision==preparation['revision']:
            preparation['policy'].stop(reason)
        self.cancel.set();self.info.update(enabled=False,status='stopped',reason=reason)
        if self.event_sequence:
            self.app.interactions.ack(self.event_sequence,'interrupted',reason=reason)
            self.event_sequence=None
        if self.process and self.process.poll() is None:self.process.terminate()
    def on_tick(self):
        if self.info['enabled'] and not available(self.app.world):self.stop('Visual lesson interrupted by body or view change')
        if self.info['enabled'] and self.cfg.get('lesson')=='perception' and self.app.world.body.eyelid_openness<=0:
            self.stop('Eyes closed')
        if self.preparation:
            pending=self.preparation;policy=pending['policy'];world=self.app.world
            if (policy is not self.app.policy or policy.revision!=pending['revision']
                    or world.body.entity_id!=pending['entity_id'] or world.view.scope_id!=pending['scope_id']):
                self.stop('Standing preparation replaced');return
            state=policy.snapshot()
            self.info['standing']={key:state.get(key) for key in
                                   ('status','actions','hold_seconds','checkpoint_sha256')}
            self.info['reason']=f"Learned standing: {state.get('actions',0)} joint actions; {state.get('hold_seconds',0)} seconds stable"
            if time.monotonic()-pending['started']>100:
                self.stop('Standing preparation deadline');return
            if state['status'] in ('stopped','error','idle'):
                self.stop('Standing preparation failed: '+state.get('reason','unavailable'));return
            if state['status']=='completed':
                from baby_arcus.body_senses import observe_body_senses
                if world.body.height<.99 or not observe_body_senses(world.body)['stable']:
                    self.stop('Standing completion did not retain stable support');return
                sequence=self.event_sequence
                self.preparation=None
                try:
                    self.control('navigate')
                    self.event_sequence=sequence
                    if self.app.audit:self.app.audit.emit('visual.standing_handoff',
                        {'event_sequence':sequence,'body_checkpoint_sha256':state.get('checkpoint_sha256')},durable=True)
                except (ContractError,OSError,ValueError) as exc:
                    self.event_sequence=sequence;self.stop('Navigation handoff failed: '+str(exc))
            return
        if self.info['enabled'] and self.cfg.get('lesson')=='navigation':
            from baby_arcus.body_senses import observe_body_senses
            if self.app.world.body.height<.99 or not observe_body_senses(self.app.world.body)['stable']:
                self.stop('Standing support changed')
    def close(self):
        self.stop('Host closed')
        if self.thread:self.thread.join(timeout=5)
    def _run(self):
        process=None;stderr=None
        run=self.project/self.cfg['output']/'sessions'/uuid.uuid4().hex
        try:
            run.mkdir(parents=True,exist_ok=False)
            with self.app.lock:self.info['session']=str(run)
            navigation=self.cfg.get('lesson')=='navigation'
            perception=self.cfg.get('lesson')=='perception'
            url=os.environ.get('ARCUS_PERCEPTION_MODEL_URL' if perception else 'ARCUS_NAVIGATION_URL' if navigation else 'ARCUS_VISUAL_URL')
            if url:
                from baby_arcus.transport import Client
                token=os.environ.get('ARCUS_VISUAL_TOKEN')
                if not token:raise ValueError('Visual service credential missing')
                client=Client(url,token,timeout=30,attempts=1);ready=client.request('GET','/ready')
                receive=lambda record:client.request('POST','/v1/visual',record)
            else:
                stderr=(run/'stderr.log').open('w',encoding='utf-8')
                process=subprocess.Popen([str(self.python),'-u','-m','baby_arcus.services.visual_worker','--config',self.config],
                    cwd=self.project,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,text=True,encoding='utf-8',bufsize=1,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                self.process=process;replies=queue.Queue()
                def reader():
                    for line in process.stdout:replies.put(line)
                    replies.put(None)
                threading.Thread(target=reader,daemon=True).start()
                def response():
                    deadline=time.monotonic()+60
                    while not self.cancel.is_set():
                        try:
                            line=replies.get(timeout=.1)
                            if line is None:raise RuntimeError('Visual worker exited')
                            data=json.loads(line)
                            if data.get('status')=='error':raise RuntimeError(data['error'])
                            return data
                        except queue.Empty:
                            if time.monotonic()>deadline:raise TimeoutError('Visual worker deadline')
                    raise InterruptedError('Visual session stopped')
                ready=response()
                def receive(record):
                    process.stdin.write(json.dumps(record)+'\n');process.stdin.flush();return response()
            if ready.get('status')!='ready':raise RuntimeError('Visual worker not ready')
            if ready.get('lesson','gaze')!=('navigation' if navigation else 'gaze'):raise RuntimeError('Worker lesson mismatch')
            if bool(ready.get('learned_budget'))!=bool(self.cfg.get('learned_budget')):raise RuntimeError('Worker resource policy mismatch')
            expected=json.loads((self.project/self.cfg['output']/'qualification.json').read_text())['checkpoint_sha256']
            if ready['checkpoint_sha256']!=expected:raise RuntimeError('Worker checkpoint mismatch')
            with self.app.lock:
                if self.cancel.is_set():return
                self.info.update(status='running',reason='Choosing eye and gaze actions',checkpoint_sha256=ready['checkpoint_sha256'],session=str(run))
            with (run/'experiences.jsonl').open('w',encoding='utf-8') as log:
                for index in range(self.cfg['decision_limit']):
                    if self.cancel.is_set():break
                    started=time.perf_counter()
                    record=observation(self.app,1-index/self.cfg['decision_limit'],navigation=navigation,perception=perception)
                    decision=receive(record)
                    if decision['id']!=record['id']:raise ValueError('Visual response identity mismatch')
                    if perception:
                        with self.app.lock:
                            if (self.cancel.is_set() or not still_current(self.app,record)
                                or self.app.world.body.eyelid_openness<=0):
                                self.stop('Perception observation became stale');break
                            if decision['checkpoint_sha256']!=expected:raise ValueError('Perception checkpoint mismatch')
                            self.info.update(decisions=index+1,observation=decision['observation'],
                                frame=record['frame'],resources=decision['resources'],reason='Learned RGB perception; no movement dispatched')
                            log.write(json.dumps({'observation':record,'decision':decision})+'\n');log.flush();os.fsync(log.fileno())
                            if self.app.audit:self.app.audit.emit('visual.perception',decision,durable=True)
                        if self.cancel.wait(self.cfg['interval_seconds']):break
                        continue
                    with self.app.lock:
                        applied=not self.cancel.is_set() and still_current(self.app,record)
                        if decision['checkpoint_sha256']!=expected:raise ValueError('Decision checkpoint mismatch')
                        if navigation:
                            from baby_arcus.visual_navigation_environment import action as navigation_action,NAMES as action_names
                            action=navigation_action(decision['action'])
                        else:
                            from baby_arcus.visual_experience import NAMES as action_names
                            action=motor_action(decision['action'],record['gaze'])
                        if decision['action_name']!=action_names[decision['action']]:raise ValueError('Action name mismatch')
                        # Write the complete observation/decision before applying any action.
                        log.write(json.dumps({'observation':record,'decision':decision,'will_apply':applied})+'\n');log.flush()
                        result=None
                        if applied and action:
                            status,result=self.app('POST','/v1/action',{'request_id':uuid.uuid4().hex,'source':'policy','action':action})
                            if status!=200:raise RuntimeError('Visual action rejected')
                        log.write(json.dumps({'observation_id':record['id'],'applied':applied,'action':action,'result':result})+'\n');log.flush()
                        if not applied:
                            self.stop('Observation became stale');break
                        self.info.update(decisions=index+1,last_action=decision['action_name'],resources=decision['resources'],
                            reason='Learned visual choices; no live weight updates')
                        if self.event_sequence:
                            self.app.interactions.ack(self.event_sequence,'delivered',model_exposure=True,understood=False,reason='Visual navigation decision received')
                        if self.app.audit:self.app.audit.emit('visual.decision',{'observation_id':record['id'],'decision':decision,'action':action},durable=True)
                        from baby_arcus.visual_replay import transition
                        after=observation(self.app,1-(index+1)/self.cfg['decision_limit'],navigation=navigation)
                        linked=transition(record,decision,action,after,True,1000*(time.perf_counter()-started))
                        self.info['resources']={**decision['resources'],'end_to_end_ms':linked['end_to_end_ms'],
                            'remaining_decisions':self.cfg['decision_limit']-index-1,'capture_ms':record['resources']['capture_ms']}
                        log.write(json.dumps(linked)+'\n');log.flush()
                        if navigation and decision['action'] in (0,6):
                            self.info['outcome']='model_reports_arrived' if decision['action']==0 else 'target_missing'
                            break
                    if self.cancel.wait(self.cfg['interval_seconds']):break
            with self.app.lock:
                if not self.cancel.is_set():
                    outcome=self.info.get('outcome','decision_limit')
                    reasons={'model_reports_arrived':'Arcus predicts he has arrived near your marker',
                             'target_missing':'Arcus cannot see the target; movement stopped',
                             'decision_limit':'Navigation allowance used; movement stopped'}
                    self.info.update(status='completed',enabled=False,reason=reasons[outcome] if navigation else 'Bounded visual lesson completed')
                    if self.event_sequence:
                        self.app.interactions.ack(self.event_sequence,'completed' if outcome=='model_reports_arrived' else 'interrupted',reason=outcome,model_exposure=True)
                        self.event_sequence=None
        except InterruptedError:pass
        except Exception as exc:
            with self.app.lock:
                if not self.cancel.is_set():
                    self.info.update(status='error',enabled=False,reason=str(exc))
                    if self.event_sequence:
                        self.app.interactions.ack(self.event_sequence,'interrupted',reason=str(exc));self.event_sequence=None
        finally:
            if process:
                if process.poll() is None:process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=3)
                process.stdin.close();process.stdout.close()
            if stderr:stderr.close()
            self.process=None
            if run.exists():
                source=run/'experiences.jsonl'
                if source.exists() and self.cfg.get('lesson')=='perception':
                    self.info['replay']={'status':'raw_perception_records','source_retained':True}
                elif source.exists():
                    from baby_arcus.visual_replay import ReplayIndex
                    try:
                        replay=ReplayIndex(run.parent.parent/'replay.sqlite3',
                            max_rows=self.cfg.get('replay_max_rows',100000),
                            max_bytes=self.cfg.get('replay_max_bytes',1024**3))
                        added=replay.ingest([source])
                        with self.app.lock:
                            self.info['replay']={**replay.manifest(),'added':added,'status':'indexed'}
                    except Exception as exc:
                        with self.app.lock:
                            self.info['replay']={'status':'error','reason':str(exc),'source_retained':True}
                (run/'summary.json').write_text(json.dumps(self.snapshot(),indent=2),encoding='utf-8')
