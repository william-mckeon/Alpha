"""Torch-free bounded observation bridge; activation requires all shared gates."""
from copy import deepcopy
import json,os,queue,subprocess,threading,time,uuid
from pathlib import Path
from baby_arcus.contracts import ContractError
from baby_arcus.shared_experience import capture,current

class SharedRuntime:
    def __init__(self,app,python,project):
        self.app=app;self.python=str(python);self.project=Path(project)
        self.config='configs/baby_arcus/shared.json';self.cfg=json.loads((self.project/self.config).read_text())
        self.info={'enabled':False,'status':'stopped','observations':0,'training':False}
        self.cancel=threading.Event();self.thread=None;self.process=None
        self.pending_request=None
        self.passage=None
        self.previous_outcome=None
        self.history=[]
        self.heard_memory=None
        self.intent_hold=0
    def snapshot(self):
        state=deepcopy(self.info);root=self.project/self.cfg['root']
        state['depth_capacity']=self.cfg.get('depth_capacity',.25)
        state['shared_only']=bool(self.cfg.get('shared_only'))
        for name,file in (('candidate','candidate.json'),('qualification','qualification.json')):
            try:state[name]=json.loads((root/file).read_text(encoding='utf-8'))
            except (OSError,ValueError):state[name]=None
        return state
    def control(self,action):
        if action=='stop':self.stop('Caregiver stopped shared observation');return self.snapshot()
        if action!='start':raise ContractError('Unknown shared control')
        if self.thread and self.thread.is_alive():raise ContractError('Shared worker still running')
        root=self.project/self.cfg['root']
        if not (root/'active.json').exists():raise ContractError('The shared candidate has not passed qualification yet.')
        try:
            manifest=json.loads((root/'active.json').read_text());report=json.loads((root/'qualification.json').read_text())
            if report.get('sha256')!=manifest['sha256'] or not all(report.get(k) is True for k in ('integration','retention','cross_modal','live')):
                raise ValueError('Shared learning gates not passed')
            from baby_arcus.shared_qualification import verify_evidence
            if report.get('candidate') != manifest:raise ValueError('Qualification candidate mismatch')
            verify_evidence(report,root)
        except (OSError,ValueError,KeyError) as exc:raise ContractError('Shared learner is not qualified: '+str(exc)) from exc
        capture(self.app)
        if self.app.language and self.app.language.busy:raise ContractError('Wait for the current language transaction before shared mode')
        for runtime in (self.app.policy,self.app.language,self.app.visual,self.app.rest,self.app.curiosity):
            if runtime:
                if runtime is self.app.language:runtime.control('stop')
                else:runtime.stop('Shared observation selected')
        self.previous_outcome=None
        self.history=[]
        self.heard_memory=None
        self.intent_hold=0
        self.cancel=threading.Event();self.info.update(enabled=True,status='loading',observations=0,generation=manifest['generation'])
        self.thread=threading.Thread(target=self._run,args=(manifest,),daemon=True);self.thread.start();return self.snapshot()
    def stop(self,reason='Shared observation stopped'):
        self.cancel.set();self.info.update(enabled=False,status='stopped',reason=reason)
        if self.process and self.process.poll() is None:self.process.terminate()
    def on_tick(self):
        if self.info['enabled'] and (self.app.world.paused or self.app.world.view.held or self.app.world.view.region!='playpen'):
            self.stop('Shared sensory scope changed')
    def close(self):
        self.stop('Host closed')
        if self.thread:self.thread.join(timeout=5)
    def _run(self,manifest):
        root=self.project/self.cfg['root'];run=root/'sessions'/uuid.uuid4().hex;run.mkdir(parents=True)
        stderr=None;memory=None;sequence=[]
        try:
            from baby_arcus.shared_replay import Replay
            replay=Replay(root/'replay.sqlite3')
            try:
                for journal in (root/'sessions').glob('*/experiences.jsonl'):replay.recover(journal)
            finally:replay.close()
            if self.cfg.get('durable_memory'):
                from baby_arcus.shared_memory import Memory
                memory=Memory(root/'memory.sqlite3',self.cfg.get('memory_budget',2048))
                for journal in (root/'sessions').glob('*/experiences.jsonl'):memory.recover(journal)
            url=os.environ.get('ARCUS_SHARED_URL')
            if url:
                from baby_arcus.transport import Client
                client=Client(url,os.environ.get('ARCUS_SHARED_TOKEN'),timeout=30,attempts=1)
                ready=client.request('GET','/ready');receive=lambda row:client.request('POST','/v1/shared/observe',row)
            else:
                stderr=(run/'stderr.log').open('w')
                module=self.cfg.get('worker_module','baby_arcus.services.shared_worker')
                if module not in ('baby_arcus.services.shared_worker','baby_arcus.services.shared_continuity_worker'):
                    raise ValueError('Unknown shared worker module')
                self.process=subprocess.Popen([self.python,'-u','-m',module,'--config',self.config],
                    cwd=self.project,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,text=True,encoding='utf-8',
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                replies=queue.Queue()
                def reader():
                    for line in self.process.stdout:replies.put(line)
                    replies.put(None)
                threading.Thread(target=reader,daemon=True).start()
                def read(timeout=60):
                    if type(timeout) not in (int,float) or not 1<=timeout<=300:
                        raise ValueError('Invalid shared worker deadline')
                    deadline=time.monotonic()+timeout
                    while not self.cancel.is_set():
                        try:
                            line=replies.get(timeout=.1)
                            if line is None:raise RuntimeError('Shared worker exited')
                            result=json.loads(line)
                            if 'error' in result:raise RuntimeError(result['error'])
                            return result
                        except queue.Empty:
                            if time.monotonic()>deadline:raise TimeoutError('Shared worker deadline')
                    raise InterruptedError()
                ready=read(self.cfg.get('startup_timeout_seconds',60))
                def receive(row):
                    self.process.stdin.write(json.dumps(row)+'\n');self.process.stdin.flush();return read()
            if any(ready[k]!=manifest[k] for k in ('generation','sha256')):raise ValueError('Shared worker identity mismatch')
            if self.cfg.get('depth_capacity') is not None and ready.get('depth_capacity')!=.25:
                raise ValueError('Shared worker did not confirm 0.25 depth')
            pending_delayed=None
            with (run/'experiences.jsonl').open('w') as log:
                for index in range(self.cfg['max_observations']):
                    if self.cancel.is_set():break
                    row=capture(self.app)
                    if ready.get('model_version',0)>=8:row['predict_outcome']=index%10==0
                    if pending_delayed:
                        before,previous=pending_delayed
                        if row['tick']<=previous['after']['tick']:
                            if self.cancel.wait(self.cfg['interval_seconds']):break
                            continue
                        from baby_arcus.shared_temporal import targets
                        try:targets(before,previous,row)
                        except ValueError as exc:
                            event={'phase':'delayed_skipped','experience_id':before['id'],'reason':str(exc)}
                        else:
                            event={'phase':'delayed_outcome','experience':before,'outcome':previous,'after':row}
                        log.write(json.dumps(event)+'\n');log.flush();os.fsync(log.fileno())
                        if event['phase']=='delayed_outcome':
                            delayed_replay=Replay(root/'replay.sqlite3')
                            try:delayed_replay.add_delayed(before,previous,row)
                            finally:delayed_replay.close()
                            self.info['delayed_outcomes']=self.info.get('delayed_outcomes',0)+1
                            if memory:
                                memory.remember_outcome(event)
                                from baby_arcus.shared_temporal import body_vector
                                self.info['last_consequence']={'forecast':previous.get('forecast'),'observed_body':body_vector(row),'horizon_ticks':row['tick']-before['tick']}
                            sequence.append((deepcopy(before),deepcopy(previous),deepcopy(row)))
                            if len(sequence)>=4:
                                from baby_arcus.shared_temporal import sequence_targets
                                try:sequence_targets(sequence)
                                except ValueError:sequence=[]
                                else:
                                    log.write(json.dumps({'phase':'sequence_outcome','transitions':sequence})+'\n');log.flush();os.fsync(log.fileno())
                                    queue_replay=Replay(root/'replay.sqlite3')
                                    try:queue_replay.add_sequence(sequence)
                                    finally:queue_replay.close()
                                    self.info['sequence_outcomes']=self.info.get('sequence_outcomes',0)+1;sequence=[]
                        else:sequence=[]
                        pending_delayed=None
                    if row['hearing']:
                        self.heard_memory=deepcopy(max(row['hearing'],key=lambda message:message.get('created_at',0)))
                        self.intent_hold=0
                    elif self.heard_memory:
                        row['hearing']=[{**self.heard_memory,'source':'remembered_hearing'}]
                        row['hearing'][0].pop('request_id',None)
                    row['history']=deepcopy(self.history)
                    if self.info.get('hearing_cursor'):row['hearing_cursor']=deepcopy(self.info['hearing_cursor'])
                    if self.passage:
                        row['ambient_hearing']={key:deepcopy(self.passage[key]) for key in ('text','source','id','file','document','offset','fingerprint')}
                        if not row['hearing']:row['hearing']=[row['ambient_hearing']]
                        row['text_source']='caregiver_and_dataset'
                    if self.previous_outcome:
                        row['previous_outcome']=deepcopy(self.previous_outcome)
                    if memory:
                        row['memory']=memory.recall(row)
                        self.info['recalled_memories']=len(row['memory'])
                    row['exploration_permitted']=bool(self.cfg.get('curiosity_enabled') and index%10==0 and not row['hearing'])
                    reply=receive(row)
                    with self.app.lock:
                        if self.cancel.is_set() or not current(self.app,row):raise InterruptedError()
                        if reply['id']!=row['id'] or any(reply[k]!=manifest[k] for k in ('generation','sha256')):raise ValueError('Shared response mismatch')
                        if self.cfg.get('depth_capacity') is not None and reply.get('depth',{}).get('capacity')!=.25:
                            raise ValueError('Shared prediction exceeded or omitted the required depth contract')
                        log.write(json.dumps({'phase':'proposed','experience':row,'prediction':reply})+'\n');log.flush();os.fsync(log.fileno())
                        result=None;executed=False
                        if self.cfg.get('execute_actions') and reply.get('proposal'):
                            self.pending_request=uuid.uuid4().hex
                            try:
                                status,result=self.app('POST','/v1/action',{'request_id':self.pending_request,'source':'policy','action':reply['proposal']})
                                executed=status==200
                            except ContractError as exc:result={'error':str(exc),'applied':False}
                            finally:self.pending_request=None
                            self.previous_outcome={'action':reply['proposal'],'executed':executed,
                                'result':result.get('event',{}).get('result',result.get('error'))}
                        for event in row.get('events',[]):
                            self.app.interactions.ack(event['sequence'],'delivered',model_exposure=True,understood=False,reason='Shared model input received')
                        for message in row['hearing']:
                            if message.get('request_id'):self.app.conversation.mark_read(message['request_id'])
                        outcome={'phase':'outcome','experience_id':row['id'],'executed':executed,'action':reply.get('proposal'),'result':result,'after':capture(self.app)}
                        if reply.get('predicted_outcome'):outcome['forecast']=reply['predicted_outcome']
                        log.write(json.dumps(outcome)+'\n');log.flush();os.fsync(log.fileno())
                        if ready.get('model_version',0)>=8 and outcome.get('action'):
                            pending_delayed=(deepcopy(row),deepcopy(outcome))
                        from baby_arcus.shared_replay import Replay
                        replay=Replay(root/'replay.sqlite3')
                        try:
                            replay.add(row,outcome)
                            replay.add_language(row,reply.get('language_example'))
                            self.info['replay']=replay.counts()
                        finally:replay.close()
                        self.info.update(status='running',observations=index+1,last_prediction=reply,executed=executed,reason='Shared prediction and verified execution; learning occurs in candidate training')
                        self.info['frame']=deepcopy(row['vision'])
                        if outcome['after']['epoch']!=row['epoch']:self.history=[]
                        else:
                            self.history.append({key:deepcopy(row[key]) for key in ('session','entity_id','scope_id','epoch','tick','senses')})
                            self.history=self.history[-2:]
                        if reply.get('expression'):
                            expressions=self.info.setdefault('expressions',[])
                            expressions.append({'text':reply['expression'],'label':'Shared model token; meaning not established'})
                            self.info['expressions']=expressions[-20:]
                        intent=reply.get('intent',{})
                        self.intent_hold=self.intent_hold+1 if intent.get('satisfied') else 0
                        if self.intent_hold>=10 or intent.get('activity')==0 or (intent.get('activity')==6 and executed) or reply.get('expression'):
                            self.heard_memory=None;self.intent_hold=0
                    if reply.get('continuity',{}).get('requires_acknowledgement') and not self.cancel.is_set():
                        acknowledged=receive({'op':'continuity_outcome','outcome':outcome})
                        if any(acknowledged.get(k)!=manifest[k] for k in ('generation','sha256')):
                            raise ValueError('Continuity acknowledgement identity mismatch')
                        log.write(json.dumps({'phase':'continuity_acknowledgement','experience_id':row['id'],
                            'result':acknowledged.get('continuity_acknowledgement')})+'\n');log.flush();os.fsync(log.fileno())
                    if reply.get('hearing_action') and not self.cancel.is_set():
                        heard=receive({'op':'hearing','action':reply['hearing_action']})
                        if heard.get('passage'):self.passage=heard['passage']
                        elif reply['hearing_action'] in ('pause','restart'):self.passage=None
                        with self.app.lock:self.info['hearing_cursor']=heard['cursor'];self.heard_memory=None
                    elif self.info.get('hearing_cursor',{}).get('playing') and not self.heard_memory and not self.cancel.is_set():
                        heard=receive({'op':'hearing','action':'listen'})
                        self.passage=heard.get('passage')
                        with self.app.lock:self.info['hearing_cursor']=heard['cursor']
                    if self.cancel.wait(self.cfg['interval_seconds']):break
                if pending_delayed:
                    log.write(json.dumps({'phase':'delayed_skipped','experience_id':pending_delayed[0]['id'],
                        'reason':'Observation ended before a valid later sensory frame'})+'\n');log.flush();os.fsync(log.fileno())
            with self.app.lock:
                if not self.cancel.is_set():self.info.update(enabled=False,status='completed')
        except InterruptedError:
            if not self.cancel.is_set():self.stop('Shared observation invalidated')
        except Exception as exc:
            with self.app.lock:
                if not self.cancel.is_set():self.info.update(enabled=False,status='error',reason=str(exc))
        finally:
            if self.process:
                if self.process.poll() is None:self.process.terminate()
                try:self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
                self.process.stdin.close();self.process.stdout.close();self.process=None
            if stderr:stderr.close()
            if memory:memory.close()
