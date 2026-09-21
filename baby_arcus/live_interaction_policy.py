"""Persistent event reception and mutually exclusive, interruptible learned actions."""
from copy import deepcopy
import hashlib,json,queue,subprocess,threading,time,uuid
from baby_arcus.live_body_policy import LiveBodyPolicy
from baby_arcus.contracts import ContractError
from baby_arcus.interaction_observation import observation
from baby_arcus.body_tools import BodyToolsApplication

class LiveInteractionPolicy(LiveBodyPolicy):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.revision=0;self.target=None;self.event_sequence=None;self.closing=False
        self.info.update(receiver='idle',last_event=None,exposure_only=True)

    def available(self):
        w=self.app.world
        return not (w.paused or w.view.held or w.view.region!='playpen' or w.body.sleep_state!='awake')

    def ensure_worker(self):
        if self.closing:return
        if not self.thread or not self.thread.is_alive():
            self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()

    def start(self,goal='standing'):
        if goal not in self.goals:raise ContractError('Goal not available in qualified checkpoint')
        if not self.available():raise ContractError('Movement requires an awake unheld body in the active playpen')
        if not self.python.is_file() or not self.checkpoint.is_file():raise ContractError('Runtime or checkpoint unavailable')
        self.stop('Replaced by new command');self.revision+=1
        self.info.update(status='loading',goal=goal,actions=0,hold_seconds=0,reason='Preparing '+goal)
        if goal=='approach':
            self.target=dict(self.app.world.environment.human)
        candidate=deepcopy(self.app.world);candidate.body.motor_mode='independent'
        self.app.commit(candidate);self.ensure_worker()
        return self.snapshot()

    def on_event(self,row):
        self.info['last_event']=row['sequence']
        if row['kind']=='call':
            if self.available():
                try:
                    hearing=row['payload'].get('hearing')
                    if hearing and not hearing['source'].get('present',True):
                        raise ContractError('Caller is not present in the playpen')
                    self.start('approach');self.event_sequence=row['sequence']
                    if hearing:self.target=deepcopy(hearing['source'])
                    self.info.update(input_modality='simulated_hearing',reason='Approaching the simulated call location')
                except ContractError as exc:self.app.interactions.ack(row['sequence'],'unsupported',reason=str(exc))
            else:
                # Deliver later as history; never move automatically on wake/drop.
                self.app.interactions.ack(row['sequence'],'interrupted',reason='Body unavailable; call again when ready')
        if self.app.world.body.sleep_state=='awake':self.ensure_worker()

    def stop(self,reason='Stopped by you'):
        self.revision+=1
        if self.info['status'] in ('loading','running'):
            self.info.update(status='stopped',reason=reason)
            if self.event_sequence:self.app.interactions.ack(self.event_sequence,'interrupted',reason=reason)
            self.emit('interaction.movement_stopped',self.snapshot())
        self.event_sequence=None
        return self.snapshot()

    def on_tick(self):
        if self.app.world.body.sleep_state=='awake' and self.app.interactions.pending():self.ensure_worker()
        if self.info['status'] not in ('running','loading'):return
        if not self.available():self.stop('Body unavailable');return
        if self.info['goal']!='approach':
            from baby_arcus.posture_goals import achieved
            from baby_arcus.body_senses import observe_body_senses
            self.info['hold_seconds']=round(self.info['hold_seconds']+.1,1) if achieved(observe_body_senses(self.app.world.body),self.info['goal']) else 0

    def close(self):
        self.closing=True;self.stop('Host closed')
        if self.process and self.process.poll() is None:self.process.terminate()
        if self.thread:self.thread.join(timeout=5)

    def _run(self):
        process=None
        try:
            with self.checkpoint.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
            if actual!=self.expected_hash:raise RuntimeError('Checkpoint differs from qualification')
            process=subprocess.Popen([str(self.python),'-u','-m','baby_arcus.services.body_predictor','--checkpoint',str(self.checkpoint)],
                cwd=self.workdir,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,bufsize=1,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            self.process=process;replies=queue.Queue()
            def reader():
                for line in process.stdout:replies.put(line)
                replies.put(None)
            threading.Thread(target=reader,daemon=True).start()
            def receive(timeout):
                line=replies.get(timeout=timeout)
                if line is None:raise RuntimeError('Predictor exited')
                return json.loads(line)
            ready=receive(40)
            if not ready.get('ready'):raise RuntimeError('Model not ready')
            with self.app.lock:self.info.update(receiver='ready',parameters=ready['parameters'],checkpoint_sha256=actual)
            tools=BodyToolsApplication(self.app);started_revision=-1;deadline=0
            while not self.closing:
                with self.app.lock:
                    if self.app.world.body.sleep_state!='awake':events=[]
                    else:events=self.app.interactions.pending(limit=4)
                    for event in list(events):
                        if time.time()-event['created_at']>300:
                            self.app.interactions.ack(event['sequence'],'expired',reason='Delivery deadline');events.remove(event)
                    active=self.info['status'] in ('loading','running') and self.available()
                    revision=self.revision
                    if active and started_revision!=revision:
                        deadline=time.monotonic()+90;started_revision=revision
                    if active and time.monotonic()>deadline:self.stop('Movement deadline');active=False
                    obs=observation(self.app.world,events,self.target if active and self.info['goal']=='approach' else None,
                                    self.info.get('input_modality','simulated_hearing'))
                    goal=self.info['goal'] if active else None
                    if active:self.info.update(status='running',reason='Model choosing actions')
                if not events and not active:
                    time.sleep(.1);continue
                process.stdin.write(json.dumps({'observation':obs,'goal':goal})+'\n');process.stdin.flush()
                response=receive(30)
                with self.app.lock:
                    for seq in response.get('received',[]):
                        if seq not in [e['sequence'] for e in events]:raise RuntimeError('Invalid event receipt')
                        previous=self.app.interactions.rows[seq-1]['status']
                        self.app.interactions.ack(seq,'delivered' if previous=='queued' else previous,model_exposure=True,understood=False)
                        event=self.app.interactions.rows[seq-1]
                        if event['kind']=='message':self.app.conversation.mark_read(event['payload']['request_id'])
                    self.emit('interaction.model_observation',{'observation':obs,'response':response,'revision':revision})
                    if not active or revision!=self.revision or not self.available():continue
                    if response.get('arrived') or (goal!='approach' and self.info['hold_seconds']>=5):
                        self.info.update(status='completed',reason='Reached your marker' if goal=='approach' else goal+' held for five seconds')
                        if self.event_sequence:self.app.interactions.ack(self.event_sequence,'completed',reason=self.info['reason'])
                        self.emit('interaction.completed',self.snapshot());continue
                    action=response.get('body_action')
                    if action:
                        if action.get('kind') not in ('joint','move'):raise RuntimeError('Action outside learned vocabulary')
                        status,result=tools('POST','/v1/tools/call',{'request_id':uuid.uuid4().hex,'name':'body_action','arguments':action})
                        if status!=200:raise RuntimeError('Action rejected')
                        self.info['actions']+=1
                        self.emit('interaction.transition',{'event_sequence':self.event_sequence,'before':obs,'action':action,'after':self.app.world.snapshot()})
                time.sleep(.12)
        except Exception as exc:
            with self.app.lock:
                if not self.closing:
                    self.stop('Predictor failure');self.info.update(status='error',receiver='error',reason=f'{type(exc).__name__}: {exc}')
                    # Avoid a tight crash/restart loop while retaining evidence.
                    for row in self.app.interactions.pending():self.app.interactions.ack(row['sequence'],'unsupported',reason='Predictor unavailable')
        finally:
            if process:
                if process.poll() is None:process.terminate()
                try:process.wait(timeout=3)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                if process.stdin:process.stdin.close()
                if process.stdout:process.stdout.close()
            self.process=None
