"""Torch-free qualified rest head; posture work stays with the learned body model."""
from copy import deepcopy
import hashlib,json,math,time,uuid,os
from pathlib import Path
from baby_arcus.contracts import ContractError
from baby_arcus.rest_environment import features,NAMES

def predict(artifact,x):
    w,b,v,c=artifact['weights']
    hidden=[math.tanh(sum(a*z for a,z in zip(row,x))+bias) for row,bias in zip(w,b)]
    scores=[sum(a*z for a,z in zip(row,hidden))+bias for row,bias in zip(v,c)]
    if not all(math.isfinite(s) for s in scores):raise ValueError('Invalid rest prediction')
    return max(range(len(scores)),key=scores.__getitem__)

class RestRuntime:
    def __init__(self,app,project,require_live=True):
        self.app=app;self.project=Path(project);self.require_live=require_live
        self.cfg=json.loads((self.project/'configs/baby_arcus/rest.json').read_text())
        self.info={'enabled':False,'status':'stopped','decisions':0,'reason':'Enable qualified rest choices',
                   'training':False,'policy_kind':'small synthetic rest head'}
        self.pending=None;self.last_tick=-100;self.artifact=None
        self.context=None;self.session=None
    def snapshot(self):return deepcopy(self.info)
    def control(self,action):
        if action=='stop':self.stop('Stopped by caregiver');return self.snapshot()
        if action!='start':raise ContractError('Unknown rest control')
        if self.info['enabled']:return self.snapshot()
        w=self.app.world
        if w.paused or w.view.held or w.view.region!='playpen':raise ContractError('Rest requires the active playpen')
        if ((self.app.visual and self.app.visual.info['enabled']) or
            (self.app.policy and self.app.policy.info['status'] in ('loading','running'))):
            raise ContractError('Finish or stop current movement before enabling rest')
        root=self.project/self.cfg['output'];path=root/'policy.json'
        try:
            report=json.loads((root/'qualification.json').read_text());sha=hashlib.sha256(path.read_bytes()).hexdigest()
            if not report['passed'] or report['checkpoint_sha256']!=sha:raise ValueError('Rest qualification mismatch')
            if self.require_live:
                live=json.loads((root/'live-qualification.json').read_text())
                if not live['passed'] or live['checkpoint_sha256']!=sha:raise ValueError('Rest live qualification missing')
            self.artifact=json.loads(path.read_text())
            if self.artifact['schema']!='arcus-rest-policy-v1' or self.artifact['actions']!=list(NAMES):raise ValueError('Rest schema mismatch')
        except (OSError,ValueError,KeyError) as exc:raise ContractError(str(exc)) from exc
        self.info.update(enabled=True,status='running',decisions=0,checkpoint_sha256=sha,reason='Learned rest choices enabled')
        self.context=(self.app.session,w.body.entity_id,w.view.scope_id)
        self.session=uuid.uuid4().hex;self.info['session']=self.session
        self.last_tick=-100;return self.snapshot()
    def stop(self,reason='Interrupted'):
        pending=self.pending;self.pending=None
        if (pending and self.app.policy is pending['owner'] and self.app.policy.revision==pending['revision']):
            self.app.policy.stop(reason)
        self.info.update(enabled=False,status='stopped',reason=reason)
    def close(self):self.stop('Host closed')
    def on_tick(self):
        if not self.info['enabled']:return
        w=self.app.world
        if self.context!=(self.app.session,w.body.entity_id,w.view.scope_id):self.stop('Rest identity or viewing scope changed');return
        if w.paused or w.view.held or w.view.region!='playpen':self.stop('Body/view unavailable');return
        if self.pending:
            p=self.app.policy
            if p is not self.pending['owner'] or p.revision!=self.pending['revision']:self.stop('Posture owner changed');return
            if p.info['status'] in ('stopped','error') or time.monotonic()>self.pending['deadline']:
                self.stop('Learned lying failed or timed out');return
            if p.info['status']!='completed':return
            self.pending=None
        elif ((self.app.visual and self.app.visual.info['enabled']) or
              (self.app.policy and self.app.policy.info['status'] in ('loading','running'))):
            self.stop('Another controller owns movement');return
        if w.tick-self.last_tick<self.cfg['interval_ticks']:return
        self.last_tick=w.tick
        if self.info['decisions']>=self.cfg['decision_limit']:self.stop('Rest decision allowance used');return
        try:
            before=features(w.body);choice=NAMES[predict(self.artifact,before)]
            decision_id=uuid.uuid4().hex
            row={'schema':'arcus-rest-decision-v2','id':decision_id,'session':self.session,
                 'entity_id':w.body.entity_id,'scope_id':w.view.scope_id,'tick':w.tick,
                 'before':before,'choice':choice,'checkpoint_sha256':self.info['checkpoint_sha256']}
            # Durable intent precedes any body action or learned-posture dispatch.
            self.record({**row,'phase':'proposed'})
            action={'rest':'rest','alert':'alert','sleep':'sleep_when_ready','wake':'wake_voluntarily'}.get(choice)
            if choice=='rest' and not before[3]:
                p=self.app.policy
                if not p or not p.expected_hash or 'lying' not in p.goals:raise ContractError('Qualified lying skill required')
                p.start('lying');self.pending={'revision':p.revision,'owner':p,'deadline':time.monotonic()+100}
            if action:
                status,result=self.app('POST','/v1/action',{'request_id':uuid.uuid4().hex,'source':'policy','action':{'kind':action}})
                if status!=200:raise ContractError(str(result))
            self.info.update(decisions=self.info['decisions']+1,last_choice=choice,reason='Learned rest choice: '+choice)
            row={**row,'phase':'applied','after':features(self.app.world.body),
                 'posture_pending':self.pending is not None}
            self.record(row)
            if self.app.audit:self.app.audit.emit('rest.decision',row,durable=True)
        except Exception as exc:self.stop(f'Rest decision failed: {type(exc).__name__}: {exc}')

    def record(self,row):
        root=self.project/self.cfg['output'];root.mkdir(parents=True,exist_ok=True)
        with (root/'decisions.jsonl').open('a',encoding='utf-8') as log:
            log.write(json.dumps(row,allow_nan=False)+'\n');log.flush();os.fsync(log.fileno())
