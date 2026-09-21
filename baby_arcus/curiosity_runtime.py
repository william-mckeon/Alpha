"""Bounded object exploration with exclusive learned-body ownership."""
from copy import deepcopy
import hashlib,json,math,os,time,uuid
from pathlib import Path
from baby_arcus.contracts import ContractError
from baby_arcus.curiosity_environment import observe,choose

class CuriosityRuntime:
    def __init__(self,app,project,require_live=True):
        self.app=app;self.project=Path(project);self.require_live=require_live
        self.cfg=json.loads((self.project/'configs/baby_arcus/curiosity.json').read_text())
        self.info={'enabled':False,'status':'stopped','decisions':0,'discoveries':0,'reason':'Add toys and enable exploration',
                   'observation':'symbolic room objects','training':False}
        self.pending=None;self.context=None
        self.blocked=set()
    def snapshot(self):return deepcopy(self.info)
    def stop(self,reason='Stopped by caregiver'):
        pending=self.pending;self.pending=None
        if pending and self.app.policy is pending['owner'] and self.app.policy.revision==pending['revision']:
            self.app.policy.stop(reason)
        self.info.update(enabled=False,status='stopped',reason=reason)
    def close(self):self.stop('Host closed')
    def control(self,action):
        if action=='stop':self.stop();return self.snapshot()
        if action=='add_toys':
            if self.info['enabled']:raise ContractError('Stop exploration before changing toys')
            if (any(runtime and runtime.info['enabled'] for runtime in (self.app.rest,self.app.visual))
                or (self.app.policy and self.app.policy.info['status'] in ('running','loading'))):
                raise ContractError('Stop other controllers before adding toys')
            world=deepcopy(self.app.world);world.environment.add_toys();world.view.epoch+=1
            self.app.commit(world);return self.snapshot()
        if action!='start':raise ContractError('Unknown exploration control')
        if self.info['enabled']:return self.snapshot()
        observe(self.app.world)
        if not self.app.world.environment.objects:raise ContractError('Add toys before exploration')
        if any(runtime and runtime.info['enabled'] for runtime in (self.app.rest,self.app.visual)):
            raise ContractError('Stop rest or vision before exploration')
        p=self.app.policy
        if not p or not p.expected_hash or 'approach' not in p.goals:raise ContractError('Qualified approach skill required')
        if p.info['status'] in ('loading','running'):raise ContractError('Stop current movement before exploration')
        root=self.project/self.cfg['output'];path=root/'policy.json'
        try:
            sha=hashlib.sha256(path.read_bytes()).hexdigest()
            for name in (['qualification.json','live-qualification.json'] if self.require_live else ['qualification.json']):
                report=json.loads((root/name).read_text())
                if report.get('passed') is not True or report.get('checkpoint_sha256')!=sha:raise ValueError('Exploration qualification mismatch')
            self.artifact=json.loads(path.read_text())
            if self.artifact['schema']!='arcus-curiosity-v1':raise ValueError('Exploration schema mismatch')
        except (OSError,ValueError,KeyError) as exc:raise ContractError(str(exc)) from exc
        w=self.app.world
        self.context=(self.app.session,w.body.entity_id,w.view.scope_id,w.environment.environment_id,w.environment.generation)
        self.blocked=set()
        self.info.update(enabled=True,status='running',decisions=0,discoveries=0,checkpoint_sha256=sha,
                         session=uuid.uuid4().hex,reason='Learned toy selection enabled',blocked_objects=[])
        return self.snapshot()
    def record(self,row):
        row={**row,'session':self.info['session'],'checkpoint_sha256':self.info['checkpoint_sha256']}
        with (self.project/self.cfg['output']/'decisions.jsonl').open('a') as log:
            log.write(json.dumps(row)+'\n');log.flush();os.fsync(log.fileno())
        if self.app.audit:self.app.audit.emit('curiosity.decision',row,durable=True)
    def on_tick(self):
        if not self.info['enabled']:return
        try:
            w=self.app.world;observations=observe(w)
            if self.context!=(self.app.session,w.body.entity_id,w.view.scope_id,w.environment.environment_id,w.environment.generation):
                raise ContractError('Exploration identity or room changed')
            if self.pending:
                p=self.app.policy
                if p is not self.pending['owner'] or p.revision!=self.pending['revision']:raise ContractError('Movement owner replaced')
                if p.info['status'] in ('error','stopped') or time.monotonic()>self.pending['deadline']:raise ContractError('Object approach failed or timed out')
                if p.info['status']!='completed':
                    position=tuple(w.environment.placements[w.body.entity_id][axis] for axis in ('x','y'))
                    if position!=self.pending['position'] or w.body.height<.99 or p.info['status']=='loading':
                        self.pending.update(position=position,progress_at=time.monotonic())
                    elif time.monotonic()-self.pending['progress_at']>self.cfg.get('stall_seconds',3):
                        key=self.pending['object_id']
                        self.record({'id':uuid.uuid4().hex,'phase':'blocked','object_id':key,'reason':'No approach progress'})
                        p.stop('Exploration approach made no progress');self.pending=None;self.blocked.add(key)
                        self.info.update(blocked_objects=sorted(self.blocked),reason='Skipped a blocked toy')
                    return
                key=self.pending['object_id'];self.pending=None
                proposal=uuid.uuid4().hex;self.record({'id':proposal,'phase':'proposed','object_id':key,'action':'inspect'})
                status,result=self.app('POST','/v1/action',{'request_id':proposal,'source':'policy','action':{'kind':'inspect_object','object_id':key}})
                if status!=200:raise ContractError(str(result))
                outcome=result['event']['result'];self.info['discoveries']+=int(outcome['new_discovery'])
                self.record({'id':proposal,'phase':'applied','outcome':outcome});return
            if ((self.app.rest and self.app.rest.info['enabled']) or (self.app.visual and self.app.visual.info['enabled'])
                or self.app.policy.info['status'] in ('running','loading')):raise ContractError('Another controller owns movement')
            if self.info['decisions']>=self.cfg['decision_limit']:self.stop('Exploration allowance used');return
            key=choose(self.artifact,[row for row in observations if row['id'] not in self.blocked])
            self.info['decisions']+=1
            self.record({'id':uuid.uuid4().hex,'phase':'selection','observations':observations,'selected':key})
            if key is None:
                self.info.update(enabled=False,status='completed',reason='No remaining unblocked toy has positive learned exploration value');return
            obj=w.environment.objects[key];position=w.environment.placements[w.body.entity_id]
            dx,dy=position['x']-obj['x'],position['y']-obj['y'];distance=math.hypot(dx,dy)
            target={'x':obj['x']+.85*dx/max(distance,.001),'y':obj['y']+.85*dy/max(distance,.001)}
            p=self.app.policy;p.start('approach');p.target=target;p.info['input_modality']='symbolic_object'
            self.pending={'owner':p,'revision':p.revision,'object_id':key,'deadline':time.monotonic()+100,
                          'position':tuple(position[axis] for axis in ('x','y')),'progress_at':time.monotonic()}
            self.info.update(reason='Investigating '+obj['name'],target=key)
        except Exception as exc:self.stop(f'Exploration failed: {type(exc).__name__}: {exc}')
