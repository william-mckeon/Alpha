"""Shared inference service: one immutable core and checkpoint generation."""
import argparse,json,os,threading
from pathlib import Path
import torch
from baby_arcus.shared_checkpoint import load

class Worker:
    def __init__(self,config,manifest=None,hearing_state_path=None):
        from arcus.tokenizer import get_tokenizer
        import importlib.metadata
        torch.set_num_threads(2);self.cfg=json.loads(Path(config).read_text())
        if importlib.metadata.version('tiktoken')!=self.cfg['tiktoken_version']:raise ValueError('Tokenizer release mismatch')
        self.root=Path(self.cfg['root']);self.manifest=manifest or json.loads((self.root/'active.json').read_text())
        if manifest is None:
            from baby_arcus.shared_qualification import verify_evidence
            report=json.loads((self.root/'qualification.json').read_text())
            if report.get('candidate')!=self.manifest:raise ValueError('Worker qualification identity mismatch')
            verify_evidence(report,self.root)
        self.model,self.data=load(self.root,self.manifest,'cuda' if torch.cuda.is_available() else 'cpu')
        from baby_arcus.shared_depth import verify_depth
        verify_depth(self.model,self.cfg)
        self.model.eval().requires_grad_(False);self.tokenizer=get_tokenizer(self.cfg['encoding']);self.lock=threading.Lock()
        self.stream=None;self.diagnostic=manifest is not None
        self.hearing_state_path=Path(hearing_state_path) if hearing_state_path is not None else self.root/'live-hearing.json'
        self.isolated_hearing=hearing_state_path is not None
        if self.diagnostic and self.isolated_hearing and self.hearing_state_path.resolve().is_relative_to(self.root.resolve()):
            raise ValueError('Diagnostic hearing cursor must be outside the candidate directory')
    def ready(self):return {'ready':True,**self.manifest,'training':False,'model_version':self.model.version,'depth_capacity':self.model.body.cfg.capacity}
    def predict(self,row):
        if row.get('op')=='hearing':return self.hearing(row['action'])
        if self.model.version>=3:
            from copy import deepcopy
            row=deepcopy(row);row['objects']=[];row['object_source']='none'
        with self.lock,torch.no_grad():
            output=self.model([row],self.tokenizer)
            if any(torch.isnan(value).any() or torch.isposinf(value).any() for value in output.values()):
                raise ValueError('Nonfinite shared prediction')
            activity=int(output['activity'][0].argmax());proposal=None;predicted_outcome=None;experimentation=None
            if activity==1:
                from baby_arcus.body_vocabulary import ACTIONS,mask
                head=('body','lying','sitting')[int(output['posture_choice'][0].argmax())]
                logits=output[head][0].clone();allowed=torch.tensor(mask(row['senses']),device=logits.device)
                logits=logits.masked_fill(~allowed,-torch.inf);proposal=ACTIONS[int(logits.argmax())]
            elif activity==2:
                name=(None,'rest','alert','sleep_when_ready','wake_voluntarily')[int(output['rest'][0].argmax())]
                proposal={'kind':name} if name else None
            elif activity==3 and row.get('objects'):
                values=output['objects'][0,:len(row['objects'])];index=int(values.argmax())
                if float(values[index])>0:
                    obj=row['objects'][index]
                    if sum(v*v for v in obj['relative'])<=1.4**2:proposal={'kind':'inspect_object','object_id':obj['id']}
                    else:
                        move=self.model.body.approach([row['senses']],[obj['relative']])
                        proposal={'kind':'move','direction':('up','down','left','right')[int(move[0].argmax())]}
            elif activity==6:
                proposal=gaze_action(row,int(output['gaze_choice'][0].argmax()))
            elif activity==7:
                from baby_arcus.body_vocabulary import ACTIONS
                import math
                relative=row.get('hearing_relative')
                if relative is not None and math.hypot(*relative)>.65:
                    if row['senses']['height']<.99 or not row['senses']['stable']:proposal=ACTIONS[int(output['body'][0].argmax())]
                    else:
                        direction=self.model.body.approach([row['senses']],[relative])
                        proposal={'kind':'move','direction':('up','down','left','right')[int(direction[0].argmax())]}
            if self.model.version>=9 and row.get('exploration_permitted') and not row['hearing'] and row['internal'][0]<.65:
                from baby_arcus.shared_causal import choose_experiment
                selected,considered=choose_experiment(self.model,self.tokenizer,row)
                experimentation={'candidates':considered,'selected':selected,'scope':'Bounded predicted sensory novelty, not general reasoning'}
                if selected:
                    proposal=selected['action'];activity=6
                    predicted_outcome={key:selected[key] for key in ('future_body','future_rgb','uncertainty')}
                    predicted_outcome.update(generation=self.manifest['generation'],horizon_ticks=3,scope='Action-conditioned ensemble prediction')
            if self.model.version>=8 and proposal and row.get('predict_outcome') and predicted_outcome is None:
                conditioned=deepcopy(row);conditioned['executed_action']=proposal
                forecast=self.model([conditioned],self.tokenizer,requested=('future_body','future_rgb','action_quality'))
                if any(not torch.isfinite(value).all() for value in forecast.values()):raise ValueError('Nonfinite outcome prediction')
                predicted_outcome={key:value[0].cpu().tolist() for key,value in forecast.items()}
                predicted_outcome.update(generation=self.manifest['generation'],horizon_ticks=conditioned.get('prediction_horizon',3),scope='Action-conditioned ensemble prediction' if self.model.version>=9 else 'Calibrated prediction; causal accuracy is not qualified')
        token=int(output['text'][0].argmax());expression=None
        special={self.tokenizer.enc.encode_single_token(value) for value in self.tokenizer.enc.special_tokens_set}
        if activity==5 and token not in special:
            try:
                raw=self.tokenizer.enc.decode_single_token_bytes(token)
                text=raw.decode('utf-8',errors='strict')
                if text and all(c.isprintable() or c in '\n\t' for c in text):expression=text
            except (KeyError,UnicodeDecodeError):pass
        observation=None
        if self.model.version>=3 and row['vision']['available']:
            from baby_arcus.object_perception_environment import arrays
            from baby_arcus.object_observation import learned_observation
            from baby_arcus.shared_experience import validate
            observation=learned_observation(arrays(validate(row)),output['perception'][0].argmax(0).cpu().numpy())
        intent={'activity':activity,'satisfied':False}
        if activity==1:
            from baby_arcus.posture_goals import achieved
            goal=('standing','lying','sitting')[int(output['posture_choice'][0].argmax())]
            intent.update(goal=goal,satisfied=achieved(row['senses'],goal))
        elif activity==7:
            import math
            intent['satisfied']=math.hypot(*row.get('hearing_relative',[1e9,1e9]))<=.65
        language_example=None
        fresh=[message for message in row['hearing'] if message.get('source')!='remembered_hearing']
        if fresh:
            message=fresh[-1];ids=self.tokenizer.encode(message.get('text',''))[-65:]
            source_id=message.get('request_id') or message.get('id') or message.get('passage_id')
            if source_id and len(ids)>=2:language_example={'source_id':source_id,'prefix':ids[:-1],'target':ids[-1]}
        from baby_arcus.shared_depth import measurement
        return {'id':row['id'],**self.manifest,'training':False,'visual_observation':observation,'depth':measurement(self.model),
            'intent':intent,'language_example':language_example,'predicted_outcome':predicted_outcome,'experimentation':experimentation,
            'scores':{k:torch.nan_to_num(v[0],neginf=-1e9).cpu().tolist() for k,v in output.items() if k not in ('hidden','aux','text','perception','future_body','future_rgb','action_quality','future_ensemble','uncertainty')},
            'text_token':token,'expression':expression,'hearing_action':(None,'listen','pause','resume','replay','restart' if self.model.version>=7 else None)[int(output['language_choice'][0].argmax())] if activity==4 else None,
            'proposal':proposal,'activity':activity,'action_executed':False}
    def hearing(self,action):
        if self.diagnostic and not self.isolated_hearing:raise ValueError('Diagnostics cannot change hearing state')
        if action not in ('listen','pause','resume','restart','replay'):raise ValueError('Invalid hearing control')
        with self.lock:
            if self.stream is None:
                from baby_arcus.language_stream import LanguageStream,inventory
                cfg=json.loads(Path(self.cfg['dataset_config']).read_text(encoding='utf-8'))
                manifest=inventory(cfg['dataset_root'],cfg['source_patterns'])
                self.stream=LanguageStream(manifest,self.tokenizer,self.hearing_state_path)
            if action in ('pause','resume','restart'):self.stream.control(action);passage=None
            else:passage=self.stream.next(limit=64,replay=action=='replay')
            return {'passage':passage,'cursor':self.stream.state,'training':False,**self.manifest}
    def close(self):
        if self.stream:self.stream.close();self.stream=None
    def __call__(self,method,path,body):
        if method=='GET' and path=='/ready':return 200,self.ready()
        if method=='POST' and path=='/v1/shared/observe':return 200,self.predict(body)
        if method=='POST' and path=='/v1/shared/hearing':return 200,self.hearing(body['action'])
        raise KeyError(path)

def gaze_action(row,choice):
    if choice==0:return None
    if choice in (5,6):return {'kind':'eyelids','openness':1 if choice==5 else 0}
    yaw,pitch=row.get('gaze',[0,0,0,0])[2:]
    if choice in (1,2):yaw+=-.25 if choice==1 else .25
    elif choice in (3,4):pitch+=-.25 if choice==3 else .25
    else:raise ValueError('Invalid gaze choice')
    return {'kind':'gaze','yaw':max(-1,min(1,yaw)),'pitch':max(-1,min(1,pitch))}

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/baby_arcus/shared.json');p.add_argument('--port',type=int)
    args=p.parse_args();worker=Worker(args.config)
    if args.port:
        from baby_arcus.transport import serve
        token=os.environ.get('ARCUS_SHARED_TOKEN')
        if not token:raise ValueError('Shared HTTP authentication required')
        server=serve('0.0.0.0',args.port,worker,token)
        try:server.serve_forever()
        finally:server.server_close()
    else:
        import sys
        print(json.dumps(worker.ready()),flush=True)
        for line in sys.stdin:
            try:result=worker.predict(json.loads(line))
            except Exception as exc:result={'error':str(exc)}
            print(json.dumps(result),flush=True)

if __name__=='__main__':main()
