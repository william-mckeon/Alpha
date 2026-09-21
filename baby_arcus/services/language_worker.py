"""Bounded language learning process. No desktop or body actions are exposed here.

Each completed decision commits weights, optimizer, RNG, cursor and receipts together.
Two checkpoint slots plus an atomic pointer retain the last accepted transaction.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

import torch
from arcus.tokenizer import get_tokenizer
from baby_arcus.body_policy import load as load_body
from baby_arcus.human_messages import validate_message
from baby_arcus.language_checkpoint import load, save
from baby_arcus.language_learning import loss, update
from baby_arcus.language_model import CHOICES, generate
from baby_arcus.language_stream import LanguageStream, atomic_json, documents, inventory
from baby_arcus.large_body_learning import file_hash


class LanguageWorker:
    def __init__(self, config):
        self.cfg=json.loads(Path(config).read_text(encoding='utf-8'))
        self.root=Path(self.cfg['language_root']);self.lock=threading.Lock();self.failed=False
        self.root.mkdir(parents=True,exist_ok=True)
        # The process owns this state directory exclusively (also on Windows).
        from baby_arcus.embodiment_store import EmbodimentStore
        self.lease=EmbodimentStore(self.root/'worker-lease')
        try:self._load()
        except BaseException:self.lease.close();raise

    def _load(self):
        torch.set_num_threads(2)
        if importlib.metadata.version('tiktoken')!=self.cfg['tiktoken_version']:
            raise ValueError('Install the pinned language tokenizer release')
        self.tokenizer=get_tokenizer(self.cfg['encoding'])
        report=json.loads((self.root/'qualification.json').read_text(encoding='utf-8'))
        self.parent=file_hash(self.cfg['body_checkpoint'])
        if not report.get('gate_passed') or report['parent_sha256']!=self.parent:
            raise ValueError('Language checkpoint is not qualified for this motor parent')
        if file_hash(self.root/'bootstrap.pt')!=report['checkpoint_sha256']:
            raise ValueError('Qualified bootstrap checkpoint changed')
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.body,_=load_body(self.cfg['body_checkpoint'])
        self.body.to(self.device).eval().requires_grad_(False)
        manifest=inventory(self.cfg['dataset_root'],self.cfg['source_patterns'])
        if manifest['fingerprint']!=json.loads((self.root/'dataset-manifest.json').read_text(encoding='utf-8'))['fingerprint']:
            raise ValueError('Dataset differs from bootstrap manifest')
        self.stream=LanguageStream(manifest,self.tokenizer,self.root/'listening.json')
        pointer=self.root/'active.json'
        selected=self.root/'bootstrap.pt'
        self.slot=0
        if pointer.exists():
            active=json.loads(pointer.read_text(encoding='utf-8'));self.slot=active['slot']
            if self.slot not in (0,1):raise ValueError('Invalid active slot')
            selected=self.root/f'live-{self.slot}.pt'
            if file_hash(selected)!=active['sha256']:raise ValueError('Active language checkpoint changed')
        self.adapter,data=load(selected,self.body.core,self.tokenizer,self.parent,self.device)
        self.meta=data['metadata'];self.optimizer=torch.optim.AdamW(self.adapter.parameters(),lr=.00005)
        # First pilot used a misleading legacy label for the pool size. Do not treat
        # that value as the number of unique tokens actually sampled by training.
        if 'unique_training_tokens' in self.meta:
            self.meta['available_training_tokens']=self.meta.pop('unique_training_tokens')
        self.optimizer.load_state_dict(data['optimizer'])
        # Bootstrap uses a higher LR; live updates deliberately use a smaller one.
        for group in self.optimizer.param_groups:group['lr']=.00005
        torch.set_rng_state(data['rng'])
        if self.device=='cuda' and data.get('cuda_rng'):torch.cuda.set_rng_state_all(data['cuda_rng'])
        self.state=deepcopy(self.meta.get('runtime',{
            'decisions':0,'training_tokens':0,'exposed_tokens':0,'accepted_updates':0,'rejected_updates':0,
            'message_receipts':{},'dataset_passages':{},'context':[], 'expressions':[],
            'last_choice':None,'last_loss':None,'last_progress':0.0,'baseline_reward':0.0,
            'policy_updates':0,'choice_counts':{name:0 for name in CHOICES}}))
        if 'stream' in self.state:
            self.stream.state=deepcopy(self.state['stream']);atomic_json(self.stream.path,self.stream.state)
        self.validation=self._validation(manifest)
        with torch.no_grad():self.reference_loss=self.evaluate()
        if not math.isfinite(self.reference_loss):raise ValueError('Invalid checkpoint validation loss')
        self.initial_loss=report['validation_loss_after']
        self.allowed_ids=[]
        special={self.tokenizer.enc.encode_single_token(value) for value in self.tokenizer.enc.special_tokens_set}
        for token in range(self.tokenizer.vocab_size):
            if token in special and token!=self.tokenizer.eot_token:continue
            try:self.tokenizer.enc.decode_single_token_bytes(token)
            except KeyError:continue
            self.allowed_ids.append(token)

    def _validation(self,manifest):
        path=self.root/'validation.json'
        if path.exists():return json.loads(path.read_text(encoding='utf-8'))
        # Compatibility with the first bootstrap; reconstruct its exact first 16 windows.
        windows=[]
        for entry in manifest['files']:
            for number,text in documents(Path(manifest['root'])/entry['path']):
                if number%10==0:
                    ids=self.tokenizer.encode('Environment:\n'+text,add_eot=True)
                    windows.extend(ids[i:i+65] for i in range(0,min(len(ids)-64,512),64))
                if len(windows)>=16 or number>=24:break
            if len(windows)>=16:break
        if not windows:raise ValueError('No held-out language windows')
        atomic_json(path,windows[:16]);return windows[:16]

    def evaluate(self):
        with torch.no_grad():
            return sum(float(loss(self.adapter,self.body.core,ids,self.device)) for ids in self.validation)/len(self.validation)

    def snapshot(self):
        receipts=self.state['message_receipts']
        return deepcopy({'status':'ready','encoding':self.cfg['encoding'],'tiktoken_version':self.cfg['tiktoken_version'],
            'listening':self.stream.state['playing'],'eof':self.stream.state['eof'],
            'cursor':{k:self.stream.state[k] for k in ('file','document','token')},
            'training_tokens':self.state['training_tokens'],'exposed_tokens':self.state['exposed_tokens'],
            'remaining_training_tokens':max(0,self.cfg['session_training_tokens']-self.state['training_tokens']),
            'budget_scope':'persisted pilot; restarting does not replenish it',
            'bootstrap_training_tokens':self.meta['training_tokens'],
            'accepted_updates':self.state['accepted_updates'],'rejected_updates':self.state['rejected_updates'],
            'dataset_unique_passage_tokens':sum(row['tokens'] for row in self.state['dataset_passages'].values()),
            'dataset_replay_tokens':sum(row['tokens']*(row['exposures']-1) for row in self.state['dataset_passages'].values()),
            'last_choice':self.state['last_choice'],'last_loss':self.state['last_loss'],
            'validation_loss':self.reference_loss,'policy_updates':self.state['policy_updates'],
            'choice_counts':self.state['choice_counts'],'decisions':self.state['decisions'],
            'message_receipts':receipts,'expressions':self.state['expressions'],
            'motor_parameters':self.meta['motor_parameters'],'language_parameters':self.meta['language_parameters'],
            'parent_sha256':self.parent,'checkpoint_slot':self.slot,
            'policy_note':'Exploratory learned choice head; curiosity and language understanding are not established.'})

    def commit(self):
        self.state['stream']=deepcopy(self.stream.state)
        meta={**self.meta,'runtime':self.state}
        next_slot=1-self.slot;path=self.root/f'live-{next_slot}.pt'
        save(path,self.adapter,self.optimizer,meta)
        atomic_json(self.root/'active.json',{'slot':next_slot,'sha256':file_hash(path),
            'decisions':self.state['decisions'],'training_tokens':self.state['training_tokens'],
            'summary':{**self.snapshot(),'checkpoint_slot':next_slot}})
        self.slot=next_slot
        with (self.root/'experience.jsonl').open('a',encoding='utf-8') as log:
            log.write(json.dumps({'time':time.time(),'decision':self.state['decisions'],
                'choice':self.state['last_choice'],'training_tokens':self.state['training_tokens'],
                'exposed_tokens':self.state['exposed_tokens'],'cursor':self.snapshot()['cursor'],
                'accepted_updates':self.state['accepted_updates'],'validation_loss':self.reference_loss,
                'last_loss':self.state['last_loss'],
                'context_passage_id':(self.stream.state.get('last') or {}).get('id'),
                'expressions':[row for row in self.state['expressions'] if row['id']==self.state['decisions']],
                'message_receipts':self.state['message_receipts']})+'\n');log.flush();os.fsync(log.fileno())

    def learn(self,ids):
        remaining=self.cfg['session_training_tokens']-self.state['training_tokens']
        ids=ids[:min(remaining+1,self.cfg['context_tokens'])]
        if len(ids)<2:return 0,0.0
        # Candidate weights and optimizer state are rolled back as one unit on rejection.
        old=deepcopy(self.adapter.state_dict());old_optimizer=deepcopy(self.optimizer.state_dict())
        with torch.no_grad():before=float(loss(self.adapter,self.body.core,ids,self.device))
        update(self.adapter,self.body.core,self.optimizer,ids,self.device)
        after_validation=self.evaluate()
        if not math.isfinite(after_validation) or after_validation>min(self.reference_loss+.05,self.initial_loss+.15):
            self.adapter.load_state_dict(old);self.optimizer.load_state_dict(old_optimizer)
            self.state['rejected_updates']+=1
            return 0,0.0
        with torch.no_grad():after=float(loss(self.adapter,self.body.core,ids,self.device))
        count=len(ids)-1
        self.state['training_tokens']+=count;self.state['accepted_updates']+=1
        self.state['last_loss']=after;self.reference_loss=after_validation
        return count,max(-1.0,min(1.0,(before-after)/max(before,1)))

    def step(self,messages):
        if len(messages)>4:raise ValueError('At most four messages per decision')
        remaining=self.cfg['session_training_tokens']-self.state['training_tokens']
        if remaining<=0 or self.state['decisions']>=256:return self.snapshot()
        heard=False
        for message in messages:
            validate_message(message)
            key=message['request_id']
            digest=hashlib.sha256(json.dumps(message,sort_keys=True).encode()).hexdigest()
            previous=self.state['message_receipts'].get(key)
            if previous and previous['sha256']!=digest:raise ValueError('Language message ID conflict')
            if previous and previous['status'] in ('trained','received_only'):continue
            ids=self.tokenizer.encode(message['sender']+':\n'+message['text'],add_eot=True)
            offset=previous['trained_tokens'] if previous else 0
            chunk=ids[offset:offset+self.cfg['context_tokens']]
            count,progress=self.learn(chunk);total=offset+count
            self.state['message_receipts'][key]={'sha256':digest,'received_tokens':len(ids),'trained_tokens':total,
                'status':'trained' if total>=len(ids)-1 else 'partially_trained' if count else 'received_only'}
            self.state['exposed_tokens']+=len(chunk);self.state['context']=chunk[-116:];heard=True
        listening=self.stream.state['playing'];can_read=not self.stream.state['eof']
        allowed=[True,listening and can_read,listening,not listening and can_read,
                 self.stream.state['last'] is not None,bool(self.state['context'])]
        index,log_prob=self.adapter.decide([float(listening),float(heard),
            min(1,self.state['last_progress']),max(0,remaining)/self.cfg['session_training_tokens']],allowed)
        choice=CHOICES[index];self.state['last_choice']=choice;self.state['choice_counts'][choice]+=1
        progress=0.0
        if choice in ('pause','resume'):self.stream.control(choice)
        elif choice in ('listen','replay'):
            passage=self.stream.next(self.cfg['passage_tokens'],replay=choice=='replay')
            if passage:
                ids=passage['tokens'];self.state['exposed_tokens']+=len(ids);self.state['context']=ids[-116:]
                count,progress=self.learn(ids)
                receipt=self.state['dataset_passages'].setdefault(passage['id'],{
                    'file':passage['file'],'document':passage['document'],'offset':passage['offset'],
                    'tokens':len(ids),'exposures':0,'trained_tokens':0})
                receipt['exposures']+=1;receipt['trained_tokens']+=count
        elif choice=='express':
            text,ids=generate(self.adapter,self.body.core,self.tokenizer,
                self.state['context']+self.tokenizer.encode('\nArcus:'),self.allowed_ids,self.cfg['max_expression_tokens'])
            if text:
                self.state['expressions'].append({'id':self.state['decisions']+1,'text':text,'tokens':ids,
                    'created_at':time.time(),'label':'Early model-generated text; meaning not established'})
                self.state['expressions']=self.state['expressions'][-100:]
        # Small policy-gradient bandit: information gain earns credit; silence remains legal.
        # A replay only earns credit for improvement, not for producing surprise/noise.
        reward=progress-.001 if choice in ('listen','replay') else 0.0
        advantage=reward-self.state['baseline_reward']
        self.optimizer.zero_grad(set_to_none=True)
        (-log_prob*advantage).backward()
        torch.nn.utils.clip_grad_norm_(self.adapter.choice.parameters(),1.0);self.optimizer.step()
        self.state['baseline_reward']=.95*self.state['baseline_reward']+.05*reward
        self.state['policy_updates']+=1;self.state['last_progress']=progress;self.state['decisions']+=1
        self.commit();return self.snapshot()

    def request(self,value):
        with self.lock:
            if self.failed:raise RuntimeError('Failed transaction; restart worker to recover the last checkpoint')
            try:return self._request(value)
            except Exception:
                self.failed=True
                raise

    def _request(self,value):
        op=value.get('op')
        if op=='status':return self.snapshot()
        if op=='step':return self.step(value.get('messages',[]))
        if op=='control':
            self.stream.control(value['action']);self.state['last_choice']='human_'+value['action']
            self.commit();return self.snapshot()
        raise ValueError('Unknown language operation')

    def close(self):
        self.stream.close();self.lease.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/baby_arcus/language.json')
    p.add_argument('--port',type=int);p.add_argument('--host',default='127.0.0.1');args=p.parse_args()
    worker=LanguageWorker(args.config)
    try:
        if args.port:
            from baby_arcus.transport import serve
            token=os.environ.get('ARCUS_LANGUAGE_TOKEN')
            if not token:raise ValueError('HTTP worker requires ARCUS_LANGUAGE_TOKEN')
            def app(method,path,body):
                if method=='GET' and path=='/health':return (503,{'error':'Restart required'}) if worker.failed else (200,worker.snapshot())
                if method=='POST' and path=='/v1/language':return 200,worker.request(body)
                raise KeyError(path)
            server=serve(args.host,args.port,app,token)
            try:server.serve_forever()
            finally:server.server_close()
        else:
            print(json.dumps(worker.snapshot()),flush=True)
            for line in sys.stdin:
                try:result=worker.request(json.loads(line))
                except Exception as exc:
                    print(json.dumps({'status':'error','error':str(exc)}),flush=True)
                    # Failed transactions must reload the last committed state before another request.
                    break
                print(json.dumps(result,ensure_ascii=True),flush=True)
    finally:worker.close()


if __name__=='__main__':main()
