"""Qualified visual adapter inference, available over stdio or authenticated HTTP."""
import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time
import torch
from baby_arcus.mode_learning import read_json,load_candidate
from baby_arcus.large_body_learning import file_hash
from baby_arcus.visual_model import VisualAdapter,NavigationAdapter,GroundedNavigationAdapter,tensors
from baby_arcus.visual_experience import NAMES

class Worker:
    def __init__(self,config):
        torch.set_num_threads(2);self.cfg=read_json(config);root=Path(self.cfg['output'])
        self.perception=self.cfg.get('lesson')=='perception'
        if self.perception:
            from baby_arcus.object_perception_learning import load
            self.cfg,self.body,self.model,self.device=load(config)
            self.model.requires_grad_(False);self.sha=file_hash(root/'perception.pt')
            self.lock=threading.Lock();self.navigation=False;self.budget_policy=None
            return
        report=read_json(root/'qualification.json')
        if not report['passed'] or file_hash(root/'visual.pt')!=report['checkpoint_sha256']:
            raise ValueError('Visual checkpoint is not qualified')
        data=torch.load(root/'visual.pt',map_location='cpu',weights_only=True)
        self.navigation=data['schema']=='arcus-navigation-v1'
        if data['schema'] not in ('arcus-visual-v1','arcus-navigation-v1') or file_hash(Path(self.cfg['parent'])/'model.pt')!=data['parent_sha256']:
            raise ValueError('Visual parent identity changed')
        self.body,language,parent=load_candidate(self.cfg['parent']);del language,parent
        if self.navigation:
            from baby_arcus.visual_navigation_environment import NAMES as navigation_names
            self.names=navigation_names
        else:self.names=NAMES
        # Preserve the first experimental navigation head's reload compatibility.
        adapter=NavigationAdapter if self.navigation and 'encoder.0.weight' in data['adapter'] else VisualAdapter
        if self.navigation and 'detector.0.weight' in data['adapter']:adapter=GroundedNavigationAdapter
        self.model=adapter(self.body.cfg.dim,len(self.names)).cuda().eval();self.model.load_state_dict(data['adapter'])
        self.body.requires_grad_(False);self.model.requires_grad_(False);self.lock=threading.Lock()
        self.sha=report['checkpoint_sha256']
        self.budget_policy=None
        if self.cfg.get('learned_budget'):
            from baby_arcus.depth_policy import MeasuredBudgetPolicy
            resource_root=root/'resources';qualification=read_json(resource_root/'qualification.json')
            if not qualification['passed'] or qualification['visual_sha256']!=self.sha or file_hash(resource_root/'policy.pt')!=qualification['checkpoint_sha256']:
                raise ValueError('Resource policy has not qualified for this visual model')
            resource=torch.load(resource_root/'policy.pt',map_location='cpu',weights_only=True)
            if resource['schema']!='arcus-measured-budget-v1' or resource['visual_sha256']!=self.sha:raise ValueError('Resource policy identity mismatch')
            self.budget_policy=MeasuredBudgetPolicy().cuda().eval().requires_grad_(False)
            self.budget_policy.load_state_dict(resource['policy'])
    def ready(self):return {'status':'ready','checkpoint_sha256':self.sha,'training':False,'scope':'playpen','lesson':'perception' if self.perception else 'navigation' if self.navigation else 'gaze','learned_budget':self.budget_policy is not None}
    def predict(self,record):
        with self.lock:
            if record['schema']!='arcus-visual-experience-v1':raise ValueError('Unsupported visual record')
            if self.perception:
                if record.get('lesson')!='perception' or record['gaze']['eyelid_openness']<=0:raise ValueError('Perception requires open eyes')
                from baby_arcus.object_observation import learned_observation
                pixels,_=tensors(record);started=time.perf_counter()
                with torch.no_grad():labels=self.model(self.body.core,pixels[None].to(self.device)).argmax(1)[0].cpu().numpy()
                return {'id':record['id'],'checkpoint_sha256':self.sha,'observation':learned_observation(pixels.numpy(),labels),
                        'resources':{'inference_ms':1000*(time.perf_counter()-started),'input_tokens':36,
                        'capacity':.5,'expert_routed_fraction':self.body.core.last_compute_fraction,'learned_budget':False}}
            if (record.get('lesson','gaze')=='navigation')!=self.navigation:raise ValueError('Visual lesson mismatch')
            pixels,state=tensors(record);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
            with torch.no_grad():
                pixels=pixels[None].cuda();state=state[None].cuda();budget=.5
                if self.budget_policy:
                    from baby_arcus.depth_policy import visual_budget_features
                    budget=self.budget_policy.capacities[int(self.budget_policy.choose_index(visual_budget_features(pixels,state)))]
                logits=self.model(self.body.core,pixels,state,budget)[0]
            if not torch.isfinite(logits).all():raise ValueError('Nonfinite visual prediction')
            torch.cuda.synchronize();elapsed=1000*(time.perf_counter()-started)
            index=int(logits.argmax());fraction=self.body.core.last_compute_fraction
            return {'id':record['id'],'action':index,'action_name':self.names[index],
                'scores':logits.cpu().tolist(),'checkpoint_sha256':self.sha,
                'resources':{'inference_ms':elapsed,'input_tokens':17,'capacity':budget,'learned_budget':self.budget_policy is not None,
                    'expert_routed_fraction':fraction,'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated()},
                'training':False}
    def __call__(self,method,path,body):
        if method=='GET' and path=='/ready':return 200,self.ready()
        if method=='POST' and path=='/v1/visual':return 200,self.predict(body)
        raise KeyError(path)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/baby_arcus/visual.json')
    parser.add_argument('--port',type=int);parser.add_argument('--host',default='127.0.0.1');args=parser.parse_args();worker=Worker(args.config)
    if args.port is not None:
        from baby_arcus.transport import serve
        token=os.environ.get('ARCUS_VISUAL_TOKEN')
        if not token:raise ValueError('Visual HTTP service requires ARCUS_VISUAL_TOKEN')
        server=serve(args.host,args.port,worker,token)
        try:server.serve_forever()
        finally:server.server_close()
    else:
        print(json.dumps(worker.ready()),flush=True)
        for line in sys.stdin:
            try:result=worker.predict(json.loads(line))
            except Exception as exc:result={'status':'error','error':str(exc)}
            print(json.dumps(result),flush=True)

if __name__=='__main__':main()
