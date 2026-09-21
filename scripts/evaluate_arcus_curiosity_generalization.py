"""Fresh seeded scene evaluation with actual body-model inference, no training."""
import hashlib,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baby_arcus.play_session import PlaySession
from baby_arcus.body_policy import load
from baby_arcus.body_controller_config import configuration
from baby_arcus.body_senses import observe_body_senses
from baby_arcus.approach_vocabulary import ACTIONS,ARRIVAL
from baby_arcus.curiosity_environment import observe,choose

def main():
    project=Path(__file__).resolve().parents[1];root=project/'runs/arcus_curiosity_v1'
    artifact=json.loads((root/'policy.json').read_text());cfg=configuration(project)
    torch.set_num_threads(2);model,_=load(cfg['checkpoint']);model.to('cuda' if torch.cuda.is_available() else 'cpu').eval()
    trials=[]
    for seed in range(14001,14013):
        w=PlaySession();w.body.motor_mode='independent';w.environment.add_toys(seed=seed)
        blocked=set();moves=0
        for _ in range(8):
            key=choose(artifact,[r for r in observe(w) if r['id'] not in blocked])
            if key is None:break
            obj=w.environment.objects[key];p=w.environment.placements[w.body.entity_id]
            dx,dy=p['x']-obj['x'],p['y']-obj['y'];distance=math.hypot(dx,dy)
            target=[obj['x']+.85*dx/distance,obj['y']+.85*dy/distance]
            stalls=0
            for step in range(120):
                relative=[target[0]-p['x'],target[1]-p['y']]
                if math.hypot(*relative)<=ARRIVAL:break
                senses=observe_body_senses(w.body)
                with torch.no_grad():action=ACTIONS[int(model.approach([senses],[relative]).argmax(-1)[0])]
                before=dict(p);w.action(action);w.step();moves+=1
                stalls=stalls+1 if p==before else 0
                if stalls>=10:break
            if math.hypot(obj['x']-p['x'],obj['y']-p['y'])<=1.4:
                w.action({'kind':'inspect_object','object_id':key})
            else:blocked.add(key)
        discovered=sum(o['discovered'] is not None for o in w.environment.objects.values())
        trials.append({'seed':seed,'discoveries':discovered,'moves':moves,'blocked':len(blocked),
                       'success':discovered==3,'no_repeated_visits':all(o['visits']<=1 for o in w.environment.objects.values())})
    successes=sum(t['success'] for t in trials)
    report={'passed':successes/len(trials)>=.9 and all(t['no_repeated_visits'] for t in trials),
            'successes':successes,'trials':trials,'minimum_success_fraction':.9,
            'checkpoint_sha256':hashlib.sha256((root/'policy.json').read_bytes()).hexdigest(),
            'body_checkpoint_sha256':cfg['expected_hash'],
            'scope':'Actual learned-body inference in accelerated seeded symbolic scenes; no HTTP or pixel grounding.'}
    (root/'generalization.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()
