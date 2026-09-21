"""Unseen approach trials and unchanged posture regressions; no evaluation training."""
import json,random,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baby_arcus.body_policy import load
from baby_arcus.large_body_learning import file_hash
from baby_arcus.play_session import PlaySession
from baby_arcus.embodiment import Embodiment
from baby_arcus.body_dynamics import pose
from baby_arcus.body_senses import observe_body_senses
from baby_arcus.body_vocabulary import ACTIONS as JOINT_ACTIONS
from baby_arcus.approach_vocabulary import ACTIONS,ARRIVAL
from baby_arcus.standing_poses import catalog as standing_catalog
from baby_arcus.lying_poses import catalog as lying_catalog,environment as lying_environment
from baby_arcus.sitting_poses import catalog as sitting_catalog,environment as sitting_environment
from baby_arcus.audit import AuditLog
from evaluate_arcus_poses import evaluate

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='runs/arcus_approach');parser.add_argument('--seed',type=int,default=9172026);args=parser.parse_args()
    torch.set_num_threads(2);root=Path(args.root)/'qualification';root.mkdir(exist_ok=False)
    path=root.parent/'postures.pt';before=file_hash(path)
    model,_=load(path);source,_=load('runs/arcus_sitting/postures.pt')
    preserved=all(torch.equal(v,model.state_dict()[k]) for k,v in source.state_dict().items());del source
    model.to('cuda').eval();rng=random.Random(args.seed);trials=[]
    with (root/'approach-transitions.jsonl').open('w') as log:
        for index in range(60):
            q=pose(0 if index%3==0 else 1)
            if index%3==1:q={k:1 if k.startswith('front') else 0 for k in q}
            w=PlaySession(Embodiment(height=.25,joint_positions=q,previous_joints=q.copy(),motor_mode='independent'))
            position=w.environment.placements[w.body.entity_id];position.update(x=rng.uniform(.5,9.5),y=rng.uniform(.5,6.5))
            target={'x':rng.uniform(.5,9.5),'y':rng.uniform(.5,6.5)}
            start=dict(position);success=False
            for step in range(240):
                relative=[target['x']-position['x'],target['y']-position['y']]
                if math.hypot(*relative)<=ARRIVAL:success=True;break
                senses=observe_body_senses(w.body)
                if senses['height']<.99 or not senses['stable']:action=JOINT_ACTIONS[model.choose(senses,'standing')]
                else:
                    with torch.no_grad():action=ACTIONS[int(model.approach([senses],[relative]).argmax(-1)[0])]
                if action:w.action(action)
                w.step()
                log.write(json.dumps({'trial':index,'step':step,'action':action,'position':dict(position),'target':target})+'\n')
            trials.append({'start':start,'target':target,'success':success,'steps':step})
            print('approach',index,success,flush=True)
    audit=AuditLog(root/'audit','interaction-qualification')
    try:
        standing=evaluate(model.choose,standing_catalog()['poses'],'standing',audit)
        lying=evaluate(lambda s:model.choose(s,'lying'),lying_catalog(9172602)['poses'],'lying',audit,lying_environment)
        sitting=evaluate(lambda s:model.choose(s,'sitting'),sitting_catalog()['poses'],'sitting',audit,sitting_environment)
    finally:audit.close()
    reloaded,_=load(path);reloaded.to('cuda').eval()
    probe=observe_body_senses(Embodiment())
    with torch.no_grad():reload_ok=torch.equal(model.approach([probe],[[2,-3]]),reloaded.approach([probe],[[2,-3]]))
    report={'checkpoint_sha256':before,'checkpoint_unchanged':before==file_hash(path),'old_skills_preserved':preserved,
            'approach_successes':sum(t['success'] for t in trials),'approach_trials':trials,
            'standing':standing,'lying':lying,'sitting':sitting,'reload_identical':reload_ok,
            'gate_passed':preserved and reload_ok and all(t['success'] for t in trials) and all(x['successes']>=18 for x in (standing,lying,sitting))}
    (root/'report.json').write_text(json.dumps(report,indent=2));print('GATE',report['gate_passed'])
if __name__=='__main__':main()
