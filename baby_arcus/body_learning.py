"""Bounded reward-only policy-gradient pilot, not a claim of 125M skill transfer."""
import argparse
import json
from pathlib import Path
import random
import torch
from torch.distributions import Categorical
from baby_arcus.body_policy import BodyPolicy,save,load
from baby_arcus.standing_environment import StandingEnvironment
from baby_arcus.standing_baselines import scripted,random_action
def assess(policy,seeds):
    results=[]
    for seed in seeds:
        env=StandingEnvironment(seed)
        while True:
            _,_,done=env.step(policy(env.observe()))
            if done:break
        results.append({"seed":seed,"success":env.success,"steps":env.steps,"height":env.session.body.height})
    return {"successes":sum(r["success"] for r in results),"episodes":len(results),"results":results}
def train(output,updates=100,seed=17,resume=None):
    from baby_arcus.audit import AuditLog
    audit=AuditLog(Path(output)/"audit","standing-training")
    try:
        return _train(output,updates,seed,resume,audit)
    except Exception as exc:
        audit.emit("training.error",{"exception_type":type(exc).__name__},durable=True)
        raise
    finally:audit.close()

def _train(output,updates,seed,resume,audit):
    audit.emit("training.started",{"updates":updates,"seed":seed,"resume":resume},durable=True)
    torch.set_num_threads(2);torch.manual_seed(seed)
    model,data=load(resume) if resume else (BodyPolicy(),None)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
    start=0
    if data:
        optimizer.load_state_dict(data["optimizer"]);torch.set_rng_state(data["rng"]);start=data["updates"]
    before=assess(model.choose,range(1000,1008))
    # On-policy one-step reward learning: explicit diagnostic, not full long-horizon PPO.
    envs=[StandingEnvironment(seed+i) for i in range(16)]
    metrics=[]
    for update in range(updates):
        senses=[e.observe() for e in envs]
        distribution=Categorical(logits=model(senses))
        actions=distribution.sample()
        rewards=[]
        for i,(env,action) in enumerate(zip(envs,actions.tolist())):
            _,reward,done=env.step(action);rewards.append(reward)
            if done:envs[i]=StandingEnvironment(1000000+seed+(start+update)*16+i)
        reward=torch.tensor(rewards)
        objective=-(distribution.log_prob(actions)*(reward-reward.mean()).detach()).mean()-.01*distribution.entropy().mean()
        optimizer.zero_grad();objective.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
        metrics.append(float(objective.detach()))
        audit.emit("training.update",{"update":start+update+1,"loss":metrics[-1],
            "observations":senses,"actions":actions.tolist(),"rewards":rewards},durable=True)
    save(Path(output)/"standing.pt",model,optimizer,start+updates)
    after=assess(model.choose,range(1000,1008))
    rng=random.Random(55)
    report={"schema":"arcus-standing-pilot-v1","updates":start+updates,
            "parameters":sum(p.numel() for p in model.parameters()),"reward_only":True,
            "algorithm":"on-policy immediate-reward policy gradient","before":before,"after":after,
            "scripted":assess(scripted,range(1000,1008)),
            "random":assess(lambda s:random_action(s,rng),range(1000,1008)),
            "idle":assess(lambda s:0,range(1000,1008)),
            "finite_losses":all(__import__("math").isfinite(x) for x in metrics),
            "mastery_claimed":False,"existing_125m_checkpoint_changed":False}
    Path(output,"report.json").write_text(json.dumps(report,indent=2))
    audit.emit("training.completed",{"checkpoint":str(Path(output)/"standing.pt"),"report":report},durable=True)
    return report
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",default="runs/arcus_standing_pilot")
    parser.add_argument("--updates",type=int,default=100);parser.add_argument("--resume")
    args=parser.parse_args()
    if not 1<=args.updates<=2000:parser.error("updates must be between 1 and 2000")
    print(json.dumps(train(args.output,args.updates,resume=args.resume)))
if __name__=="__main__":main()
