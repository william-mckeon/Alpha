"""Reward optimization for a spatial action head on the frozen Arcus body model."""
import argparse,json,random,math
from pathlib import Path
import torch
from baby_arcus.body_policy import BodyPolicy,load,save
from baby_arcus.body_vocabulary import encode
from baby_arcus.body_dynamics import pose
from baby_arcus.large_body_learning import file_hash

def train(source_path,output):
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(919);rng=random.Random(919)
    source,_=load(source_path);source_hash=file_hash(source_path)
    model=BodyPolicy(source.cfg,lying=True,sitting=True,approach=True)
    model.load_state_dict(source.state_dict(),strict=False)
    for name,p in model.named_parameters():p.requires_grad_(name.startswith('approach_actor.'))
    model=model.to('cuda');model.eval()
    senses={'joint_positions':pose(1),'height':1}
    with torch.no_grad():hidden=model.core.trunk(torch.tensor([encode(senses)],device='cuda'))[:,-1].detach()
    hidden=torch.nn.functional.normalize(hidden,dim=-1)*.01
    # Remove an arbitrary initial location-independent preference.
    torch.nn.init.zeros_(model.approach_actor.weight);torch.nn.init.zeros_(model.approach_actor.bias)
    optimizer=torch.optim.AdamW(model.approach_actor.parameters(),lr=.02)
    directions=torch.tensor([[0,-.32],[0,.32],[-.32,0],[.32,0]],device='cuda')
    with (root/'training.jsonl').open('w') as f:
        for update in range(1200):
            xy=torch.tensor([[rng.uniform(-9,9),rng.uniform(-6,6)] for _ in range(128)],device='cuda')
            reward=xy.norm(dim=-1)[:,None]-(xy[:,None,:]-directions).norm(dim=-1)
            features=torch.cat((hidden.expand(128,-1),xy),dim=-1)
            logits=model.approach_actor(features)
            prob=logits.softmax(-1)
            loss=-(prob*reward).sum(-1).mean()+.003*(prob*prob.clamp_min(1e-9).log()).sum(-1).mean()
            optimizer.zero_grad();loss.backward();optimizer.step()
            if update%100==0:f.write(json.dumps({'update':update,'expected_reward':float((-loss).detach())})+'\n')
    model.cpu()
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in source.state_dict().items())
    assert source_hash==file_hash(source_path)
    save(root/'postures.pt',model,optimizer,1200)
    manifest={'checkpoint_sha256':file_hash(root/'postures.pt'),'source_sha256':source_hash,
              'old_skills_preserved':True,'trunk_weights_preserved':True,'updates':1200,
              'parameters':sum(p.numel() for p in model.parameters()),
              'method':'expected immediate distance-progress reward over four actions; frozen trunk'}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();train(a.source,a.output)
