"""Small learned novelty/cost value head; separate from the main Arcus trunk."""
import json,hashlib,random
from pathlib import Path
import torch
from baby_arcus.curiosity_environment import utility,score

def main():
    cfg=json.loads(Path('configs/baby_arcus/curiosity.json').read_text());root=Path(cfg['output'])
    root.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.manual_seed(cfg['seed'])
    def data(seed,n):
        rng=random.Random(seed)
        return [[float(rng.randrange(2)),rng.random(),rng.random(),rng.random()] for _ in range(n)]
    rows=data(cfg['seed'],5000);x=torch.tensor(rows);y=torch.tensor([[utility(r)] for r in rows])
    model=torch.nn.Sequential(torch.nn.Linear(4,16),torch.nn.Tanh(),torch.nn.Linear(16,1))
    optimizer=torch.optim.Adam(model.parameters(),lr=.005)
    for _ in range(2000):
        ids=torch.randint(len(x),(128,));loss=torch.nn.functional.mse_loss(model(x[ids]),y[ids])
        optimizer.zero_grad();loss.backward();optimizer.step()
    artifact={'schema':'arcus-curiosity-v1','weights':[p.detach().tolist() for p in model.parameters()],
              'observation':'symbolic room objects; not pixel recognition','training':'designer utility regression'}
    path=root/'policy.json';path.write_text(json.dumps(artifact))
    final=data(cfg['seed']+1000,2000)
    predicted=[score(artifact,row) for row in final]
    with torch.no_grad():reference=model(torch.tensor(final)).flatten().tolist()
    error=max(abs(a-b) for a,b in zip(predicted,reference))
    agreement=sum((p>0)==(utility(r)>0) for r,p in zip(final,predicted))/len(final)
    report={'passed':agreement>=.98 and error<1e-5,'positive_value_agreement':agreement,
            'reload_max_error':error,'final_cases':len(final),'parameters':sum(p.numel() for p in model.parameters()),
            'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (root/'qualification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()
