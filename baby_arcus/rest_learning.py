"""Fit a small action-value head to the declared synthetic rest reward curriculum."""
import hashlib,json
from pathlib import Path
import torch
from baby_arcus.rest_environment import samples,rewards,NAMES

def main():
    cfg=json.loads(Path('configs/baby_arcus/rest.json').read_text())
    root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(cfg['seed'])
    model=torch.nn.Sequential(torch.nn.Linear(5,32),torch.nn.Tanh(),torch.nn.Linear(32,5))
    x=torch.tensor(samples(cfg['seed'],6000));y=torch.tensor([rewards(row) for row in x.tolist()])
    optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    for step in range(5000):
        ids=torch.randint(len(x),(128,));loss=torch.nn.functional.mse_loss(model(x[ids]),y[ids])
        optimizer.zero_grad();loss.backward();optimizer.step()
    final=samples(cfg['seed']+1000,4000)
    with torch.no_grad():prediction=model(torch.tensor(final)).argmax(-1).tolist()
    correct=sum(p==max(range(5),key=lambda i:rewards(row)[i]) for row,p in zip(final,prediction))
    artifact={'schema':'arcus-rest-policy-v1','actions':NAMES,'weights':[p.detach().tolist() for p in model.parameters()],
              'training':'synthetic action-value regression; separate small head, not Arcus trunk training'}
    path=root/'policy.json';path.write_text(json.dumps(artifact))
    from baby_arcus.rest_runtime import predict
    reload_ok=all(predict(artifact,row)==p for row,p in zip(final,prediction))
    report={'passed':correct/len(final)>=.97 and reload_ok,'accuracy':correct/len(final),'trials':len(final),
            'reload_identical':reload_ok,'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'parameters':sum(p.numel() for p in model.parameters()),'seed':cfg['seed'],'final_seed':cfg['seed']+1000,
            'scope':'synthetic reward agreement only; live qualification required'}
    (root/'qualification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()
