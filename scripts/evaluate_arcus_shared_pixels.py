"""Evaluate the decoder owned by the shared checkpoint, not its pretraining copy."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baby_arcus.shared_checkpoint import load
from baby_arcus.object_perception_learning import dataset,assess
from baby_arcus.language_stream import atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--confirmation',action='store_true')
    p.add_argument('--seed',type=int);p.add_argument('--count',type=int);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());root=Path(cfg['root']);manifest=json.loads((root/'candidate.json').read_text())
    torch.set_num_threads(2);device='cuda' if torch.cuda.is_available() else 'cpu';model,_=load(root,manifest,device)
    model.eval().requires_grad_(False)
    if model.version<3:raise ValueError('Shared checkpoint has no pixel decoder')
    seed,count=(78291,1000) if a.confirmation else (58291,200)
    if a.seed is not None:seed=a.seed
    if a.count is not None:count=a.count
    if count<20 or (a.confirmation and count<1000):raise ValueError('Insufficient pixel evaluation examples')
    report=assess(model.perception,model.core,dataset(seed,count),device)
    report.update(candidate=manifest,seed=seed,split='confirmation' if a.confirmation else 'validation',training_updates=0)
    report['passed']=report['count_accuracy']>=.9 and report['ball_iou']>=.75 and report['surface_color_accuracy']>=.9 and report['count_accuracy']-report['blank_count_accuracy']>=.2
    from baby_arcus.shared_qualification import source_snapshot
    report['runtime_sources']=source_snapshot()
    atomic_json(root/(report['split']+'-pixels.json'),report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
