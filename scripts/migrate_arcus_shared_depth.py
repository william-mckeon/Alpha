"""Create a separate 0.25-depth candidate; never replace active weights."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baby_arcus.shared_checkpoint import load,save,restore_optimizer
from baby_arcus.shared_depth import set_depth,verify_depth
from baby_arcus.language_stream import atomic_json

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();cfg=json.loads(Path(args.config).read_text());root=Path(args.output)
    root.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((Path(cfg['root'])/'candidate.json').read_text())
    model,data=load(cfg['root'],manifest)
    previous=model.body.cfg.capacity;set_depth(model);verify_depth(model)
    optimizer=restore_optimizer(model,data,cfg['learning_rate'])
    data['progress'].setdefault('receipts',[]).append({'migration':'depth_capacity','from':previous,'to':.25,'source':manifest,'training_updates':0})
    result=save(root,model,optimizer,data['progress'])
    atomic_json(root/'candidate.json',result)
    atomic_json(root/'config.json',dict(cfg,root=root.as_posix(),depth_capacity=.25))
    print(json.dumps({'candidate':result,'previous_capacity':previous,'capacity':.25,'active_changed':False}),flush=True)

if __name__=='__main__':main()
