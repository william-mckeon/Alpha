"""Save an unqualified candidate with contextual motor residual disabled."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.shared_checkpoint import load,save,restore_optimizer
from baby_arcus.language_stream import atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());source=json.loads((Path(cfg['root'])/'candidate.json').read_text())
    model,data=load(cfg['root'],source)
    if model.version!=4:raise ValueError('Migration requires schema v4')
    model.version=5;optimizer=restore_optimizer(model,data,1e-7)
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    progress=data['progress'];progress['receipts'].append({'migration':'v4-to-v5-retained-motor-path','source':source,'updates':0})
    manifest=save(root,model,optimizer,progress)
    atomic_json(root/'candidate.json',manifest);atomic_json(root/'config.json',dict(cfg,root=str(root)))
    atomic_json(root/'migration.json',{'source_root':cfg['root'],'source':source,'candidate':manifest,'qualified':False})
    print(json.dumps(manifest))

if __name__=='__main__':main()
