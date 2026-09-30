"""Read-only historical checkpoint inventory. Never deserialize model weights."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.language_stream import atomic_json

def audit(root):
    root=Path(root).resolve();groups={}
    for p in root.rglob('*.pt'):
        if p.is_symlink() or not p.resolve().is_relative_to(root):continue
        groups.setdefault(str(p.parent),[]).append(p)
    result=[]
    for folder,paths in sorted(groups.items()):
        ordered=sorted(paths,key=lambda p:(p.stat().st_mtime_ns,p.name),reverse=True)
        result.append({'folder':folder,'count':len(paths),'logical_bytes':sum(p.stat().st_size for p in paths),
                       'newest_by_mtime':[str(p) for p in ordered[:2]],
                       'candidate_pointer':str(Path(folder)/'candidate.json') if (Path(folder)/'candidate.json').exists() else None,
                       'deletion_eligible':False,'reason':'Verify training lineage, durable hashes, update order and dependencies before deleting; mtime alone is insufficient.'})
    return {'root':str(root),'read_only':True,'deleted':0,'groups':result}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=audit(a.root);atomic_json(a.output,result);print(json.dumps({'groups':len(result['groups']),'deleted':0}))
