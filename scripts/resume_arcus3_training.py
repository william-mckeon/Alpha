"""Validate a durable resume and print the exact selected generation; no auto-launch."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from arcus3.config import read,safe_child
from arcus3.checkpoint import digest
from arcus3.expanded_checkpoint import verify

def select(root,checkpoint_root=None):
    root=Path(root);base=Path(checkpoint_root) if checkpoint_root else root/'checkpoints'
    pointer=read(base/'latest.json');cp=safe_child(base,pointer['generation'])
    if digest(cp/'manifest.json')!=pointer['manifest_sha256']:raise ValueError('Latest pointer mismatch')
    manifest=read(cp/'manifest.json')
    if manifest.get('campaign')!='backbone-adaptation-v1':raise ValueError('Different campaign')
    verify(cp,manifest['parent_sha256'])
    return {'checkpoint':str(cp.resolve()),'manifest_sha256':digest(cp/'manifest.json'),'launch_started':False,
            'note':'Pass this generation as ResumePath to an authorized new adaptation session; old pause flags remain intact.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--checkpoint-root')
    p.add_argument('--checkpoint');p.add_argument('--policy');p.add_argument('--adaptation-config');p.add_argument('--output')
    p.add_argument('--restart-from-zero',action='store_true');a=p.parse_args()
    if a.output:
        if not all((a.checkpoint,a.policy,a.adaptation_config)):p.error('Recovery output requires checkpoint, policy and adaptation config')
        from arcus3.checkpoint_recovery import prepare
        result=prepare(a.root,a.checkpoint,a.policy,a.adaptation_config,a.output,a.restart_from_zero)
    else:result=select(a.root,a.checkpoint_root)
    print(json.dumps(result,indent=2))
