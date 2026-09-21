"""Publish only a candidate with intact, measured qualification evidence."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.shared_checkpoint import promote

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--rollback',action='store_true',help='Restore the previous qualified generation after rechecking its evidence')
    args=parser.parse_args();root=Path(json.loads(Path(args.config).read_text())['root'])
    manifest=json.loads((root/('previous-active.json' if args.rollback else 'candidate.json')).read_text())
    generation=manifest['generation']
    if len(generation)!=32 or any(c not in '0123456789abcdef' for c in generation):raise ValueError('Invalid generation')
    report=json.loads((root/(generation+'.qualification.json' if args.rollback else 'reviewed-qualification.json')).read_text())
    promote(root,manifest,report)
    print(json.dumps({'active':manifest,'qualified':True}))

if __name__=='__main__':main()
