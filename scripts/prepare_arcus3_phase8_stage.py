"""Bounded initial-stage acquisition from the already reviewed sample sources."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.prepare_arcus3_phase8_sample import prepare

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--donor',required=True)
    p.add_argument('--max-tokens',type=int,default=10000000);p.add_argument('--max-length',type=int,default=8192);a=p.parse_args()
    result=prepare(a.root,a.donor,a.max_tokens,a.max_length,stage=True)
    print(json.dumps(result,indent=2))
    if not result['complete']:raise SystemExit(2)
