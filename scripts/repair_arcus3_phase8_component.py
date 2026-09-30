"""Retry one failed reviewed source without rereading successful source shards."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import scripts.prepare_arcus3_phase8_sample as preparation

def main(a):
    original=preparation.read
    def read(path):
        result=original(path)
        if str(path)=='configs/arcus3/phase8_sample.json':result={**result,'sources':{a.category:result['sources'][a.category]}}
        return result
    preparation.read=read;preparation.SHARES={a.category:1.0}
    result=preparation.prepare(a.root,a.donor,a.max_tokens,8192,stage=True)
    print(json.dumps(result,indent=2))
    if not result['complete']:raise SystemExit(2)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--donor',required=True)
    p.add_argument('--category',required=True,choices=list(preparation.SHARES));p.add_argument('--max-tokens',type=int,required=True);main(p.parse_args())
