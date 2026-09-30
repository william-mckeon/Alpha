"""CPU metadata inspection of explicitly mounted historical PyTorch archives."""
import argparse,json
from pathlib import Path

def main(a):
    import torch
    records=[]
    for p in sorted(Path(a.root).glob('*.pt')):
        try:
            state=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
            scalars={k:v for k,v in state.items() if isinstance(v,(str,int,float,bool,type(None)))}
            progress=state.get('progress',{})
            progress_scalars={k:v for k,v in progress.items() if isinstance(v,(str,int,float,bool,type(None)))} if isinstance(progress,dict) else {}
            records.append({'name':p.name,'bytes':p.stat().st_size,'scalars':scalars,'progress':progress_scalars,'keys':list(state)})
            del state
        except Exception as e:records.append({'name':p.name,'error':type(e).__name__})
    Path(a.output).write_text(json.dumps(records,indent=2));print(json.dumps({'inspected':len(records),'errors':sum('error' in r for r in records)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);main(p.parse_args())
