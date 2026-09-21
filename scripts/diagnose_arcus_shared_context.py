"""Compare logged runtime predictions with and without bounded body history."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baby_arcus.shared_checkpoint import load
from arcus.tokenizer import get_tokenizer

parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
cfg=json.loads(Path(args.config).read_text());root=Path(cfg['root'])
torch.set_num_threads(2);model,_=load(root,json.loads((root/'candidate.json').read_text()),'cuda')
model.eval();tokenizer=get_tokenizer(cfg['encoding']);results=[]
with torch.inference_mode():
    for path in (root/'runtime-evidence').glob('*.jsonl'):
        for line in path.open():
            event=json.loads(line)
            if event['phase']!='proposed' or not event['experience'].get('history'):continue
            row=event['experience'];text=row['hearing'][-1]['text'] if row['hearing'] else ''
            if sum(r['text']==text for r in results)>=3:continue
            full=int(model([row],tokenizer,requested=('activity',))['activity'][0].argmax())
            row['history']=[]
            absent=int(model([row],tokenizer,requested=('activity',))['activity'][0].argmax())
            results.append({'text':text,'history':full,'no_history':absent})
(root/'context-diagnostic.json').write_text(json.dumps(results));print(json.dumps(results))
