"""Explain abstention on validation scenes without changing model or gates."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from baby_arcus.shared_continuity_model import load_candidate
from baby_arcus.shared_continuity_curriculum import encode_scene
from baby_arcus.shared_identity_context import pair_features, uncertainty
from baby_arcus.language_stream import atomic_json

p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--scenes',type=int,default=128)
a=p.parse_args();cfg=json.loads(Path(a.config).read_text());root=Path(cfg['root'])
manifest=json.loads((root/'candidate.json').read_text());torch.set_num_threads(2)
model,_=load_candidate(root,manifest,'cuda');model.eval().requires_grad_(False)
from arcus.tokenizer import get_tokenizer
tokenizer=get_tokenizer(cfg['encoding']);rows=[]
for index in range(a.scenes):
    frames,ambiguous=encode_scene(model,tokenizer,index,'validation',primed=True)
    risks=[];missing=0;features=[]
    for frame in frames:
        inventory=frame['inventory']
        for query,label in zip(frame['descriptors'],frame['labels']):
            if not label:continue
            risk=uncertainty(model,frame['hidden'][None].cuda(),inventory,query)
            risks.append(risk)
            pairs=[pair_features(x,y,query) for i,x in enumerate(inventory) for y in inventory[i+1:]]
            missing+=not pairs or max(sum(v*v for v in pair[3:5]) for pair in pairs)<.04**2
            if risk>.1 and pairs:
                with torch.no_grad():
                    scores=model.uncertainty_logits(frame['hidden'][None].cuda().expand(len(pairs),-1),torch.tensor(pairs,device='cuda')).sigmoid()
                features.append(pairs[int(scores.argmax())])
    rows.append({'scene':index,'ambiguous':ambiguous,'risks':risks,'no_uniqueness_evidence':missing,'high_risk_pairs':features})
    if (index+1)%32==0:print(json.dumps({'scenes':index+1}),flush=True)
atomic_json(root/'validation-risk-audit.json',{'candidate':manifest,'rows':rows})
print(json.dumps({'distinct_queries':sum(len(r['risks']) for r in rows if not r['ambiguous']),
 'distinct_abstained':sum(sum(x>.1 for x in r['risks']) for r in rows if not r['ambiguous']),
 'distinct_no_evidence':sum(r['no_uniqueness_evidence'] for r in rows if not r['ambiguous']),
 'worst_distinct':sorted([r for r in rows if not r['ambiguous']],key=lambda r:sum(x>.1 for x in r['risks']),reverse=True)[:12]}))
