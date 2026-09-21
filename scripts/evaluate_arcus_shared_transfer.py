"""Held-out command and cross-modal checks; no optimizer or automatic promotion."""
import argparse,json,sys,base64,hashlib
from pathlib import Path
from copy import deepcopy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baby_arcus.shared_checkpoint import load
from baby_arcus.shared_curriculum import example,COMMANDS,SEEDS
from baby_arcus.language_stream import atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--count',type=int,default=200)
    p.add_argument('--split',choices=('training','validation','confirmation'),default='validation');p.add_argument('--seed',type=int);a=p.parse_args()
    if a.count<20:p.error('At least 20 examples required')
    cfg=json.loads(Path(a.config).read_text());root=Path(cfg['root']);manifest=json.loads((root/'candidate.json').read_text())
    seed=a.seed if a.seed is not None else cfg.get('confirmation_seed') if a.split=='confirmation' else None
    if seed is not None:
        if a.split!='confirmation' or seed<0 or (seed-SEEDS['training'])%1009==0:raise ValueError('Invalid independent confirmation seed')
        SEEDS[a.split]=seed
    torch.set_num_threads(2);model,_=load(root,manifest,'cuda' if torch.cuda.is_available() else 'cpu');model.eval().requires_grad_(False)
    from arcus.tokenizer import get_tokenizer
    tokenizer=get_tokenizer(cfg['encoding']);results={};per_command={};errors=[]
    with torch.no_grad():
        for family in ('commands','color_reference','rest'):
            observations=[example(i,a.split,family,paired=cfg.get('paired_curriculum',False)) for i in range(a.count)]
            conditions=('full','no_rgb','no_hearing','shuffled_rgb') if family=='color_reference' else ('full',)
            for condition in conditions:
                correct=[]
                for i,(source,target,meta) in enumerate(observations):
                    row=deepcopy(source)
                    if condition=='no_rgb':row['vision'].update(available=False,image_base64='',sha256=hashlib.sha256(b'').hexdigest())
                    if condition=='no_hearing':row['hearing']=[]
                    if condition=='shuffled_rgb':row['vision']=deepcopy(observations[(i+1)%len(observations)][0]['vision'])
                    output=model([row],tokenizer,requested=tuple(target))
                    ok=all(int(output[key][0].argmax())==(max(range(len(value)),key=value.__getitem__) if isinstance(value,list) else value) for key,value in target.items());correct.append(ok)
                    if not ok and condition=='full':errors.append({'family':family,'index':i,'hearing':row['hearing'],'internal':row['internal'],'target':target,'predicted':{key:int(value[0].argmax()) for key,value in output.items()}})
                    if family=='commands':per_command.setdefault(COMMANDS[i%len(COMMANDS)][0],[]).append(ok)
                results[family+':'+condition]=sum(correct)/len(correct)
    full=results['color_reference:full'];negative=max(results['color_reference:'+c] for c in ('no_rgb','no_hearing','shuffled_rgb'))
    report={'candidate':manifest,'split':a.split,'seed':SEEDS[a.split],'examples_per_condition':a.count,'accuracy':results,
        'per_command':{k:sum(v)/len(v) for k,v in per_command.items()},
        'command_gate':a.split!='training' and results['commands:full']>=.9 and min(sum(v)/len(v) for v in per_command.values())>=.8,
        'cross_modal_gate':a.split!='training' and full>=.9 and full-negative>=.25,'training_updates':0,
        'rest_gate':a.split!='training' and results['rest:full']>=.9,
        'scope':'Simple scripted command/gaze lessons in rendered playpen; no general reasoning claim'}
    from baby_arcus.shared_qualification import source_snapshot
    report['runtime_sources']=source_snapshot()
    atomic_json(root/(a.split+'-transfer.json'),report);print(json.dumps(report),flush=True)
    atomic_json(root/(a.split+'-errors.json'),{'candidate':manifest,'errors':errors})

if __name__=='__main__':main()
