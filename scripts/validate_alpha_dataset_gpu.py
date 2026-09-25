"""Read-only GPU loss check for prepacked dataset samples, inside the learner container."""
import argparse
import json
import time
from pathlib import Path
import sys
sys.path.insert(0, '/app')
from baby_arcus.runtime_contract import require_gpu
from baby_arcus.gpu_job_control import gpu_job


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    require_gpu()
    import torch
    from arcus.tokenizer import get_tokenizer
    from baby_arcus.shared_checkpoint import load
    from baby_arcus.shared_factory import read_config,verify_run
    from baby_arcus.shared_curriculum import example
    from baby_arcus.shared_objectives import loss
    torch.set_num_threads(2)
    cfg=read_config('configs/baby_arcus/alpha_three_stage_learner.container.json')
    root=Path(cfg['root']); candidate=json.loads((root/'candidate.json').read_text())
    samples=json.loads(Path(args.samples).read_text())
    if not 1<=len(samples)<=8: raise ValueError('Bounded sample set required')
    started=time.monotonic()
    report={'complete':False,'device':'cuda','training_updates':0,'candidate':candidate,'samples':[]}
    with gpu_job():
        print(json.dumps({'event':'dataset_validation','stage':'loading_gpu_checkpoint'}),flush=True)
        model,data=load(root,candidate,'cuda'); verify_run(cfg,data); del data
        model.eval(); tokenizer=get_tokenizer(cfg['encoding'])
        for index,item in enumerate(samples):
            row,_,_=example(index,'training','commands')
            row['hearing']=[]; row.pop('language_prefix_ids',None)
            with torch.no_grad():
                total,metrics=loss(model,tokenizer,row,item['target'])
            result={'source':item['source'],'loss':float(total),'metrics':metrics,
                    'finite':bool(torch.isfinite(total)), 'device':str(next(model.parameters()).device)}
            report['samples'].append(result)
            print(json.dumps({'event':'dataset_validation','sample':index,**result}),flush=True)
        if json.loads((root/'candidate.json').read_text()) != candidate:
            raise ValueError('Candidate changed during read-only validation')
        report.update(complete=True,seconds=time.monotonic()-started,
                      peak_gpu_bytes=torch.cuda.max_memory_allocated(),checkpoint_unchanged=True)
    Path(args.output).write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
