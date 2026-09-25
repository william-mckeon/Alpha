"""Disposable real 16K-packed SFT updates on the preserved fresh checkpoint."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from arcus.tokenizer import get_tokenizer
    from baby_arcus.shared_checkpoint import load, digest
    from baby_arcus.shared_curriculum import example
    from baby_arcus.shared_objectives import step
    from baby_arcus.sft_dataset import windows
    from baby_arcus.runtime_contract import require_gpu
    from baby_arcus.gpu_job_control import gpu_job
    require_gpu(); torch.set_num_threads(2); torch.cuda.set_per_process_memory_fraction(.70)
    root=Path('/parent'); pointer=json.loads((root/'candidate.json').read_text())
    assert pointer['updates']==0
    report={'candidate':pointer,'updates_saved':0,'results':[]}
    with gpu_job():
        model,data=load(root,pointer,'cuda'); del data
        model.core.gradient_checkpointing=True
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5)
        tokenizer=get_tokenizer('o200k_base')
        row,_,_=example(1,'training','commands'); row['hearing']=[]
        categories=set()
        for line in Path('/dataset/records.jsonl').open():
            record=json.loads(line)
            if record['split']!='training': continue
            source=record['source']
            category='local' if source.startswith('local:') else 'hf' if source.startswith('hf:') else 'tool' if source=='alpha:tool-discovery-v1' else None
            if category is None or category in categories: continue
            window=next(windows(record,tokenizer,16384))
            torch.cuda.reset_peak_memory_stats()
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                result=step(model,optimizer,tokenizer,row,{'sft':window})
            report['results'].append({'category':category,'input_tokens':len(window['ids'])-1,
                                      'peak_allocated_bytes':torch.cuda.max_memory_allocated(),**result})
            categories.add(category)
            if len(categories)==3: break
        report['complete']=len(categories)==3
        report['checkpoint_unchanged']=digest(root/(pointer['generation']+'.pt'))==pointer['sha256']
        Path('/evidence/real-sft.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
        if not report['complete']: raise RuntimeError('Missing real-data category')


if __name__=='__main__': main()
