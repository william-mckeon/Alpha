"""Read-only conversational probe, sharing the developmental inference path."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    import torch
    from baby_arcus.runtime_contract import require_gpu
    from baby_arcus.gpu_job_control import gpu_job
    from baby_arcus.shared_checkpoint import load,digest
    from baby_arcus.conversation_probe import respond
    from arcus.tokenizer import get_tokenizer
    p=argparse.ArgumentParser();p.add_argument('--prompt',default='Hi, how are you?')
    p.add_argument('--root',default='/model');p.add_argument('--output',default='greeting.json');a=p.parse_args()
    require_gpu();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.70)
    root=Path(a.root);pointer=json.loads((root/'candidate.json').read_text())
    if not (root/'pause-training').exists():raise ValueError('Pause training before an ad hoc probe')
    with gpu_job():
        model,data=load(root,pointer,'cuda');del data;model.eval().requires_grad_(False)
        result={'candidate':pointer,**respond(model,get_tokenizer('o200k_base'),a.prompt)}
        result['checkpoint_unchanged']=digest(root/(pointer['generation']+'.pt'))==pointer['sha256']
        target=Path('/output')/a.output
        if target.exists():raise ValueError('Use a fresh output path')
        target.write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
