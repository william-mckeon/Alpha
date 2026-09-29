"""Read-only standalone package verification; preserves any existing export."""
import argparse
import gc
import json
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from arcus3.donor import load,verify
from arcus3.checkpoint import digest
from arcus3.config import read,deadline,check_live
from baby_arcus.gpu_job_control import gpu_job

def main(args):
    if os.environ.get('ARCUS3_CONTROLLED_DOCKER')!='1' or not Path('/.dockerenv').exists():raise RuntimeError('Docker CUDA required')
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from verify_arcus3_conversion import measure,compare
    spec=read('/app/configs/arcus3/alpha_3_release.json');end=deadline(args.deadline);out=Path(args.output)
    if digest(Path(args.converted)/'manifest.json')!=spec['parent_manifest_sha256']:raise ValueError('Parent mismatch')
    verify(args.donor);torch.set_num_threads(2)
    with gpu_job():
        torch.cuda.set_per_process_memory_fraction(.7)
        model,tok=load(args.donor,converted=args.converted);before=measure(model,tok)
        del model;gc.collect();torch.cuda.empty_cache();check_live(end,out)
        model,info=AutoModelForCausalLM.from_pretrained(args.package,trust_remote_code=True,local_files_only=True,
            torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,attn_implementation='sdpa',output_loading_info=True)
        model=model.to('cuda').eval();tok=AutoTokenizer.from_pretrained(args.package,local_files_only=True)
        if any(info[k] for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):raise ValueError('Export key mismatch')
        count=sum(p.numel() for p in model.parameters())
        if count!=spec['unique_parameters']:raise ValueError('Export parameter mismatch')
        report={'complete':True,'release':spec,'parity':compare(before,measure(model,tok)),
                'loading':info,'unique_parameters':count,'peak_cuda_bytes':torch.cuda.max_memory_allocated()}
        check_live(end,out);(out/'verification.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({'complete':True,'parameters':count}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--donor',default='/donor');p.add_argument('--converted',required=True)
    p.add_argument('--package',required=True);p.add_argument('--output',default='/output');p.add_argument('--deadline',required=True)
    p.add_argument('--max-new-tokens',type=int,default=128);main(p.parse_args())
