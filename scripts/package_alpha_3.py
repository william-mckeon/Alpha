"""Export only verified Phase 5 inference weights; smoke-test the standalone package."""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from arcus3.donor import load,verify
from arcus3.checkpoint import digest
from arcus3.config import read,deadline,check_live
from baby_arcus.gpu_job_control import gpu_job

def main(args):
    if os.environ.get('ARCUS3_CONTROLLED_DOCKER')!='1' or not Path('/.dockerenv').exists():raise RuntimeError('Docker CUDA required')
    end=deadline(args.deadline);out=Path(args.output);package=out/'package'
    if package.exists():raise ValueError('Fresh package required')
    spec=read('/app/configs/arcus3/alpha_3_release.json')
    if digest(Path(args.converted)/'manifest.json')!=spec['parent_manifest_sha256']:raise ValueError('Release parent mismatch')
    verify(args.donor)
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from arcus3.hf_model import Alpha3Config
    from verify_arcus3_conversion import measure,compare
    torch.set_num_threads(2)
    with gpu_job():
        torch.cuda.set_per_process_memory_fraction(.7)
        model,tok=load(args.donor,converted=args.converted);tok.pad_token=tok.eos_token;tok.padding_side='left'
        before=measure(model,tok);check_live(end,out)
        model.config=Alpha3Config(**model.config.to_dict())
        model.save_pretrained(package,safe_serialization=True,max_shard_size='1GB');tok.save_pretrained(package)
        cfg=read(package/'config.json');cfg.update(_name_or_path='',architectures=['Alpha3ForCausalLM'],auto_map={'AutoConfig':'modeling_alpha3.Alpha3Config','AutoModelForCausalLM':'modeling_alpha3.Alpha3ForCausalLM'})
        (package/'config.json').write_text(json.dumps(cfg,indent=2))
        shutil.copyfile('/app/arcus3/hf_model.py',package/'modeling_alpha3.py');shutil.copyfile('/app/arcus3/routing.py',package/'routing.py')
        shutil.copyfile('/app/docs/ALPHA_3_MODEL_CARD.md',package/'README.md')
        shutil.copyfile('/app/NOTICE',package/'NOTICE')
        license_path=Path(args.donor)/'files/LICENSE'
        if license_path.exists():shutil.copyfile(license_path,package/'LICENSE')
        else:shutil.copyfile('/app/LICENSE',package/'LICENSE')
        shutil.copyfile(Path(args.donor)/'files/README.md',package/'DONOR_MODEL_CARD.md')
        del model;gc.collect();torch.cuda.empty_cache();check_live(end,out)
        model,loading=AutoModelForCausalLM.from_pretrained(package,trust_remote_code=True,local_files_only=True,
            torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,attn_implementation='sdpa',output_loading_info=True)
        model=model.to('cuda').eval();tok=AutoTokenizer.from_pretrained(package,local_files_only=True)
        if any(loading[k] for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):raise ValueError('Export loading mismatch')
        if sum(p.numel() for p in model.parameters())!=spec['unique_parameters']:raise ValueError('Export size mismatch')
        parity=compare(before,measure(model,tok));check_live(end,out)
        report={'complete':True,'release':spec,'parity':parity,'loading':loading,'unique_parameters':sum(p.numel() for p in model.parameters()),'peak_cuda_bytes':torch.cuda.max_memory_allocated()}
        (package/'verification.json').write_text(json.dumps(report,indent=2))
        # Dynamic module imports may leave bytecode; only enumerate intended files.
        allowed={p.name for p in package.iterdir() if p.is_file() and p.suffix in ('.json','.safetensors','.py','.md','.txt')}
        allowed.update({'NOTICE','LICENSE'})
        files={n:{'sha256':digest(package/n),'bytes':(package/n).stat().st_size} for n in sorted(allowed)}
        (package/'manifest.json').write_text(json.dumps({'release':spec,'files':files},indent=2))
        (out/'package-report.json').write_text(json.dumps(report,indent=2));print(json.dumps({'complete':True,'files':len(files),'bytes':sum(v['bytes'] for v in files.values())}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--donor',default='/donor');p.add_argument('--converted',required=True)
    p.add_argument('--output',default='/output');p.add_argument('--deadline',required=True);p.add_argument('--max-new-tokens',type=int,default=128);main(p.parse_args())
