"""Construct an isolated initialization artifact and verify it in Docker CUDA."""
import argparse
import gc
import json
import os
import resource
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from arcus3.config import authorize,read,deadline,check_live,validate_conversion
from arcus3.donor import verify,load
from arcus3.checkpoint import save_conversion,digest
from baby_arcus.gpu_job_control import gpu_job


def main(args):
    if os.environ.get('ARCUS3_CONTROLLED_DOCKER')!='1' or not Path('/.dockerenv').exists():
        raise RuntimeError('Controlled Docker CUDA required')
    authorize(read('/app/configs/arcus3/project.json'),'conversion')
    cfg=validate_conversion(read('/app/configs/arcus3/architecture.json'))
    end=deadline(args.deadline);out=Path(args.output);verify(args.donor)
    import torch
    from arcus3.model import expand,inventory
    from verify_arcus3_conversion import measure,compare
    from baby_arcus.language_stream import atomic_json
    torch.set_num_threads(2);start=time.monotonic()
    with gpu_job():
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        torch.cuda.set_per_process_memory_fraction(.7)
        model,tokenizer=load(args.donor)
        tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side='left'
        check_live(end,out);before=measure(model,tokenizer)
        expand(model,cfg['layers']);model.eval()
        check_live(end,out);after=measure(model,tokenizer)
        parity=compare(before,after);info=inventory(model,cfg['layers'])
        if info['unique_parameters']!=cfg['expected_parameters'] or not info['independent_experts'] or not info['tied_embeddings']:
            raise ValueError('Conversion inventory failed')
        clone_equal=all(torch.equal(a,b) for i in cfg['layers']
            for a,b in zip(model.model.layers[i].mlp.experts[0].parameters(),model.model.layers[i].mlp.experts[1].parameters()))
        if not clone_equal:raise ValueError('Expert copy mismatch')
        check_live(end,out);artifact=out/'converted';save_conversion(artifact,model,cfg,args.donor)
        del model;gc.collect();torch.cuda.empty_cache()
        check_live(end,out);model,tokenizer=load(args.donor,converted=artifact)
        tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side='left'
        reloaded=compare(before,measure(model,tokenizer));check_live(end,out)
        report={'schema':'arcus3-conversion-results-v1','training_updates':0,
            'inventory':info,'clone_weights_equal':clone_equal,'parity':parity,'reload_parity':reloaded,
            'manifest_sha256':digest(artifact/'manifest.json'),'architecture':cfg,
            'seconds':time.monotonic()-start,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),
            'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'limitation':'Zero initialized routers select expert 0; cloned expert 1 is independent but untrained. No learned specialization or depth routing.'}
        atomic_json(out/'conversion-report.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--donor',default='/donor');p.add_argument('--output',default='/output')
    p.add_argument('--deadline',required=True);p.add_argument('--max-new-tokens',type=int,default=128)
    main(p.parse_args())
