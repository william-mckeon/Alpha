"""Read-only CUDA trace and inference-export parity on a paused checkpoint."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def validate(root, output):
    import torch
    from arcus.tokenizer import get_tokenizer
    from baby_arcus.runtime_contract import require_gpu
    from baby_arcus.gpu_job_control import gpu_job
    from baby_arcus.shared_checkpoint import load,digest
    from baby_arcus.conversation_probe import respond
    from baby_arcus.routing_trace import RoutingTrace
    from scripts.package_alpha_hf import LOADER
    from safetensors.torch import save_model
    require_gpu();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.70)
    root,output=Path(root),Path(output)
    if not (root/'pause-training').exists():raise ValueError('Training must remain paused')
    output.mkdir(parents=True,exist_ok=False)
    pointer=json.loads((root/'candidate.json').read_text())
    with gpu_job():
        model,data=load(root,pointer,'cuda');model.eval().requires_grad_(False)
        metadata={'body_config':data['body_config'],'schema_version':int(data['schema'].rsplit('v',1)[1]),
                  'vocab_size':data['vocab_size'],'text_dim':data['text_dim'],
                  'integrated_motor':data.get('integrated_motor',False),
                  'experiment_depth_capacity':data.get('experiment_depth_capacity')}
        del data
        tokenizer=get_tokenizer('o200k_base')
        # Warm up before comparing observational overhead.
        respond(model,tokenizer,'Hi, how are you?')
        torch.cuda.reset_peak_memory_stats()
        baseline=respond(model,tokenizer,'Hi, how are you?')
        baseline_peak=torch.cuda.max_memory_allocated()
        rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
        torch.cuda.reset_peak_memory_stats()
        with RoutingTrace(model) as trace:
            traced=respond(model,tokenizer,'Hi, how are you?')
        same_rng=torch.equal(rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state())
        traced_peak=torch.cuda.max_memory_allocated()
        if traced['token_ids']!=baseline['token_ids'] or not same_rng:raise ValueError('Trace changed generation or RNG')
        def fingerprints(m):
            return {k:hashlib.sha256(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                    for k,v in m.state_dict().items()}
        before=fingerprints(model)
        save_model(model,output/'model.safetensors')
        (output/'alpha_config.json').write_text(json.dumps(metadata,indent=2))
        (output/'load_alpha.py').write_text(LOADER)
        del model;torch.cuda.empty_cache()
        loader={};exec(compile(LOADER,'load_alpha.py','exec'),loader)
        reloaded,_=loader['load_alpha'](output,device='cuda')
        same_tensors=fingerprints(reloaded)==before
        export=respond(reloaded,tokenizer,'Hi, how are you?')
        if not same_tensors or export['token_ids']!=baseline['token_ids']:raise ValueError('Export changed model')
        result={'candidate':pointer,'trace_token_parity':True,'trace_rng_parity':same_rng,
                'export_tensor_parity':same_tensors,'export_token_parity':True,
                'baseline_seconds':baseline['seconds'],'traced_seconds':traced['seconds'],
                'baseline_peak_bytes':baseline_peak,'traced_peak_bytes':traced_peak,
                'events':len(trace.events),'checkpoint_unchanged':digest(root/(pointer['generation']+'.pt'))==pointer['sha256'],
                'release_eligible':False,'note':'Local validation artifact, not a 60k release package.'}
        if not result['checkpoint_unchanged']:raise ValueError('Source checkpoint changed')
        (output/'validation.json').write_text(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps(validate(a.root,a.output),indent=2))
