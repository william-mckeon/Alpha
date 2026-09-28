"""Fresh production-size CUDA benchmark; synthetic inputs are not training data."""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def benchmark(cfg,output,microsteps=1):
    import torch
    import torch.distributed as dist
    from baby_arcus.runtime_contract import require_gpu
    from baby_arcus.gpu_job_control import gpu_job
    from baby_arcus.nanotron_adapter import create_model,ArcusForTraining,train_microbatches
    from baby_arcus.foundation_schedule import accumulation,estimate
    from baby_arcus.foundation_checkpoint import save,load
    from baby_arcus.model_inventory import assert_inventory
    from baby_arcus.language_stream import atomic_json
    from tests.baby_arcus.foundation_fixtures import State
    require_gpu();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.7)
    output=Path(output)
    if output.exists():raise ValueError('Preserve existing benchmark')
    report={'complete':False,'synthetic_inputs':True,'production_training':False,'requested_microsteps':microsteps}
    def write():atomic_json(output,report)
    write()
    with gpu_job():
        torch.manual_seed(cfg['seed'])
        dist.init_process_group('gloo',init_method='file://'+tempfile.mktemp(),rank=0,world_size=1)
        try:
            model=create_model(cfg)
            report['parameters']=assert_inventory(model,cfg['required_unique_parameters'])['unique_parameters']
            report['context']=cfg['sequence_length'];write()
            optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate']/cfg['warmup_steps'],betas=tuple(cfg['betas']),eps=cfg['epsilon'],weight_decay=cfg['weight_decay'],foreach=False)
            x=torch.randint(2,cfg['vocab_size'],(1,cfg['sequence_length']),device='cuda');mask=torch.ones_like(x,dtype=torch.bool)
            batches=({'input_ids':x,'label_ids':x.roll(-1,1),'input_mask':mask,'label_mask':mask} for _ in range(microsteps))
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.bfloat16):outputs=train_microbatches(ArcusForTraining(model),batches,microsteps,dist.group.WORLD)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip_grad'],error_if_nonfinite=True);optimizer.step()
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            report.update(seconds=elapsed,nll=sum(float(o['nll']) for o in outputs)/microsteps,gradient_norm=float(norm),
                          measured_tokens=microsteps*cfg['sequence_length'],peak_cuda_allocated=torch.cuda.max_memory_allocated(),peak_cuda_reserved=torch.cuda.max_memory_reserved(),gpu=torch.cuda.get_device_name())
            report['projection']=estimate(elapsed,report['measured_tokens'])
            report['recipe_accumulation']=accumulation(cfg['sequence_length'])
            report['full_global_batch_measured']=microsteps==report['recipe_accumulation']
            report['estimated_recipe_step_seconds']=elapsed*report['recipe_accumulation']/microsteps
            write()
            with tempfile.TemporaryDirectory(prefix='foundation-save-') as directory:
                state=State();start=time.perf_counter();pointer=save(directory,model,optimizer,{'updates':1},state,cfg)
                report['checkpoint_seconds']=time.perf_counter()-start
                report['checkpoint_bytes']=(Path(directory)/(pointer['generation']+'.pt')).stat().st_size
                before=model.language.embedding.weight[:16].detach().clone()
                with torch.no_grad():model.language.embedding.weight[:16].zero_()
                load(directory,model,optimizer,state,cfg)
                torch.testing.assert_close(before,model.language.embedding.weight[:16],rtol=0,atol=0)
                report['checkpoint_restore_passed']=True
            report['complete']=True;write()
        except Exception as exc:
            report.update(error_type=type(exc).__name__,error=str(exc));write();raise
        finally:dist.destroy_process_group()
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/baby_arcus/arcus_128m_smollm2_pretrain.yaml');p.add_argument('--output',required=True);p.add_argument('--microsteps',type=int,default=1);a=p.parse_args()
    if not 1<=a.microsteps<=64:raise ValueError('Bound benchmark to 1..64 microsteps')
    print(json.dumps(benchmark(json.loads(Path(a.config).read_text()),a.output,a.microsteps),indent=2))
