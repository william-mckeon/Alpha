"""Pinned donor LightEval protocol; keep model capacity separate from benchmark limits."""
import argparse,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from arcus3.config import read,check_live
from arcus3.checkpoint import digest
from baby_arcus.language_stream import atomic_json

def run(a):
    if os.environ.get('ARCUS3_CONTROLLED_DOCKER')!='1' or not Path('/.dockerenv').exists():raise RuntimeError('Docker CUDA only')
    import torch
    from accelerate import Accelerator
    from lighteval.models.base_model import BaseModel
    from lighteval.utils.utils import EnvConfig
    from lighteval.pipeline import Pipeline,PipelineParameters,ParallelismManager
    from lighteval.logging.evaluation_tracker import EvaluationTracker,EnhancedJSONEncoder
    from arcus3.donor import load
    from arcus3.tokenizer_contract import contract
    from scripts.prepare_arcus3_benchmarks import install
    benchmarks=install(a.benchmarks)
    from baby_arcus.gpu_job_control import gpu_job
    from datetime import datetime
    end=datetime.fromisoformat(a.deadline.replace('Z','+00:00'));out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    upstream=Path('/app/vendor/smollm2');pin=read(upstream/'manifest.json')
    for name,value in pin['files'].items():
        if digest(upstream/name)!=value:raise ValueError('Donor evaluation code changed')
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.7)
    with gpu_job(),torch.inference_mode():
        model,tok=load(a.donor,converted=a.converted,expanded=a.expanded)
        model.config._commit_hash=read(Path(a.donor)/'manifest.json')['revision']
        capacity=contract(a.donor)['context_tokens']
        if model.config.max_position_embeddings!=capacity:raise ValueError('Donor context mismatch')
        wrapped=BaseModel.from_model(model,EnvConfig(cache_dir='/eval-cache'),accelerator=Accelerator(),
            tokenizer_name=str(Path(a.donor)/'files'),use_chat_template=True,add_special_tokens=False)
        # Official donor evaluation uses 2048; this never changes model capacity.
        wrapped._max_length=2048
        class Deadline:
            def __call__(self,*args,**kwargs):check_live(end,out)
        guard=Deadline()
        for method in ('loglikelihood','loglikelihood_single_token','greedy_until'):
            original=getattr(wrapped,method)
            def checked(requests,*args,_original=original,**kwargs):
                result=[]
                for request in requests:
                    guard();result.extend(_original([request],*args,**kwargs))
                return result
            setattr(wrapped,method,checked)
        import vendor.smollm2.tasks as task_module
        params=PipelineParameters(launcher_type=ParallelismManager.ACCELERATE,env_config=EnvConfig(cache_dir='/output/eval-cache'),
            custom_tasks_directory=task_module,override_batch_size=1,use_chat_template=True,
            max_samples={'light':16,'developmental':128,'full':None}[a.tier])
        tasks=','.join((upstream/'smollm2_instruct.txt').read_text().splitlines())
        tracker=EvaluationTracker(output_dir=str(out),save_details=True,push_to_hub=False)
        pipeline=Pipeline(tasks,params,tracker,model=wrapped)
        pipeline.evaluate();pipeline.save_and_push_results()
        atomic_json(out/'donor-scores.json',{'complete':True,'tier':a.tier,'protocol':pin,
            'checkpoint_sha256':digest(Path(a.expanded)/'manifest.json') if a.expanded else 'donor',
            'model_context':capacity,'donor_context':capacity,'benchmark_context':2048,'max_samples_per_task':params.max_samples,
            'lighteval_revision':read('/app/configs/arcus3/production.json')['lighteval_revision'],
            'benchmark_manifest_sha256':digest(Path(a.benchmarks)/'manifest.json'),
            'runtime_adjustments':['Blackwell-compatible Torch','local donor revision supplied','unused metrics disabled','Torch typing alias restored','generation padding longest; reserve one generation position','chat detail fingerprint serialized as UTF-8'],
            'results':json.loads(json.dumps(pipeline.get_results(),cls=EnhancedJSONEncoder)),
            'limitations':'Interim subsets are not published full benchmark scores; no paid judges.'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--donor',default='/donor');p.add_argument('--benchmarks',default='/benchmarks');p.add_argument('--converted');p.add_argument('--expanded')
    p.add_argument('--output',default='/output');p.add_argument('--deadline',required=True);p.add_argument('--tier',choices=['light','developmental','full'],default='light')
    a=p.parse_args()
    try:run(a)
    except Exception as e:atomic_json(Path(a.output)/'error.json',{'type':type(e).__name__});raise
