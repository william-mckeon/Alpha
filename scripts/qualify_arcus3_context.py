"""Disposable sequential production training-memory probes; no campaign updates."""
import argparse,json,os,sys,time,gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main(a):
    if os.environ.get('ARCUS3_CONTROLLED_DOCKER')!='1' or not Path('/.dockerenv').exists():raise RuntimeError('Docker CUDA only')
    import torch
    from arcus3.donor import load,verify
    from arcus3.adapters import train_added_experts
    from arcus3.backbone_adaptation import update
    from arcus3.learning_rate import build_optimizer,initial_state,apply_for_update
    from arcus3.distillation import targets
    from arcus3.config import read,deadline,check_live
    from arcus3.campaign import validate
    from arcus3.tokenizer_contract import contract
    from baby_arcus.gpu_job_control import gpu_job
    from baby_arcus.language_stream import atomic_json
    end=deadline(a.deadline);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    cfg=validate(read(a.config));identity=contract(a.donor)
    if a.activation_checkpointing:cfg['activation_checkpointing']=True
    if a.loss_chunk_size:cfg['loss_chunk_size']=a.loss_chunk_size
    if a.teaching_chunk_size:cfg['teaching_chunk_size']=a.teaching_chunk_size
    if a.expert_chunk_size:cfg['expert_chunk_size']=a.expert_chunk_size
    result={'schema':'arcus3-context-probe-v1','campaign_updates':0,'activation_checkpointing':cfg.get('activation_checkpointing',False),'tokenizer_contract':identity,'measurements':[],
            'note':'Repeated diagnostic text, not training corpus or context-quality evidence. No checkpoint or resume qualification.'}
    verify(a.donor);torch.set_num_threads(2);torch.manual_seed(cfg['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(a.flash_attention);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(not a.flash_attention)
    result['attention_backend']='flash' if a.flash_attention else 'math'
    result['loss_chunk_size']=a.loss_chunk_size
    result['teaching_chunk_size']=a.teaching_chunk_size
    result['expert_chunk_size']=a.expert_chunk_size
    with gpu_job():
        torch.cuda.set_per_process_memory_fraction(.7)
        model,tok=load(a.donor,converted=a.converted);result['trainable_parameters']=train_added_experts(model)
        optimizer=build_optimizer(model,cfg);scheduler=initial_state(cfg);input_tokens=0
        fragment=tok('This is a temporary memory qualification sequence. Python loops repeat operations. ',add_special_tokens=False)['input_ids']
        for length in a.lengths:
            check_live(end,out)
            if not 2<=length<=identity['context_tokens']:raise ValueError('Probe exceeds backbone context')
            ids=(fragment*((length+len(fragment)-1)//len(fragment)))[:length]
            torch.cuda.reset_peak_memory_stats();start=time.monotonic()
            try:
                optimizer.zero_grad(set_to_none=True)
                # Same token positions and target format; disposable capacity probe only.
                with torch.no_grad():
                    teacher=targets(model(torch.tensor([ids],device='cuda'),use_cache=False).logits[0,:-1],cfg['teacher_top_k'])
                next_scheduler=apply_for_update(optimizer,cfg,scheduler,input_tokens,len(ids))
                metrics=update(model,optimizer,{'input_ids':ids,'labels':ids},teacher,cfg)
                input_tokens+=len(ids)
                if next_scheduler is not None:scheduler=next_scheduler
                torch.cuda.synchronize()
                item={'length':length,'passed':True,'seconds':time.monotonic()-start,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'gradients':metrics['gradient_groups']}
            except torch.OutOfMemoryError as error:
                import traceback
                item={'length':length,'passed':False,'reason':'CUDA out of memory','detail':str(error),'traceback':traceback.format_exc(),'peak_cuda_bytes':torch.cuda.max_memory_allocated()}
            result['measurements'].append(item);atomic_json(out/'context-report.json',result);print(json.dumps(item),flush=True)
            if not item['passed']:break
            gc.collect();torch.cuda.empty_cache()
        result['full_context_memory_passed']=any(x['length']==8192 and x['passed'] for x in result['measurements'])
        atomic_json(out/'context-report.json',result)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n,d in [('donor','/donor'),('converted','/converted'),('output','/output'),('config','/app/configs/arcus3/backbone_adaptation.json')]:p.add_argument('--'+n,default=d)
    p.add_argument('--deadline',required=True);p.add_argument('--activation-checkpointing',action='store_true');p.add_argument('--flash-attention',action='store_true');p.add_argument('--loss-chunk-size',type=int,default=0);p.add_argument('--teaching-chunk-size',type=int,default=0);p.add_argument('--expert-chunk-size',type=int,default=0);p.add_argument('--lengths',type=int,nargs='+',default=[512,1024,2048,4096,8192]);main(p.parse_args())
