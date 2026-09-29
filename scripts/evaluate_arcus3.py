"""Read-only Docker CUDA baseline, with original donor template and frozen inputs."""
import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from arcus3.config import authorize, read, deadline, check_live, REVISION
from arcus3.donor import verify, load
from arcus3.evaluation import load_suite, sha, aggregate, masked_nll, messages_for, score, summarize
from arcus3.orchestration import invoke_messages
from baby_arcus.gpu_job_control import gpu_job


def run(args):
    if os.environ.get('ARCUS3_CONTROLLED_DOCKER')!='1' or not Path('/.dockerenv').exists():
        raise RuntimeError('Controlled Docker CUDA required')
    authorize(read('/app/configs/arcus3/project.json'),'inference')
    end = deadline(args.deadline)
    out = Path(args.output)
    manifest,prompts = load_suite('/app')
    cfg = read('/app/configs/arcus3/evaluation.json')
    if args.max_new_tokens != cfg['max_new_tokens']:
        raise ValueError('Frozen token budget mismatch')
    verify(args.donor)
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList
    torch.set_num_threads(2)
    class Stop(StoppingCriteria):
        def __call__(self,*args,**kwargs):
            check_live(end,out)
            return False
    rows, language = [], []
    started = time.monotonic()
    with gpu_job(), torch.inference_mode():
        if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
        torch.cuda.set_per_process_memory_fraction(.7)
        model,tokenizer = load(args.donor)
        for item in manifest['language']:
            check_live(end,out)
            # Raw text, all tokens except the first are targets. No chat-template PPL.
            ids = tokenizer(item['text'],add_special_tokens=False,return_tensors='pt').input_ids.to('cuda')
            if not 2 <= ids.shape[1] <= cfg['max_input_tokens']: raise ValueError('Language budget exceeded')
            metrics = masked_nll(model(input_ids=ids,use_cache=False).logits,ids,1)
            language.append({**item,**metrics,'input_ids':ids[0].tolist(),'target_start':1})
        for item in prompts:
            check_live(end,out)
            messages = messages_for(item,cfg)
            serialized = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            inputs = tokenizer(serialized,return_tensors='pt',add_special_tokens=False).to('cuda')
            if inputs.input_ids.shape[1] > cfg['max_input_tokens']: raise ValueError('Prompt exceeds frozen budget')
            begin = time.monotonic()
            def generate(incoming):
                if incoming != messages: raise ValueError('Changed messages')
                return model.generate(**inputs,max_new_tokens=cfg['max_new_tokens'],do_sample=False,
                    use_cache=True,pad_token_id=tokenizer.eos_token_id,stopping_criteria=StoppingCriteriaList([Stop()]))
            result = invoke_messages(generate,messages)
            torch.cuda.synchronize()
            ids = result[0,inputs.input_ids.shape[1]:].tolist()
            response = tokenizer.decode(ids,skip_special_tokens=True)
            rows.append({**item,'messages':messages,'serialized_prompt':serialized,'input_ids':inputs.input_ids[0].tolist(),
                         'output_ids':ids,'response':response,'seconds':time.monotonic()-begin,
                         'truncated':len(ids)==cfg['max_new_tokens'] and ids[-1]!=tokenizer.eos_token_id,
                         'metrics':score(item,response)})
            (out/'transcripts.json').write_text(json.dumps(rows,indent=2))
            print(json.dumps({'completed':item['id'],'seconds':rows[-1]['seconds']}),flush=True)
        report = {'schema':'arcus3-baseline-v1','complete_generation':True,'execution_complete':False,
                  'suite_sha256':sha('/app/evaluation/arcus3/baseline-v1.json'),
                  'settings_sha256':sha('/app/configs/arcus3/evaluation.json'),'tokenizer_revision':REVISION,
                  'precision':'bfloat16','orchestration':'langchain-runnable-in-langgraph',
                  'unique_parameters':sum(p.numel() for p in model.parameters()),
                  'language':aggregate(language),'language_records':language,'categories':summarize(rows),
                  'resources':{'seconds':time.monotonic()-started,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),
                               'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024},
                  'limitations':manifest['limitations']}
        (out/'scores.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--donor',default='/donor'); parser.add_argument('--output',default='/output')
    parser.add_argument('--deadline',required=True); parser.add_argument('--max-new-tokens',type=int,default=128)
    run(parser.parse_args())
