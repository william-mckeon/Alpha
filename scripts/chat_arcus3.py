"""Bounded request-file chat or the Phase 3 live integration probes, Docker CUDA only."""
import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from arcus3.config import authorize, read, deadline, check_live, validate_application
from arcus3.evaluation import sha
from arcus3.donor import load, verify
from arcus3.model import depth_metadata
from arcus3.chat_model import ArcusChatModel
from arcus3.memory import ConversationMemory
from arcus3.orchestration import application_turn
from arcus3.tools import SCHEMAS
from baby_arcus.gpu_job_control import gpu_job

PROBES=[{'session':'greeting','prompt':'Hi, how are you?','tools':False},
        {'session':'memory','prompt':'Remember that my favorite color is violet. Reply briefly.','tools':False},
        {'session':'memory','prompt':'What is my favorite color? Answer briefly.','tools':False},
        {'session':'echo','prompt':'Use echo with text hello, then tell me what the tool returned.','tools':True},
        {'session':'arithmetic','prompt':'Use calculate to add 17 and 25, then tell me the result.','tools':True}]


def run(args):
    if os.environ.get('ARCUS3_CONTROLLED_DOCKER')!='1' or not Path('/.dockerenv').exists():
        raise RuntimeError('Controlled Docker CUDA required')
    authorize(read('/app/configs/arcus3/project.json'),'inference')
    cfg=validate_application(read('/app/configs/arcus3/application.json'))
    if args.max_new_tokens!=cfg['max_new_tokens']: raise ValueError('Unapproved output budget')
    end=deadline(args.deadline);out=Path(args.output)
    requests=read(args.requests) if args.requests else PROBES
    if not isinstance(requests,list) or not 1<=len(requests)<=8: raise ValueError('1..8 requests required')
    for row in requests:
        if (not isinstance(row,dict) or set(row)!={'session','prompt','tools'} or type(row['tools']) is not bool
            or not isinstance(row['prompt'],str) or not 1<=len(row['prompt'])<=4000):
            raise ValueError('Invalid request')
    verify(args.donor)
    import torch
    from transformers import StoppingCriteria,StoppingCriteriaList
    from langchain_core.messages import SystemMessage,HumanMessage,messages_to_dict
    torch.set_num_threads(2)
    class Stop(StoppingCriteria):
        def __call__(self,*args,**kwargs): check_live(end,out);return False
    transcripts=[];generations=[];started=time.monotonic()
    memory=ConversationMemory(cfg['max_turns'],cfg['max_history_chars'],out/'sessions' if args.persist_memory else None)
    with gpu_job(),torch.inference_mode():
        if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
        torch.cuda.set_per_process_memory_fraction(.7)
        model,tokenizer=load(args.donor,args.adapter,args.converted,args.expanded)
        def generate(messages):
            check_live(end,out)
            serialized=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            inputs=tokenizer(serialized,add_special_tokens=False,return_tensors='pt').to('cuda')
            count=inputs.input_ids.shape[1]
            if count>cfg['max_input_tokens']: raise ValueError('History exceeds input token budget; start a new session')
            begin=time.monotonic()
            result=model.generate(**inputs,max_new_tokens=cfg['max_new_tokens'],do_sample=False,use_cache=True,
                                  pad_token_id=tokenizer.eos_token_id,stopping_criteria=StoppingCriteriaList([Stop()]))
            torch.cuda.synchronize();ids=result[0,count:].tolist()
            record={'response':tokenizer.decode(ids,skip_special_tokens=True),'output_ids':ids,
                    'input_ids':inputs.input_ids[0].tolist(),'serialized_prompt':serialized,
                    'seconds':time.monotonic()-begin,'truncated':len(ids)==cfg['max_new_tokens'] and ids[-1]!=tokenizer.eos_token_id,
                    'usage':{'input_tokens':count,'output_tokens':len(ids),'total_tokens':count+len(ids)}}
            generations.append(record);return record
        chat=ArcusChatModel(backend=generate)
        for request in requests:
            prior=memory.get(request['session'])
            messages=[SystemMessage(content='You are Arcus, a helpful AI assistant.')]+prior+[HumanMessage(content=request['prompt'])]
            state=application_turn(chat.bind_tools(SCHEMAS) if request['tools'] else chat,messages,
                lambda:check_live(end,out),cfg['max_model_calls'],cfg['max_tool_calls'])
            # Never carry incomplete call/result sequences into a future turn.
            if state['status']=='complete': memory.add(request['session'],state['messages'][1+len(prior):])
            transcripts.append({'request':request,'status':state['status'],'model_calls':state['model_calls'],
                                'tool_calls':state['tool_calls'],'messages':messages_to_dict(state['messages'])})
            (out/'application-transcripts.json').write_text(json.dumps(transcripts,indent=2))
            print(json.dumps({'session':request['session'],'status':state['status'],'tool_calls':state['tool_calls']}),flush=True)
        selected_delta=args.expanded or args.adapter
        checkpoint_updates=read(Path(selected_delta)/'manifest.json')['updates'] if selected_delta else 0
        report={'schema':'arcus3-application-v1','adapter':args.adapter,'expanded_manifest_sha256':sha(Path(args.expanded)/'manifest.json') if args.expanded else None,'conversion_manifest_sha256':sha(Path(args.converted)/'manifest.json') if args.converted else None,'training_updates':checkpoint_updates,'updates_this_invocation':0,'unique_parameters':sum(p.numel() for p in model.parameters()),
                'requests':transcripts,'generations':generations,'persist_memory':args.persist_memory,'depth':depth_metadata(model),
                'seconds':time.monotonic()-started,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),
                'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
        (out/'application-report.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--donor',default='/donor');p.add_argument('--output',default='/output')
    p.add_argument('--deadline',required=True);p.add_argument('--max-new-tokens',type=int,default=128)
    p.add_argument('--requests');p.add_argument('--persist-memory',action='store_true')
    p.add_argument('--adapter');p.add_argument('--converted');p.add_argument('--expanded')
    run(p.parse_args())
