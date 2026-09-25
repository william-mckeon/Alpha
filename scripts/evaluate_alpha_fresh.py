"""Read-only CUDA diagnostics for fresh SFT and unseen tool names."""
import argparse
import json
import math
import time
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def evaluation_identity(config):
    import hashlib
    def sha(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
        return h.hexdigest()
    cfg=json.loads(Path(config).read_text())
    files=[Path(__file__),Path('scripts/evaluate_alpha_coding.py'),
           Path('baby_arcus/coding_curriculum.py'),Path('baby_arcus/tool_discovery_curriculum.py'),
           Path('baby_arcus/coding_policy.py'),Path('/review/records.jsonl')]
    return {'config':cfg,'files':{str(p):sha(p) for p in files},'sft_windows':4,'tools':3,'coding_actions':4,'generation_tokens':128}


def reusable(report, pointer, identity):
    return (report.get('complete') is True and report.get('checkpoint_unchanged') is True
            and report.get('candidate')==pointer and report.get('evaluation_identity')==identity
            and report.get('coding_execution_evaluated') is True)


def evaluate(config, output, initial=False, coding=False):
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from arcus.tokenizer import get_tokenizer
    from baby_arcus.shared_factory import read_config, verify_run
    from baby_arcus.shared_checkpoint import load, digest
    from baby_arcus.shared_curriculum import example
    from baby_arcus.shared_objectives import loss
    from baby_arcus.sft_dataset import windows
    from baby_arcus.coding_policy import decide
    from baby_arcus.coding_tools import DEFINITIONS
    from baby_arcus.tool_catalog import ToolCatalog
    from baby_arcus.tool_context import ToolContext
    from baby_arcus.tool_search import search
    from baby_arcus.tool_discovery_curriculum import tasks
    from baby_arcus.runtime_contract import require_gpu
    from baby_arcus.gpu_job_control import gpu_job
    from baby_arcus.local_agent_dataset import IDENTITY
    require_gpu(); torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.70)
    started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    cfg=read_config(config);root=Path(cfg['root']);output=Path(output)
    if output.exists(): raise ValueError('Use a fresh evaluation output')
    pointer_name='initial.json' if initial else 'candidate.json'
    pointer=json.loads((root/pointer_name).read_text())
    report={'evaluation_identity':evaluation_identity(config),'candidate':pointer,'complete':False,'sft':[],'unseen_tools':[],
            'coding_execution_evaluated':False,'mastery_established':False}
    with gpu_job():
        model,data=load(root,pointer,'cuda');verify_run(cfg,data);del data
        parameter_count=sum(p.numel() for p in model.parameters())
        model.eval().requires_grad_(False); tokenizer=get_tokenizer(cfg['encoding'])
        row,_,_=example(0,'training','commands');row['hearing']=[]
        with torch.inference_mode(), sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            for line in Path('/review/records.jsonl').open():
                record=json.loads(line)
                if record['split']!='validation':continue
                window=next(windows(record,tokenizer,16384))
                _, metrics=loss(model,tokenizer,row,{'sft':window})
                value=float(metrics['sft'])
                if not math.isfinite(value):raise RuntimeError('Nonfinite validation loss')
                report['sft'].append({'record_sha256':__import__('hashlib').sha256(line.encode()).hexdigest(),
                                      'input_tokens':len(window['ids']) if 'ids' in window else len(window.get('tokens',[])),
                                      'target_tokens':sum(window['mask'][1:]),'nll':value})
                if len(report['sft'])==4:break
            for task in list(tasks('validation'))[:3]:
                catalog=ToolCatalog([DEFINITIONS[0],task['tool']]);context=ToolContext(catalog)
                messages=[{'role':'system','content':IDENTITY},{'role':'user','content':task['query']+': '+next(iter(task['arguments'].values()))}]
                item={'tool':task['tool']['name'],'searched':False,'solved':False,'decisions':[]}
                for _ in range(3):
                    decision=decide(model,tokenizer,row,messages,context.definitions(),128)
                    call=decision.get('call');item['decisions'].append({'status':decision['status'],'call':call})
                    try:
                        if call is None:raise ValueError('No valid call')
                        context.resolve(call)
                        if call['name']=='tool_search':
                            result=search(catalog,**call['arguments']);context.accept(result);item['searched']=True
                        else:
                            value=next(iter(call['arguments'].values()))
                            operation=task['query'].split()[0]
                            result={'text':value[::-1] if operation=='reverse' else value.upper() if operation=='uppercase' else value.lower()}
                            item['solved']=result['text']==task['expected']
                    except ValueError as exc:result={'error':str(exc)}
                    messages.extend([{'role':'assistant','content':json.dumps(call) if call else 'Invalid call'},
                                     {'role':'tool','content':json.dumps(result)}])
                    if item['solved']:break
                report['unseen_tools'].append(item)
        report['checkpoint_unchanged']=digest(root/(pointer['generation']+'.pt'))==pointer['sha256']
        report['complete']=len(report['sft'])==4 and report['checkpoint_unchanged']
    if coding:
        del model
        torch.cuda.empty_cache()
        from scripts.evaluate_alpha_coding import evaluate as evaluate_coding
        from scripts.audit_alpha_fresh_holdouts import NAMES
        report['coding']=evaluate_coding(config,output.parent/'coding',tasks=NAMES,checkpoint_pointer=pointer_name)
        report['coding_execution_evaluated']=True
        report['complete'] = report['complete'] and report['coding']['complete'] and report['coding']['checkpoint_unchanged']
    report.update(seconds=time.monotonic()-started,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                  parameters=parameter_count,
                  context_tokens=16384,depth_capacity=pointer['depth_capacity'],
                  limitations='Four validation windows and three tasks per cohort; not proof of 16K competence or general coding ability.')
    from baby_arcus.language_stream import atomic_json
    output.parent.mkdir(parents=True,exist_ok=True);atomic_json(output,report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--initial',action='store_true');p.add_argument('--coding',action='store_true')
    a=p.parse_args();print(json.dumps(evaluate(a.config,a.output,a.initial,a.coding)))
