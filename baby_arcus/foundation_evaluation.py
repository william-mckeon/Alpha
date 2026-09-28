"""Read-only, bounded base-model evaluations, separate from agent and RL tasks."""
import json
from contextlib import contextmanager
from pathlib import Path
import torch


@contextmanager
def read_only(model):
    training = model.training
    # Evaluation must not perturb the training RNG or model mode on return.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        model.eval()
        try:
            with torch.no_grad():yield
        finally:model.train(training)


def score_ids(model, ids, start=1):
    if not 1 <= start < len(ids) or len(ids)-1 > model.body.cfg.max_seq_len:
        raise ValueError('Invalid scoring span/context')
    x=torch.tensor([ids[:-1]],device=next(model.parameters()).device)
    y=torch.tensor([ids[1:]],device=x.device);y[:,:start-1]=-100
    residual=model.language.embedding.weight.new_zeros((1,model.language.embedding.embedding_dim))
    with torch.autocast('cuda',dtype=torch.bfloat16):
        nll=model.language.loss(model.core,x,y,residual)
    return {'nll':float(nll),'target_tokens':len(ids)-start}


def completion(model, tokenizer, prompt, limit=128):
    ids=tokenizer.encode(prompt)
    if not ids or len(ids)+limit>model.body.cfg.max_seq_len or not 1<=limit<=128:
        raise ValueError('Generation budget/context exceeded')
    result=[]
    # Use the proven uncached language path for parity; bounded at 128 new tokens.
    for _ in range(limit):
        x=torch.tensor([ids+result],device=next(model.parameters()).device)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model.language(model.core,x,last_only=True)
        token=int(logits[0,-1].argmax())
        if token==tokenizer.eot_token:break
        result.append(token)
    return {'prompt':prompt,'response':tokenizer.decode(result),'token_ids':result,'truncated':len(result)==limit}


def evaluate(model, tokenizer, corpus, sources, settings):
    from baby_arcus.foundation_data import PackedStream
    from baby_arcus.evaluation_metrics import domain_language_metrics
    windows=[];coverage={};generations={}
    with read_only(model):
        for source in sources:
            stream=None
            try:
                stream=PackedStream(corpus,[{**source,'weight':1}],settings['sequence_length'],split='validation',repeat=False)
                measured=0
                for _ in range(settings['max_windows_per_domain']):
                    try:x,y=stream.next_window()
                    except StopIteration:break
                    windows.append({'domain':source['name'],**score_ids(model,x+[y[-1]])});measured+=1
                coverage[source['name']]={'windows':measured,'available':True}
            except ValueError as exc:coverage[source['name']]={'windows':0,'available':False,'reason':str(exc)}
            finally:
                if stream:stream.close()
        for track,key in [('base-completion','base_prompts'),('chat-diagnostic','chat_prompts')]:
            generations[track]=[completion(model,tokenizer,p,settings['max_generated_tokens']) for p in settings[key]]
        choices=[]
        for task in settings['choices']:
            context=tokenizer.encode(task['prompt'])
            scores=[score_ids(model,context+tokenizer.encode(choice),len(context)) for choice in task['choices']]
            predicted=min(range(len(scores)),key=lambda i:scores[i]['nll']*scores[i]['target_tokens'])
            choices.append({**task,'scores':scores,'predicted':predicted,'correct':predicted==task['answer']})
        retrieval=[]
        for position in settings['retrieval_positions']:
            prefix=tokenizer.encode('The secret word is lavender.\n')
            filler=tokenizer.encode('Ordinary text with no secret word.\n')
            tokens=(prefix+filler*(position//len(filler)+1))[:position]
            prompt=tokenizer.decode(tokens)+'\nWhat is the secret word? Answer:'
            row=completion(model,tokenizer,prompt,min(16,settings['max_generated_tokens']))
            retrieval.append({**row,'requested_prefix_tokens':position,'actual_prompt_tokens':len(tokenizer.encode(prompt)),'contains_expected_word':'lavender' in row['response'].lower()})
    return {'schema':'arcus-foundation-evaluation-v1','language':domain_language_metrics(windows,[s['name'] for s in sources]),
            'coverage':coverage,'generations':generations,'choice_diagnostics':choices,'long_context':retrieval,
            'benchmark_scores':None,'limitations':'Small diagnostic fixtures are not benchmark scores. A configured 16k window does not establish 16k competence. No evaluation feedback is trained.'}
