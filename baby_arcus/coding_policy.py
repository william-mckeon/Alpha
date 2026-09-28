"""Bounded tool-call decoding by Alpha's existing shared core and language adapter."""
import json
import torch
from baby_arcus.coding_contracts import validate_call
from baby_arcus.contracts import decode


def system_prompt(definitions):
    from baby_arcus.local_agent_dataset import IDENTITY
    return (IDENTITY + '\nChoose one tool call as JSON with name, version and arguments. '
            'Available definitions: ' + json.dumps(definitions,sort_keys=True,separators=(',',':')))


def prepare_prompt(messages, definitions, tokenizer, budget):
    """Keep bootstrap and recent schemas, without duplicating them in observations."""
    from baby_arcus.conversation_format import pack
    from baby_arcus.contracts import digest
    history=[]
    for index,item in enumerate(messages):
        if index==0 and item.get('role')=='system': continue
        item=dict(item)
        if item['role']=='tool':
            try:
                result=json.loads(item['content'])
                if isinstance(result,dict) and isinstance(result.get('tools'),list):
                    # Full results and their hashes remain in the episode receipt.
                    # Repeating catalog/content hashes here used enough context to
                    # evict the very schema that discovery had just returned.
                    result={'tools':[{'name':tool['name'],'version':tool['version']} for tool in result['tools']],
                            'status':result.get('status','matches'),'schemas':'current system definitions'}
                    item['content']=json.dumps(result,separators=(',',':'))
            except (ValueError,KeyError,TypeError): pass
        history.append(item)
    # Discovery orders recently returned tools first; old schemas are evicted first.
    for count in range(len(definitions),0,-1):
        selected=definitions[:count]
        try:
            packed,ids=pack([{'role':'system','content':system_prompt(selected)},*history],tokenizer,budget)
            return packed,ids,selected
        except ValueError:
            if count==1: raise
    raise ValueError('No bootstrap definition available')


def decide(model, tokenizer, row, messages, definitions, max_new_tokens=128, cancelled=lambda: False):
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 256:
        raise ValueError('Invalid generation budget')
    if cancelled(): return {'status':'cancelled','call':None}
    from baby_arcus.conversation_format import pack
    from baby_arcus.contracts import digest
    context = model.body.cfg.max_seq_len
    try:
        packed, ids, selected = prepare_prompt(messages,definitions,tokenizer,context-max_new_tokens)
    except ValueError as exc:
        return {'status':'context_exhausted', 'error':str(exc), 'call':None}
    evidence = {'input_sha256':digest(ids), 'input_ids':ids, 'input_messages':packed,
                'prompt_tokens':len(ids), 'definitions_sha256':digest(definitions),
                'selected_definitions':selected}
    sensory = dict(row, hearing=[])
    sensory.pop('language_prefix_ids',None)
    model.eval()
    with torch.no_grad():
        hidden = model([sensory],tokenizer,requested=('hidden',))['hidden']
        residual = torch.nn.functional.linear(model.text_context(hidden),model.language.embedding.weight)
        generated = []
        from arcus.kv_cache import KVCache
        cache=KVCache(context) if model.core.supports_cache() else None
        for _ in range(max_new_tokens):
            if cancelled(): return {**evidence,'status':'cancelled','call':None}
            current=([generated[-1]] if cache is not None and generated else ids+generated)
            tokens = torch.tensor([current],device=hidden.device)
            logits = (model.language.cached_logits(model.core,tokens,cache) if cache is not None
                      else model.language(model.core,tokens,last_only=True)[:, -1]) + residual
            generated.append(int(logits.argmax(-1)[0]))
            text = tokenizer.decode(generated)
            try:
                import html
                call = decode(html.unescape(text).encode())
                validate_call(call)
                return {**evidence, 'status':'call', 'call':call, 'text':text, 'generated_tokens':len(generated)}
            except (ValueError, TypeError, KeyError):
                pass
            if '\n</assistant>' in text:
                return {**evidence,'status':'invalid_call','reason':'end_of_turn_without_action','call':None,
                        'text':text,'generated_tokens':len(generated)}
    text=tokenizer.decode(generated)
    from baby_arcus.sft_target_contract import classify
    kind=classify(text)
    reason='external_transcript' if kind=='external_transcript' else 'generation_budget_exhausted'
    try:
        json.loads(__import__('html').unescape(text));reason='invalid_action_schema'
    except ValueError:pass
    return {**evidence, 'status':'invalid_call', 'reason':reason,'call':None, 'text':text, 'generated_tokens':len(generated)}
