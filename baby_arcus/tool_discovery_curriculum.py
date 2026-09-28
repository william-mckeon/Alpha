"""Deterministic schema-reading lessons; disjoint tool names across splits."""
import json
from baby_arcus.coding_tools import definition, DEFINITIONS
from baby_arcus.tool_catalog import ToolCatalog
from baby_arcus.tool_search import search
from baby_arcus.local_agent_dataset import IDENTITY


def tasks(split):
    if split not in ('training', 'validation', 'test'):
        raise ValueError('Unknown split')
    prefix = {'training': 'practice', 'validation': 'validation', 'test': 'heldout'}[split]
    for index in range(120 if split == 'training' else 24):
        suffix = chr(97 + index // 26) + chr(97 + index % 26)
        argument = ('text', 'value', 'input_text')[index % 3]
        operation = ('reverse', 'uppercase', 'lowercase')[index % 3]
        name = prefix + '_' + suffix
        tool = definition(name, f'{operation} the supplied text and return the transformed text.',
                          {argument: {'type': 'string', 'maxLength': 1000}}, [argument])
        value = f'Arcus {split} {suffix} AbCd'
        output = value[::-1] if operation == 'reverse' else value.upper() if operation == 'uppercase' else value.lower()
        yield {'tool': tool, 'query': operation + ' text', 'arguments': {argument: value},
               'expected': output, 'split': split}


def records(split, tokenizer=None, context=16384, generation_tokens=128):
    if tokenizer is not None:
        yield from action_records(split, tokenizer, context, generation_tokens)
        return
    for task in tasks(split):
        catalog = ToolCatalog([DEFINITIONS[0], task['tool']])
        search_call = {'name': 'tool_search', 'version': 1, 'arguments': {'query': task['query']}}
        result = search(catalog, task['query'])
        call = {'name': task['tool']['name'], 'version': 1, 'arguments': task['arguments']}
        catalog.resolve(call)
        messages = [
            {'role': 'system', 'content': IDENTITY + '\nDiscover tools using this definition: ' + json.dumps(DEFINITIONS[0])},
            {'role': 'user', 'content': task['query'] + ': ' + next(iter(task['arguments'].values()))},
            {'role': 'assistant', 'content': json.dumps(search_call)},
            {'role': 'tool', 'content': json.dumps(result)},
            {'role': 'assistant', 'content': json.dumps(call)},
            {'role': 'tool', 'content': json.dumps({'text': task['expected']})},
            {'role': 'assistant', 'content': task['expected']},
        ]
        yield {'version': 1, 'source': 'alpha:tool-discovery-v1',
               'group': 'tool-family:' + task['tool']['name'], 'split': split, 'messages': messages}


def action_records(split, tokenizer, context=16384, generation_tokens=128):
    """Execute discovery before teaching invocation, using the live prompt path."""
    from baby_arcus.tool_context import ToolContext
    from baby_arcus.coding_policy import prepare_prompt
    from baby_arcus.sft_source_lessons import action_record
    for task in tasks(split):
        catalog=ToolCatalog([DEFINITIONS[0],task['tool']]);tools=ToolContext(catalog)
        history=[{'role':'user','content':task['query']+': '+next(iter(task['arguments'].values()))}]
        calls=[{'name':'tool_search','version':1,'arguments':{'query':task['query']}},
               {'name':task['tool']['name'],'version':1,'arguments':task['arguments']}]
        for call in calls:
            record=action_record(history,tools.definitions(),call,tokenizer,context,generation_tokens,
                                 'alpha:discovery-correction','tool-family:'+task['tool']['name'],split)
            tools.resolve(call)
            if call['name']=='tool_search':
                result=search(catalog,**call['arguments']);tools.accept(result)
            else:
                value=next(iter(call['arguments'].values()));operation=task['query'].split()[0]
                output=value[::-1] if operation=='reverse' else value.upper() if operation=='uppercase' else value.lower()
                if output!=task['expected']:raise ValueError('Teacher outcome mismatch')
                result={'text':output}
            yield record
            history.extend([{'role':'assistant','content':__import__('json').dumps(call)},
                            {'role':'tool','content':__import__('json').dumps(result)}])
