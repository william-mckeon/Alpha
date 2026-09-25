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
        value = f'Arcus {suffix} AbCd'
        output = value[::-1] if operation == 'reverse' else value.upper() if operation == 'uppercase' else value.lower()
        yield {'tool': tool, 'query': operation + ' text', 'arguments': {argument: value},
               'expected': output, 'split': split}


def records(split):
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
