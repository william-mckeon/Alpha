"""Lossless structured import; flattened corpora cannot reconstruct tool roles."""
import json
from baby_arcus.sft_validation import validate


def import_messages(messages, source, group, split='training', adapter=None, target_last=False):
    if adapter is not None:
        if adapter != 'opencode-v1': raise ValueError('Unknown source adapter')
        from baby_arcus.sft_source_adapters import adapt_messages
        return adapt_messages(messages,source,group,split,target_last)
    from baby_arcus.coding_tools import DEFINITIONS
    from baby_arcus.tool_catalog import ToolCatalog
    catalog = ToolCatalog(DEFINITIONS)
    normalized = []
    pending = None
    for item in messages:
        if set(item) - {'role','content','tool_calls','tool_call_id','name','train'}:
            raise ValueError('Unsupported message fields; write an explicit source adapter')
        calls = item.get('tool_calls')
        if calls:
            if item['role'] != 'assistant' or pending is not None or len(calls) != 1 or item.get('content'):
                raise ValueError('Quarantine parallel, mixed-content or unmatched tool call')
            call = calls[0]
            if call.get('type') != 'function' or not isinstance(call.get('id'),str) or not call['id']:
                raise ValueError('Tool call identity required')
            function = call['function']
            args = function['arguments']
            if isinstance(args,str): args=json.loads(args)
            alpha_call = {'name':function['name'],'version':1,'arguments':args}
            catalog.resolve(alpha_call)  # Foreign schemas never silently become Alpha actions.
            content=json.dumps(alpha_call,sort_keys=True)
            pending=call['id']
        elif item['role'] == 'tool':
            if pending is None or item.get('tool_call_id') != pending:
                raise ValueError('Tool result does not match preceding call')
            content=item.get('content'); pending=None
        else:
            if pending is not None or item.get('tool_call_id') or item.get('name'):
                raise ValueError('Unmatched or unsupported tool protocol')
            content=item.get('content')
        normalized.append({'role':item['role'],'content':content,
                           **({'train':item['train']} if 'train' in item else {})})
    if pending is not None: raise ValueError('Tool call has no observed result')
    return validate({'version':1,'source':source,'group':group,'split':split,'messages':normalized})
