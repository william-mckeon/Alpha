"""Explicit OpenCode transcript adapter. Import never grants tool execution access."""
import json
from baby_arcus.contracts import digest
from baby_arcus.sft_validation import validate

VERSION = 'opencode-v1'


def adapt_call(call):
    from baby_arcus.coding_tools import DEFINITIONS
    from baby_arcus.tool_catalog import ToolCatalog
    if call.get('type') != 'function' or not isinstance(call.get('id'),str) or not call['id']:
        raise ValueError('source_call_identity')
    fn = call['function']; args = fn['arguments']
    args = json.loads(args) if isinstance(args,str) else dict(args)
    name = fn['name']
    # Exact-string replacement is equivalent only when replace_all is false.
    if name == 'edit_file':
        if set(args)-{'path','old_string','new_string','replace_all'} or args.get('replace_all',False) is not False:
            raise ValueError('unsupported_edit_semantics')
        args={'path':args['path'],'old':args['old_string'],'new':args['new_string']}; name='patch_file'
        if not args['old']: raise ValueError('empty_patch_target')
    elif name not in ('read_file','write_file'):
        raise ValueError('unsupported_source_tool:'+str(name))
    # Do not discard offset/limit, turn regex into literal search, or remap host paths.
    converted={'name':name,'version':1,'arguments':args}
    ToolCatalog(DEFINITIONS).resolve(converted)
    if args.get('path') != 'solution.py':
        raise ValueError('outside_current_lesson_workspace')
    return converted


def adapt_messages(messages, source, group, split='training', target_last=False):
    normalized=[]; pending=None; targets=[]
    for index,item in enumerate(messages):
        if set(item)-{'role','content','tool_calls','tool_call_id','name','train'}:
            raise ValueError('unsupported_source_message_fields')
        role=item['role']; calls=item.get('tool_calls')
        train=item.get('train',True) and (not target_last or index==len(messages)-1)
        if calls:
            if role!='assistant' or pending is not None or len(calls)!=1:
                raise ValueError('unmatched_or_parallel_source_call')
            call=calls[0]
            if call.get('type')!='function' or not isinstance(call.get('id'),str) or not call['id']:
                raise ValueError('source_call_identity')
            if train:
                action=adapt_call(call)
            else:
                # Historical external actions are evidence, never executable Alpha targets.
                # Preserve their actual arguments instead of pretending they used Alpha tools.
                if not isinstance(call.get('function'),dict): raise ValueError('source_call_function')
                action={'source_only_tool_call':call,'executable_by_alpha':False}
            pending=call['id']
            if item.get('content'):
                normalized.append({'role':'assistant','content':item['content'],'train':False})
            normalized.append({'role':'assistant','content':json.dumps(action,sort_keys=True),'train':train})
            if train: targets.append('action_prediction')
        elif role=='tool':
            if pending is None or item.get('tool_call_id')!=pending:
                raise ValueError('mismatched_source_observation')
            normalized.append({'role':'tool','content':item.get('content')}); pending=None
        else:
            if pending is not None or item.get('tool_call_id') or item.get('name'):
                raise ValueError('missing_source_observation')
            normalized.append({'role':role,'content':item.get('content'),
                               **({'train':train} if role=='assistant' else {})})
            if role=='assistant' and train: targets.append('text')
    # A terminal action can be a prediction target; it is not a successful execution.
    if pending is not None and (not normalized or not normalized[-1].get('train')):
        raise ValueError('unobserved_history_action')
    from baby_arcus.sft_target_contract import annotate
    return validate(annotate({'version':1,'source':source,'group':group,'split':split,'messages':normalized,
                     'provenance':{'adapter':VERSION,'input_sha256':digest(messages),
                                   'target_kinds':targets,'terminal_result_observed':pending is None}}))
