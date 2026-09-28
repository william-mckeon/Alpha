"""Source-derived literal-edit drills, not relabelled repository conversations.

The old/new text is copied verbatim into a disposable lesson. A deterministic
teacher discovers and executes the real tools. This teaches tool mechanics;
it makes no assertion that the original repository change was correct.
"""
import json
from pathlib import Path
from baby_arcus.contracts import digest
from baby_arcus.sft_validation import validate

ADAPTER = 'opencode-literal-lesson-v1'


def action_record(history, definitions, call, tokenizer, context, generation_tokens, source, group, split):
    """A single runtime-identical action target; observations are never targets."""
    from baby_arcus.coding_policy import prepare_prompt
    from baby_arcus.coding_contracts import validate_call
    from baby_arcus.conversation_format import content
    from baby_arcus.sft_dataset import windows
    validate_call(call)
    target={'role':'assistant','content':json.dumps(call,sort_keys=True),'target_kind':'action_prediction'}
    target_tokens=tokenizer.encode(content(target)+'\n</assistant>\n')
    if len(target_tokens)>generation_tokens:raise ValueError('Action exceeds inference budget')
    packed,ids,selected=prepare_prompt(history,definitions,tokenizer,context-generation_tokens)
    if call['name'] not in {d['name'] for d in selected}:raise ValueError('Target tool not visible')
    record={'version':1,'source':source,'group':group,'split':split,
            'messages':[dict(m,train=False) if m['role']=='assistant' else dict(m) for m in packed]+[target]}
    sequence=next(windows(record,tokenizer,context))
    prefix=[i for i,m in zip(sequence['ids'],sequence['mask']) if not m]
    if prefix!=ids:raise ValueError('Training/runtime prompt mismatch')
    return record


def extract(row):
    completion = row.get('completion', {})
    calls = completion.get('tool_calls', [])
    if len(calls) != 1:
        raise ValueError('requires_one_edit_or_write')
    call = calls[0]
    if call.get('type') != 'function':
        raise ValueError('requires_function_call')
    fn = call['function']
    args = fn['arguments']
    args = json.loads(args) if isinstance(args, str) else args
    if not isinstance(args, dict) or not isinstance(args.get('path'), str) or not args['path']:
        raise ValueError('requires_source_path')
    if fn['name'] == 'edit_file':
        if (set(args) - {'path', 'old_string', 'new_string', 'replace_all'}
                or args.get('replace_all', False) is not False):
            raise ValueError('unsupported_edit_semantics')
        old, new = args['old_string'], args['new_string']
        if not isinstance(old, str) or not old:
            raise ValueError('empty_patch_target')
        kind = 'patch_file'
    elif fn['name'] == 'write_file':
        if set(args) != {'path', 'content'}:
            raise ValueError('unsupported_write_semantics')
        old, new, kind = '', args['content'], 'write_file'
    else:
        raise ValueError('not_an_edit_or_write')
    if not isinstance(new, str) or new == old:
        raise ValueError('empty_or_unchanged_edit')
    if max(len(old.encode()), len(new.encode())) > 12000:
        raise ValueError('lesson_exceeds_file_budget')
    # Validate content before any disposable workspace write (including secrets).
    validate({'version': 1, 'source': 'opencode:lesson-screen', 'group': 'screen',
              'split': 'training', 'messages': [
                  {'role': 'user', 'content': old or 'Empty practice file.'},
                  {'role': 'assistant', 'content': new or 'Delete the selected text.'}]})
    return {'kind': kind, 'old': old, 'new': new, 'source_path': args['path']}


def demonstrate(spec, workspace, tokenizer, source, group, split, provenance, adapter=ADAPTER):
    from baby_arcus.coding_environment import CodingEnvironment
    from baby_arcus.coding_tools import CodingTools
    from baby_arcus.coding_policy import prepare_prompt
    from baby_arcus.conversation_format import content
    from baby_arcus.sft_dataset import packing_report
    # CodingEnvironment enforces a fresh, scoped file path; source paths are metadata only.
    workspace = Path(workspace)
    if workspace.exists():
        raise ValueError('A fresh lesson workspace is required')
    env = CodingEnvironment(workspace, 'positive_sum')
    env.write('solution.py', spec['old'])
    tools = CodingTools(env)
    if spec['kind'] == 'patch_file':
        instruction = ('Literal-edit drill in a practice copy. In solution.py replace exactly '
                       + json.dumps(spec['old']) + ' with ' + json.dumps(spec['new'])
                       + '. Discover tools, inspect, edit, and read back. Do not execute code.')
        edit_args = {'path': 'solution.py', 'old': spec['old'], 'new': spec['new']}
        query = 'patch exact occurrence'
    else:
        instruction = ('Literal-write drill in a practice copy. Set solution.py to '
                       + json.dumps(spec['new'])
                       + '. Discover tools, inspect, write, and read back. Do not execute code.')
        edit_args = {'path': 'solution.py', 'content': spec['new']}
        query = 'write replace file'
    calls = [
        ('tool_search', {'query': 'read inspect file', 'limit': 1}),
        ('read_file', {'path': 'solution.py'}),
        ('tool_search', {'query': query, 'limit': 1}),
        (spec['kind'], edit_args),
        ('tool_search', {'query': 'read inspect file', 'limit': 1}),
        ('read_file', {'path': 'solution.py'}),
    ]
    history = [{'role': 'user', 'content': instruction}]
    records, events = [], []
    for name, arguments in calls:
        action = {'name': name, 'version': 1, 'arguments': arguments}
        target = {'role': 'assistant', 'content': json.dumps(action, sort_keys=True)}
        # Match current runtime's 128-token decoding budget, including end marker.
        if len(tokenizer.encode(content(target) + '\n</assistant>\n')) > 128:
            raise ValueError('lesson_action_exceeds_decode_budget')
        packed, ids, selected = prepare_prompt(history, tools.context.definitions(), tokenizer, 384)
        if name not in {definition['name'] for definition in selected}:
            raise ValueError('required_tool_schema_did_not_fit:' + name)
        result = tools.execute(action)
        events.append({'call': action, 'result': result, 'input_sha256': digest(ids)})
        messages = [dict(m, train=False) if m['role'] == 'assistant' else dict(m) for m in packed]
        messages.append(target)
        records.append({'version': 1, 'source': source, 'group': group, 'split': split,
                        'messages': messages, 'provenance': {
                            'adapter': adapter, 'input_sha256': provenance['input_sha256'],
                            'target_kinds': ['action_prediction'], 'terminal_result_observed': True,
                            'lesson': {**provenance['lesson'], 'source_path': spec['source_path'],
                                       'teacher': 'deterministic', 'original_task_solved': False,
                                       'code_executed': False}}})
        history.extend([target, {'role': 'tool', 'content': json.dumps(result, sort_keys=True)}])
    if events[-1]['result']['content'] != spec['new']:
        raise ValueError('lesson_readback_mismatch')
    verified = digest({'content': events[-1]['result']['content']})
    for record in records:
        record['provenance']['lesson']['verified_content_sha256'] = verified
        validate(record)
    # Context eviction can yield the same discovery example twice in one lesson.
    # Keep its executed receipts, but do not double-weight an identical target.
    unique = {digest(record['messages']): record for record in records}
    duplicate_targets = len(records) - len(unique)
    records = list(unique.values())
    packed = packing_report(records, tokenizer)
    if packed['quarantined']:
        raise ValueError('lesson_packing_failed:' + packed['quarantined'][0]['reason'])
    return records, {'events': events, 'readback_verified': True, 'packing': packed,
                     'duplicate_targets_omitted': duplicate_targets,
                     'original_task_solved': False, 'code_executed': False}
