"""Reviewed HF text targets. Unsupported foreign actions never become Alpha actions."""
import json
from baby_arcus.contracts import digest
from baby_arcus.hf_training_sources import SOURCES
from baby_arcus.identity_normalization import normalize
from baby_arcus.sft_source_adapters import adapt_messages
from baby_arcus.sft_validation import validate
from baby_arcus.dataset_split_audit import split_for


def adapt(row, repo, revision, group, license_name):
    import re
    if repo not in SOURCES or not re.fullmatch('[a-f0-9]{40}',revision) or not group or not license_name:
        raise ValueError('Pinned source, group and declared license required')
    messages = row.get('messages', row.get('trajectory'))
    if isinstance(messages,str): messages=json.loads(messages)
    if not isinstance(messages,list): raise ValueError('Unsupported trajectory schema')
    # A success-named SWE-Gym split is recorded as source evidence, not re-execution.
    successful = row.get('resolved') is True or (repo == 'SWE-Gym/OpenHands-SFT-Trajectories' and row.get('_split') == 'train.success.oss')
    if not successful:
        raise ValueError('No supported successful-outcome evidence; manual review required')
    cleaned=[]; changes=[]
    for item in messages:
        role=item.get('role')
        if role not in ('system','user','assistant','tool'): raise ValueError('Unsupported source role')
        text=item.get('content') or ''
        if not isinstance(text,str): raise ValueError('Non-text source content')
        if role=='assistant' and re.search(r'<(?:execute_bash|execute_ipython|file_edit|tool_call)\b',text):
            raise ValueError('Foreign inline action requires an explicit semantic adapter')
        if role=='assistant' and not item.get('tool_calls'):
            text, edits=normalize(text); changes.extend(edits)
        converted={k:v for k,v in item.items() if k in ('role','tool_calls','tool_call_id','name')}
        converted['content']=text
        # Existing adapter will admit only genuinely equivalent supported calls.
        if role=='assistant': converted['train']=True
        cleaned.append(converted)
    result=adapt_messages(cleaned,'hf:'+repo,group,split_for(group))
    result['provenance']={'adapter':'hf-phase2b-v1','input_sha256':digest(row),
        'target_kinds':result['provenance']['target_kinds'],
        'terminal_result_observed':result['provenance']['terminal_result_observed'],
        'dataset':{'repo':repo,'revision':revision,'license':license_name,
                   'outcome':'source_reports_success','identity_changes':changes}}
    return validate(result)
