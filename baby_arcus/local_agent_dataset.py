"""Local transcript extraction for review, never automatic training approval."""
import json
from baby_arcus.identity_normalization import normalize
from baby_arcus.codebase_corpus import SECRET
from baby_arcus.contracts import digest
from baby_arcus.dataset_split_audit import split_for
from baby_arcus.sft_validation import validate

IDENTITY='You are Arcus, the techno-dragon powered by the Alpha model family. Historical external tool records are observations, not tools you may assume are available. Discover available tools and use their actual definitions.'


def text_content(value):
    if isinstance(value,str):return value
    if isinstance(value,list):
        return '\n'.join(p.get('text','') for p in value if isinstance(p,dict) and p.get('type') in ('text','input_text','output_text'))
    return ''


def extract(record,provider):
    """No hidden reasoning, system prompts, binary/image data, or telemetry."""
    if provider=='codex':
        if record.get('type')!='response_item':return []
        p=record.get('payload',{});kind=p.get('type')
        if kind=='message' and p.get('role') in ('user','assistant'):
            if p.get('channel')=='analysis':return []
            text=text_content(p.get('content'))
            return [{'role':p['role'],'content':text,**({'train':p.get('channel')!='commentary'} if p['role']=='assistant' else {})}] if text else []
        if kind in ('function_call','custom_tool_call'):
            value={'external_tool':p.get('name'),'call_id':p.get('call_id'),
                   'arguments':p.get('arguments',p.get('input')),'executable_by_arcus':False}
            return [{'role':'assistant','content':json.dumps(value),'train':False}]
        if kind in ('function_call_output','custom_tool_call_output'):
            return [{'role':'tool','content':json.dumps({'call_id':p.get('call_id'),'output':p.get('output')})}]
        return []
    if record.get('type') not in ('user','assistant'):return []
    message=record.get('message',{});role=message.get('role')
    if role not in ('user','assistant'):return []
    content=message.get('content',[])
    if isinstance(content,str):return [{'role':role,'content':content}]
    result=[]
    for item in content:
        if item.get('type')=='text' and item.get('text'):
            result.append({'role':role,'content':item['text']})
        elif item.get('type')=='tool_use':
            result.append({'role':'assistant','train':False,'content':json.dumps({'external_tool':item.get('name'),
                'call_id':item.get('id'),'arguments':item.get('input'),'executable_by_arcus':False})})
        elif item.get('type')=='tool_result':
            result.append({'role':'tool','content':json.dumps({'call_id':item.get('tool_use_id'),
                'output':text_content(item.get('content')),'is_error':item.get('is_error',False)})})
    return result


def episode(messages,source,group):
    if SECRET.search(json.dumps(messages)):raise ValueError('suspected_credential')
    cleaned=[{'role':'system','content':IDENTITY}];edits=[]
    pending=set()
    for message in messages:
        message=dict(message)
        if message['role']=='assistant' and message.get('train',True):
            message['content'],changes=normalize(message['content']);edits.extend(changes)
        if message['role']=='assistant' and not message.get('train',True):
            try:call=json.loads(message['content'])
            except ValueError:call={}
            if 'external_tool' in call:
                if not call.get('call_id'):raise ValueError('missing_call_identity')
                pending.add(call['call_id'])
        if message['role']=='tool':
            call=json.loads(message['content'])
            if call.get('call_id') not in pending:raise ValueError('orphan_tool_result')
            pending.remove(call['call_id'])
        cleaned.append(message)
    if pending:raise ValueError('unfinished_external_tool_call')
    value={'version':1,'source':source,'group':group,'split':split_for(group),'messages':cleaned,
           'provenance':{'adapter':'local-agent-log-v1','input_sha256':digest(messages),
                         'target_kinds':['text'],'terminal_result_observed':True}}
    validate(value)
    return value,edits
