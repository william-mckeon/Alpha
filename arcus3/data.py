"""Bounded supervised data normalization, overlap exclusion and assistant masks."""
import hashlib
import json
import re
from difflib import SequenceMatcher

def normalized(text):return ' '.join(re.findall(r'\w+',text.lower()))

def identity(messages):
    return hashlib.sha256(json.dumps(messages,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def encode_record(tokenizer, record, max_length, forbidden):
    """Shared preparation for reviewed text or chat; never supervise rejected preferences."""
    if 'messages' in record:
        if excluded(record['messages'],forbidden):return []
        row=encode(tokenizer,record['messages'],max_length)
        return [row] if row else []
    text=record.get('text')
    if not isinstance(text,str):raise ValueError('Normalize tools/preferences explicitly before preparation')
    if excluded([{'content':text}],forbidden):return []
    ids=tokenizer(text,add_special_tokens=False)['input_ids'];rows=[]
    for offset in range(0,len(ids),max_length):
        part=ids[offset:offset+max_length]
        if len(part)<2:continue
        rows.append({'input_ids':part,'labels':[-100]+part[1:],'target_tokens':len(part)-1,
                     'sha256':hashlib.sha256(json.dumps(part).encode()).hexdigest()})
    return rows

def excluded(messages, forbidden):
    texts=[normalized(m['content']) for m in messages]
    for text in texts:
        for item in forbidden:
            item=normalized(item)
            if not item:continue
            # Short generic answers like "5" are not useful substring exclusions.
            if text==item or (len(item)>=24 and item in text):return True
            if len(item)>=24 and SequenceMatcher(None,text,item).ratio()>=.85:return True
    return False

def encode(tokenizer,messages,max_length):
    if not messages or messages[-1]['role']!='assistant':raise ValueError('Assistant completion required')
    ids=[];labels=[]
    for index,message in enumerate(messages):
        if message['role'] not in ('system','user','assistant') or not isinstance(message['content'],str):
            raise ValueError('Invalid message')
        full=tokenizer.apply_chat_template(messages[:index+1],tokenize=True,add_generation_prompt=False)
        if full[:len(ids)]!=ids:raise ValueError('Non-prefix chat template')
        new=[-100]*(len(full)-len(ids))
        if message['role']=='assistant':
            prefix=tokenizer.apply_chat_template(messages[:index],tokenize=True,add_generation_prompt=True)
            if full[:len(prefix)]!=prefix:raise ValueError('Assistant prefix mismatch')
            for offset in range(len(prefix),len(full)):new[offset-len(ids)]=full[offset]
        ids=full;labels+=new
    if len(ids)>max_length or sum(t!=-100 for t in labels[1:])==0:return None
    return {'input_ids':ids,'labels':labels,'target_tokens':sum(t!=-100 for t in labels[1:]),'sha256':identity(messages)}
