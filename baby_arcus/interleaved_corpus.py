"""Deterministic per-shard cursors so large files cannot starve other sources."""
import copy
from baby_arcus.sustained_curriculum import corpus_windows


def windows(manifest, tokenizer, cursor, length):
    if cursor and cursor.get('schema')!='interleaved-corpus-v1':
        file=cursor.get('file',0)
        cursor={'schema':'interleaved-corpus-v1','index':file+1,'files':{str(file):dict(cursor,file=0)},'exhausted':list(range(file))}
    state=copy.deepcopy(cursor or {'schema':'interleaved-corpus-v1','index':0,'files':{}})
    streams={};exhausted=set(state.get('exhausted',[]));count=len(manifest['files'])
    while len(exhausted)<count:
        index=state['index']%count;state['index']+=1
        if index in exhausted:continue
        if index not in streams:
            single=dict(manifest,files=[manifest['files'][index]])
            streams[index]=corpus_windows(single,tokenizer,state['files'].get(str(index),{}),length)
        try:ids,after=next(streams[index])
        except StopIteration:
            exhausted.add(index);state['exhausted']=sorted(exhausted);continue
        state['files'][str(index)]=after
        yield ids,copy.deepcopy(state)
