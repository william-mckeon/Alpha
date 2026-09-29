"""Immutable local JSONL shards with bounded memory and exact byte cursors.

Data acquisition is separate. A missing shard fails closed, never changes sampling.
The manifest order is the sampling policy; no hidden shuffle or packing buffer.
"""
import copy
import json
from pathlib import Path
from arcus3.checkpoint import digest
from arcus3.config import read, safe_child

class CorpusStream:
    def __init__(self, root, state=None):
        self.root=Path(root);self.manifest=read(self.root/'manifest.json')
        if self.manifest.get('schema')!='arcus3-corpus-v1' or not self.manifest.get('shards'):
            raise ValueError('Pinned corpus manifest required')
        if sum(s['bytes'] for s in self.manifest['shards'])>self.manifest.get('cache_limit_bytes',21474836480):
            raise ValueError('Prepared corpus exceeds bounded cache allocation')
        self.sha=digest(self.root/'manifest.json')
        self.state=copy.deepcopy(state or {'manifest_sha256':self.sha,'shard':0,'offset':0,'epoch':0,'records':0})
        if self.state['manifest_sha256']!=self.sha:raise ValueError('Corpus identity changed')
        if not 0<=self.state['shard']<len(self.manifest['shards']) or self.state['offset']<0:
            raise ValueError('Invalid cursor')
        self.verified=set()

    def snapshot(self):return copy.deepcopy(self.state)

    def next(self):
        for _ in range(len(self.manifest['shards'])+1):
            item=self.manifest['shards'][self.state['shard']]
            path=safe_child(self.root,item['path'])
            if item['path'] not in self.verified:
                if path.stat().st_size!=item['bytes'] or digest(path)!=item['sha256']:raise ValueError('Shard changed')
                self.verified.add(item['path'])
            if self.state['offset']>item['bytes']:raise ValueError('Cursor outside shard')
            with path.open('rb') as f:
                f.seek(self.state['offset']);line=f.readline()
                if line:self.state['offset']=f.tell()
            if line:
                row=json.loads(line);self.state['records']+=1
                if len(row['input_ids'])<2 or len(row['labels'])!=len(row['input_ids']):raise ValueError('Invalid token record')
                return row
            self.state['offset']=0;self.state['shard']+=1
            if self.state['shard']==len(self.manifest['shards']):
                self.state['shard']=0;self.state['epoch']+=1
        raise ValueError('Empty corpus')
