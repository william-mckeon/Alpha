"""Resumable, fingerprinted reading of local DatasetForge shards. Source is read-only."""
import hashlib,io,json,os
from pathlib import Path
import zstandard

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.pending')
    with temp.open('w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

def inventory(root,patterns):
    root=Path(root).resolve();paths=sorted({p.resolve() for pattern in patterns for p in root.glob(pattern)})
    if not paths:raise ValueError('No dataset shards matched')
    if any(not p.is_relative_to(root) for p in paths):raise ValueError('Dataset path escapes source root')
    files=[{'path':p.relative_to(root).as_posix(),'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in paths]
    identity=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    return {'root':str(root),'files':files,'fingerprint':identity,'fingerprint_method':'relative names, sizes and modification times'}

def documents(path):
    with open(path,'rb') as raw,zstandard.ZstdDecompressor().stream_reader(raw) as reader:
        with io.TextIOWrapper(reader,encoding='utf-8') as stream:
            index=-1
            while True:
                line=stream.readline(8_000_001)
                if not line:break
                index+=1
                if len(line)>8_000_000:raise ValueError('Oversized dataset record')
                row=json.loads(line)
                text=row.get('text') or row.get('content')
                if isinstance(text,str) and text.strip():yield index,text

class LanguageStream:
    def __init__(self,manifest,tokenizer,state_path):
        self.manifest=manifest;self.tokenizer=tokenizer;self.path=Path(state_path)
        self.state={'fingerprint':manifest['fingerprint'],'file':0,'document':0,'token':0,'playing':True,'exposures':0,'last':None,'eof':False}
        if self.path.exists():
            self.state=json.loads(self.path.read_text(encoding='utf-8'))
            if self.state['fingerprint']!=manifest['fingerprint']:raise ValueError('Dataset changed; choose a new listening state')
        self.iterator=None;self.cached=None

    def control(self,action):
        if action not in ('pause','resume','restart'):raise ValueError('Unknown listening control')
        if action=='restart':
            self.close();self.state.update(file=0,document=0,token=0,eof=False);self.iterator=None;self.cached=None
        self.state['playing']=action!='pause' and not self.state['eof'];atomic_json(self.path,self.state)

    def next(self,limit=64,replay=False):
        if not 1<=limit<=1024:raise ValueError('Passage limit must be 1..1024')
        if replay:return self.state['last']
        if not self.state['playing'] or self.state['eof']:return None
        while self.state['file']<len(self.manifest['files']):
            if self.iterator is None:
                entry=self.manifest['files'][self.state['file']]
                path=Path(self.manifest['root'])/entry['path']
                stat=path.stat()
                if stat.st_size!=entry['size'] or stat.st_mtime_ns!=entry['mtime_ns']:raise ValueError('Dataset shard changed')
                self.iterator=documents(path)
            if self.cached is None:
                for number,text in self.iterator:
                    if number<self.state['document']:continue
                    # Whole documents, deterministically held out, never enter live listening.
                    if number%10==0:continue
                    self.state['document']=number
                    self.cached=self.tokenizer.encode(text);break
                if self.cached is None:
                    self.state.update(file=self.state['file']+1,document=0,token=0);self.iterator=None;continue
            start=self.state['token'];ids=self.cached[start:start+limit]
            if not ids:self.cached=None;self.state.update(document=self.state['document']+1,token=0);continue
            passage={'source':'dataset','file':self.manifest['files'][self.state['file']]['path'],
                     'document':self.state['document'],'offset':start,'tokens':ids,
                     'text':self.tokenizer.decode(ids),'fingerprint':self.manifest['fingerprint']}
            passage['id']=hashlib.sha256(json.dumps({k:v for k,v in passage.items() if k!='text'},sort_keys=True).encode()).hexdigest()
            self.state.update(token=start+len(ids),last=passage,exposures=self.state['exposures']+len(ids))
            atomic_json(self.path,self.state);return passage
        self.state.update(eof=True,playing=False);atomic_json(self.path,self.state);return None

    def close(self):
        if self.iterator:self.iterator.close()


class AcknowledgedLanguageStream:
    """Single-owner offer/ack wrapper; legacy eager streams remain unchanged.

    Acceptance receipt must already be durable in the consumer journal. A crash
    before ack repeats the same offer, never skips it. Consumer deduplicates ID.
    """
    def __init__(self, manifest, tokenizer, state_path):
        self.manifest, self.tokenizer = manifest, tokenizer
        self.path = Path(state_path)
        self.pending = self.path.with_suffix('.offer.json')
        self.scratch = self.path.with_suffix('.scratch.json')
        self.playing = True

    def offer(self, limit=65):
        state = json.loads(self.path.read_text()) if self.path.exists() else None
        if state and not state.get('playing', True):
            return None
        if state and state.get('replay_requested') and state.get('last'):
            return state['last']
        if self.pending.exists():
            pending = json.loads(self.pending.read_text())
            if (not state or state.get('epoch', 0) == pending['passage'].get('epoch', 0)) and (not state or state.get('accepted_id') != pending['passage']['id']):
                return pending['passage']
            self.pending.unlink()
        # Scratch progress is never the acknowledged cursor.
        if state:
            atomic_json(self.scratch, state)
        elif self.scratch.exists():
            self.scratch.unlink()
        stream = LanguageStream(self.manifest, self.tokenizer, self.scratch)
        try:
            passage = stream.next(limit)
            if passage:
                passage['epoch'] = stream.state.get('epoch', 0)
                passage['id'] = hashlib.sha256(json.dumps({k: v for k, v in passage.items() if k not in ('id', 'text')}, sort_keys=True).encode()).hexdigest()
                stream.state['last'] = passage
                atomic_json(self.pending, {'passage': passage, 'after': stream.state})
            return passage
        finally:
            stream.close()

    def ack(self, passage_id, receipt):
        if receipt.get('source_id') != passage_id or receipt.get('durable') is not True:
            raise ValueError('A durable matching consumer receipt is required')
        state = json.loads(self.path.read_text()) if self.path.exists() else {}
        if state.get('accepted_id') == passage_id:
            if state.get('replay_requested'):
                state['replay_requested'] = False
                atomic_json(self.path, state)
            return False
        pending = json.loads(self.pending.read_text())
        if pending['passage']['id'] != passage_id:
            raise ValueError('Acknowledgement does not match offered passage')
        atomic_json(self.path, {**pending['after'], 'control_receipts': state.get('control_receipts', {}),
                               'playing': state.get('playing', pending['after']['playing']), 'accepted_id': passage_id})
        return True

    def control(self, action, request_id=None):
        if action not in ('pause', 'resume', 'restart', 'replay'):
            raise ValueError('Unknown hearing control')
        stream = LanguageStream(self.manifest, self.tokenizer, self.path)
        try:
            receipts = stream.state.setdefault('control_receipts', {})
            if request_id in receipts:
                if receipts[request_id] != action:
                    raise ValueError('Hearing control identity conflict')
                return
            if action == 'replay':
                stream.state['replay_requested'] = True
            if action == 'restart':
                stream.state.update(file=0, document=0, token=0, eof=False, last=None)
                stream.state['epoch'] = stream.state.get('epoch', 0) + 1
                stream.state.pop('accepted_id', None)
                stream.state['replay_requested'] = False
            if action != 'replay':
                stream.state['playing'] = action != 'pause' and not stream.state['eof']
            if request_id:
                receipts[request_id] = action
            atomic_json(self.path, stream.state)
        finally:
            stream.close()
        if action == 'restart' and self.pending.exists():
            self.pending.unlink()
