"""Durable, idempotent outcome replay with session-level evaluation isolation."""
import hashlib,json,sqlite3
from copy import deepcopy
from baby_arcus.shared_experience import validate

def partition(row):
    lesson=row.get('lesson_provenance',{})
    if lesson.get('split') in ('validation','confirmation'):return 'evaluation'
    for message in row.get('hearing',[])+[row.get('ambient_hearing',{})]:
        if message.get('source')=='dataset' and type(message.get('document')) is int and message['document']%10==0:
            return 'evaluation'
    if lesson.get('split')=='training':return 'training'
    # Every frame from one session stays in the same partition.
    key=str(row['session'])
    return 'evaluation' if int(hashlib.sha256(key.encode()).hexdigest()[:8],16)%10==0 else 'training'

class Replay:
    def __init__(self,path):
        from pathlib import Path
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(path,timeout=10)
        self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS samples (id TEXT PRIMARY KEY, digest TEXT NOT NULL, partition TEXT NOT NULL, task TEXT NOT NULL, record TEXT NOT NULL)')
        self.db.execute('CREATE INDEX IF NOT EXISTS samples_partition_task ON samples(partition,task)')

    def add(self,row,outcome,targets=None,task='prediction'):
        validate(row)
        if outcome.get('experience_id')!=row['id']:raise ValueError('Outcome identity mismatch')
        after=outcome['after'];validate(after)
        if any(row[k]!=after[k] for k in ('session','entity_id','scope_id')) or after['tick']<row['tick']:
            raise ValueError('Outcome crossed a sensory scope')
        own_view_change=(outcome.get('action') or {}).get('kind') in ('gaze','head','eyelids','move','sleep_when_ready','wake_voluntarily')
        if after['epoch']!=row['epoch'] and not (own_view_change and after['epoch']==row['epoch']+1):
            raise ValueError('Outcome crossed an unexplained visual epoch')
        if outcome.get('executed') is not True:return False
        record=deepcopy(row);record['eligibility'].update(executed=True,training=partition(row)=='training')
        record['executed_action']=deepcopy(outcome.get('action'))
        label=targets if targets is not None else {'prediction':after['internal']}
        if not label:raise ValueError('Empty replay target')
        payload=json.dumps({'row':record,'targets':label,'task':task},sort_keys=True,allow_nan=False)
        fingerprint=hashlib.sha256(payload.encode()).hexdigest()
        with self.db:
            old=self.db.execute('SELECT digest FROM samples WHERE id=?',(row['id'],)).fetchone()
            if old:
                if old[0]!=fingerprint:raise ValueError('Conflicting replay delivery')
                return False
            self.db.execute('INSERT INTO samples VALUES (?,?,?,?,?)',(row['id'],fingerprint,partition(row),task,payload))
        return True

    def select(self,limit=20,consumed=()):
        if type(limit) is not int or not 1<=limit<=1000:raise ValueError('Invalid replay limit')
        # Keep payload memory proportional to the requested batch, not the
        # accumulated history. A temporary exclusion table also avoids SQLite's
        # parameter-count limit for long consumed-ID histories.
        self.db.execute('CREATE TEMP TABLE IF NOT EXISTS consumed_ids (id TEXT PRIMARY KEY)')
        self.db.execute('DELETE FROM consumed_ids')
        self.db.executemany('INSERT OR IGNORE INTO consumed_ids VALUES (?)',((key,) for key in consumed))
        groups={}
        tasks=[row[0] for row in self.db.execute("SELECT DISTINCT task FROM samples WHERE partition='training' ORDER BY task")]
        for task in tasks:
            rows=self.db.execute("SELECT record FROM samples s WHERE partition='training' AND task=? AND NOT EXISTS (SELECT 1 FROM consumed_ids c WHERE c.id=s.id) ORDER BY s.rowid LIMIT ?",(task,limit))
            groups[task]=[json.loads(payload) for (payload,) in rows]
        result=[]
        while len(result)<limit and any(groups.values()):
            for task in sorted(groups):
                if groups[task] and len(result)<limit:result.append(groups[task].pop(0))
        return result

    def add_language(self,row,example):
        """Observed words can train even when no body action was taken."""
        validate(row)
        if not example:return False
        prefix=example.get('prefix');target=example.get('target');source=example.get('source_id')
        if not isinstance(source,str) or not source or not isinstance(prefix,list) or not 1<=len(prefix)<=64:
            raise ValueError('Invalid language exposure')
        if any(type(i) is not int or i<0 for i in prefix+[target]):raise ValueError('Invalid language tokens')
        key='language-'+hashlib.sha256(source.encode()).hexdigest()
        semantic=json.dumps(example,sort_keys=True);fingerprint=hashlib.sha256(semantic.encode()).hexdigest()
        record=deepcopy(row);record['id']=key;record['language_prefix_ids']=prefix
        record['eligibility'].update(executed=False,training=partition(row)=='training')
        payload=json.dumps({'row':record,'targets':{'text':target},'task':'language'},sort_keys=True,allow_nan=False)
        with self.db:
            old=self.db.execute('SELECT digest FROM samples WHERE id=?',(key,)).fetchone()
            if old:
                if old[0]!=fingerprint:raise ValueError('Conflicting language exposure')
                return False
            self.db.execute('INSERT INTO samples VALUES (?,?,?,?,?)',(key,fingerprint,partition(row),'language',payload))
        return True

    def counts(self):return dict(self.db.execute('SELECT partition,COUNT(*) FROM samples GROUP BY partition'))
    def add_delayed(self,row,outcome,later):
        from baby_arcus.shared_temporal import targets
        labels=targets(row,outcome,later);record=deepcopy(row);record['id']='delayed-'+row['id']
        split='evaluation' if 'evaluation' in (partition(row),partition(later)) else 'training'
        record['eligibility'].update(executed=outcome['executed'],training=split=='training')
        record['executed_action']=deepcopy(outcome['action'])
        record['prediction_horizon']=later['tick']-row['tick']
        quality={'accepted':outcome['executed'],'result':deepcopy(outcome.get('result')),
            'horizon_ticks':later['tick']-outcome['after']['tick'],'vision_target_available':'future_rgb' in labels}
        payload=json.dumps({'row':record,'targets':labels,'task':'delayed_outcome','quality':quality},sort_keys=True,allow_nan=False)
        fingerprint=hashlib.sha256(payload.encode()).hexdigest()
        with self.db:
            old=self.db.execute('SELECT digest FROM samples WHERE id=?',(record['id'],)).fetchone()
            if old:
                if old[0]!=fingerprint:raise ValueError('Conflicting delayed outcome')
                return False
            self.db.execute('INSERT INTO samples VALUES (?,?,?,?,?)',(record['id'],fingerprint,split,'delayed_outcome',payload))
        return True

    def recover(self,path):
        """Recover committed outcomes if the host stopped between log and queue writes."""
        from pathlib import Path
        pending={};added=0
        with Path(path).open(encoding='utf-8') as stream:
            for line in stream:
                if not line.endswith('\n'):break  # interrupted final write was never committed
                event=json.loads(line)
                if event['phase']=='proposed':pending[event['experience']['id']]=event
                elif event['phase']=='outcome':
                    proposed=pending.pop(event['experience_id'],None)
                    if proposed is None:raise ValueError('Outcome without a logged proposal')
                    row=proposed['experience'];added+=self.add(row,event)
                    added+=self.add_language(row,proposed.get('prediction',{}).get('language_example'))
                elif event['phase']=='delayed_outcome':
                    added+=self.add_delayed(event['experience'],event['outcome'],event['after'])
                elif event['phase']=='sequence_outcome':added+=self.add_sequence(event['transitions'])
        return added

    def add_sequence(self,transitions):
        from baby_arcus.shared_temporal import sequence_targets
        row,labels=sequence_targets(transitions)
        split='evaluation' if any(partition(value)=='evaluation' for before,_,later in transitions for value in (before,later)) else 'training'
        row['id']='sequence-'+row['id']+'-'+transitions[-1][2]['id']
        row['eligibility']['training']=split=='training'
        payload=json.dumps({'row':row,'targets':labels,'task':'sequence_outcome'},sort_keys=True,allow_nan=False)
        fingerprint=hashlib.sha256(payload.encode()).hexdigest()
        with self.db:
            old=self.db.execute('SELECT digest FROM samples WHERE id=?',(row['id'],)).fetchone()
            if old:
                if old[0]!=fingerprint:raise ValueError('Conflicting sequence outcome')
                return False
            self.db.execute('INSERT INTO samples VALUES (?,?,?,?,?)',(row['id'],fingerprint,split,'sequence_outcome',payload))
        return True
    def close(self):self.db.close()
