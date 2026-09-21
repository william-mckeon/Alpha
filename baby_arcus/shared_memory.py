"""Bounded durable sensory memory, isolated by entity, environment and holdout."""
import hashlib,json,math,sqlite3
from pathlib import Path
from baby_arcus.shared_experience import validate
from baby_arcus.shared_replay import partition


class Memory:
    def __init__(self,path,limit=2048):
        if type(limit) is not int or not 8<=limit<=100000:raise ValueError('Invalid memory budget')
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        self.limit=limit;self.db=sqlite3.connect(path,timeout=10)
        self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS memories (id TEXT PRIMARY KEY, namespace TEXT NOT NULL, partition TEXT NOT NULL, digest TEXT NOT NULL, value TEXT NOT NULL)')

    @staticmethod
    def namespace(row):
        environment=row.get('environment_id')
        if not isinstance(environment,str) or not environment:raise ValueError('Persistent memory requires environment identity')
        return json.dumps([row['entity_id'],environment],separators=(',',':'))

    def remember(self,row,features,action=None,error=None,generation=None):
        validate(row)
        if len(features)!=72 or any(type(v) not in (int,float) or not math.isfinite(v) for v in features):raise ValueError('Invalid memory features')
        if error is not None and (not math.isfinite(error) or error<0):raise ValueError('Invalid prediction error')
        value={'features':features,'action':action,'error':error,'generation':generation,'tick':row['tick'],'session':row['session']}
        payload=json.dumps(value,sort_keys=True,allow_nan=False);digest=hashlib.sha256(payload.encode()).hexdigest()
        namespace=self.namespace(row);split=partition(row)
        with self.db:
            old=self.db.execute('SELECT namespace,partition,digest FROM memories WHERE id=?',(row['id'],)).fetchone()
            if old:
                if old!=(namespace,split,digest):raise ValueError('Conflicting memory event')
                return False
            self.db.execute('INSERT INTO memories VALUES (?,?,?,?,?)',(row['id'],namespace,split,digest,payload))
            self.db.execute('DELETE FROM memories WHERE namespace=? AND partition=? AND id NOT IN (SELECT id FROM memories WHERE namespace=? AND partition=? ORDER BY rowid DESC LIMIT ?)',(namespace,split,namespace,split,self.limit))
        return True

    def recall(self,row,limit=8):
        validate(row)
        if type(limit) is not int or not 1<=limit<=8:raise ValueError('Invalid recall budget')
        values=self.db.execute('SELECT id,value FROM memories WHERE namespace=? AND partition=? AND id!=? ORDER BY rowid DESC LIMIT ?',
            (self.namespace(row),partition(row),row['id'],limit)).fetchall()
        return [{'id':key,**json.loads(value)} for key,value in reversed(values)]

    def close(self):self.db.close()

    def remember_outcome(self,event):
        from baby_arcus.shared_temporal import targets
        before=event['experience'];outcome=event['outcome'];later=event['after']
        labels=targets(before,outcome,later);forecast=outcome.get('forecast') or {};error=None
        if forecast.get('future_body') is not None and forecast.get('horizon_ticks')==later['tick']-before['tick']:
            predicted=forecast['future_body']
            if len(predicted)!=20:raise ValueError('Invalid remembered prediction')
            error=sum((a-b)**2 for a,b in zip(predicted,labels['future_body']))/20
        return self.remember(before,sensory_features(before),outcome['action'],error,forecast.get('generation'))

    def recover(self,path):
        added=0
        with Path(path).open(encoding='utf-8') as stream:
            for line in stream:
                if not line.endswith('\n'):break
                event=json.loads(line)
                if event.get('phase')=='delayed_outcome' and event.get('experience',{}).get('environment_id'):
                    added+=self.remember_outcome(event)
        return added


def sensory_features(row):
    from io import BytesIO
    from PIL import Image
    from baby_arcus.shared_temporal import body_vector
    raw=validate(row);rgb=[0.0]*48
    if raw:
        with Image.open(BytesIO(raw)) as image:rgb=[v/255 for v in image.convert('RGB').resize((4,4)).tobytes()]
    return body_vector(row)+row.get('gaze',[0,0,0,0])+rgb
