"""Durable ordered interaction receipts; never replay motor commands on restart."""
from copy import deepcopy
import json
import os
from pathlib import Path
import time
from baby_arcus.contracts import ContractError

class InteractionStore:
    def __init__(self, root=None):
        self.path=Path(root)/'interactions.jsonl' if root else None
        self.rows=[]
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding='utf-8').splitlines():
                entry=json.loads(line)
                if entry['operation']=='event':self.rows.append(entry['row'])
                else:
                    self.rows[entry['sequence']-1].update(entry['receipt'])
        self.by_id={r['request_id']:r for r in self.rows}
        # Persisted commands are history, not authorization to move after restart.
        for row in list(self.rows):
            if not row.get('model_exposure') and row['status'] not in ('expired','unsupported'):self.ack(row['sequence'],'expired',reason='Host restarted')

    def write(self,entry):
        if self.path:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            with self.path.open('a',encoding='utf-8') as f:
                f.write(json.dumps(entry,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())

    def append(self,request_id,kind,payload,tick,entity_id):
        old=self.by_id.get(request_id)
        if old:
            if old['kind']!=kind or old['payload']!=payload:raise ContractError('Interaction ID conflict')
            return deepcopy(old)
        row={'sequence':len(self.rows)+1,'request_id':request_id,'kind':kind,
             'payload':deepcopy(payload),'tick':tick,'entity_id':entity_id,
             'created_at':time.time(),'status':'queued'}
        self.write({'operation':'event','row':row});self.rows.append(row);self.by_id[request_id]=row
        return deepcopy(row)

    def ack(self,sequence,status,**details):
        if type(sequence) is not int or not 1<=sequence<=len(self.rows):raise ContractError('Unknown interaction')
        if status not in ('delivered','completed','interrupted','expired','unsupported'):raise ContractError('Invalid receipt')
        receipt={'status':status,**details}
        self.write({'operation':'receipt','sequence':sequence,'receipt':receipt})
        self.rows[sequence-1].update(receipt)

    def pending(self,limit=16):
        return deepcopy([r for r in self.rows if r['status'] in ('queued','interrupted','completed') and not r.get('model_exposure')][:limit])

    def snapshot(self):return deepcopy(self.rows[-100:])
