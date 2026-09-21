"""Atomic bounded conversation queue. Received by interface is not read by a model."""
from copy import deepcopy
from pathlib import Path
import os
import time
from baby_arcus.contracts import canonical,decode,ContractError
from baby_arcus.human_messages import validate_message,SENDERS
class ConversationStore:
    def __init__(self,root=None):
        self.path=Path(root)/"conversation.json" if root else None
        self.rows=[]
        if self.path and self.path.exists():
            payload=decode(self.path.read_bytes())
            if payload.get("version")!=1 or not isinstance(payload.get("messages"),list): raise ContractError("Invalid conversation")
            self.rows=payload["messages"]
            if len(self.rows)>500: raise ContractError("Conversation over capacity")
            ids=set()
            for row in self.rows:
                validate_message({k:row[k] for k in ("request_id","sender","text")})
                if row["request_id"] in ids or row["status"] not in ("queued","available"): raise ContractError("Invalid conversation receipt")
                ids.add(row["request_id"])
    def commit(self,rows):
        if self.path:
            pending=self.path.with_suffix(".pending")
            with pending.open("wb") as stream:
                stream.write(canonical({"version":1,"messages":rows}));stream.flush();os.fsync(stream.fileno())
            os.replace(pending,self.path)
        self.rows=rows
    def send(self,value,awake):
        validate_message(value)
        for row in self.rows:
            if row["request_id"]==value["request_id"]:
                if any(row[k]!=value[k] for k in value): raise ContractError("Message ID conflict")
                return deepcopy(row)
        if len(self.rows)>=500: raise ContractError("Conversation full; export history before starting a new session")
        row={**value,"sender_name":SENDERS[value["sender"]],"created_at":time.time(),
             "status":"available" if awake else "queued","model_read":False}
        self.commit([*self.rows,row]);return deepcopy(row)
    def release(self,awake):
        if awake and any(r["status"]=="queued" for r in self.rows):
            self.commit([{**r,"status":"available"} for r in self.rows])
    def snapshot(self,model=False):
        return deepcopy([r for r in self.rows if not model or r["status"]=="available"])

    def mark_read(self,request_id):
        self.commit([{**r,'model_read':True} if r['request_id']==request_id else r for r in self.rows])
