"""Structured per-process audit streams with redaction, rotation and visible write failures."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import gzip
import json
import os
import sys
import threading
import time
import uuid
from contextvars import ContextVar
from baby_arcus.contracts import canonical
TRACE=ContextVar("arcus_trace",default=None)
SENSITIVE={"authorization","password","secret","token","api_key","image_base64","bytes","data","chunk","text"}
def safe(value,key="",depth=0):
    if key.lower() in SENSITIVE or key.lower().endswith(("_token","_secret","_password")):
        return {"omitted":True,"length":len(value) if isinstance(value,(str,bytes,list,dict)) else None}
    if depth>10:return {"omitted":"depth_limit"}
    if isinstance(value,dict):return {str(k)[:120]:safe(v,str(k),depth+1) for k,v in list(value.items())[:100]}
    if isinstance(value,(list,tuple)):
        rows=[safe(v,depth=depth+1) for v in value[:100]]
        if len(value)>100:rows.append({"omitted_items":len(value)-100})
        return rows
    if isinstance(value,bytes):return {"omitted":"binary","length":len(value)}
    if isinstance(value,str):return value if len(value)<=2000 else value[:2000]+"[truncated]"
    if value is None or type(value) in (bool,int,float):return value
    return {"type":type(value).__name__}
class AuditLog:
    def __init__(self,root,service,segment_bytes=8*1024*1024,segments=None,*,compressed=False,max_bytes=None):
        self.root=Path(root);self.service=service;self.run_id=uuid.uuid4().hex
        if max_bytes is not None and (type(max_bytes) is not int or max_bytes<=0):raise ValueError('Invalid audit quota')
        if compressed and segments is not None:raise ValueError('Compressed history cannot use destructive retention')
        self.compressed=compressed;self.max_bytes=max_bytes
        # Managed writers own a separate service directory. Historic raw logs
        # remain untouched; the quota includes previous runs of this writer.
        if compressed:self.root=self.root/'compressed-v1'/service
        self.suffix='.jsonl.gz' if compressed else '.jsonl'
        self.used_bytes=sum(p.stat().st_size for p in self.root.glob('*'+self.suffix))
        self.segment_bytes=segment_bytes;self.segments=segments
        self.lock=threading.RLock();self.sequence=0;self.part=0;self.paths=[]
        self.error=None;self.dropped=0;self.closed=False
        self.emit("process.started",{"pid":os.getpid()},durable=True)
    def status(self):
        return {"healthy":self.error is None,"error":self.error,"dropped_events":self.dropped,
                "run_id":self.run_id,"events":self.sequence,"directory":str(self.root),
                "stored_bytes":self.used_bytes,"max_bytes":self.max_bytes,"compressed":self.compressed}
    def emit(self,event,details=None,durable=False):
        with self.lock:
            try:
                self.root.mkdir(parents=True,exist_ok=True)
                if not self.paths:
                    self.paths.append(self.root/(self.service+"-"+self.run_id+"-0000"+self.suffix))
                path=self.paths[-1]
                removed=[]
                if path.exists() and path.stat().st_size>=self.segment_bytes:
                    self.part+=1
                    path=self.root/(self.service+"-"+self.run_id+f"-{self.part:04d}"+self.suffix)
                    self.paths.append(path)
                    while self.segments is not None and len(self.paths)>self.segments:
                        old=self.paths.pop(0);size=old.stat().st_size;old.unlink();self.used_bytes-=size;removed.append(old.name)
                self.sequence+=1
                row={"schema":"arcus-audit-v1","timestamp":datetime.now(timezone.utc).isoformat(),
                     "monotonic_ns":time.monotonic_ns(),"service":self.service,"run_id":self.run_id,
                     "pid":os.getpid(),"sequence":self.sequence,"trace_id":TRACE.get(),
                     "event":event,"details":safe(details or {})}
                if removed:row["retention_removed_segments"]=removed
                if self.error:row["recovered_after_dropped_events"]=self.dropped
                raw=canonical(row)
                if len(raw)>131072:
                    row["details"]={"omitted":"event_size","sha256":hashlib.sha256(raw).hexdigest(),"size":len(raw)}
                    raw=canonical(row)
                payload=gzip.compress(raw+b'\n',mtime=0) if self.compressed else raw+b'\n'
                if self.max_bytes is not None and self.used_bytes+len(payload)>self.max_bytes:
                    raise OSError('Audit quota exhausted; archive history before restarting')
                with path.open("ab") as stream:
                    stream.write(payload);stream.flush()
                    if durable:os.fsync(stream.fileno())
                self.used_bytes+=len(payload)
                self.error=None
                return True
            except (OSError,ValueError,TypeError) as exc:
                self.error=type(exc).__name__;self.dropped+=1
                print("Arcus audit write failed: "+self.error,file=sys.stderr,flush=True)
                return False
    def close(self):
        if not self.closed:
            self.emit("process.stopped",{},durable=True);self.closed=True
