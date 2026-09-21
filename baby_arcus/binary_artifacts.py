"""Chunked binary transport shared by CPU orchestration and GPU workers."""
import base64
import hashlib
import os
from pathlib import Path
import tempfile
import uuid
from baby_arcus.vocabulary import VOCABULARY_HASH
from baby_arcus.contracts import ContractError

class Repository:
    CHUNK = 256*1024

    def __init__(self, client):
        self.client = client

    def put_file(self,path,metadata=None,heartbeat=lambda:None):
        refs = []
        sha = hashlib.sha256()
        size = 0
        upload = uuid.uuid4().hex
        with open(path,"rb") as handle:
            while data := handle.read(self.CHUNK):
                heartbeat()
                sha.update(data)
                size += len(data)
                record = self.client.request("POST","/v1/artifacts",{
                    "schema_version":1,"request_id":upload+"-"+str(len(refs)),
                    "kind":"checkpoint_chunk","payload":{"data":base64.b64encode(data).decode()}})
                refs.append(record["artifact_id"])
        root = {"format":1,"sha256":sha.hexdigest(),"size":size,"chunks":refs,
                "vocabulary_hash":VOCABULARY_HASH,"metadata":metadata or {}}
        heartbeat()
        return self.client.request("POST","/v1/artifacts",{
            "schema_version":1,"request_id":upload+"-manifest","kind":"checkpoint_manifest",
            "payload":root})["artifact_id"]

    def get_file(self,checkpoint_id,path,heartbeat=lambda:None):
        record = self.client.request("GET","/v1/artifacts/"+checkpoint_id)
        root = record["payload"]
        if record["manifest"]["kind"] != "checkpoint_manifest" or root.get("vocabulary_hash") != VOCABULARY_HASH or root.get("format") != 1:
            raise ContractError("Not a compatible checkpoint manifest")
        if type(root.get("size")) is not int or not 0<root["size"]<=4*1024**3:
            raise ContractError("Checkpoint size exceeds initial resource limit")
        if not isinstance(root.get("chunks"),list) or len(root["chunks"]) != (root["size"]+self.CHUNK-1)//self.CHUNK:
            raise ContractError("Invalid checkpoint chunk count")
        sha = hashlib.sha256()
        size = 0
        path = Path(path)
        descriptor,tmp = tempfile.mkstemp(dir=path.parent,prefix=".download-")
        try:
            with os.fdopen(descriptor,"wb") as handle:
                for ref in root["chunks"]:
                    heartbeat()
                    chunk = self.client.request("GET","/v1/artifacts/"+ref)
                    if chunk["manifest"]["kind"] != "checkpoint_chunk":
                        raise ContractError("Wrong checkpoint chunk kind")
                    data = base64.b64decode(chunk["payload"]["data"],validate=True)
                    if len(data)>self.CHUNK:
                        raise ContractError("Oversized checkpoint chunk")
                    handle.write(data)
                    sha.update(data)
                    size += len(data)
                handle.flush()
                os.fsync(handle.fileno())
            if size != root["size"] or sha.hexdigest() != root["sha256"]:
                raise ContractError("Checkpoint integrity failure")
            os.replace(tmp,path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return root
