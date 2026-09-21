"""Conservative lease ownership: expiry blocks reuse until worker unload is confirmed."""
import json
from pathlib import Path
import threading
import time
import uuid
from baby_arcus.artifacts import Conflict
from baby_arcus.contracts import canonical

class ResourceManager:
    def __init__(self,path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.lock = threading.RLock()
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {"generation":0,"lease":None}
        # An orphaned owner must be explicitly unloaded; never silently grant a second GPU model.
        if self.state["lease"]:
            self.state["lease"]["expires"] = 0
            self._save()

    def _save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_bytes(canonical(self.state))
        tmp.replace(self.path)

    def acquire(self,owner,ttl=60):
        with self.lock:
            if self.state["lease"] is not None:
                raise Conflict("Resource already owned, expired, or awaiting unload")
            self.state["generation"] += 1
            lease = {"id":uuid.uuid4().hex,"owner":owner,"generation":self.state["generation"],
                     "expires":time.time()+ttl}
            self.state["lease"] = lease
            self._save()
            return dict(lease)

    def check(self,lease_id,owner,renew=False):
        with self.lock:
            lease = self.state["lease"]
            if not lease or lease["id"] != lease_id or lease["owner"] != owner or lease["expires"]<time.time():
                raise Conflict("Invalid, expired, or stale lease")
            if renew:
                lease["expires"] = time.time()+60
                self._save()
            return dict(lease)

    def release(self,lease_id,unloaded):
        with self.lock:
            if not unloaded or not self.state["lease"] or self.state["lease"]["id"] != lease_id:
                raise Conflict("Release requires matching confirmed-unloaded owner")
            self.state["lease"] = None
            self._save()
