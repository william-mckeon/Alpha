"""Artifact service routes."""
from baby_arcus.artifacts import ArtifactStore
from baby_arcus.contracts import envelope, WORLD_VERSION
import shutil

class ArtifactApplication:
    def __init__(self, root):
        self.store = ArtifactStore(root)

    def __call__(self, method, path, body):
        if method=="GET" and path=="/v1/storage":
            return 200,{"bytes":sum(p.stat().st_size for p in self.store.root.iterdir() if p.is_file()),
                        "free_bytes":shutil.disk_usage(self.store.root).free}
        if method == "GET" and path in ("/health","/ready"):
            return 200, {"service":"artifacts","schema_version":1,"world_version":WORLD_VERSION,"ready":True}
        if method == "POST" and path == "/v1/artifacts":
            envelope(body,("kind","payload"))
            return 200,self.store.publish(body["request_id"],body["kind"],body["payload"])
        if method == "GET" and path.startswith("/v1/artifacts/"):
            return 200,self.store.read(path[len("/v1/artifacts/"):])
        raise KeyError(path)
