"""Shared unittest helpers; pytest is not required."""
from contextlib import contextmanager
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import threading
from baby_arcus.artifacts import ArtifactStore
from baby_arcus.transport import Client, serve
from baby_arcus.services.artifacts import ArtifactApplication

@contextmanager
def running_service(kind, root, artifact_url=None, token="test-private-token", port=0):
    command = [sys.executable,"-m","baby_arcus.cli","serve",kind,"--host","127.0.0.1",
               "--port",str(port),"--state-root",str(root)]
    if artifact_url:
        command += ["--artifact-url",artifact_url]
    env = dict(os.environ, BABY_ARCUS_TOKEN=token)
    process = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                               text=True,env=env)
    try:
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(process.stderr.read())
        record = json.loads(line)
        client = Client("http://127.0.0.1:"+str(record["port"]),token=token)
        client.request("GET","/ready")
        yield client, process
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        process.stdout.close()
        process.stderr.close()

class LocalArtifacts:
    def __init__(self, root):
        self.store = ArtifactStore(root)
    def request(self, method, path, body=None):
        if path == "/ready":
            return {"schema_version":1,"ready":True}
        return self.store.publish(body["request_id"],body["kind"],body["payload"])
