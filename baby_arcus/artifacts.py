"""Content-addressed local artifacts with manifest-last publication and receipts."""
import hashlib
import os
from pathlib import Path
import re
import sqlite3
import tempfile
from contextlib import contextmanager
from baby_arcus.contracts import canonical, decode, digest, identifier, ContractError

class Conflict(ContractError):
    pass

class CorruptArtifact(RuntimeError):
    pass

class ArtifactStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS receipts (request_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, response BLOB NOT NULL)")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.root / "receipts.sqlite", timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def _path(self, artifact_id, suffix):
        if not isinstance(artifact_id,str) or not re.fullmatch(r"[0-9a-f]{64}",artifact_id):
            raise ContractError("Invalid artifact ID")
        return self.root / (artifact_id + suffix)

    def _atomic(self, path, data):
        descriptor, name = tempfile.mkstemp(dir=self.root, prefix=".pending-")
        try:
            with os.fdopen(descriptor,"wb") as file:
                file.write(data)
                file.flush()
                os.fsync(file.fileno())
            os.replace(name,path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def publish(self, request_id, kind, payload):
        identifier(request_id)
        identifier(kind)
        body = canonical({"kind":kind,"payload":payload})
        artifact_id = hashlib.sha256(body).hexdigest()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT fingerprint,response FROM receipts WHERE request_id=?", (request_id,)).fetchone()
            if row:
                if row[0] != artifact_id:
                    raise Conflict("Request ID already used with different content")
                # A successful receipt must not mask disk corruption.
                self.read(artifact_id)
                return decode(row[1])
            blob = self._path(artifact_id, ".blob")
            manifest = {"schema_version":1,"artifact_id":artifact_id,"sha256":artifact_id,
                        "size":len(body),"kind":kind}
            if blob.exists():
                if blob.read_bytes() != body:
                    raise CorruptArtifact("Existing content hash mismatch")
            else:
                self._atomic(blob,body)
            self._atomic(self._path(artifact_id,".json"),canonical(manifest))
            db.execute("INSERT INTO receipts VALUES (?,?,?)",(request_id,artifact_id,canonical(manifest)))
            return manifest

    def read(self, artifact_id):
        manifest_path = self._path(artifact_id,".json")
        if not manifest_path.exists():
            raise KeyError("Artifact is not published")
        try:
            manifest = decode(manifest_path.read_bytes())
            if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
                raise ContractError("Invalid manifest schema")
        except ContractError as exc:
            raise CorruptArtifact("Published manifest is corrupt") from exc
        blob_path = self._path(artifact_id,".blob")
        if not blob_path.exists():
            raise CorruptArtifact("Published blob is missing")
        body = blob_path.read_bytes()
        if manifest.get("artifact_id") != artifact_id or manifest.get("sha256") != artifact_id or hashlib.sha256(body).hexdigest() != artifact_id or len(body) != manifest.get("size"):
            raise CorruptArtifact("Artifact hash or manifest mismatch")
        content = decode(body)
        if content.get("kind") != manifest.get("kind"):
            raise CorruptArtifact("Artifact kind mismatch")
        return {"manifest":manifest,"payload":content["payload"]}
