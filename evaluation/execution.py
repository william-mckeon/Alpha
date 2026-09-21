"""Source identity for controlled workers and readiness evidence."""
import hashlib
import json
from pathlib import Path
import subprocess
from evaluation.provider import EvaluationBlocked


def source_fingerprint(root):
    files = sorted((Path(root) / "evaluation").glob("*.py"))
    if not files:
        raise ValueError("evaluation execution sources are missing")
    mapping = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    return hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()


def inspect_worker_image(root, tag):
    image = json.loads(subprocess.check_output(["docker", "image", "inspect", tag], text=True))[0]
    label = (image["Config"].get("Labels") or {}).get("arcus.evaluation.source")
    if label != source_fingerprint(root):
        raise EvaluationBlocked("worker image contains stale or unverified execution sources; rebuild " + tag)
    return image["Id"]
