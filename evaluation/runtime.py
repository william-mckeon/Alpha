"""Validate the separated official-worker lock index without installing runtimes."""
from __future__ import annotations

import hashlib
from pathlib import Path


def validate_lock_index(root: Path, index: dict, revisions: dict[str, str]) -> list[str]:
    errors = []
    entries = index.get("harnesses", {})
    if index.get("format") != "arcus-separated-harness-lock-index-v1" or set(entries) != {"function_calling", "repository", "mcp"}:
        return ["runtime lock index has an invalid format or harness set"]
    for harness_id, entry in entries.items():
        if entry.get("source_revision") != revisions.get(harness_id):
            errors.append(f"runtime lock index source revision differs: {harness_id}")
        raw = entry.get("requirements", "")
        if not isinstance(raw, str) or not raw:
            errors.append(f"runtime requirements path missing: {harness_id}")
            continue
        path = (root / raw).resolve()
        if not path.is_relative_to((root / "evaluation").resolve()) or not path.is_file():
            errors.append(f"runtime requirements path unsafe or missing: {harness_id}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != entry.get("sha256"):
            errors.append(f"runtime requirements hash differs: {harness_id}")
    return errors
