"""Explicit own-code source allowlist. Never ingest run logs or execute code."""
from pathlib import Path
import re
from baby_arcus.dataset_split_audit import content_hash, split_for

EXCLUDED = {'.git', '.codex', '.claude', 'runs', 'vendor', 'node_modules', 'artifacts',
            '__pycache__', 'dist', 'build', 'fixtures', 'site-packages', 'logs'}
SUFFIXES = {'.py', '.js', '.ts', '.tsx', '.jsx', '.go', '.rs', '.md'}
SECRET = re.compile(r'-----BEGIN .*PRIVATE KEY|\bhf_[A-Za-z0-9]{20,}|\bsk-[A-Za-z0-9_-]{20,}|(?i:api_key|password|access_token)\s*[:=]\s*["\'][^"\']{12,}')


def collect(root, paths, group, max_bytes=65536):
    root = Path(root).resolve()
    if not paths or not group:
        raise ValueError('Explicit file allowlist and repository identity required')
    for relative in sorted(set(paths)):
        path = root / relative
        if path.is_symlink() or not path.is_file():
            continue
        resolved = path.resolve()
        if root not in resolved.parents:
            raise ValueError('Codebase path escape')
        parts = resolved.relative_to(root).parts
        if any(p.lower() in EXCLUDED or p.lower().startswith(('.venv', '.env')) for p in parts):
            continue
        if path.suffix.lower() not in SUFFIXES or path.stat().st_size > max_bytes:
            continue
        text = path.read_text(encoding='utf-8')
        if not text.strip() or SECRET.search(text):
            continue
        yield {'text': text, 'group': group, 'split': split_for(group),
               'source': 'own-code', 'path': path.relative_to(root).as_posix(),
               'sha256': __import__('hashlib').sha256(path.read_bytes()).hexdigest(),
               'content_hash': content_hash(text)}
