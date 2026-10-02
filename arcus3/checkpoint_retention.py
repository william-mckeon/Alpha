"""Explicit checkpoint retention with immutable, separately protected parents."""
import json
from pathlib import Path
from arcus3.checkpoint import digest
from arcus3.expanded_checkpoint import verify
from baby_arcus.language_stream import atomic_json

POLICY = 'latest-two-plus-major-evaluations-v1'
FILES = {'manifest.json', 'state.pt', 'delta.safetensors'}


def safe_generation(root, name):
    root = Path(root).resolve()
    if Path(name).name != name or not name.startswith('step-'):
        raise ValueError('Unsafe retention generation')
    path = root / name
    if path.is_symlink() or getattr(path, 'is_junction', lambda: False)() or path.resolve().parent != root:
        raise ValueError('Checkpoint must remain directly inside managed root')
    return path


def _entries(root):
    index = Path(root) / 'retention.json'
    if not index.exists():
        return []
    data = json.loads(index.read_text())
    if data['policy'] != POLICY:
        raise ValueError('Unknown retention policy')
    entries = data['entries']
    if len({e['generation'] for e in entries}) != len(entries):
        raise ValueError('Duplicate retention generation')
    return entries


def _meta(root, entry):
    path = safe_generation(root, entry['generation'])
    if not path.is_dir() or digest(path / 'manifest.json') != entry['manifest_sha256']:
        raise ValueError('Retention manifest missing or changed')
    meta = json.loads((path / 'manifest.json').read_text())
    if meta.get('retention_policy') != POLICY:
        raise ValueError('Checkpoint not opted into retention')
    return path, meta


def preflight(root, milestone_limit=None, parent=None, config=None):
    """Read-only metadata check, also called before model loading/optimizer updates."""
    if milestone_limit not in (None, 2):
        raise ValueError('Production retains two milestones')
    entries = _entries(root)
    for entry in entries:
        path, meta = _meta(root, entry)
        if parent is not None and meta['parent_sha256'] != parent:
            raise ValueError('Retention parent mismatch')
        if config is not None and meta.get('config_sha256') != config:
            raise ValueError('Retention config mismatch')
        if entry.get('protected_parent'):
            if meta.get('updates') != 0:
                raise ValueError('Only explicit zero-update initialization may be protected')
        elif milestone_limit is not None and meta.get('retention_milestone_limit') != milestone_limit:
            raise ValueError('Production retention needs an isolated checkpoint root or explicitly protected initialization')
        if set(p.name for p in path.iterdir()) != FILES:
            raise ValueError('Unexpected checkpoint contents')
        for name in FILES:
            file = path / name
            if file.is_symlink() or file.resolve().parent != path:
                raise ValueError('Unsafe checkpoint file')
    return entries


def protect_initialization(root, checkpoint, parent, config):
    """Explicit opt-in only; never rewrite an immutable checkpoint manifest."""
    path = safe_generation(root, Path(checkpoint).name)
    if Path(checkpoint).resolve() != path:
        raise ValueError('Initialization must be in this checkpoint root')
    meta = verify(path, parent, config=config)
    if meta.get('updates') != 0 or meta.get('production') or meta.get('retention_policy') != POLICY:
        raise ValueError('Expected non-production zero-update initialization')
    entries = preflight(root, parent=parent, config=config)
    entry = {'generation': path.name, 'manifest_sha256': digest(path / 'manifest.json'),
             'pinned': True, 'protected_parent': True}
    entries = [e for e in entries if e['generation'] != path.name] + [entry]
    atomic_json(Path(root) / 'retention.json', {'policy': POLICY, 'entries': entries})
    return entry


def register_and_prune(root, new, milestone_limit=None):
    root = Path(root).resolve()
    new = safe_generation(root, Path(new).name)
    meta = json.loads((new / 'manifest.json').read_text())
    entries = preflight(root, milestone_limit, meta['parent_sha256'], meta.get('config_sha256'))
    if meta.get('retention_policy') != POLICY:
        raise ValueError('Checkpoint not opted into retention')
    if milestone_limit is not None and meta.get('retention_milestone_limit') != milestone_limit:
        raise ValueError('New checkpoint retention mismatch')
    if not any(e['generation'] == new.name for e in entries):
        entries.append({'generation': new.name, 'manifest_sha256': digest(new / 'manifest.json'),
                        'pinned': meta.get('retention_pinned', False)})
    rolling = [e for e in entries if not e.get('protected_parent')]
    milestones = [e for e in rolling if e['pinned']]
    if milestone_limit is not None:
        milestones = milestones[-milestone_limit:]
    keep = {e['generation'] for e in rolling[-2:] + milestones}
    keep.update(e['generation'] for e in entries if e.get('protected_parent'))
    # Validate the full deletion plan and both recovery payloads before touching files.
    for entry in entries:
        path, m = _meta(root, entry)
        if {p.name for p in path.iterdir()} != FILES:
            raise ValueError('Unexpected checkpoint contents')
        for name in FILES:
            if (path / name).is_symlink() or (path / name).resolve().parent != path:
                raise ValueError('Unsafe checkpoint file')
    for entry in rolling[-2:]:
        path, m = _meta(root, entry)
        verify(path, m['parent_sha256'])
    deleted = []
    for entry in entries:
        if entry['generation'] in keep:
            continue
        path = safe_generation(root, entry['generation'])
        for name in FILES:
            (path / name).unlink()
        path.rmdir()
        deleted.append(entry['generation'])
    atomic_json(root / 'retention.json', {'policy': POLICY, 'entries': [e for e in entries if e['generation'] in keep]})
    if deleted:
        with (root / 'retention-events.jsonl').open('a') as f:
            f.write(json.dumps({'deleted': deleted, 'latest': new.name}) + '\n')
    return deleted
