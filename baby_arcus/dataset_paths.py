"""Logical dataset identities resolve independently of approved content hashes."""
import json
import os
import re
from pathlib import Path


def resolve_root(manifest):
    name = manifest.get('dataset_id')
    if name is None:
        return Path(manifest['root']).resolve()
    if not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_-]{0,63}', name):
        raise ValueError('Invalid logical dataset identity')
    mounts = json.loads(os.environ.get('ALPHA_DATASET_MOUNTS', '{}'))
    if name not in mounts:
        raise ValueError('Dataset mount not configured: ' + name)
    return Path(mounts[name]).resolve()
