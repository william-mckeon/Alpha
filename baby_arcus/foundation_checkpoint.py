"""Atomic immutable checkpoints with config, corpus, RNG and cursor identity."""
import hashlib
import json
import os
import random
import uuid
from pathlib import Path


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def save(root, model, optimizer, progress, stream, cfg):
    import torch
    from baby_arcus.shared_checkpoint import HashingWriter
    from baby_arcus.language_stream import atomic_json
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    generation = uuid.uuid4().hex
    pending, destination = root/(generation+'.pending'), root/(generation+'.pt')
    payload = {'schema': 'arcus-foundation-v1', 'config': cfg, 'config_hash': config_hash(cfg),
               'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
               'progress': progress, 'stream': stream.state_dict(), 'rng': torch.get_rng_state(),
               'cuda_rng': torch.cuda.get_rng_state_all(), 'python_rng': random.getstate()}
    with pending.open('xb') as out:
        writer = HashingWriter(out)
        torch.save(payload, writer)
        out.flush()
        os.fsync(out.fileno())
    os.replace(pending, destination)
    pointer = {'generation': generation, 'sha256': writer.hash.hexdigest(),
               'updates': progress['updates'], 'config_hash': config_hash(cfg)}
    atomic_json(root/'candidate.json', pointer)
    return pointer


def load(root, model, optimizer, stream, cfg, restore_rng=True):
    import torch
    from baby_arcus.shared_checkpoint import digest
    root = Path(root)
    pointer = json.loads((root/'candidate.json').read_text())
    generation = pointer['generation']
    if len(generation) != 32 or any(c not in '0123456789abcdef' for c in generation):
        raise ValueError('Invalid checkpoint generation')
    path = root/(generation+'.pt')
    if pointer['config_hash'] != config_hash(cfg) or digest(path) != pointer['sha256']:
        raise ValueError('Checkpoint hash or immutable configuration mismatch')
    data = torch.load(path, map_location='cpu', weights_only=True)
    if data['schema'] != 'arcus-foundation-v1' or data['config_hash'] != config_hash(cfg):
        raise ValueError('Checkpoint schema/configuration mismatch')
    model.load_state_dict(data['model'])
    optimizer.load_state_dict(data['optimizer'])
    stream.load_state_dict(data['stream'])
    if restore_rng:
        torch.set_rng_state(data['rng'])
        torch.cuda.set_rng_state_all(data['cuda_rng'])
        random.setstate(data['python_rng'])
    return data['progress'], pointer
