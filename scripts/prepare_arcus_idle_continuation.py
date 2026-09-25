"""Preserve the release and continue its full optimizer/RNG/curriculum state."""
import argparse
import json
import shutil
import sqlite3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baby_arcus.language_stream import atomic_json
from baby_arcus.shared_checkpoint import digest
from baby_arcus.shared_factory import read_config

GENERATION = 'efa75913a35a499483975736e57f84f6'
SHA256 = '9e6f8e21bc4b5e70d94d92cfcdff2872980e28c6585ebcaabad11a0fbb810520'


def prepare(config):
    cfg = read_config(config)
    root = Path(cfg['root'])
    source = Path('runs/test2/depth100-seed-2101')
    if root.exists() and any(root.iterdir()):
        raise ValueError('Continuation destination must be empty; never overwrite a run')
    original = json.loads((source/'experiment.json').read_text())
    for key in ('seed', 'preset', 'text_dim', 'depth_capacity', 'encoding', 'tiktoken_version', 'learning_rate', 'dataset_config'):
        if cfg[key] != original[key]:
            raise ValueError('Continuation changes training configuration: '+key)
    checkpoint = source/(GENERATION+'.pt')
    if digest(checkpoint) != SHA256:
        raise ValueError('Alpha-1.0.0 source hash mismatch')
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, root/checkpoint.name)
    if digest(root/checkpoint.name) != SHA256:
        raise ValueError('Continuation checkpoint copy mismatch')
    for name in ('experiment.json', 'initial.json', 'dataset.json', 'hearing.json'):
        if (source/name).exists():
            shutil.copy2(source/name, root/name)
    if (source/'world.sqlite').exists():
        with sqlite3.connect(f'file:{(source/"world.sqlite").resolve().as_posix()}?mode=ro', uri=True) as src:
            with sqlite3.connect(root/'world.sqlite') as dest:
                src.backup(dest)
    atomic_json(root/'candidate.json', {'generation': GENERATION, 'sha256': SHA256,
                'updates': 37000, 'receipt_count': 37000, 'depth_capacity': 1.0})
    atomic_json(root/'continuation.json', {'release': 'Alpha-1.0.0', 'source_generation': GENERATION,
                'source_sha256': SHA256, 'source_root': str(source), 'initial_updates': 37000,
                'method': 'scripts/train_arcus_to_baseline.py', 'optimizer_and_rng_preserved': True})
    (root/'pause-training').touch()
    return {'prepared': str(root), 'updates': 37000, 'source_unchanged': digest(checkpoint) == SHA256}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/baby_arcus/alpha_idle.json')
    print(json.dumps(prepare(parser.parse_args().config)))
