"""Record a read-only DatasetForge source selection for a fresh run."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baby_arcus.shared_factory import read_config
from baby_arcus.language_stream import inventory, atomic_json
from baby_arcus.shared_checkpoint import digest


def prepare(config, maximum_files=None):
    cfg = read_config(config)
    root = Path(cfg['root'])
    if not (root / 'initial.json').exists():
        raise ValueError('Initialize the isolated run first')
    path = root / 'dataset.json'
    if path.exists():
        raise ValueError('Dataset manifest already exists; never replace an in-use source')
    source = json.loads(Path(cfg['dataset_config']).read_text())
    manifest = inventory(source['dataset_root'], source['source_patterns'])
    if maximum_files is not None:
        if maximum_files < 1:
            raise ValueError('Maximum files must be positive')
        manifest['files'] = manifest['files'][:maximum_files]
    for entry in manifest['files']:
        entry['sha256'] = digest(Path(manifest['root']) / entry['path'])
    import hashlib
    manifest['fingerprint'] = hashlib.sha256(json.dumps(manifest['files'], sort_keys=True).encode()).hexdigest()
    manifest.update(fingerprint_method='relative names, sizes, timestamps and SHA256',
                    encoding=cfg['encoding'], tiktoken_version=cfg['tiktoken_version'],
                    provenance={'source_config': cfg['dataset_config'], 'license': 'retain source corpus terms; not inferred'})
    atomic_json(path, manifest)
    return {'path': str(path), 'files': len(manifest['files']), 'fingerprint': manifest['fingerprint']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/baby_arcus/test2.json')
    parser.add_argument('--maximum-files', type=int)
    args = parser.parse_args()
    print(json.dumps(prepare(args.config, args.maximum_files)), flush=True)
