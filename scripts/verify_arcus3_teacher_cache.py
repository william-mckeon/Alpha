"""Verify a sealed teacher cache against every prepared training record on CPU."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arcus3.checkpoint import digest
from arcus3.config import read, safe_child
from arcus3.tokenizer_contract import validate_data
from baby_arcus.language_stream import atomic_json


def verify_cache(data, teacher, donor):
    from safetensors import safe_open
    data, teacher, donor = map(Path, (data, teacher, donor))
    manifest = read(data / 'manifest.json')
    cache = read(teacher / 'manifest.json')
    validate_data(donor, manifest)
    if cache['data_sha256'] != digest(data / 'manifest.json'):
        raise ValueError('Teacher data identity mismatch')
    if cache['donor_manifest_sha256'] != digest(donor / 'manifest.json'):
        raise ValueError('Teacher donor identity mismatch')
    expected = {}
    tokens = 0
    for shard in manifest['shards']:
        path = safe_child(data, shard['path'])
        if digest(path) != shard['sha256']:
            raise ValueError('Prepared shard changed')
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                row = json.loads(line)
                expected[row['sha256'] + '.safetensors'] = len(row['input_ids']) - 1
                tokens += len(row['input_ids'])
    if set(expected) != set(cache['files']):
        raise ValueError('Incomplete teacher coverage')
    for name, positions in expected.items():
        path = safe_child(teacher, name)
        if digest(path) != cache['files'][name]:
            raise ValueError('Teacher file hash mismatch')
        with safe_open(str(path), framework='pt', device='cpu') as tensors:
            if tensors.get_slice('indices').get_shape() != [positions, cache['top_k']]:
                raise ValueError('Teacher target shape mismatch')
    if tokens != manifest['input_tokens_per_pass'] or tokens != cache['teacher_input_tokens']:
        raise ValueError('Teacher exposure mismatch')
    return {'verified': True, 'records': manifest['records'], 'unique_records':len(expected), 'input_tokens': tokens,
            'data_sha256': digest(data / 'manifest.json'),
            'teacher_sha256': digest(teacher / 'manifest.json'),
            'donor_sha256': cache['donor_manifest_sha256'], 'top_k': cache['top_k']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('data', 'teacher', 'donor', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    result = verify_cache(args.data, args.teacher, args.donor)
    atomic_json(args.output, result)
    print(json.dumps(result))
