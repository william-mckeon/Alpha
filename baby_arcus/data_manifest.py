"""Content-addressed selected DatasetForge shards; creation does not approve data."""
from pathlib import Path
from baby_arcus.language_stream import inventory
import hashlib
from baby_arcus.contracts import digest


def file_digest(path, cancelled=lambda: False):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            if cancelled():
                raise InterruptedError('Corpus verification interrupted')
            block = stream.read(1024 * 1024)
            if not block:
                return value.hexdigest()
            value.update(block)


def build(root, patterns, languages, fixture=False, dataset_id=None, cancelled=lambda: False):
    if not languages or set(languages) - {'Python','JavaScript','Go','Rust'}:
        raise ValueError('Only the selected coding languages are eligible for review')
    result = inventory(root, patterns)
    for entry in result['files']:
        entry['sha256'] = file_digest(Path(result['root'])/entry['path'], cancelled)
    result.update(schema='alpha-coding-corpus-v1',fixture=fixture,languages=sorted(languages))
    result['fingerprint'] = digest(result['files'])
    result['fingerprint_method'] = 'content SHA256, path, size and modification time'
    if dataset_id is not None:
        result['schema'] = 'alpha-coding-corpus-v2'
        result['dataset_id'] = dataset_id
        result.pop('root')
        for entry in result['files']:
            entry.pop('mtime_ns', None)
            entry.pop('mtime', None)
        result['fingerprint'] = digest(result['files'])
        result['fingerprint_method'] = 'content SHA256, relative path and size'
    return result


def validate_manifest(manifest, verify_files=False, cancelled=lambda: False):
    if manifest.get('schema') not in ('alpha-coding-corpus-v1','alpha-coding-corpus-v2','alpha-coding-corpus-v3') or type(manifest.get('fixture')) is not bool:
        raise ValueError('Unsupported corpus manifest')
    if not manifest.get('languages') or set(manifest['languages']) - {'Python','JavaScript','Go','Rust'}:
        raise ValueError('Ineligible coding source')
    if not manifest.get('files') or digest(manifest['files']) != manifest.get('fingerprint'):
        raise ValueError('Corpus identity mismatch')
    if manifest['schema'] in ('alpha-coding-corpus-v2','alpha-coding-corpus-v3') and not manifest.get('dataset_id'):
        raise ValueError('Logical dataset identity required')
    from baby_arcus.dataset_paths import resolve_root
    root = resolve_root(manifest) if verify_files else None
    for item in manifest['files']:
        if manifest['schema'] == 'alpha-coding-corpus-v3' and item.get('split') not in ('training','validation','test'):
            raise ValueError('Explicit file split required')
        if cancelled():
            raise InterruptedError('Corpus verification interrupted')
        relative = Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts or '\\' in item['path'] or ':' in item['path']:
            raise ValueError('Corpus path escape')
        if verify_files:
            path = (root/relative).resolve()
            if root not in path.parents or file_digest(path, cancelled) != item['sha256']:
                raise ValueError('Approved corpus content changed or escaped root')
    return manifest
