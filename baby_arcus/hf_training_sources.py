"""Public revision-pinned dataset acquisition with bounded downloads."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import urllib.request

SOURCES = {'SWE-Gym/OpenHands-SFT-Trajectories', 'SWE-bench/SWE-smith-trajectories',
           'nebius/SWE-rebench-openhands-trajectories', 'OpenHands/openhands-feedback'}


def fetch(repo, revision, filename, output, max_bytes=32*1024*1024):
    if repo not in SOURCES or not re.fullmatch('[a-f0-9]{40}', revision):
        raise ValueError('Allowlisted HF source and immutable commit required')
    relative = PurePosixPath(filename)
    if relative.is_absolute() or '..' in relative.parts or not re.fullmatch(r'[\w./-]+', filename):
        raise ValueError('Invalid dataset filename')
    path = Path(output)
    if path.exists():
        raise ValueError('Do not overwrite retained data')
    if not 0 < max_bytes <= 256*1024*1024:
        raise ValueError('Invalid download budget')
    url = f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}'
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    sha = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=30) as response, path.open('xb') as stream:
            while block := response.read(min(65536, max_bytes-count+1)):
                count += len(block)
                if count > max_bytes:
                    raise ValueError('Download budget exceeded')
                stream.write(block); sha.update(block)
    except Exception:
        # Keep partial evidence, but never let it masquerade as completed input.
        if path.exists():
            path.rename(path.with_name(path.name+'.incomplete'))
        raise
    receipt = {'repo':repo, 'revision':revision, 'file':filename, 'sha256':sha.hexdigest(), 'bytes':count}
    path.with_name(path.name+'.receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    return receipt


def rows(path, limit=100):
    if type(limit) is not int or not 1 <= limit <= 10000:
        raise ValueError('Bounded row limit required')
    path = Path(path)
    if path.suffix == '.parquet':
        import pyarrow.parquet as pq
        count = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=min(limit,32)):
            for row in batch.to_pylist():
                yield row
                count += 1
                if count >= limit: return
    elif path.suffix == '.jsonl':
        with path.open(encoding='utf-8') as stream:
            for _ in range(limit):
                line = stream.readline(1048577)
                if not line: break
                if len(line) > 1048576: raise ValueError('Oversize source record')
                yield json.loads(line)
    else:
        raise ValueError('Supported source files are parquet and JSONL')
