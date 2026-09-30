"""Reclaim only registered, consumed production batches not referenced by checkpoints."""
import json,sqlite3
from pathlib import Path
from arcus3.config import read
from arcus3.checkpoint import digest
from baby_arcus.language_stream import atomic_json

def reclaim(cache,checkpoint_root,protected=()):
    cache=Path(cache).resolve();checkpoints=Path(checkpoint_root).resolve();keep=set(protected)
    for path in checkpoints.glob('step-*/manifest.json'):keep.add(read(path)['data_sha256'])
    db=sqlite3.connect(cache/'acquisition.sqlite')
    try:
        if not db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='batches'").fetchone():return []
        removed=[]
        for location,sha in db.execute('SELECT path,manifest FROM batches'):
            root=Path(location)
            if not root.exists() or sha in keep:continue
            if root.is_symlink() or root.resolve()!=root.absolute() or root.resolve().parent!=cache or not root.name.startswith('batch-'):raise ValueError('Unsafe managed batch')
            if digest(root/'manifest.json')!=sha:raise ValueError('Changed managed batch')
            teacher=cache/('teacher-'+root.name.removeprefix('batch-'))
            if not (teacher/'manifest.json').exists():continue
            if read(teacher/'manifest.json')['data_sha256']!=sha:raise ValueError('Teacher batch mismatch')
            planned=[]
            for directory in (root,teacher):
                if directory.is_symlink() or directory.resolve()!=directory.absolute() or directory.resolve().parent!=cache:raise ValueError('Unsafe managed cache directory')
                files=list(directory.iterdir())
                allowed=({'manifest.json','provenance.json','progress.json'}|{s['path'] for s in read(root/'manifest.json')['shards']}) if directory==root else ({'manifest.json','identity.json','completed.jsonl','progress.json'}|set(read(teacher/'manifest.json')['files']))
                for p in files:
                    if not p.is_file() or p.is_symlink() or p.resolve().parent!=directory.resolve():raise ValueError('Unsafe cache entry')
                    if p.name not in allowed:raise ValueError('Unknown cache file')
                planned.append((directory,files))
            evidence=cache/'receipts';evidence.mkdir(exist_ok=True)
            atomic_json(evidence/(root.name+'.json'),{'data':read(root/'manifest.json'),'teacher':read(teacher/'manifest.json')})
            for directory,files in planned:
                for p in files:p.unlink()
                directory.rmdir()
            removed.append(root.name)
        return removed
    finally:db.close()
