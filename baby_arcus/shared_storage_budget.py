"""Fail closed before storage growth; never silently erase experiment evidence."""
from pathlib import Path
import shutil


def check(root, maximum, reserve=0):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    size = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    if size + reserve > maximum or shutil.disk_usage(root).free < reserve:
        raise RuntimeError('Experiment storage budget exhausted; archive explicitly before resuming')
    return size
