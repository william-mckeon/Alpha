"""Fail closed before storage growth; never silently erase experiment evidence."""
from pathlib import Path
import shutil
import os


def check(root, maximum, reserve=0):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    # Keep full reconciliation, but avoid Path allocation and repeated stat calls.
    pending=[root];size=0
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):pending.append(entry.path)
                elif entry.is_file():size+=entry.stat().st_size
    if size + reserve > maximum or shutil.disk_usage(root).free < reserve:
        raise RuntimeError('Experiment storage budget exhausted; archive explicitly before resuming')
    return size
