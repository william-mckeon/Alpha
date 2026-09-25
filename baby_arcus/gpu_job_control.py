"""OS-owned cross-run lock; all Alpha containers mount the same control volume."""
import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from baby_arcus.process_lock import ProcessLock

_owned = ContextVar('alpha_gpu_owned', default=False)


@contextmanager
def gpu_job():
    if _owned.get():
        yield
        return
    directory = os.environ.get('ALPHA_JOB_CONTROL')
    if not directory:
        raise RuntimeError('Shared ALPHA_JOB_CONTROL mount required for model jobs')
    lock = ProcessLock(Path(directory) / 'gpu.lock')
    token = _owned.set(True)
    try:
        yield
    finally:
        _owned.reset(token)
        lock.close()


def serialized(function):
    from functools import wraps
    @wraps(function)
    def wrapped(*args, **kwargs):
        with gpu_job():
            return function(*args, **kwargs)
    return wrapped
