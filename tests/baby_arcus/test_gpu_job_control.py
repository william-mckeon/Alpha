import os
import tempfile
import unittest
from unittest.mock import patch
from baby_arcus.gpu_job_control import gpu_job
from baby_arcus.process_lock import ProcessLock
from baby_arcus.artifacts import Conflict
from pathlib import Path


class OwnershipTests(unittest.TestCase):
    def test_os_lock_exclusion_and_release(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'ALPHA_JOB_CONTROL':tmp}):
            with gpu_job():
                with gpu_job(): pass
                with self.assertRaises(Conflict): ProcessLock(Path(tmp)/'gpu.lock')
            lock=ProcessLock(Path(tmp)/'gpu.lock'); lock.close()

    def test_missing_shared_mount_fails_closed(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                with gpu_job(): pass
