import tempfile
import unittest
import json
import os
import time
from pathlib import Path
from unittest.mock import patch
from baby_arcus.runtime_contract import require_container, require_checkpoint, require_gpu


class RuntimeTests(unittest.TestCase):
    def test_controlled_mode_checks_actual_cgroups_without_claiming_hardware_clearance(self):
        env = {'ALPHA_GPU_MODE': 'controlled-docker', 'ALPHA_RUNTIME_IMAGE': 'sha256:'+'a'*64,
               'ALPHA_HOST_FINGERPRINT': 'b'*64}
        limits = {'memory.max': str(10*1024**3), 'pids.max': '256', 'cpu.max': '200000 100000'}
        def read(path, *args, **kwargs): return limits[path.name]
        with patch.dict(os.environ, env), patch('baby_arcus.runtime_contract.require_container'), patch.object(Path, 'read_text', read):
            report = require_gpu()
            self.assertFalse(report['host_stability_established'])
            self.assertFalse(report['training_authorized'])
            for field in limits:
                original = limits[field]
                limits[field] = 'max'
                with self.assertRaises(RuntimeError): require_gpu()
                limits[field] = original
            with patch('baby_arcus.runtime_contract.require_container', side_effect=RuntimeError('native')):
                with self.assertRaises(RuntimeError): require_gpu()

    def test_gpu_receipts_are_scoped_fresh_and_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); scope={'image_id':'sha256:'+'a'*64,'host_fingerprint':'b'*64}
            env={'ALPHA_RUNTIME_IMAGE':scope['image_id'],'ALPHA_HOST_FINGERPRINT':scope['host_fingerprint'],
                 'ALPHA_HOST_QUALIFICATION':str(root/'host.json'),'ALPHA_GPU_QUALIFICATION':str(root/'gpu.json')}
            def write(age=0,complete=True):
                for kind in ('host','gpu'):
                    (root/(kind+'.json')).write_text(json.dumps({'schema':'alpha-'+kind+'-qualification-v1',
                        'scope':scope,'created_at':time.time()-age,'complete':complete,'checks':{'measured':True}}))
            with patch.dict(os.environ,env),patch('baby_arcus.runtime_contract.require_container'):
                write(); self.assertTrue(require_gpu()['complete'])
                write(age=90000)
                with self.assertRaises(RuntimeError): require_gpu()
                write(complete=False)
                with self.assertRaises(RuntimeError): require_gpu()
                write()
                with patch.dict(os.environ,{'ALPHA_RUNTIME_IMAGE':'sha256:'+'c'*64}):
                    with self.assertRaises(RuntimeError): require_gpu()

    def test_direct_cuda_checkpoint_blocked_before_deserialization(self):
        from baby_arcus.shared_checkpoint import load
        with tempfile.TemporaryDirectory() as tmp:
            generation='a'*32; (Path(tmp)/(generation+'.pt')).write_bytes(b'fixture')
            with patch('baby_arcus.runtime_contract.require_gpu',side_effect=RuntimeError('unqualified')),patch('torch.load') as deserialize:
                with self.assertRaises(RuntimeError): load(tmp,{'generation':generation,'sha256':'unused'},'cuda:0')
                deserialize.assert_not_called()

    def test_factory_blocks_direct_cuda(self):
        from baby_arcus.shared_factory import create
        with patch('baby_arcus.runtime_contract.require_gpu',side_effect=RuntimeError('unqualified')):
            with self.assertRaises(RuntimeError): create({'preset':'tiny','depth_capacity':1.},256,'cuda')
    def test_environment_variable_alone_cannot_authorize_native_execution(self):
        with patch('baby_arcus.runtime_contract.identity', return_value={'container':False,'profile':'alpha-container-v1'}):
            with self.assertRaises(RuntimeError): require_container()

    def test_large_checkpoint_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.pt'
            with path.open('wb') as stream: stream.truncate(64*1024*1024+1)
            with patch('baby_arcus.runtime_contract.require_container', side_effect=RuntimeError('blocked')):
                with self.assertRaises(RuntimeError): require_checkpoint(path)

    def test_small_cpu_fixture_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'tiny.pt'; path.write_bytes(b'fixture')
            require_checkpoint(path)
