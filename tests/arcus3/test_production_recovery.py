"""Model-free recovery checks using serialized optimizer/RNG metadata."""
import json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from arcus3.checkpoint import digest
from arcus3.checkpoint_recovery import inspect_state,prepare
from arcus3.production import identity
from arcus3.config import read


class RecoveryTests(unittest.TestCase):
    def fixture(self,root,updates=0):
        import torch
        cfg={'model_label':'alpha3.2.1','parent_sha256':'parent','nll_regression_limit':.2}
        cp=root/'step-fixture';cp.mkdir()
        state={'campaign':'backbone-adaptation-v1','parent_sha256':'parent','config_sha256':identity(cfg),'data_sha256':'data',
               'updates':updates,'input_tokens':updates*4,'target_tokens':updates*3,'cursor':updates,'config':cfg,
               'optimizer':{'state':{1:{'step':1}} if updates else {},'param_groups':[]},'python_rng':(),
               'torch_rng':torch.zeros(1,dtype=torch.uint8),'cuda_rng':[],'stream':{'offset':updates},
               'accumulation_position':0,'evaluation_completed':[],'evaluation_pending':['baseline-full']}
        torch.save(state,cp/'state.pt');(cp/'delta.safetensors').write_bytes(b'fixture')
        m={k:state[k] for k in ('parent_sha256','config_sha256','data_sha256','updates')}
        m.update(schema='arcus3-expanded-delta-v1',model_label='alpha3.2.1',files={n:digest(cp/n) for n in ('state.pt','delta.safetensors')})
        (cp/'manifest.json').write_text(json.dumps(m));return cp,cfg,state

    def test_restart_accepts_zero_and_rejects_trained_state(self):
        with tempfile.TemporaryDirectory() as d:
            cp,cfg,_=self.fixture(Path(d))
            result=inspect_state(cp,'parent',identity(cfg),True)
            self.assertTrue(result['optimizer_empty']);self.assertEqual(result['state']['cursor'],0)
        with tempfile.TemporaryDirectory() as d:
            cp,cfg,_=self.fixture(Path(d),1)
            with self.assertRaisesRegex(ValueError,'zero-update'):inspect_state(cp,'parent',identity(cfg),True)

    def test_tampering_and_wrong_model_config_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            cp,cfg,_=self.fixture(Path(d))
            with self.assertRaises(ValueError):inspect_state(cp,'wrong',identity(cfg))
            with self.assertRaises(ValueError):inspect_state(cp,'parent','wrong')
            (cp/'state.pt').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):inspect_state(cp,'parent',identity(cfg))

    def test_recovery_preserves_accepted_evals_counters_queue_and_pause(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source';source.mkdir();(source/'pause-training').write_text('user pause')
            data=root/'data';teacher=root/'teacher';data.mkdir();teacher.mkdir()
            for p in (data,teacher):(p/'manifest.json').write_text('{}')
            policy=read('configs/arcus3/production.json');policy_path=root/'policy.json';policy_path.write_text(json.dumps(policy))
            cfg={'parent_sha256':'parent'};config_path=root/'cfg.json';config_path.write_text(json.dumps(cfg))
            state={'updates':1,'input_tokens':4,'target_tokens':3,'data_sha256':digest(data/'manifest.json'),
                   'teacher_sha256':digest(teacher/'manifest.json'),'production':{'policy_sha256':identity(policy)},
                   'evaluation_pending':[],'evaluation_completed':[{'update':0,'nll':2.2}],'stream':{'offset':43}}
            (source/'controller-state.json').write_text(json.dumps({'checkpoint':'old','state':{},'data':str(data),'teacher':str(teacher),
                 'evaluation':'stale','transition':'stale','queued_batches':['queued']}))
            verified={'state':state,'checkpoint_manifest_sha256':'sha','optimizer_empty':False}
            with patch('arcus3.checkpoint_recovery.inspect_state',return_value=verified):
                result=prepare(source,root/'cp',policy_path,config_path,root/'out')
            saved=read(root/'out/controller-state.json')
            self.assertEqual(saved['state'],state);self.assertEqual(saved['queued_batches'],['queued'])
            self.assertIsNone(saved['transition']);self.assertIsNone(saved['evaluation'])
            self.assertFalse(result['launch_started']);self.assertTrue((source/'pause-training').exists())
