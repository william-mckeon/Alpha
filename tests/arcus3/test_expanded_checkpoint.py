"""Docker CUDA save-failure boundaries using a disposable tiny model."""
import json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch

@unittest.skipUnless(os.environ.get('ARCUS3_CONTROLLED_DOCKER')=='1','Docker CUDA only')
class SaveTests(unittest.TestCase):
    def test_durable_save_retention_failure_and_exact_restore(self):
        import torch,random
        from baby_arcus.gpu_job_control import gpu_job
        from arcus3.expanded_checkpoint import save,restore,verify,CheckpointRetentionError
        from arcus3.checkpoint import digest
        with gpu_job(),tempfile.TemporaryDirectory() as d:
            torch.cuda.set_per_process_memory_fraction(.7)
            model=torch.nn.Linear(2,2).cuda();opt=torch.optim.AdamW(model.parameters(),lr=1e-5)
            model(torch.ones(1,2,device='cuda')).sum().backward();opt.step();opt.zero_grad()
            state={'updates':1,'parent_sha256':'p','data_sha256':'d','config_sha256':'c','input_tokens':4,'target_tokens':3,
                   'retention_policy':'latest-two-plus-major-evaluations-v1','production':{'campaign_id':'test'}}
            original={n:p.clone() for n,p in model.named_parameters()}
            with patch('arcus3.checkpoint_retention.register_and_prune',side_effect=OSError('injected retention failure')):
                with self.assertRaises(CheckpointRetentionError) as error:save(d,model,opt,state)
            cp=error.exception.checkpoint;verify(cp,'p','d','c')
            pointer=json.loads((Path(d)/'latest.json').read_text())
            self.assertEqual(pointer['manifest_sha256'],digest(cp/'manifest.json'))
            self.assertFalse(json.loads((Path(d)/'last-save.json').read_text())['retention_complete'])
            expected=(torch.rand(3,device='cuda'),torch.rand(3),random.random())
            with torch.no_grad():
                for p in model.parameters():p.add_(1)
            restored=restore(cp,model,opt,'p','d','c')
            self.assertTrue(all(torch.equal(original[n],p) for n,p in model.named_parameters()))
            self.assertTrue(torch.equal(torch.rand(3,device='cuda'),expected[0]));self.assertTrue(torch.equal(torch.rand(3),expected[1]))
            self.assertEqual(random.random(),expected[2]);self.assertEqual(restored['updates'],1)
            # A payload write failure cannot advance the authoritative latest pointer.
            with patch('safetensors.torch.save_file',side_effect=OSError('injected write failure')):
                with self.assertRaises(OSError):save(d,model,opt,{**state,'updates':2})
            self.assertEqual(json.loads((Path(d)/'latest.json').read_text()),pointer)

    def test_incompatible_retention_rejected_before_new_generation(self):
        import torch
        from baby_arcus.gpu_job_control import gpu_job
        from arcus3.expanded_checkpoint import save
        with gpu_job(),tempfile.TemporaryDirectory() as d:
            model=torch.nn.Linear(1,1).cuda();opt=torch.optim.AdamW(model.parameters())
            state={'updates':0,'parent_sha256':'p','data_sha256':'d','config_sha256':'c',
                   'retention_policy':'latest-two-plus-major-evaluations-v1'}
            save(d,model,opt,state);before={p.name for p in Path(d).iterdir()}
            with self.assertRaisesRegex(ValueError,'isolated'):save(d,model,opt,{**state,'updates':1,'production':{'x':1}})
            self.assertEqual(before,{p.name for p in Path(d).iterdir()})
            from arcus3.checkpoint_retention import protect_initialization
            cp=Path(d)/json.loads((Path(d)/'latest.json').read_text())['generation']
            protect_initialization(d,cp,'p','c')
            for step in (1,2,3,4):save(d,model,opt,{**state,'updates':step,'production':{'x':1}})
            self.assertTrue(cp.exists());self.assertEqual(len(list(Path(d).glob('step-*'))),3)
