import json
import copy
import tempfile
import unittest
from pathlib import Path
from arcus3.checkpoint import digest
from arcus3.checkpoint_recovery import inspect_state, verify_fresh_alpha322_initialization
from arcus3.config import read
from arcus3.learning_rate import identity, initial_state
from arcus3.production import validate_policy
from scripts.run_arcus3_production import bind_alpha322_initialization
from scripts.qualify_arcus3_production import adaptation_for_policy


class Alpha322LineageTests(unittest.TestCase):
    def test_policy_is_4t_and_not_launch_ready(self):
        policy=validate_policy(read('configs/arcus3/production_alpha322.json'))
        self.assertEqual(policy['ceiling_input_tokens'],4_000_000_000_000)
        self.assertEqual(policy['joint_evaluation_input_tokens'],7_000_000)
        self.assertEqual(policy['evaluation'],{'light':7_000_000,'developmental':7_000_000,'full':7_000_000})
        self.assertFalse(policy['launch_ready'])
        with self.assertRaisesRegex(ValueError,'calibration receipt'):
            validate_policy({**policy,'launch_ready':True})
        with self.assertRaisesRegex(ValueError,'exact adaptation configuration'):
            adaptation_for_policy(policy,None)
        cfg=adaptation_for_policy(policy,'configs/arcus3/backbone_adaptation_alpha322.json')
        self.assertEqual(cfg['lineage']['id'],policy['lineage_id'])

    def test_checkpoint_lineage_is_immutable(self):
        import torch
        cfg=read('configs/arcus3/backbone_adaptation_alpha322.json')
        with tempfile.TemporaryDirectory() as temp:
            cp=Path(temp)/'step-0-fixture';cp.mkdir()
            state={'campaign':'backbone-adaptation-v1','parent_sha256':cfg['parent_sha256'],
                   'config_sha256':identity(cfg),'data_sha256':'data','updates':0,'input_tokens':0,
                   'target_tokens':0,'cursor':0,'config':cfg,'scheduler':initial_state(cfg),
                   'optimizer':{'state':{},'param_groups':[]},'python_rng':(),
                   'torch_rng':torch.zeros(1,dtype=torch.uint8),'cuda_rng':[],
                   'stream':{'offset':0},'accumulation_position':0,'evaluation_completed':[],
                   'evaluation_pending':['baseline-full']}
            torch.save(state,cp/'state.pt');(cp/'delta.safetensors').write_bytes(b'fixture')
            manifest={'schema':'arcus3-expanded-delta-v1','parent_sha256':cfg['parent_sha256'],
                      'config_sha256':identity(cfg),'data_sha256':'data','updates':0,
                      'model_label':'alpha3.2.2','lineage_id':'wrong-lineage',
                      'files':{name:digest(cp/name) for name in ('state.pt','delta.safetensors')}}
            (cp/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'lineage'):
                inspect_state(cp,cfg['parent_sha256'],identity(cfg),restart=True)

    def test_independent_zero_update_initialization_verification(self):
        import torch
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);receipt=root/'selection.json'
            receipt.write_text(json.dumps({'schema':'arcus3-alpha322-schedule-selection-v1',
                'lineage_id':'alpha3.2.2-wsd-001','selected_warmup_input_tokens':1_350_000,
                'campaign_updates':0}))
            cfg=copy.deepcopy(read('configs/arcus3/backbone_adaptation_alpha322.json'))
            cfg['campaign_enabled']=True
            cfg['learning_rate_schedule']['selection']={'status':'qualified',
                'selected_warmup_input_tokens':1_350_000,'receipt_sha256':digest(receipt)}
            cfg_path=root/'adaptation.json';cfg_path.write_text(json.dumps(cfg))
            cp=root/'step-0-fixture';cp.mkdir();scheduler=initial_state(cfg)
            state={'campaign':'backbone-adaptation-v1','parent_sha256':cfg['parent_sha256'],
                   'config_sha256':identity(cfg),'data_sha256':'data','teacher_sha256':'teacher',
                   'updates':0,'input_tokens':0,'target_tokens':0,'cursor':0,'config':cfg,
                   'scheduler':scheduler,'optimizer':{'state':{},'param_groups':[]},'python_rng':(),
                   'torch_rng':torch.zeros(1,dtype=torch.uint8),'cuda_rng':[],
                   'stream':{'records':0},'accumulation_position':0,'evaluation_completed':[],
                   'evaluation_pending':[],'evaluation_deferred_until_input_tokens':7_000_000,
                   'retention_policy':'latest-two-plus-major-evaluations-v1'}
            torch.save(state,cp/'state.pt');(cp/'delta.safetensors').write_bytes(b'fixture')
            manifest={'schema':'arcus3-expanded-delta-v1','parent_sha256':cfg['parent_sha256'],
                      'config_sha256':identity(cfg),'data_sha256':'data','updates':0,
                      'campaign':'backbone-adaptation-v1','model_label':'alpha3.2.2',
                      'lineage_id':cfg['lineage']['id'],
                      'learning_rate_schedule_sha256':scheduler['schedule_sha256'],
                      'learning_rate_input_tokens':0,'learning_rate_phase':'warmup',
                      'files':{name:digest(cp/name) for name in ('state.pt','delta.safetensors')}}
            (cp/'manifest.json').write_text(json.dumps(manifest))
            result=verify_fresh_alpha322_initialization(cp,cfg_path,receipt,root/'verified.json')
            self.assertTrue(result['optimizer_empty'] and result['payload_hashes_verified'])
            self.assertEqual(result['updates'],0)
            self.assertTrue((root/'verified.json').exists())

    def test_production_requires_verified_untouched_step_zero_parent(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace=Path(temp);owned=workspace/'runs/arcus3/init';owned.mkdir(parents=True)
            receipt=owned/'selection.json';receipt.write_text(json.dumps({}))
            cfg=copy.deepcopy(read('configs/arcus3/backbone_adaptation_alpha322.json'))
            cfg['campaign_enabled']=True
            cfg['learning_rate_schedule']['selection']={'status':'qualified',
                'selected_warmup_input_tokens':1_350_000,'receipt_sha256':digest(receipt)}
            cfg_path=workspace/'adaptation.json';cfg_path.write_text(json.dumps(cfg))
            cp=owned/'step-0';cp.mkdir();(cp/'manifest.json').write_text('{}')
            manifest_sha=digest(cp/'manifest.json')
            report={'checkpoint_manifest_sha256':manifest_sha,'state':{
                'updates':0,'input_tokens':0,'target_tokens':0,'cursor':0,
                'evaluation_pending':[],'evaluation_completed':[],
                'evaluation_deferred_until_input_tokens':7_000_000,
                'production':None,'stream':{'records':0},'scheduler':initial_state(cfg)}}
            verification=owned/'independent-verification.json'
            verification.write_text(json.dumps({
                'schema':'arcus3-alpha322-initialization-verification-v1','model_label':'alpha3.2.2',
                'lineage_id':cfg['lineage']['id'],'checkpoint':str(cp.resolve()),
                'checkpoint_manifest_sha256':manifest_sha,'config_file_sha256':digest(cfg_path),
                'config_sha256':identity(cfg),'calibration_receipt_sha256':digest(receipt),
                'warmup_input_tokens':1_350_000,'optimizer_empty':True,'payload_hashes_verified':True,
                'updates':0,'input_tokens':0,'target_tokens':0,'launch_started':False,
                'evaluation_deferred_until_input_tokens':7_000_000}))
            evidence=bind_alpha322_initialization(workspace,verification,cp,report,cfg,cfg_path)
            self.assertEqual(evidence[str(verification.resolve())],digest(verification))
            bad=copy.deepcopy(report);bad['state']['updates']=1
            with self.assertRaisesRegex(ValueError,'untouched step-zero'):
                bind_alpha322_initialization(workspace,verification,cp,bad,cfg,cfg_path)
