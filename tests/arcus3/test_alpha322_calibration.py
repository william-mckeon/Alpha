import json
import copy
import tempfile
import unittest
from pathlib import Path
from arcus3.config import read
from scripts.calibrate_arcus3_alpha322_schedule import assess,validate_protocol


class Alpha322CalibrationTests(unittest.TestCase):
    def test_all_arms_same_fresh_start_and_explicit_selection(self):
        cfg=read('configs/arcus3/alpha322_schedule_calibration.json')
        with tempfile.TemporaryDirectory() as temp:
            cfg_path=Path(temp)/'config.json';cfg_path.write_text(json.dumps(cfg));cfg['_path']=str(cfg_path)
            paths=[]
            for arm in cfg['arms']:
                path=Path(temp)/(arm['id']+'.json')
                path.write_text(json.dumps({'schema':'arcus3-alpha322-warmup-arm-v1','disposable':True,
                    'model_label':'alpha3.2.2','lineage_id':'alpha3.2.2-wsd-001',
                    'arm_id':arm['id'],'warmup_input_tokens':arm['warmup_input_tokens'],'input_tokens':cfg['tokens_per_arm'],
                    'initialization_manifest_sha256':'i','initialization_recipe_sha256':'recipe',
                    'data_sha256':'d','record_order_sha256':'r',
                    'complete':True,'qualification_exact_replay':True,'scheduler_qualification_sha256':'q','frozen_unchanged':True,
                    'nll_before':2.25,'nll_after':2.251,
                    'gradient_groups':{'expert':1,'router':1,'gate':1}}));paths.append(path)
            result=assess(cfg,paths,1_350_000)
            self.assertEqual(result['selected_warmup_input_tokens'],1_350_000)
            broken=json.loads(paths[0].read_text());broken['data_sha256']='other';paths[0].write_text(json.dumps(broken))
            with self.assertRaisesRegex(ValueError,'did not share initialization'):
                assess(cfg,paths,1_350_000)

    def test_protocol_rejects_weakened_safety_gates(self):
        cfg=read('configs/arcus3/alpha322_schedule_calibration.json')
        changes={
            'maximum_nll_regression':1.0,'peak_learning_rate':2e-5,
            'fresh_initialization_required':False,'same_records_and_order_required':False,
            'scheduler_replay_qualification_required':False,'frozen_backbone_required':False,
            'all_gradient_groups_required':False,'selection_mode':'automatic-lowest-loss',
        }
        for field,value in changes.items():
            with self.subTest(field=field):
                changed=copy.deepcopy(cfg);changed[field]=value
                with self.assertRaisesRegex(ValueError,'grid/provenance'):
                    validate_protocol(changed)
