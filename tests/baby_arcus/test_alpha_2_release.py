import json
import tempfile
import unittest
from pathlib import Path
from scripts.release_alpha_2 import validate_evidence,verify_package,digest

class ReleaseTests(unittest.TestCase):
    def test_inference_package_includes_observation_assets_and_not_runs(self):
        from scripts.release_alpha_2 import copy_inference_sources
        with tempfile.TemporaryDirectory() as d:
            copy_inference_sources(d)
            for name in ('arcus-lying.png','arcus-body.png','arcus-sitting.png'):
                self.assertEqual(digest(Path(d,'baby_arcus/web',name)),digest(Path('baby_arcus/web',name)))
            self.assertTrue(Path(d,'baby_arcus/conversation_probe.py').is_file())
            self.assertFalse(Path(d,'runs').exists())
            self.assertTrue(Path(d,'LICENSE').is_file())

    def test_explicit_snapshot_identity_and_evidence(self):
        from scripts.release_alpha_2 import validate_snapshot
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'saved.pt').write_bytes(b'fixture')
            pointer={'generation':'saved','updates':53192,'sha256':digest(root/'saved.pt')}
            (root/'candidate.json').write_text(json.dumps(pointer));(root/'pause-training').touch()
            spec={'run_root':d,'private':True,'repo_id':'Islanderintel/Alpha-2.0',
                  'authorization':'user-requested-current-checkpoint-publication',
                  'expected_candidate':pointer.copy(),'required_updates':53192}
            common={'candidate':pointer,'complete':True,'checkpoint_unchanged':True}
            for key,extra in {
                'developmental_report':{'records':[{}]*36,'executor_controls':{'correct_passed':True,'incorrect_rejected':True},'summary':{},'identity':{}},
                'mapping_report':{'traces':[{}]*36,'inventory':{'unique_parameters':128353994}},
                'agent_report':{'scores':{}}}.items():
                path=root/(key+'.json');path.write_text(json.dumps({**common,**extra}));spec[key]=str(path)
            self.assertEqual(validate_snapshot(spec)[0],pointer)
            spec['expected_candidate']['updates']=60000
            with self.assertRaisesRegex(ValueError,'identity'):validate_snapshot(spec)
            spec['expected_candidate']=pointer.copy();spec['authorization']=''
            with self.assertRaisesRegex(ValueError,'authorization'):validate_snapshot(spec)

    def test_public_repository_rejected_before_upload(self):
        import sys,types
        from unittest.mock import MagicMock,patch
        from scripts.release_alpha_2 import publish
        api=MagicMock();api.model_info.return_value.private=False
        fake=types.ModuleType('huggingface_hub');fake.HfApi=lambda **kw:api;fake.hf_hub_download=MagicMock()
        manifest={'source_checkpoint':{},'validated_tensor_identity':True,'validated_greedy_parity':True}
        with patch.dict(sys.modules,{'huggingface_hub':fake}),patch.dict('os.environ',{'HF_TOKEN':'test-only'}),patch('scripts.release_alpha_2.validate_evidence',return_value=({},{})),patch('scripts.release_alpha_2.verify_package',return_value=manifest):
            with self.assertRaisesRegex(ValueError,'public'):
                publish({'destination':'unused','repo_id':'Islanderintel/Alpha-2.0'})
        api.upload_folder.assert_not_called()

    def test_early_release_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d,'candidate.json').write_text(json.dumps({'updates':53192}))
            with self.assertRaisesRegex(ValueError,'60000'):
                validate_evidence({'run_root':d,'private':True,'repo_id':'Islanderintel/Alpha-2.0','required_updates':60000})

    def test_tampering_and_unlisted_files_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'weights').write_bytes(b'example')
            (root/'manifest.json').write_text(json.dumps({'files':{'weights':digest(root/'weights')}}))
            verify_package(root)
            (root/'secret.txt').write_text('not for publication')
            with self.assertRaises(ValueError):verify_package(root)
