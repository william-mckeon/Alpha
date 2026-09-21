import tempfile,unittest
from pathlib import Path
import torch
from tests.baby_arcus.test_shared_learning import fixture,Tokenizer
from baby_arcus.shared_checkpoint import save,load,promote

class SharedCheckpointTests(unittest.TestCase):
    def test_production_worker_rejects_flags_without_measured_evidence(self):
        import json
        from unittest.mock import patch
        from baby_arcus.services.shared_worker import Worker
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);manifest={'generation':'a'*32,'sha256':'b'*64}
            config=root/'config.json';config.write_text(json.dumps({'root':str(root),'tiktoken_version':'test'}))
            (root/'active.json').write_text(json.dumps(manifest))
            (root/'qualification.json').write_text(json.dumps({'candidate':manifest,
                **dict.fromkeys(('integration','retention','cross_modal','live'),True)}))
            with patch('importlib.metadata.version',return_value='test'):
                with self.assertRaisesRegex(ValueError,'Missing measured qualification evidence'):Worker(config)

    def test_qualified_publication_preserves_reviewable_rollback(self):
        import json
        from tests.baby_arcus.test_shared_qualification import passing_records
        from baby_arcus.shared_qualification import compile_report
        model,_=fixture();optimizer=torch.optim.AdamW(model.parameters())
        from baby_arcus.shared_depth import set_depth
        set_depth(model)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);generations=[]
            for step in range(2):
                manifest=save(root,model,optimizer,{'updates':step});generations.append(manifest)
                evidence=root/manifest['generation'];evidence.mkdir();paths={}
                for name,value in passing_records().items():
                    path=evidence/(name+'.json');path.write_text(json.dumps(dict(value,candidate=manifest)));paths[name]=path
                promote(root,manifest,compile_report(root,manifest,paths))
            self.assertEqual(json.loads((root/'active.json').read_text()),generations[1])
            previous=json.loads((root/'previous-active.json').read_text())
            self.assertEqual(previous,generations[0])
            report=json.loads((root/(previous['generation']+'.qualification.json')).read_text())
            promote(root,previous,report)
            self.assertEqual(json.loads((root/'active.json').read_text()),generations[0])
            self.assertEqual(json.loads((root/'previous-active.json').read_text()),generations[1])

    def test_v6_uses_retained_motor_path_with_one_core(self):
        from baby_arcus.shared_model import SharedModel
        from arcus.model import ArcusMoDE
        current,row=fixture();model=SharedModel(current.body,current.language,version=6).eval()
        with torch.no_grad():
            model.body_context.bias.fill_(1000)
            out=model([row],Tokenizer())
            expected=model.body([row['senses']])
        finite=torch.isfinite(out['body'])
        torch.testing.assert_close(out['body'][finite],expected[finite])
        self.assertEqual(sum(isinstance(m,ArcusMoDE) for m in model.modules()),1)
        with tempfile.TemporaryDirectory() as root:
            manifest=save(root,model,torch.optim.AdamW(model.parameters()),{'updates':0})
            restored,_=load(root,manifest);self.assertEqual(restored.version,6)

    def test_v3_preserves_language_path_and_grouped_optimizer(self):
        from baby_arcus.shared_model import SharedModel
        from baby_arcus.shared_checkpoint import restore_optimizer
        from arcus.model import ArcusMoDE
        current,row=fixture();model=SharedModel(current.body,current.language,version=3)
        self.assertEqual(sum(isinstance(module,ArcusMoDE) for module in model.modules()),1)
        tokenizer=Tokenizer();ids=torch.tensor([tokenizer.encode('hello')])
        with torch.no_grad():
            expected=model.language(model.core,ids)[:,-1]
            actual=model([row],tokenizer)['text']
        torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)
        params=list(model.parameters());optimizer=torch.optim.AdamW([{'params':params[:3],'lr':.0001},{'params':params[3:],'lr':.001}])
        with tempfile.TemporaryDirectory() as root:
            manifest=save(root,model,optimizer,{'updates':0})
            restored,data=load(root,manifest);resumed=restore_optimizer(restored,data,.1)
            self.assertEqual(restored.version,3)
            self.assertEqual([g['lr'] for g in resumed.param_groups],[.0001,.001])

    def test_legacy_shared_generation_remains_readable(self):
        from baby_arcus.shared_model import SharedModel
        current,_=fixture();model=SharedModel(current.body,current.language,version=1)
        with tempfile.TemporaryDirectory() as root:
            manifest=save(root,model,torch.optim.AdamW(model.parameters()),{'updates':0})
            restored,data=load(root,manifest)
            self.assertEqual(restored.version,1)
            self.assertEqual(data['schema'],'arcus-shared-v1')
            self.assertFalse(hasattr(restored,'gaze_choice'))

    def test_resumed_optimizer_matches_next_update(self):
        from baby_arcus.shared_learning import update
        model,row=fixture();optimizer=torch.optim.AdamW(model.parameters(),lr=.0001)
        target=[{'body':0,'text':42}]
        update(model,optimizer,[row],Tokenizer(),target)
        with tempfile.TemporaryDirectory() as root:
            manifest=save(root,model,optimizer,{'updates':1,'cursor':{'token':64}})
            restored,data=load(root,manifest)
            resumed=torch.optim.AdamW(restored.parameters(),lr=.0001);resumed.load_state_dict(data['optimizer'])
            torch.set_rng_state(data['rng']);a=update(model,optimizer,[row],Tokenizer(),target)
            torch.set_rng_state(data['rng']);b=update(restored,resumed,[row],Tokenizer(),target)
            self.assertEqual(a,b)
            for key,value in model.state_dict().items():self.assertTrue(torch.equal(value,restored.state_dict()[key]),key)

    def test_roundtrip_and_gated_promotion(self):
        model,row=fixture();optimizer=torch.optim.AdamW(model.parameters())
        with tempfile.TemporaryDirectory() as root:
            manifest=save(root,model,optimizer,{'updates':0,'cursor':{'token':42},'receipts':[]})
            restored,data=load(root,manifest)
            self.assertTrue(torch.equal(model([row],Tokenizer())['hidden'],restored([row],Tokenizer())['hidden']))
            self.assertEqual(data['progress']['cursor']['token'],42)
            with self.assertRaises(ValueError):promote(root,manifest,{'sha256':manifest['sha256'],'integration':True})
            self.assertFalse((Path(root)/'active.json').exists())
    def test_corrupt_and_traversal_rejected(self):
        model,row=fixture()
        with tempfile.TemporaryDirectory() as root:
            manifest=save(root,model,torch.optim.AdamW(model.parameters()),{})
            (Path(root)/(manifest['generation']+'.pt')).write_bytes(b'broken')
            with self.assertRaises(ValueError):load(root,manifest)
            with self.assertRaises(ValueError):load(root,{'generation':'../bad','sha256':'none'})
