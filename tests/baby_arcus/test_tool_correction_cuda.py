"""Opt-in disposable CUDA fixture; never updates the production checkpoint."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.environ.get('ALPHA_CORRECTION_CUDA_TEST')=='1','explicit bounded Docker CUDA fixture only')
class CorrectionCudaTests(unittest.TestCase):
    def test_new_plan_preserves_parent_and_resumes_all_streams(self):
        import importlib.metadata
        import torch
        import zstandard
        from baby_arcus.shared_factory import create
        from baby_arcus.shared_checkpoint import save,digest,load
        from baby_arcus.data_staging import StagingStore
        from baby_arcus.data_manifest import build
        from baby_arcus.three_stage_training import train
        from baby_arcus.training_session import SESSION
        from tests.baby_arcus.test_three_stage_continuation import Tokenizer
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(.7)
        with tempfile.TemporaryDirectory(dir='runs/test2',prefix='correction-cuda-fixture-') as folder:
            root=Path(folder).resolve();corpus=root/'corpus';corpus.mkdir()
            (corpus/'text.jsonl.zst').write_bytes(zstandard.ZstdCompressor().compress(
                ((json.dumps({'text':'A simple language and coding example. '*100})+'\n')*2).encode()))
            (root/'language.json').write_text(json.dumps({'dataset_root':str(corpus),'source_patterns':['*.zst']}))
            cfg={'schema':'arcus-test2-v1','initialization':'random','root':str(root),'seed':2101,
                 'preset':'tiny','text_dim':16,'depth_capacity':1.,'encoding':'o200k_base',
                 'tiktoken_version':importlib.metadata.version('tiktoken'),'learning_rate':.00001,
                 'context_tokens':512,'dataset_config':str(root/'language.json'),
                 'three_stage_config':str(root/'training.json'),'max_storage_bytes':1024**3,'cuda_memory_fraction':.7}
            for name in ('config.json','experiment.json'):(root/name).write_text(json.dumps(cfg))
            model=create(cfg,256,device='cuda');optimizer=torch.optim.AdamW(model.parameters(),lr=.00001)
            parent=save(root,model,optimizer,{'initialization':'random','sources':{},'updates':40000,
                'trained_tokens':0,'receipts':[],'three_stage':{'plan_hash':'old-plan','index':9}})
            del model,optimizer
            (root/'candidate.json').write_text(json.dumps(parent))
            (root/'three-stage-continuation.json').write_text(json.dumps({'fixture':True,'sha256':parent['sha256']}))
            store=StagingStore(root/'review.sqlite','test-review-secret-12345678',fixture=True)
            batch=store.stage([{'version':1,'source':'fixture:correction','group':'one','split':'training',
                'messages':[{'role':'user','content':'Hello'},{'role':'assistant','content':'Hello, I am Arcus.'}]}])
            source=store.stage_sources(build(corpus,['*.zst'],['Python'],True))
            for item in (batch,source):store.review(item,'approved','fixture','test-review-secret-12345678')
            store.close()
            plan=json.loads(Path('configs/baby_arcus/alpha_tool_correction_training.json').read_text())
            plan.update(fixture=True,training_enabled=True,source_sha256=parent['sha256'],
                staging_store=str(root/'review.sqlite'),approved_batches=[batch],approved_source_manifests=[source],
                context_tokens=512,language_window_tokens=64,interleave_corpus_files=False,
                mixture=['language','sft','coding_corpus'],token_budget=10000)
            (root/'training.json').write_text(json.dumps(plan))
            with patch('baby_arcus.three_stage_training.get_tokenizer',return_value=Tokenizer()), \
                 patch('baby_arcus.three_stage_training.model_device',return_value='cuda'):
                first=train(str(root/'config.json'),2,checkpoint_every=1)
                self.assertEqual(first['candidate']['updates'],40002)
                second=train(str(root/'config.json'),1,checkpoint_every=1)
                self.assertEqual(second['candidate']['updates'],40003)
            self.assertEqual(digest(root/(parent['generation']+'.pt')),parent['sha256'])
            model,data=load(root,second['candidate'],'cuda')
            self.assertEqual(set(data['progress']['three_stage']['streams']),{'language','sft','coding_corpus'})
            self.assertEqual(data['progress']['previous_three_stage'][0]['plan_hash'],'old-plan')
            self.assertEqual(data['progress']['three_stage']['sft_cursor'],1)
            self.assertTrue(data['optimizer']['state'])
            self.assertIn('fixture:correction/text',data['progress']['three_stage']['target_exposures'])
            old_state=data['progress']['three_stage']
            old_index=old_state['index']
            plan['migration']={'previous_plan_hash':old_state['plan_hash'],'at_updates':40003}
            plan['token_budget']=11000
            del model,data;SESSION.clear()
            (root/'training.json').write_text(json.dumps(plan))
            with patch('baby_arcus.three_stage_training.get_tokenizer',return_value=Tokenizer()), \
                 patch('baby_arcus.three_stage_training.model_device',return_value='cuda'):
                third=train(str(root/'config.json'),1,checkpoint_every=1)
            model,data=load(root,third['candidate'],'cuda')
            self.assertEqual(data['progress']['three_stage']['index'],old_index+1)
            self.assertEqual(data['progress']['three_stage']['sft_cursor'],1)
            self.assertTrue(data['optimizer']['state'])
            self.assertEqual(digest(root/(parent['generation']+'.pt')),parent['sha256'])
            del model,data;SESSION.clear()
