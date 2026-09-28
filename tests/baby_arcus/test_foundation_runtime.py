import tempfile
import unittest
from datetime import datetime,timezone
from pathlib import Path
from baby_arcus.foundation_schedule import stop_reason

class RuntimeTests(unittest.TestCase):
    def test_failed_lock_open_has_safe_cleanup(self):
        from baby_arcus.process_lock import ProcessLock
        lock=ProcessLock.__new__(ProcessLock)
        lock.close()

    def test_pause_and_deadline(self):
        with tempfile.TemporaryDirectory() as d:
            now=datetime(2026,9,28,tzinfo=timezone.utc)
            self.assertEqual(stop_reason(d,'2026-09-27T00:00:00Z',now),'deadline')
            self.assertIsNone(stop_reason(d,'2026-09-29T00:00:00Z',now))
            Path(d,'pause-training').touch()
            self.assertEqual(stop_reason(d,'2026-09-29T00:00:00Z',now),'user_pause')
    def test_training_refuses_expired_deadline_before_model_creation(self):
        from scripts.train_arcus_smollm2 import run
        with tempfile.TemporaryDirectory() as d:
            result=run({'updates':2,'world_size':1,'micro_batch':1},[],None,Path(d)/'foundation'/'fixture',1,'2000-01-01T00:00:00Z')
            self.assertEqual(result['status'],'paused')

    def test_end_to_end_train_evaluate_resume(self):
        import json
        from dataclasses import asdict
        from unittest.mock import patch
        from datetime import timedelta
        from tests.baby_arcus.foundation_fixtures import tiny_model,corpus,Tokenizer
        from scripts.train_arcus_smollm2 import run
        model=tiny_model()
        cfg=json.loads(Path('configs/baby_arcus/arcus_128m_smollm2_pretrain.yaml').read_text())
        cfg.update(body_config=asdict(model.body.cfg),vocab_size=128,text_dim=8,
                   required_unique_parameters=sum(p.numel() for p in model.parameters()),
                   sequence_length=32,tokens_per_update=64,updates=4,warmup_steps=1,
                   decay_start=2,decay_steps=2,checkpoint_every=1,evaluate_every=1,trace_every=1)
        del model
        settings={'sequence_length':32,'max_windows_per_domain':1,'max_generated_tokens':4,
                  'base_prompts':['Hi'],'chat_prompts':['User: Hi Assistant:'],'choices':[], 'retrieval_positions':[]}
        with tempfile.TemporaryDirectory() as d,patch('arcus.tokenizer.get_tokenizer',return_value=Tokenizer()):
            path=Path(d)/'data.sqlite';corpus(path);root=Path(d)/'foundation'/'run'
            deadline=(datetime.now(timezone.utc)+timedelta(minutes=3)).isoformat()
            sources=[{'name':'a','weight':.5},{'name':'b','weight':.5}]
            first=run(cfg,sources,path,root,1,deadline,eval_settings=settings)
            second=run(cfg,sources,path,root,1,deadline,resume=True,eval_settings=settings)
            self.assertEqual(first['candidate']['updates'],1);self.assertEqual(second['candidate']['updates'],2)
            self.assertTrue((root/'evaluation-1.json').exists());self.assertTrue((root/'evaluation-2.json').exists())
            self.assertNotEqual(first['candidate']['sha256'],second['candidate']['sha256'])
            self.assertTrue((root/'routing-2.json').exists())
