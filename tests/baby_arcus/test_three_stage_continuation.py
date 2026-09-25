"""CPU-only end-to-end continuation of a tiny, explicitly synthetic learner."""
import importlib.metadata
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
import zstandard
from baby_arcus.shared_factory import create
from baby_arcus.shared_checkpoint import save, digest, load
from baby_arcus.language_stream import atomic_json
from baby_arcus.data_staging import StagingStore
from baby_arcus.data_manifest import build
from baby_arcus.shared_idle_training import train_idle


class Tokenizer:
    vocab_size=256
    eot_token=0
    def encode(self,text): return list(text.encode())
    def decode(self,ids): return bytes(ids).decode(errors='replace')


class ContinuationTests(unittest.TestCase):
    def test_all_streams_retry_pause_and_checkpoint_lineage(self):
        self.run_continuation()

    @unittest.skipUnless(__import__('os').environ.get('ALPHA_PHASE2B_CUDA_TEST') == '1', 'explicit Docker CUDA fixture only')
    def test_language_only_cuda_no_motor(self):
        self.run_continuation(language_only=True)

    def run_continuation(self, language_only=False):
        torch.set_num_threads(1)
        base=Path('runs/test2'); base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='three-stage-unit-',dir=base) as tmp:
            root=Path(tmp).resolve(); corpus=root/'corpus'; corpus.mkdir()
            raw='\n'.join(json.dumps({'text':'A small coding example with values and functions.'}) for _ in range(4))
            (corpus/'fixture.jsonl.zst').write_bytes(zstandard.ZstdCompressor().compress(raw.encode()))
            dataset=root/'dataset-config.json'
            atomic_json(dataset,{'dataset_root':str(corpus),'source_patterns':['*.zst']})
            plan=json.loads(Path('configs/baby_arcus/alpha_three_stage.json').read_text())
            store=StagingStore(root/'review.sqlite','fixture-review-token-123456',fixture=True)
            record={'version':1,'source':'fixture:continuation','group':'tiny-episode','split':'training',
                    'messages':[{'role':'user','content':'Say hello.'},{'role':'assistant','content':'Hello.'}]}
            batch=store.stage([record]); sources=store.stage_sources(build(corpus,['*.zst'],['Python'],True))
            for identity in (batch,sources): store.review(identity,'approved','fixture','fixture-review-token-123456')
            store.close()
            plan.update(fixture=True,training_enabled=True,staging_store=str(root/'review.sqlite'),
                        approved_batches=[batch],approved_source_manifests=[sources],
                        mixture=['embodied','coding_corpus','sft'],token_budget=1000)
            if language_only:
                plan.update(schema='alpha-phase2b-v1', preserve_embodied_schedule=False,
                            mixture=['language','coding_corpus','sft'])
            atomic_json(root/'plan.json',plan)
            cfg={'schema':'arcus-test2-v1','initialization':'random','root':str(root),
                 'seed':2101,'preset':'tiny','text_dim':16,'depth_capacity':1.,'encoding':'o200k_base',
                 'tiktoken_version':importlib.metadata.version('tiktoken'),'learning_rate':.00001,
                 'dataset_config':str(dataset),'three_stage_config':str(root/'plan.json'),
                 'max_storage_bytes':1024**3,
                 'indexed_corpus':language_only,
                 'training_cache_bytes':64*1024**2 if language_only else 0,
                 'idle_learning':{'auto_resume':True,'idle_seconds':60,'chunk_updates':3,'checkpoint_every':3,'session_updates':6}}
            atomic_json(root/'config.json',cfg); atomic_json(root/'experiment.json',cfg)
            model=create(cfg,256); optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'])
            initial=save(root,model,optimizer,{'initialization':'random','sources':{},'updates':0,'trained_tokens':0,'receipts':[]})
            atomic_json(root/'candidate.json',initial)
            atomic_json(root/'three-stage-continuation.json',{'fixture':True,'sha256':initial['sha256']})
            del model,optimizer
            from contextlib import ExitStack
            with ExitStack() as stack:
                stack.enter_context(patch('baby_arcus.three_stage_training.get_tokenizer',return_value=Tokenizer()))
                if language_only:
                    self.assertTrue(torch.cuda.is_available())
                    stack.enter_context(patch('baby_arcus.three_stage_training.model_device',return_value='cuda'))
                    stack.enter_context(patch('baby_arcus.three_stage_training.MotorStream',side_effect=AssertionError('Motor simulation forbidden')))
                result=train_idle(str(root/'config.json'),3,'fixture-three-streams')
                self.assertTrue(result['job_complete'])
                self.assertEqual(result['candidate']['updates'],3)
                repeated=train_idle(str(root/'config.json'),3,'fixture-three-streams')
                self.assertTrue(repeated['already_trained'])
                if language_only:
                    with patch('baby_arcus.training_session.load',side_effect=AssertionError('Warm chunk must reuse model and optimizer')):
                        result=train_idle(str(root/'config.json'),1,'fixture-warm-chunk')
                    self.assertEqual(result['candidate']['updates'],4)
                (root/'pause-training').touch()
                paused=train_idle(str(root/'config.json'),3,'fixture-paused')
                self.assertEqual(paused['stop_reason'],'paused')
                self.assertEqual(paused['candidate'],result['candidate'])
            self.assertEqual(digest(root/(initial['generation']+'.pt')),initial['sha256'])
            _,data=load(root,result['candidate'])
            expected={'language','coding_corpus','sft'} if language_only else {'embodied','coding_corpus','sft'}
            self.assertEqual(set(data['progress']['three_stage']['streams']),expected)
            self.assertEqual(data['progress']['three_stage']['sft_cursor'],1)
            self.assertIn('optimizer',data); self.assertIn('rng',data)
            from baby_arcus.training_session import SESSION
            SESSION.clear()
