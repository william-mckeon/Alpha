import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import zstandard
from arcus.tokenizer import get_tokenizer
from baby_arcus.body_policy import BodyPolicy
from baby_arcus.language_model import LanguageAdapter, generate
from baby_arcus.language_learning import update
from baby_arcus.language_stream import LanguageStream, atomic_json, inventory
from baby_arcus.language_runtime import LanguageRuntime
from baby_arcus.services.playroom import PlayroomApplication


class LanguageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tokenizer=get_tokenizer('o200k_base')

    def runtime(self,app):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        root=Path(temporary.name)
        (root/'language.json').write_text(json.dumps({'decision_interval_seconds':5,
            'language_root':'language','session_training_tokens':4096}))
        return LanguageRuntime(app,__file__,root,config='language.json')

    def test_unicode_and_literal_special_tokens(self):
        text='Hello Arcus 🐉 — café 你好 <|endoftext|>'
        ids=self.tokenizer.encode(text)
        self.assertEqual(self.tokenizer.decode(ids),text)
        self.assertNotIn(self.tokenizer.eot_token,ids)

    def test_atomic_json_retries_transient_permission_error(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'status.json';path.write_text('{"old":true}',encoding='utf-8')
            real_replace=os.replace;attempts=[]
            def transient(source,destination):
                attempts.append((source,destination))
                if len(attempts)<3:raise PermissionError('simulated OneDrive lock')
                return real_replace(source,destination)
            with patch('baby_arcus.language_stream.os.replace',side_effect=transient), \
                 patch('baby_arcus.language_stream.time.sleep'):
                atomic_json(path,{'new':True})
            self.assertEqual(json.loads(path.read_text(encoding='utf-8')),{'new':True})
            self.assertEqual(len(attempts),3)
            self.assertEqual(list(Path(folder).glob('*.pending')),[])

    def test_stream_cursor_pause_replay_restart_and_holdout(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source';source.mkdir()
            records=[{'text':'HELD OUT ONLY'}, {'text':'one two three four five six seven'}, {'content':'next story'}]
            (source/'a.jsonl.zst').write_bytes(zstandard.ZstdCompressor().compress(
                ''.join(json.dumps(row)+'\n' for row in records).encode()))
            manifest=inventory(source,['*.jsonl.zst']);path=root/'cursor.json'
            stream=LanguageStream(manifest,self.tokenizer,path)
            first=stream.next(2);self.assertEqual(first['document'],1)
            cursor=dict(stream.state);stream.control('pause');self.assertIsNone(stream.next())
            self.assertEqual(stream.next(replay=True),first);stream.close()
            restored=LanguageStream(manifest,self.tokenizer,path)
            self.assertFalse(restored.state['playing']);restored.control('resume')
            second=restored.next(2);self.assertEqual(second['offset'],2)
            restored.control('restart');self.assertEqual(restored.next(2)['tokens'],first['tokens'])
            while restored.next(64):pass
            self.assertTrue(restored.state['eof']);restored.control('resume')
            self.assertFalse(restored.state['playing']);restored.close()
            altered=dict(manifest,fingerprint='different')
            with self.assertRaisesRegex(ValueError,'Dataset changed'):LanguageStream(altered,self.tokenizer,path)

    def test_language_training_changes_adapter_and_preserves_body(self):
        torch.manual_seed(5);body=BodyPolicy().eval().requires_grad_(False)
        adapter=LanguageAdapter(body.cfg.dim,32,text_dim=8)
        optimizer=torch.optim.AdamW(adapter.parameters(),lr=.005)
        before={k:v.clone() for k,v in body.state_dict().items()}
        embedding=adapter.embedding.weight.clone()
        tokens=torch.tensor([[1,2,3]])
        torch.testing.assert_close(body.core.trunk(tokens),body.core.trunk_embedded(body.core.token_embed(tokens)),rtol=0,atol=0)
        update(adapter,body.core,optimizer,[1,2,3,4,5],'cpu')
        self.assertFalse(torch.equal(embedding,adapter.embedding.weight))
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in body.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in body.parameters()))

    def test_generation_is_model_output_and_filters_invalid_utf8(self):
        class Output:
            def parameters(self):return iter([torch.zeros(1)])
            def __call__(self,core,tokens):return torch.zeros(1,tokens.shape[1],self.vocab)
        model=Output();model.vocab=self.tokenizer.vocab_size
        token=self.tokenizer.encode(' hello')[0]
        text,ids=generate(model,None,self.tokenizer,[1],[token],3)
        self.assertEqual(text,'hello hello hello');self.assertEqual(ids,[token]*3)

    def test_language_controls_do_not_change_body_or_start_when_asleep(self):
        app=PlayroomApplication()
        runtime=self.runtime(app);app.language=runtime
        with patch.object(runtime,'ensure_worker'):
            before=app.world.body.record()
            command={'request_id':'start-language','action':'start'}
            app('POST','/v1/language/control',command);app('POST','/v1/language/control',command)
            self.assertEqual(before,app.world.body.record())
            runtime.info.update(status='ready')
            app.world.body.sleep_state='asleep';runtime.on_tick();self.assertTrue(runtime.commands.empty())
            app.world.body.sleep_state='awake';app.world.paused=True;runtime.on_tick();self.assertTrue(runtime.commands.empty())
            app.world.paused=False;runtime.on_tick();self.assertFalse(runtime.commands.empty())
            runtime.control('stop');self.assertFalse(runtime.enabled)
        app.close()

    def test_checkpoint_rejects_wrong_parent_and_tokenizer(self):
        from baby_arcus.language_checkpoint import save,load
        import importlib.metadata
        body=BodyPolicy();adapter=LanguageAdapter(body.cfg.dim,self.tokenizer.vocab_size,4)
        optimizer=torch.optim.AdamW(adapter.parameters())
        meta={'parent_sha256':'abc','encoding':'o200k_base','vocab_size':self.tokenizer.vocab_size,
              'text_dim':4,'tiktoken_version':importlib.metadata.version('tiktoken')}
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'checkpoint.pt';save(path,adapter,optimizer,meta)
            with self.assertRaisesRegex(ValueError,'parent mismatch'):load(path,body.core,self.tokenizer,'wrong')
            restored,_=load(path,body.core,self.tokenizer,'abc')
            torch.testing.assert_close(restored.embedding.weight,adapter.embedding.weight)

    def test_validation_regression_rolls_back_weights_and_optimizer(self):
        from baby_arcus.services.language_worker import LanguageWorker
        worker=object.__new__(LanguageWorker)
        worker.body=BodyPolicy().eval().requires_grad_(False)
        worker.adapter=LanguageAdapter(worker.body.cfg.dim,32,8)
        worker.optimizer=torch.optim.AdamW(worker.adapter.parameters(),lr=.01)
        worker.device='cpu';worker.cfg={'session_training_tokens':10,'context_tokens':8}
        worker.state={'training_tokens':0,'rejected_updates':0}
        worker.reference_loss=1;worker.initial_loss=1
        original={k:v.clone() for k,v in worker.adapter.state_dict().items()}
        with patch.object(worker,'evaluate',return_value=100):
            self.assertEqual(worker.learn([1,2,3,4]),(0,0.0))
        self.assertEqual(worker.state['training_tokens'],0)
        self.assertEqual(worker.state['rejected_updates'],1)
        self.assertEqual(len(worker.optimizer.state),0)
        self.assertTrue(all(torch.equal(value,original[key]) for key,value in worker.adapter.state_dict().items()))

    def test_start_is_idempotent_and_does_not_stall_ready_worker(self):
        app=PlayroomApplication();runtime=self.runtime(app)
        runtime.enabled=True;runtime.info.update(status='ready')
        with patch.object(runtime,'ensure_worker'):
            self.assertEqual(runtime.control('start')['status'],'ready')
        runtime.close();app.close()

    def test_saved_receipts_visible_before_worker_starts(self):
        app=PlayroomApplication();runtime=self.runtime(app)
        runtime.log_root.mkdir()
        (runtime.log_root/'active.json').write_text(json.dumps({'training_tokens':20,'decisions':3,
            'summary':{'training_tokens':20,'decisions':3,'message_receipts':{'hello':{'trained_tokens':20}},
                       'expressions':[{'text':'hello 🐉 “Arcus” 你好'}]}},ensure_ascii=False),encoding='utf-8')
        restored=LanguageRuntime(app,__file__,runtime.project,config='language.json')
        self.assertFalse(restored.enabled);self.assertEqual(restored.snapshot()['status'],'stopped')
        self.assertEqual(restored.snapshot()['message_receipts']['hello']['trained_tokens'],20)
        self.assertEqual(restored.snapshot()['expressions'][0]['text'],'hello 🐉 “Arcus” 你好')
        restored.close();runtime.close();app.close()


if __name__=='__main__':unittest.main()
