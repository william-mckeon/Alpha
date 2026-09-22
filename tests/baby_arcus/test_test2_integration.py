"""Fresh training and graph failure contracts, using reduced test shapes."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from baby_arcus.shared_factory import create
from baby_arcus.shared_objectives import step
from baby_arcus.shared_checkpoint import save, load, restore_optimizer
from baby_arcus.shared_curriculum import example
from baby_arcus.services.playroom import PlayroomApplication
from baby_arcus.tool_registry import ToolRegistry
from baby_arcus.interaction_graph import InteractionGraph
from baby_arcus.language_stream import AcknowledgedLanguageStream


class Tokenizer:
    vocab_size = 512
    eot_token = 1
    def encode(self, text):
        return [2 + b for b in text.encode()]
    def decode(self, ids):
        return ''.join(chr(max(32, i-2)) for i in ids)


class TestFresh(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def cfg(self):
        return {'depth_capacity': .25, 'seed': 2101, 'preset': 'tiny', 'text_dim': 16}

    def test_random_factory_no_parents_and_repeatable(self):
        with patch('baby_arcus.body_policy.load', side_effect=AssertionError('loaded parent')):
            a = create(self.cfg(), 512)
            b = create(self.cfg(), 512)
        self.assertTrue(a.integrated_motor)
        self.assertTrue(all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items()))
        self.assertTrue(all(block.capacity == .25 for block in a.core.blocks))
        self.assertEqual(len({id(p) for p in a.parameters()}), len(list(a.parameters())))

    def test_joint_learning_and_checkpoint_resume(self):
        tokenizer = Tokenizer()
        model = create(self.cfg(), 512)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        row, target, _ = example(0, family='color_reference')
        target['tokens'] = [1, 80, 81, 82]
        before = model.core.token_embed.weight.detach().clone()
        result = step(model, optimizer, tokenizer, row, target)
        self.assertTrue(torch.isfinite(torch.tensor(result['loss'])))
        self.assertFalse(torch.equal(before, model.core.token_embed.weight))
        self.assertGreater(result['gradient_norms']['core'], 0)
        self.assertGreater(result['gradient_norms']['language'], 0)
        self.assertGreater(result['gradient_norms']['rgb'], 0)
        self.assertEqual(result['trained_tokens'], 3)
        with tempfile.TemporaryDirectory() as tmp:
            manifest = save(tmp, model, optimizer, {'updates': 1, 'receipts': []})
            resumed, data = load(tmp, manifest)
            self.assertTrue(resumed.integrated_motor)
            restored = restore_optimizer(resumed, data, .001)
            step(model, optimizer, tokenizer, row, target)
            torch.set_rng_state(data['rng'])
            step(resumed, restored, tokenizer, row, target)
            self.assertTrue(all(torch.equal(v, resumed.state_dict()[k]) for k, v in model.state_dict().items()))

    def test_full_depth_same_initial_weights_and_guarded_resume(self):
        from baby_arcus.shared_depth import verify_depth
        shallow = create(self.cfg(), 512)
        full = create({**self.cfg(), 'depth_capacity': 1.0}, 512)
        self.assertTrue(all(torch.equal(v, full.state_dict()[k]) for k, v in shallow.state_dict().items()))
        self.assertEqual(verify_depth(full), 1.0)
        row, target, _ = example(0, family='color_reference')
        optimizer = torch.optim.AdamW(full.parameters(), lr=.001)
        step(full, optimizer, Tokenizer(), row, target)
        self.assertEqual(float(full.core.last_compute_fraction), 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            manifest = save(tmp, full, optimizer, {'updates': 1, 'receipts': []})
            resumed, data = load(tmp, manifest)
            self.assertEqual(verify_depth(resumed), 1.0)
            restored = restore_optimizer(resumed, data, .001)
            step(full, optimizer, Tokenizer(), row, target)
            torch.set_rng_state(data['rng'])
            step(resumed, restored, Tokenizer(), row, target)
            self.assertTrue(all(torch.equal(v, resumed.state_dict()[k]) for k, v in full.state_dict().items()))
            resumed.core.blocks[0].capacity = .25
            with self.assertRaises(ValueError):
                resumed([row], Tokenizer())
        with self.assertRaises(ValueError):
            verify_depth(shallow, {'depth_capacity': 1.0})

    def test_heldout_refused(self):
        model = create(self.cfg(), 512)
        row, target, _ = example(0, 'confirmation')
        with self.assertRaisesRegex(ValueError, 'Held-out'):
            step(model, torch.optim.AdamW(model.parameters()), Tokenizer(), row, target)

    def test_rgb_and_hearing_change_integrated_motor_output(self):
        model = create(self.cfg(), 512).eval()
        row, _, _ = example(0, family='color_reference')
        changed = copy.deepcopy(row)
        changed['hearing'] = [{'text': 'a different cue'}]
        with torch.no_grad():
            before = model([row], Tokenizer(), requested=('body',))['body']
            after = model([changed], Tokenizer(), requested=('body',))['body']
        finite = torch.isfinite(before) & torch.isfinite(after)
        self.assertFalse(torch.equal(before[finite], after[finite]))
        changed = copy.deepcopy(row)
        import hashlib
        changed['vision'].update(available=False, image_base64='', sha256=hashlib.sha256(b'').hexdigest())
        with torch.no_grad():
            without_rgb = model([changed], Tokenizer(), requested=('body',))['body']
        finite = torch.isfinite(before) & torch.isfinite(without_rgb)
        self.assertFalse(torch.equal(before[finite], without_rgb[finite]))

    def test_storage_limit_fails_without_deleting(self):
        from baby_arcus.shared_storage_budget import check
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'evidence'
            path.write_bytes(b'preserve')
            with self.assertRaises(RuntimeError):
                check(tmp, 8, reserve=1)
            self.assertEqual(path.read_bytes(), b'preserve')

    def test_pre_restart_unexecuted_decision_expires(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = PlayroomApplication()
            tools = ToolRegistry(app, Path(tmp) / 'world.sqlite')
            with self.assertRaisesRegex(ValueError, 'expired'):
                tools.execute('stale', {'kind': 'eyelids', 'openness': 1}, session='old-session')
            self.assertEqual(app.world.tick, 0)
            tools.close()
            app.close()

    def test_action_receipt_survives_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = PlayroomApplication()
            path = Path(tmp) / 'world.sqlite'
            tools = ToolRegistry(app, path)
            action = {'kind': 'eyelids', 'openness': .5}
            first = tools.execute('one', action)
            tick = app.world.tick
            tools.close()
            tools = ToolRegistry(app, path)
            self.assertEqual(first, tools.execute('one', action))
            self.assertEqual(tick, app.world.tick)
            with self.assertRaises(ValueError):
                tools.execute('one', {'kind': 'eyelids', 'openness': 1})
            tools.close()
            app.close()

    def test_graph_crash_after_execution_does_not_repeat_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = PlayroomApplication()
            tools = ToolRegistry(app, Path(tmp) / 'world.sqlite')
            fail = [True]
            def execute(identity, decision, row):
                result = tools.execute(identity, decision['action'])
                if fail[0]:
                    fail[0] = False
                    raise RuntimeError('crash after durable action')
                return result
            graph = InteractionGraph(Path(tmp) / 'graph.sqlite', lambda: {},
                                     lambda r: {'action': {'kind': 'eyelids', 'openness': 1}}, execute, lambda s: None)
            with self.assertRaises(RuntimeError):
                graph.invoke('one')
            tick = app.world.tick
            graph.close()
            graph = InteractionGraph(Path(tmp) / 'graph.sqlite', lambda: {},
                                     lambda r: self.fail('decision should be cached'), execute, lambda s: None)
            graph.invoke('one')
            self.assertEqual(tick, app.world.tick)
            graph.close()
            tools.close()
            app.close()

    def test_acknowledged_hearing_never_skips_offer(self):
        import zstandard
        from baby_arcus.language_stream import inventory
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = '\n'.join(json.dumps({'text': text}) for text in ['holdout', 'hello world hello', 'second'])
            (root / 'sample.jsonl.zst').write_bytes(zstandard.ZstdCompressor().compress(data.encode()))
            manifest = inventory(root, ['*.zst'])
            stream = AcknowledgedLanguageStream(manifest, Tokenizer(), root / 'cursor.json')
            first = stream.offer(5)
            self.assertEqual(first['document'], 1)
            self.assertFalse((root / 'cursor.json').exists())
            stream = AcknowledgedLanguageStream(manifest, Tokenizer(), root / 'cursor.json')
            self.assertEqual(first, stream.offer(5))
            with self.assertRaises(ValueError):
                stream.ack(first['id'], {'durable': False})
            stream.ack(first['id'], {'source_id': first['id'], 'durable': True})
            self.assertNotEqual(first['id'], stream.offer(5)['id'])
            stream.control('restart', request_id='restart-1')
            restarted = stream.offer(5)
            stream.control('restart', request_id='restart-1')
            self.assertEqual(restarted, stream.offer(5))
            self.assertEqual(first['tokens'], restarted['tokens'])
            self.assertNotEqual(first['id'], restarted['id'])
            stream.ack(restarted['id'], {'source_id': restarted['id'], 'durable': True})
            stream.control('pause')
            self.assertIsNone(stream.offer(5))
            stream.control('resume')
            stream.control('replay')
            self.assertEqual(restarted, stream.offer(5))
            stream.ack(restarted['id'], {'source_id': restarted['id'], 'durable': True})
            self.assertNotEqual(restarted['id'], stream.offer(5)['id'])


if __name__ == '__main__':
    unittest.main()
