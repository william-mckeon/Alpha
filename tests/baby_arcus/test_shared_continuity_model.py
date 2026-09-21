import tempfile
import unittest
from pathlib import Path
import torch
from tests.baby_arcus.test_shared_learning import fixture, Tokenizer
from baby_arcus.shared_continuity_model import ContinuityModel, load_candidate
from baby_arcus.shared_checkpoint import save, restore_optimizer
from baby_arcus.shared_depth import set_depth, verify_depth
from baby_arcus.shared_learning import configure_training


class ContinuityModelTests(unittest.TestCase):
    def test_shared_core_gradients_and_exact_optimizer_recovery(self):
        configure_training()
        old, row = fixture()
        set_depth(old)
        model = ContinuityModel(old.body, old.language, 11)
        from arcus.model import ArcusMoDE
        self.assertEqual(sum(isinstance(m, ArcusMoDE) for m in model.modules()), 1)
        self.assertEqual(verify_depth(model), .25)
        row['objects'] = []
        a, b = torch.rand(1, 11), torch.rand(1, 11)
        gaze = torch.rand(1, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        def step(current, opt):
            opt.zero_grad(set_to_none=True)
            hidden = current([row], Tokenizer(), requested=('hidden',))['hidden']
            loss = current.association_logits(hidden, a, b).square().mean()+current.search_logits(hidden, a, gaze).square().mean()
            loss = loss+current.uncertainty_logits(hidden, torch.ones(1, 7)).square().mean()
            loss.backward()
            opt.step()
            return float(loss.detach())
        step(model, optimizer)
        self.assertGreater(float(model.core.token_embed.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.object_association[-1].weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.object_search[-1].weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.identity_uncertainty[-1].weight.grad.abs().sum()), 0)
        with tempfile.TemporaryDirectory() as root:
            manifest = save(root, model, optimizer, {'updates': 1})
            from baby_arcus.shared_checkpoint import load
            normal, _ = load(root, manifest)
            self.assertEqual(normal.version, 11)
            self.assertTrue(torch.equal(normal.identity_uncertainty[-1].weight, model.identity_uncertainty[-1].weight))
            restored, data = load_candidate(root, manifest)
            resumed = restore_optimizer(restored, data, .001)
            state = torch.get_rng_state()
            original_loss = step(model, optimizer)
            torch.set_rng_state(state)
            resumed_loss = step(restored, resumed)
            self.assertEqual(original_loss, resumed_loss)
            self.assertEqual([k for k, v in model.state_dict().items() if not torch.equal(v, restored.state_dict()[k])], [])
            invalid = dict(manifest, depth_capacity=1)
            with self.assertRaises(ValueError):
                load_candidate(root, invalid)

    def test_normal_learner_updates_all_identity_heads_and_rejects_missing_inputs(self):
        from baby_arcus.shared_learning import update
        old, row = fixture()
        set_depth(old)
        model = ContinuityModel(old.body, old.language, 11)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        with self.assertRaises(ValueError):
            update(model, optimizer, [row], Tokenizer(), [{'identity_risk': [1.0]}])
        row['identity_pair'] = [[.1]*11, [.2]*11]
        row['search_query'] = [.1]*13
        row['identity_context'] = [.2]*7
        loss = update(model, optimizer, [row], Tokenizer(),
                      [{'identity_match': [1.0], 'visual_search': [0.0], 'identity_risk': [1.0]}])
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        for head in (model.object_association, model.object_search, model.identity_uncertainty):
            self.assertGreater(float(head[-1].weight.grad.abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
