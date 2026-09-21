import unittest
from types import SimpleNamespace
import torch
from arcus.moe import BatchedExperts
from baby_arcus.shared_pathways import NeuronProbe, NeuronUpdate, top_masks, shared_masks, matched_random


class PathwaysTests(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(182)
        experts = BatchedExperts(2, 3, 5)
        model = SimpleNamespace(core=SimpleNamespace(blocks=[SimpleNamespace(moe=SimpleNamespace(experts=experts))]))
        return model, experts, torch.randn(1, 2, 4, 3)

    def test_noop_gradient_padding_and_exception_cleanup(self):
        model, experts, inputs = self.fixture()
        original = experts(inputs)
        with NeuronProbe(model, collect=True) as probe:
            actual = experts(inputs)
            self.assertTrue(torch.equal(original, actual))
            actual.square().sum().backward()
            self.assertGreater(float(probe.salience[0].sum()), 0)
        with NeuronProbe(model, collect=True) as padded:
            experts(torch.zeros_like(inputs)).sum().backward()
        self.assertEqual(float(padded.activation[0].sum()), 0)
        self.assertEqual(float(padded.salience[0].sum()), 0)
        with self.assertRaises(RuntimeError):
            with NeuronProbe(model):
                raise RuntimeError('simulated failure')
        self.assertEqual(len(experts._forward_hooks), 0)
        self.assertTrue(torch.equal(original, experts(inputs)))

    def test_known_shared_channel_contributes_to_two_outputs(self):
        model, experts, inputs = self.fixture()
        with torch.no_grad():
            experts.down_proj.zero_()
            experts.down_proj[:, 2, :2] = 1
        mask = torch.zeros(2, 5, dtype=torch.bool)
        mask[:, 2] = True
        original = experts(inputs)
        with NeuronProbe(model, {0: mask}):
            changed = experts(inputs)
        self.assertGreater(float(original[..., :2].detach().abs().sum()), 0)
        self.assertEqual(float(changed.detach().abs().sum()), 0)
        self.assertTrue(torch.equal(original, experts(inputs)))

    def test_selection_excludes_zero_and_controls_match(self):
        scores = {0: torch.tensor([[5., 4., 0., 0.], [0., 0., 0., 0.]])}
        a = top_masks(scores, .5)
        b = {0: torch.tensor([[True, False, True, False], [False]*4])}
        shared = shared_masks([a, b])
        self.assertEqual(shared[0].sum().item(), 1)
        controls = matched_random(shared, 2)
        self.assertTrue(torch.equal(shared[0].sum(1), controls[0].sum(1)))
        self.assertTrue(torch.equal(controls[0], matched_random(shared, 2)[0]))
        self.assertFalse(a[0][1].any())

    def test_update_only_selected_rows_and_exact_restore_on_error(self):
        model, experts, inputs = self.fixture()
        experts(inputs).square().sum().backward()
        original = experts.down_proj.detach().clone()
        mask = torch.zeros(2, 5, dtype=torch.bool)
        mask[0, 1] = True
        with self.assertRaises(RuntimeError):
            with NeuronUpdate(model, {0: mask}, .01):
                self.assertTrue(torch.equal(original[~mask], experts.down_proj[~mask]))
                self.assertAlmostEqual(float((original-experts.down_proj.detach()).norm()), .01, places=6)
                raise RuntimeError('interruption')
        self.assertTrue(torch.equal(original, experts.down_proj))

    def test_invalid_masks_and_update_norm(self):
        model, _, _ = self.fixture()
        with self.assertRaises(ValueError):
            NeuronProbe(model, {0: torch.zeros(2, 5)})
        with self.assertRaises(ValueError):
            NeuronUpdate(model, {}, 0)

    def test_restart_comparison_rejects_behavior_or_transfer_drift(self):
        from copy import deepcopy
        from scripts.verify_arcus_pathways_report import compare
        report = {'candidate': {'sha256': 'example'}, 'config': {}, 'cohorts': {},
                  'raw_confirmation': {'baseline': {'rest': {'losses': [1.], 'correct': [1]}}},
                  'transfer': {'commands:shared': {'rest': {'loss_delta': .02}}}}
        self.assertEqual(compare(report, deepcopy(report), {}, {}), 0)
        changed = deepcopy(report)
        changed['raw_confirmation']['baseline']['rest']['correct'] = [0]
        with self.assertRaises(ValueError):
            compare(report, changed, {}, {})
        changed = deepcopy(report)
        changed['transfer']['commands:shared']['rest']['loss_delta'] = -.02
        with self.assertRaises(ValueError):
            compare(report, changed, {}, {})

    def test_frozen_upstream_still_collects_loss_salience(self):
        model, experts, inputs = self.fixture()
        experts.requires_grad_(False)
        with NeuronProbe(model, collect=True) as probe:
            experts(inputs).square().mean().backward()
        self.assertGreater(float(probe.salience[0].sum()), 0)
        self.assertTrue(all(p.grad is None for p in experts.parameters()))


if __name__ == '__main__':
    unittest.main()
