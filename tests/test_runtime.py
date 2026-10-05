import unittest


class RuntimeTests(unittest.TestCase):
    def test_depth_gates(self):
        import torch
        from arcus3.depth import FullDepthGate, ReducedDepthGate

        hidden = torch.tensor([[[2.0, 0.0], [-2.0, 0.0]]])
        full = FullDepthGate(2, "cpu")
        self.assertTrue(torch.equal(full(hidden), hidden))
        self.assertEqual(full.last_observation["executed_slots"], 2)

        reduced = ReducedDepthGate(2, "cpu", threshold=0.5)
        with torch.no_grad():
            reduced.weight[0, 0] = 10
        self.assertEqual(reduced.decisions(hidden).tolist(), [[True, False]])

    def test_selective_experts_routes_each_token_once(self):
        import torch
        from torch import nn
        from arcus3.routing import SelectiveExperts

        class Expert(nn.Module):
            def __init__(self, scale):
                super().__init__()
                self.gate_proj = nn.Linear(2, 2, bias=False)
                with torch.no_grad():
                    self.gate_proj.weight.copy_(torch.eye(2) * scale)

            def forward(self, value):
                return self.gate_proj(value)

        block = SelectiveExperts(Expert(1.0))
        with torch.no_grad():
            block.experts[1].gate_proj.weight.copy_(torch.eye(2) * 2)
            block.router.weight.copy_(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
        hidden = torch.tensor([[[2.0, 1.0], [-2.0, 1.0]]])
        output = block(hidden)
        self.assertTrue(torch.equal(output[0, 0], hidden[0, 0]))
        self.assertTrue(torch.equal(output[0, 1], hidden[0, 1] * 2))
        self.assertEqual(block.last_counts, [1, 1])
        self.assertEqual(
            set(block.state_dict()),
            {
                "experts.0.gate_proj.weight",
                "experts.1.gate_proj.weight",
                "router.weight",
            },
        )

    def test_config_rejects_architecture_drift(self):
        try:
            from arcus3.hf_model import Alpha3Config
        except ModuleNotFoundError as error:
            if error.name == "transformers":
                self.skipTest("transformers is not installed")
            raise
        with self.assertRaises(ValueError):
            Alpha3Config(selected_layers=[0])
        with self.assertRaises(ValueError):
            Alpha3Config(depth_enabled=True, depth_capacity=0.5)
        config = Alpha3Config(arcus_release={"model": "Alpha 3.2.2"})
        self.assertEqual(config.arcus_release["model"], "Alpha 3.2.2")

    def test_tiny_transformers_model_constructs_and_runs(self):
        try:
            import torch
            from arcus3.hf_model import Alpha3Config, Alpha3ForCausalLM
        except ModuleNotFoundError as error:
            if error.name == "transformers":
                self.skipTest("transformers is not installed")
            raise
        config = Alpha3Config(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=24,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
            depth_enabled=True,
        )
        model = Alpha3ForCausalLM(config).eval()
        with torch.no_grad():
            result = model(torch.tensor([[1, 2, 3]]), use_cache=False)
        self.assertEqual(tuple(result.logits.shape), (1, 3, 32))
        self.assertEqual(
            sum(hasattr(layer.mlp, "router") for layer in model.model.layers), 6
        )


if __name__ == "__main__":
    unittest.main()
