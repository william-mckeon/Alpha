"""
scripts/phase0_recon.py

Phase 0 recon — pin the Arcus wrapper to the REAL Qwen3-MoE API.

The Phase 2 wrapper subclasses HuggingFace's Qwen3 MoE decoder layer and inserts a
MoD router in front of the MoE block. The exact attribute names and forward
signatures shift between `transformers` versions, so we read them from the
INSTALLED package instead of trusting memory. This script prints everything the
wrapper has to bind to, and proves a small random-init Qwen3-MoE instantiates and
forwards on the dev box (no 30B weights, no network).

Run:
    python scripts/phase0_recon.py

It is read-only: it builds a tiny random-init model in memory and inspects classes.
Nothing is downloaded or written.
"""

from __future__ import annotations

import inspect
import sys


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    section("environment")
    try:
        import torch
        import transformers
    except Exception as exc:  # pragma: no cover - environment probe
        print(f"FAILED to import torch/transformers: {exc}")
        print("Install: pip install torch --index-url "
              "https://download.pytorch.org/whl/cu128 ; pip install -r requirements.txt")
        return 1

    print(f"python       {sys.version.split()[0]}")
    print(f"torch        {torch.__version__}")
    print(f"transformers {transformers.__version__}")
    print(f"cuda         available={torch.cuda.is_available()} "
          f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    # ---- Qwen3-MoE classes from the installed transformers ----
    section("qwen3-moe classes (installed)")
    try:
        from transformers.models.qwen3_moe import modeling_qwen3_moe as M
    except Exception as exc:
        print(f"FAILED to import Qwen3-MoE modeling: {exc}")
        print("Your transformers is too old for Qwen3-MoE. Need >= 4.51.")
        return 1

    print(f"modeling file: {inspect.getfile(M)}")
    decoder_cls = M.Qwen3MoeDecoderLayer
    moe_cls = M.Qwen3MoeSparseMoeBlock
    print(f"decoder layer: {decoder_cls.__name__}")
    print(f"  forward{inspect.signature(decoder_cls.forward)}")
    print(f"moe block    : {moe_cls.__name__}")
    print(f"  forward{inspect.signature(moe_cls.forward)}")

    # ---- small random-init config (the Phase 0 / 5080 substrate) ----
    section("small random-init Qwen3-MoE (dev config)")
    cfg = M.Qwen3MoeConfig(
        vocab_size=2048,
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        num_experts=8,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        max_position_embeddings=512,
    )
    for field in (
        "num_hidden_layers", "hidden_size", "num_attention_heads",
        "num_key_value_heads", "num_experts", "num_experts_per_tok",
        "moe_intermediate_size", "norm_topk_prob", "decoder_sparse_step",
        "mlp_only_layers",
    ):
        print(f"  {field:22s} = {getattr(cfg, field, '<absent>')}")

    model = M.Qwen3MoeForCausalLM(cfg).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  instantiated: {n_params/1e6:.2f}M params")

    # which decoder layers actually carry an MoE block (vs a dense MLP)?
    layer0 = model.model.layers[0]
    mlp_type = type(layer0.mlp).__name__
    gate_attr = "gate" if hasattr(getattr(layer0, "mlp", None), "gate") else "<none>"
    print(f"  layer[0].mlp = {mlp_type}  (expert router attr: '{gate_attr}')")

    # ---- MoE gate (expert router) internals — bound by MoDGatedMoE._record_balance ----
    section("moe gate (expert router) internals")
    gate = layer0.mlp.gate
    print(f"gate type    : {type(gate).__name__}")
    print(f"gate forward : {inspect.signature(type(gate).forward)}")
    print(f"gate params  : {[n for n, _ in gate.named_parameters()]}")
    print(f"gate children: {[n for n, _ in gate.named_children()] or '(none)'}")

    def _desc(o):
        if isinstance(o, tuple):
            return "tuple of " + ", ".join(
                f"{type(t).__name__}{tuple(t.shape)}" if hasattr(t, "shape") else type(t).__name__
                for t in o)
        return f"{type(o).__name__}{tuple(o.shape)}" if hasattr(o, "shape") else type(o).__name__

    for shape in ((4, cfg.hidden_size), (1, 4, cfg.hidden_size)):
        try:
            print(f"gate(randn{shape}) -> {_desc(gate(torch.randn(*shape)))}")
            break
        except Exception as exc:
            print(f"gate(randn{shape}) raised {type(exc).__name__}: {exc}")

    # ---- tiny forward (proves the substrate runs) ----
    section("forward smoke")
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        out = model(x)
    print(f"  input  {tuple(x.shape)} -> logits {tuple(out.logits.shape)}")
    assert out.logits.shape == (2, 16, cfg.vocab_size), "unexpected logits shape"

    section("recon OK")
    print("Bind the Phase 2 wrapper to the signatures above. Confirm:")
    print("  - decoder layer forward arg order (hidden_states, position_embeddings, ...)")
    print("  - MoE block returns (hidden_states, router_logits) or just hidden_states")
    print("  - the expert-router attribute name on the MoE block ('gate' above)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
