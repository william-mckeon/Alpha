"""
arcus/model_config.py

Arcus MoDE model config + the Alpha-ladder scale presets (boenet Phase-4 report §7).

`tiny` is what runs on the 5080 / in unit tests (pipeline validation). `alpha-0.1`
onward are cloud targets; their sizes are APPROXIMATE — tune `expert_hidden`/`n_layers`
to land on the param target once the tokenizer's vocab is fixed (o200k's ~200k vocab
makes the tied embedding a big share at 1.3B, smaller as you scale up).
"""

from __future__ import annotations

from dataclasses import dataclass

from arcus.backbone import BackboneConfig
from arcus.moe import MoEConfig


@dataclass
class ModelConfig:
    vocab_size: int = 100_277          # cl100k_base; overridden from the tokenizer
    dim: int = 1536
    n_heads: int = 16
    n_kv_heads: int = 8
    head_dim: int = 96
    n_layers: int = 20
    expert_hidden: int = 2048          # SwiGLU inner width per expert
    max_seq_len: int = 4096
    rope_theta: float = 1_000_000.0
    norm_eps: float = 1e-6
    qk_norm: bool = True
    tie_embeddings: bool = True
    # --- MoDE knobs (boenet's validated config) ---
    n_experts: int = 4
    moe_top_k: int = 1
    capacity: float = 0.5              # MoD: fraction of tokens that reach the MoE
    capacity_factor: float = 1.0       # MoE per-expert buffer
    lb_loss_weight: float = 0.01

    def to_backbone_config(self) -> BackboneConfig:
        return BackboneConfig(
            dim=self.dim, n_heads=self.n_heads, n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim, n_layers=self.n_layers, mlp_hidden=self.expert_hidden,
            max_seq_len=self.max_seq_len, rope_theta=self.rope_theta,
            norm_eps=self.norm_eps, qk_norm=self.qk_norm,
        )

    def to_moe_config(self) -> MoEConfig:
        return MoEConfig(
            dim=self.dim, expert_hidden=self.expert_hidden, n_experts=self.n_experts,
            top_k=self.moe_top_k, capacity_factor=self.capacity_factor,
            lb_loss_weight=self.lb_loss_weight,
        )


# Scale ladder. Sizes approximate; `tiny` is exact-as-written (tests / 5080 check).
PRESETS = {
    "tiny": dict(dim=256, n_heads=8, n_kv_heads=2, head_dim=32, n_layers=4,
                 expert_hidden=512, max_seq_len=512, n_experts=4),
    # Mirrors boenet's "medium" architecture (the config that tied dense in MoDE):
    # dim 256, 8 heads (MHA: n_kv_heads=n_heads), 4 layers, mlp_ratio 4 -> hidden 1024,
    # 4 experts. Modern substrate (RoPE/RMSNorm/SwiGLU/QK-norm) + o200k is ours.
    "boenet-medium": dict(dim=256, n_heads=8, n_kv_heads=8, head_dim=32, n_layers=4,
                          expert_hidden=1024, max_seq_len=512, n_experts=4),
    "0.5b": dict(dim=1024, n_heads=16, n_kv_heads=4, head_dim=64, n_layers=12,
                 expert_hidden=2560, max_seq_len=2048, n_experts=4),               # ~0.5B (cl100k)
    # The base run: same backbone as 0.5b but 8 FULL (2560-wide) experts — the grow-params
    # capacity step. 128k context declared (GPT-OSS-120B parity); we still train short.
    "0.9b": dict(dim=1024, n_heads=16, n_kv_heads=4, head_dim=64, n_layers=12,
                 expert_hidden=2560, max_seq_len=131072, n_experts=8),             # ~889M (cl100k)
    # Crossing 1B by GROWING experts (the capacity play) — 10 full 2560-wide experts on the
    # same backbone. On 16GB this trains with a spill to shared RAM: the ceiling is the fp32
    # optimizer states, NOT routing. Measure first: scripts/memcheck.py --preset 1b.
    "1b": dict(dim=1024, n_heads=16, n_kv_heads=4, head_dim=64, n_layers=12,
               expert_hidden=2560, max_seq_len=131072, n_experts=10),             # ~1.08B (cl100k)
    "alpha-0.1": dict(dim=1536, n_heads=16, n_kv_heads=8, head_dim=96, n_layers=20,
                      expert_hidden=2048, max_seq_len=4096, n_experts=4),          # ~1.3B
    "alpha-0.5": dict(dim=3072, n_heads=24, n_kv_heads=8, head_dim=128, n_layers=28,
                      expert_hidden=4096, max_seq_len=4096, n_experts=8),          # ~7-13B
    "alpha-1.0": dict(dim=6144, n_heads=48, n_kv_heads=8, head_dim=128, n_layers=48,
                      expert_hidden=8192, max_seq_len=4096, n_experts=16),         # ~70-86B
}


def get_config(preset: str, vocab_size: int, **overrides) -> ModelConfig:
    """Build a ModelConfig from a preset, with the tokenizer's vocab and any overrides."""
    if preset not in PRESETS:
        raise ValueError(f"unknown preset '{preset}'; choices: {list(PRESETS)}")
    fields = dict(PRESETS[preset], vocab_size=vocab_size)
    fields.update(overrides)
    return ModelConfig(**fields)
