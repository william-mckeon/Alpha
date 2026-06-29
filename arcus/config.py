"""
arcus/config.py

Training-side config. The MODEL architecture config lives in `arcus/model_config.py`
(ModelConfig + the Alpha presets); this file holds the training hyperparameters,
mirroring boenet's.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    """End-to-end training hyperparameters (boenet's). The whole model trains — no
    freeze. `router_lr_mult` gives the routers (MoD + MoE) the dedicated higher LR that
    boenet found necessary (scale-suppressed router gradient). `capacity_penalty_weight=0`
    leaves the MoD capacity purely structural (fixed-K)."""
    capacity: float = 0.5
    router_lr_mult: float = 30.0
    lr: float = 3e-4
    weight_decay: float = 0.01
    adamw_betas: tuple = (0.9, 0.95)
    adamw_eps: float = 1e-8
    warmup_steps: int = 200
    epochs: int = 1
    seq_len: int = 512          # 5080-friendly default; the 200k-vocab logits scale with batch*seq
    batch_size: int = 2
    grad_clip: float = 1.0
    capacity_penalty_weight: float = 0.0
    seed: int = 0
    grad_accum: int = 1          # micro-batches per optimizer step (traditional large effective batch)
    amp: bool = True             # bf16 autocast (fp32 master weights kept)
    grad_checkpoint: bool = True # trade compute for activation memory
    save_every_steps: int = 0    # mid-epoch checkpoint+upload cadence (0 = end-of-epoch only)

    def __post_init__(self) -> None:
        if not (0.0 < self.capacity <= 1.0):
            raise ValueError(f"capacity must be in (0, 1], got {self.capacity}")
