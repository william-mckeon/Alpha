"""
arcus/generate.py

Autoregressive sampling — how you hear the model talk.

`val_ppl` tells you the model is learning; it does NOT tell you whether the text is
coherent. Fluency (specs/0008-fluency-pretraining.md) is a qualitative gate: you sample
the model and read it. This module is that check.

`generate` is a plain decode loop — no KV cache; it re-feeds the growing sequence each
step (O(T^2), fine for the short samples a fluency check needs). `load_model` rebuilds an
ArcusMoDE from the checkpoint that arcus/hf_upload.py:save_checkpoint wrote (config.json +
model.safetensors, bf16, tied head dropped) and RE-TIES the head so the artifact you
uploaded is exactly the one you sample.
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Optional

import torch
import torch.nn.functional as F

from arcus.model import ArcusMoDE
from arcus.model_config import ModelConfig

__all__ = ["generate", "load_model"]


def load_model(ckpt_dir: str, device: str = "cpu", dtype: torch.dtype = torch.float32):
    """Rebuild an ArcusMoDE + its tokenizer name from a saved checkpoint dir.

    Reverses arcus/hf_upload.py:save_checkpoint — reads config.json (which carries every
    ModelConfig field plus serving metadata we ignore), builds the model, loads
    model.safetensors (or the pytorch_model.bin fallback), and RE-TIES head.weight to the
    embedding: the tie is dropped from the bf16 file, and `.to()` can otherwise un-share it.
    Returns (model, encoding_name, cfg).
    """
    with open(os.path.join(ckpt_dir, "config.json"), encoding="utf-8") as f:
        raw = json.load(f)
    field_names = {f.name for f in dataclasses.fields(ModelConfig)}
    cfg = ModelConfig(**{k: v for k, v in raw.items() if k in field_names})
    model = ArcusMoDE(cfg)

    st_path = os.path.join(ckpt_dir, "model.safetensors")
    bin_path = os.path.join(ckpt_dir, "pytorch_model.bin")
    if os.path.exists(st_path):
        from safetensors.torch import load_file
        sd = load_file(st_path)
    elif os.path.exists(bin_path):
        sd = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"no model.safetensors or pytorch_model.bin in {ckpt_dir}")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    # head.weight is tied to token_embed.weight and dropped from the bf16 file -> expected
    # missing. Anything else missing/unexpected is a real mismatch worth surfacing.
    leftover = [k for k in missing if k != "head.weight"]
    if leftover or unexpected:
        raise RuntimeError(f"checkpoint mismatch in {ckpt_dir}: "
                           f"missing={leftover} unexpected={list(unexpected)}")

    model.to(device=device, dtype=dtype)
    if cfg.tie_embeddings:
        model.head.weight = model.token_embed.weight     # re-tie (survives .to())
    model.eval()
    return model, raw.get("tokenizer", "o200k_base"), cfg


def _filter_logits(logits: torch.Tensor, top_k: Optional[int], top_p: Optional[float]) -> torch.Tensor:
    """Mask a 1-D logits vector [V] to the top-k and/or top-p (nucleus) set."""
    if top_k:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        cut = cumprobs > top_p
        cut[1:] = cut[:-1].clone()     # shift so the first token crossing p is kept
        cut[0] = False
        logits = logits.index_fill(0, sorted_idx[cut], float("-inf"))
    return logits


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 128,
             temperature: float = 0.8, top_k: Optional[int] = None,
             top_p: Optional[float] = None, device: str = "cpu",
             seed: Optional[int] = None, stop_at_eot: bool = True) -> str:
    """Sample a continuation of `prompt`. `temperature <= 0` is greedy (deterministic).
    Returns the FULL decoded text (prompt + continuation)."""
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    max_ctx = model.cfg.max_seq_len
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    if ids.size(1) == 0:
        ids = torch.tensor([[tokenizer.eot_token]], dtype=torch.long, device=device)
    eot = tokenizer.eot_token

    for _ in range(max_new_tokens):
        cond = ids[:, -max_ctx:]                          # never exceed the RoPE cache
        logits = model(cond)[0, -1, :].float()            # [V]
        if temperature <= 0.0:
            next_id = int(logits.argmax(-1))
        else:
            logits = _filter_logits(logits / temperature, top_k, top_p)
            next_id = int(torch.multinomial(F.softmax(logits, dim=-1), num_samples=1))
        ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
        if stop_at_eot and next_id == eot:
            break

    return tokenizer.decode(ids[0].tolist())
