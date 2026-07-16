"""
scripts/memcheck.py — show WHERE the GPU memory goes for a preset at a given batch size.

The point this answers: MoDE saves COMPUTE (MoD drops ~half the tokens, MoE routes each
kept token to 1 expert) — but not parameter/optimizer STORAGE. Every expert's weights +
Adam states stay resident regardless of routing. This prints the breakdown with MoD
ACTIVE (see the kept-token fraction), so fit-vs-spill is settled empirically, not by my
arithmetic.

  python scripts/memcheck.py --preset 0.9b --batch_size 2
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcus.tokenizer import get_tokenizer
from arcus.model_config import get_config
from arcus.model import ArcusMoDE
from arcus.optim import build_param_groups, build_optimizer


def gb(n_bytes: float) -> float:
    return n_bytes / 1e9


def main() -> int:
    ap = argparse.ArgumentParser(description="GPU memory breakdown for an Arcus preset")
    ap.add_argument("--preset", default="0.9b")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--vram_gb", type=float, default=16.0, help="your card's VRAM for the verdict")
    a = ap.parse_args()

    if not torch.cuda.is_available():
        print("[memcheck] needs CUDA")
        return 1
    dev = "cuda"

    tok = get_tokenizer("o200k_base")
    cfg = get_config(a.preset, vocab_size=tok.vocab_size)
    model = ArcusMoDE(cfg).to(dev)
    model.gradient_checkpointing = True
    model.train()

    torch.cuda.synchronize()
    w0 = torch.cuda.memory_allocated()                      # weights only, before optimizer

    opt = build_optimizer(build_param_groups(model, 0.01, 30.0, 3e-4), 3e-4, (0.9, 0.95), 1e-8)
    B, T = a.batch_size, a.seq_len
    for _ in range(a.steps):
        x = torch.randint(0, cfg.vocab_size, (B, T), device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, cfg.vocab_size), x.view(-1)) + model.last_aux_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    static = torch.cuda.memory_allocated()                 # weights + AdamW states (grads zeroed)
    peak = torch.cuda.max_memory_allocated()               # + transient grads/activations at peak
    kept = float(model.last_compute_fraction)

    print(f"\n[{a.preset}] params={model.num_parameters()/1e6:.1f}M  experts={cfg.n_experts}  "
          f"MoD capacity={cfg.capacity}  batch_size={B} seq_len={T}")
    print(f"  MoD kept-token fraction      : {kept:.2f}   "
          f"(~{(1-kept)*100:.0f}% of tokens SKIP the experts — MoDE is active)")
    print("  ---- where the VRAM goes ----")
    print(f"  expert+model weights (resident): {gb(w0):6.2f} GB   all {cfg.n_experts} experts, always loaded")
    print(f"  + AdamW optimizer states       : {gb(static - w0):6.2f} GB   m+v, fp32, for every param")
    print(f"  + grads/activations (transient): {gb(peak - static):6.2f} GB   MoDE shrinks THIS slice")
    over = gb(peak) - a.vram_gb
    verdict = "FITS in VRAM" if gb(peak) <= a.vram_gb else \
        f"OVER {a.vram_gb:.0f} GB by {over:.1f} GB -> spills to shared system RAM (slow)"
    print(f"  = PEAK                         : {gb(peak):6.2f} GB / {a.vram_gb:.0f}   ->  {verdict}")
    # projection: 8-bit AdamW (specs/0007) quantizes the m+v states ~4x (L40S; fp32 params kept)
    adamw_gb = gb(static - w0)
    proj_peak = gb(peak) - adamw_gb * 0.75
    print(f"  ~ with --optimizer adamw8bit   : AdamW {adamw_gb:5.2f} -> ~{adamw_gb / 4:4.2f} GB, "
          f"projected peak ~{proj_peak:5.2f} GB (L40S)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
