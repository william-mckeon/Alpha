"""
scripts/run_dense_mod.py

The dense MoD experiment — the first real quality finding (Phase 3.5).

Pipeline:
  1. load a pretrained DENSE Qwen3 (default Qwen3-0.6B) + its tokenizer;
  2. measure base perplexity on a real corpus  <- the matched control;
  3. wrap with MoD at the target capacity;
  4. train the WHOLE model end-to-end (no freeze) with boenet's hyperparameters;
  5. measure perplexity again and report MoD-vs-base + the compute fraction.

This validates whether a pretrained model survives depth-routing at real scale. It
is the D, not full MoDE (a dense model has no experts). Honest by construction: the
base number is measured the same way as the MoD number.

Run (on the 5080, in the venv):
    python scripts/run_dense_mod.py --model Qwen/Qwen3-0.6B --capacity 0.5 --epochs 1
"""

from __future__ import annotations

import argparse
import json

import torch

from arcus.config import MoDConfig, TrainConfig
from arcus.data import text_dataloaders
from arcus.eval import perplexity
from arcus.qwen_dense import wrap_qwen3_dense
from arcus.train import train_end_to_end


def main() -> int:
    ap = argparse.ArgumentParser(description="Dense MoD experiment (Phase 3.5)")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--capacity", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--router_lr_mult", type=float, default=50.0)
    ap.add_argument("--dataset", default="wikitext")
    ap.add_argument("--subset", default="wikitext-2-raw-v1")
    ap.add_argument("--max_docs", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[load] {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto").to(device)
    model.gradient_checkpointing_enable()

    train_b, val_b = text_dataloaders(
        tok, seq_len=args.seq_len, batch_size=args.batch_size,
        dataset=args.dataset, subset=args.subset, max_docs=args.max_docs)
    print(f"[data] train batches={len(train_b)} val batches={len(val_b)}")

    # ---- matched control: base perplexity BEFORE any change ----
    base_ppl = perplexity(model, val_b, device)
    print(f"[base] dense val perplexity = {base_ppl:.3f}")

    # ---- wrap + train end-to-end (no freeze) ----
    wrap_qwen3_dense(model, MoDConfig(capacity=args.capacity))
    tcfg = TrainConfig(capacity=args.capacity, lr=args.lr,
                       router_lr_mult=args.router_lr_mult,
                       epochs=args.epochs, seq_len=args.seq_len,
                       batch_size=args.batch_size)
    history = train_end_to_end(model, tcfg, train_b, val_b, device=device)

    mod_ppl = history[-1]["val_ppl"]
    frac = history[-1]["compute_fraction"]
    print("\n" + "=" * 60)
    print(f"base (dense)      val ppl : {base_ppl:.3f}")
    print(f"MoD @cap={args.capacity:.2f}  val ppl : {mod_ppl:.3f}")
    print(f"compute fraction          : {frac:.3f}")
    print(f"quality delta             : {mod_ppl - base_ppl:+.3f} ppl")
    print("=" * 60)
    print("__SUMMARY__ " + json.dumps({
        "model": args.model, "capacity": args.capacity,
        "base_ppl": base_ppl, "mod_ppl": mod_ppl,
        "compute_fraction": frac, "history": history,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
