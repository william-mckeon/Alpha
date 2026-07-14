"""
scripts/eval_ppl.py

Perplexity of a checkpoint on the DIVERSE, HELD-OUT val set (all domains) — the *trustworthy*
quality number, unlike the training-time `val_ppl` that was confounded by a single-domain val
slice (see docs/RESULTS.md). Use it to compare models honestly: the 0.5B seed vs the from-scratch
1B, or a grown-1B vs the from-scratch 1B (the growth calibration).

  # a local checkpoint:
  python scripts/eval_ppl.py --ckpt runs/arcus_grown_1b/checkpoint
  # or pull one straight from HF:
  python scripts/eval_ppl.py --repo Islanderintel/arcus-alpha-v0.5-0.5b
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from arcus.generate import load_model
from arcus.tokenizer import get_tokenizer
from arcus.data import build_val_set
from arcus.eval import perplexity
from arcus.hf_utils import resolve_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser(description="Perplexity on the diverse held-out val set (all domains)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ckpt", help="local checkpoint dir (config.json + model.safetensors)")
    g.add_argument("--repo", help="HF repo id to download the checkpoint from")
    ap.add_argument("--shards", default="alpha dataset", help="corpus root (for the held-out val set)")
    ap.add_argument("--val_docs", type=int, default=64, help="held-out val docs per shard")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--token", default=None, help="HF token (else uses the HF_TOKEN env var)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.repo:
        print(f"[repo] {args.repo}")
    ckpt = resolve_checkpoint(repo=args.repo, ckpt=args.ckpt, token=args.token)

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model, encoding, cfg = load_model(ckpt, device=args.device, dtype=dtype)
    tok = get_tokenizer(encoding)
    print(f"[model] {model.num_parameters() / 1e6:.1f}M params | {cfg.n_experts} experts | "
          f"dim {cfg.dim} | tokenizer {encoding} | device {args.device}")

    print(f"[val] building held-out set: {args.val_docs} docs/shard from '{args.shards}' ...")
    val = build_val_set(args.shards, tok, seq_len=args.seq_len,
                        batch_size=args.batch_size, val_docs_per_shard=args.val_docs)
    if not val:
        raise SystemExit(f"empty val set — check --shards '{args.shards}'")
    n_tok = sum(int(labels.numel()) for _, labels in val)
    ppl = perplexity(model, val, args.device)
    print(f"\n=== val perplexity = {ppl:.3f}   ({len(val)} batches, {n_tok:,} tokens, all domains) ===")


if __name__ == "__main__":
    main()
