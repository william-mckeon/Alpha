"""
scripts/train_arcus.py

Train an Arcus MoDE model from scratch on the alpha dataset.

  python scripts/train_arcus.py --preset tiny --max_tokens 2000000 --epochs 1

The matched dense baseline (boenet discipline: always compare to dense) comes from
the SAME code with `--dense`, which sets n_experts=1 + capacity=1.0 → a plain dense
model of matched size. Run both and compare val perplexity at equal settings.

On the 5080 this validates the from-scratch pipeline at small scale; the real Alpha
0.1 (~1.3B) trains on cloud with a streaming loader.
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch

from arcus.tokenizer import get_tokenizer
from arcus.model_config import get_config, PRESETS
from arcus.model import ArcusMoDE
from arcus.config import TrainConfig
from arcus.data import packed_batches
from arcus.train import train_end_to_end


def main() -> int:
    ap = argparse.ArgumentParser(description="Train Arcus MoDE from scratch on the alpha dataset")
    ap.add_argument("--preset", default="tiny", choices=list(PRESETS))
    ap.add_argument("--shards", default="./alpha dataset", help="root of the *.jsonl.zst shards")
    ap.add_argument("--encoding", default="cl100k_base", help="cl100k_base (default) or o200k_base")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--max_seq_len", type=int, default=None,
                    help="model context window / RoPE cache size; independent of --seq_len "
                         "(the training length). e.g. 131072 to declare GPT-OSS-120B's 128k context")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_tokens", type=int, default=5_000_000)
    ap.add_argument("--capacity", type=float, default=None,
                    help="MoD keep fraction; default from the preset (0.5)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--router_lr_mult", type=float, default=30.0, help="boenet's router-LR fix")
    ap.add_argument("--warmup_epochs", type=float, default=1.0, help="boenet used 1 epoch")
    ap.add_argument("--n_experts", type=int, default=None,
                    help="number of experts; default from the preset (0.9b=8, 0.5b=4)")
    ap.add_argument("--expert_hidden", type=int, default=None,
                    help="per-expert SwiGLU width; lower it when raising --n_experts to stay "
                         "param-matched (e.g. 8 experts x 1280 ~= 4 experts x 2560)")
    ap.add_argument("--lb_loss_weight", type=float, default=None,
                    help="Switch load-balance weight (preset default 0.01); raise if experts collapse")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dense", action="store_true",
                    help="matched dense baseline: n_experts=1, capacity=1.0 (MoE+MoD off)")
    ap.add_argument("--run_name", default=None, help="runs/<run_name>/ for epoch log + checkpoint")
    ap.add_argument("--grad_accum", type=int, default=1, help="micro-batches per optimizer step")
    ap.add_argument("--hf_repo", default=None, help="HF repo id to upload each epoch (backup/serving)")
    ap.add_argument("--save_every_steps", type=int, default=0,
                    help="also checkpoint+upload every N optimizer steps (0 = end-of-epoch only); "
                         "crash protection for long single-epoch runs")
    ap.add_argument("--no_amp", action="store_true", help="disable bf16 AMP")
    ap.add_argument("--no_grad_ckpt", action="store_true", help="disable gradient checkpointing")
    args = ap.parse_args()

    # Seed everything BEFORE model init + data, so runs reproduce and the MoDE-vs-dense
    # pair is seed-matched (removes the ~1.5% run-to-run init noise).
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device} | seed {args.seed}")

    tok = get_tokenizer(args.encoding)
    # Context window (RoPE cache) is decoupled from the training sequence length: --max_seq_len
    # sets the model's declared context (e.g. 131072 to match GPT-OSS-120B's 128k), which only
    # sizes a cheap RoPE buffer + the served config.json — we still TRAIN at the much smaller
    # --seq_len. If unset, keep the preset's window (but never below --seq_len).
    if args.max_seq_len:
        context_len = args.max_seq_len
    else:
        context_len = max(PRESETS[args.preset].get("max_seq_len", 4096), args.seq_len)
    if context_len < args.seq_len:
        raise SystemExit(f"--max_seq_len {context_len} cannot be below --seq_len {args.seq_len}")
    # Only override preset fields the user explicitly set (None = "keep the preset's value"),
    # so e.g. --preset 0.9b keeps its 8 experts instead of being clobbered by a CLI default.
    overrides = dict(max_seq_len=context_len)
    if args.capacity is not None:
        overrides["capacity"] = args.capacity
    if args.n_experts is not None:
        overrides["n_experts"] = args.n_experts
    if args.expert_hidden is not None:
        overrides["expert_hidden"] = args.expert_hidden
    if args.lb_loss_weight is not None:
        overrides["lb_loss_weight"] = args.lb_loss_weight
    if args.dense:
        overrides.update(n_experts=1, capacity=1.0)
    cfg = get_config(args.preset, vocab_size=tok.vocab_size, **overrides)
    model = ArcusMoDE(cfg)
    kind = "dense baseline" if args.dense else "MoDE"
    print(f"[model] {args.preset} {kind} | params={model.num_parameters()/1e6:.1f}M | "
          f"vocab={cfg.vocab_size} dim={cfg.dim} layers={cfg.n_layers} experts={cfg.n_experts} "
          f"cap={cfg.capacity} ctx={cfg.max_seq_len} (train seq_len={args.seq_len})")

    print(f"[data] tokenizing up to {args.max_tokens:,} tokens from {args.shards} ...")
    train_b, val_b = packed_batches(args.shards, tok, seq_len=args.seq_len,
                                    batch_size=args.batch_size, max_tokens=args.max_tokens)
    print(f"[data] train batches={len(train_b)} val batches={len(val_b)}")
    if not train_b:
        print("[data] no batches — is the shard path right and are shards present?")
        return 1

    steps_per_epoch = max(1, len(train_b) // max(1, args.grad_accum))
    warmup_steps = max(1, int(args.warmup_epochs * steps_per_epoch))
    tcfg = TrainConfig(capacity=cfg.capacity, lr=args.lr, epochs=args.epochs,
                       seq_len=args.seq_len, batch_size=args.batch_size,
                       weight_decay=args.weight_decay, router_lr_mult=args.router_lr_mult,
                       warmup_steps=warmup_steps, seed=args.seed,
                       grad_accum=args.grad_accum, amp=not args.no_amp,
                       grad_checkpoint=not args.no_grad_ckpt,
                       save_every_steps=args.save_every_steps)

    run_name = args.run_name or f"{args.preset}_{'dense' if args.dense else 'mode'}_s{args.seed}"
    run_dir = os.path.join("runs", run_name)
    log_path = os.path.join(run_dir, "epochs.csv")
    save_dir = os.path.join(run_dir, "checkpoint")
    print(f"[run] {run_dir} | grad_accum {args.grad_accum} amp {not args.no_amp} "
          f"ckpt {not args.no_grad_ckpt}" + (f" | hf {args.hf_repo}" if args.hf_repo else ""))
    history = train_end_to_end(model, tcfg, train_b, val_b, device=device,
                               log_path=log_path, save_dir=save_dir,
                               hf_repo=args.hf_repo, encoding=args.encoding)

    last = history[-1]
    print("\n" + "=" * 60)
    print(f"{kind}: val ppl = {last['val_ppl']:.3f} | compute fraction = {last['compute_fraction']:.3f}")
    print("=" * 60)
    print("__SUMMARY__ " + json.dumps({
        "preset": args.preset, "dense": args.dense, "capacity": cfg.capacity,
        "seed": args.seed, "params_M": round(model.num_parameters() / 1e6, 2),
        "history": history,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
