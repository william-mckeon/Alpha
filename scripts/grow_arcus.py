"""
scripts/grow_arcus.py

Grow a trained Arcus checkpoint into a bigger one by ADDING experts (specs/0010, Stage 1).
Warm-copies existing experts with a dormant router init (near-lossless@grow) and writes a new
checkpoint dir that `train_arcus.py --init_from <out>` continues training.

  # from the LATEST checkpoint on HF (preferred — never grows a stale local copy):
  python scripts/grow_arcus.py --repo Islanderintel/arcus-alpha-v0.5-0.5b --out runs/arcus_grown_1b/checkpoint --to_experts 8

  # or from a local checkpoint dir (e.g. the live training output on the same pod):
  python scripts/grow_arcus.py --ckpt runs/arcus_0.5b_fluency/checkpoint --out runs/arcus_grown_1b/checkpoint --to_experts 8

  (single line so it pastes cleanly in both PowerShell and bash — bash '\' line-continuations break in PowerShell)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from arcus.generate import load_model
from arcus.grow import grow_experts
from arcus.hf_upload import save_checkpoint
from arcus.hf_utils import resolve_checkpoint, repo_last_modified
from arcus.model import ArcusMoDE


def main() -> None:
    ap = argparse.ArgumentParser(description="Grow an Arcus checkpoint by adding experts (Stage 1)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--repo", help="HF repo id to grow from — downloads the LATEST checkpoint (not local)")
    src.add_argument("--ckpt", help="local checkpoint dir (config.json + model.safetensors)")
    ap.add_argument("--out", required=True, help="output dir for the grown checkpoint")
    ap.add_argument("--token", default=None, help="HF token for --repo (else uses the HF_TOKEN env var)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--to_experts", type=int, help="grow up to this many experts total")
    g.add_argument("--add", type=int, help="add this many experts")
    ap.add_argument("--source", default="roundrobin", help="'roundrobin' or an expert index to copy")
    ap.add_argument("--dormant_margin", type=float, default=8.0,
                    help="router-logit gap of each new expert below its source (high = nearer lossless, "
                         "low = faster differentiation)")
    args = ap.parse_args()

    if args.repo:      # grow from the LATEST HF checkpoint (gated + freshness-printed), never stale-local
        print(f"[repo] {args.repo} | latest checkpoint pushed {repo_last_modified(args.repo, args.token, required=False)}")
    ckpt = resolve_checkpoint(repo=args.repo, ckpt=args.ckpt, token=args.token)
    model, encoding, cfg = load_model(ckpt, device="cpu", dtype=torch.float32)
    add = args.add if args.add is not None else args.to_experts - cfg.n_experts
    if add < 1:
        raise SystemExit(f"nothing to grow: current n_experts={cfg.n_experts}, requested add={add}")
    source = int(args.source) if args.source.lstrip("-").isdigit() else args.source

    sd2, cfg2 = grow_experts(model.state_dict(), cfg, add=add, source=source,
                             dormant_margin=args.dormant_margin)
    grown = ArcusMoDE(cfg2)
    grown.load_state_dict(sd2, strict=False)
    if cfg2.tie_embeddings:
        grown.head.weight = grown.token_embed.weight
    save_checkpoint(grown, cfg2, encoding, args.out)

    print(f"[grow] {cfg.n_experts} -> {cfg2.n_experts} experts | "
          f"{model.num_parameters() / 1e6:.1f}M -> {grown.num_parameters() / 1e6:.1f}M params "
          f"| margin {args.dormant_margin} | wrote {args.out}")
    print(f"       continue: python scripts/train_arcus.py --init_from {args.out} --stream ...")


if __name__ == "__main__":
    main()
