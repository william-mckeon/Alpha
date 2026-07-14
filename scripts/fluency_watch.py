"""
scripts/fluency_watch.py

One standing check for a running Arcus model: pull the latest checkpoint, squash the HF repo
history (reclaim storage), sample the COMMUNICATION-fluency prompts, and append a timestamped
entry to a running log. Run it on a schedule (every few hours / daily) to get a dated timeline
of the model learning to talk — while keeping the repo storage clean.

  $env:HF_TOKEN = "<token>"
  python scripts/fluency_watch.py                 # the 0.5B; squash + sample + log
  python scripts/fluency_watch.py --no-squash     # sample + log only (leave history alone)
  python scripts/fluency_watch.py --repo Islanderintel/arcus-alpha-v0.1-1b

Stage 0 tests COMMUNICATION fluency — coherent language, not code and not knowledge.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from arcus.generate import load_model, generate
from arcus.tokenizer import get_tokenizer
from arcus.hf_utils import resolve_checkpoint, repo_last_modified

DEFAULT_REPO = "Islanderintel/arcus-alpha-v0.5-0.5b"
PROMPTS = [
    "The key idea behind gradient descent is",
    "In my opinion, the best way to learn a new skill is",
    "Let me explain how a computer stores information.",
    "The most interesting thing about the ocean is that",
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Pull + squash + communication-fluency sample + log")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--log", default="runs/fluency_log.md", help="markdown log to append to")
    ap.add_argument("--no-squash", action="store_true", help="skip squashing the HF history")
    ap.add_argument("--max_new_tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--token", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from huggingface_hub import HfApi
    token = args.token or os.environ.get("HF_TOKEN")

    # gate access + date the ENTRY by the checkpoint's push time (fatal if the repo can't be read —
    # squashing/sampling a repo you have no access to is pointless)
    updated = repo_last_modified(args.repo, token=token, required=True)
    print(f"[repo] {args.repo} | last updated {updated}")

    if not args.no_squash:
        try:
            HfApi(token=token).super_squash_history(
                repo_id=args.repo, repo_type="model",
                commit_message="fluency_watch: squash history (keep latest)")
            print("[squash] history collapsed to the latest checkpoint")
        except Exception as exc:
            print(f"[squash] skipped: {type(exc).__name__}: {exc}")

    print("[download] fetching latest checkpoint...")
    local_dir = resolve_checkpoint(repo=args.repo, token=token)

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model, encoding, cfg = load_model(local_dir, device=args.device, dtype=dtype)
    tok = get_tokenizer(encoding)
    header = f"{model.num_parameters() / 1e6:.1f}M params | {cfg.n_experts} experts | dim {cfg.dim}"
    print(f"[model] {header} | device {args.device}")

    samples = []
    for p in PROMPTS:
        text = generate(model, tok, p, max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature, top_k=args.top_k,
                        device=args.device, seed=args.seed)
        samples.append((p, text))
        print(f"\n--- {p}\n{text}")

    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    with open(args.log, "a", encoding="utf-8") as f:
        f.write(f"\n## {args.repo} — checkpoint updated {updated}\n\n")
        f.write(f"_{header} · temp {args.temperature} top-k {args.top_k}_\n\n")
        for p, text in samples:
            quoted = "\n> ".join(text.splitlines()) or text
            f.write(f"**{p}**\n\n> {quoted}\n\n")
    print(f"\n[log] appended to {args.log}")


if __name__ == "__main__":
    main()
