"""
scripts/check_fluency.py

Pull the LATEST checkpoint of an Arcus model from HuggingFace and sample it — the "can it
talk yet?" check (Stage 0, specs/0008) you can run from your desk anytime.

Each run fetches the repo's current weights (snapshot_download only re-downloads files that
changed since last time), reloads the model, and generates from a few prompts so you can read
the fluency yourself. Run it again in a few days and you get a further-trained checkpoint —
watch the repetition loops break and the sentences hold together as it climbs its token budget.

  # set your token, then check the 0.5B:
  #   PowerShell:  $env:HF_TOKEN = "<your token>"
  python scripts/check_fluency.py

  # a different repo / your own prompt / greedy decoding:
  python scripts/check_fluency.py --repo Islanderintel/arcus-alpha-v0.1-1b --prompt "def quicksort(a):" --temperature 0
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
# Communication-fluency prompts (Stage 0 tests COHERENT LANGUAGE, not code or knowledge).
DEFAULT_PROMPTS = [
    "The key idea behind gradient descent is",
    "In my opinion, the best way to learn a new skill is",
    "Let me explain how a computer stores information.",
    "The most interesting thing about the ocean is that",
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Download the latest Arcus checkpoint from HF and sample it")
    ap.add_argument("--repo", default=DEFAULT_REPO, help="HF repo id (default: the 0.5B)")
    ap.add_argument("--prompt", action="append", default=None,
                    help="prompt (repeatable); default = a built-in STEM/code set")
    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8, help="<=0 is greedy/deterministic")
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--token", default=None, help="HF token (else uses the HF_TOKEN env var)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")

    # recency — when was the latest checkpoint pushed? (so you know how fresh this is)
    updated = repo_last_modified(args.repo, token=token, required=False)
    print(f"[repo] {args.repo}" + (f" | last updated {updated}" if updated else ""))

    print("[download] fetching the latest checkpoint (only changed files re-download)...")
    local_dir = resolve_checkpoint(repo=args.repo, token=token)   # gates access — no stale cache on a 401

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model, encoding, cfg = load_model(local_dir, device=args.device, dtype=dtype)
    tok = get_tokenizer(encoding)
    print(f"[model] {model.num_parameters() / 1e6:.1f}M params | {cfg.n_experts} experts | "
          f"dim {cfg.dim} | tokenizer {encoding} | device {args.device}\n")

    prompts = args.prompt or DEFAULT_PROMPTS
    for i, p in enumerate(prompts, 1):
        text = generate(model, tok, p, max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                        device=args.device, seed=args.seed)
        print(f"--- prompt {i}/{len(prompts)} " + "-" * 50)
        print(text)
        print()


if __name__ == "__main__":
    main()
