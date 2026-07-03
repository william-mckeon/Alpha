"""
scripts/sample_arcus.py

Hear the model talk — load a trained checkpoint and print a sampled continuation.
This is the Stage 0 fluency check (specs/0008-fluency-pretraining.md): `val_ppl` says the
model is learning; this says whether the text is coherent.

  python scripts/sample_arcus.py --ckpt runs/arcus_0.5b_fluency/checkpoint \
      --prompt "The key idea behind gradient descent is" \
      --max_new_tokens 160 --temperature 0.8 --top_k 50

Point --ckpt at any dir holding config.json + model.safetensors (what the trainer writes
each save, and what --hf_repo uploads). --temperature 0 is greedy/deterministic.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from arcus.generate import generate, load_model
from arcus.tokenizer import get_tokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample text from a trained Arcus checkpoint")
    ap.add_argument("--ckpt", required=True, help="checkpoint dir (config.json + model.safetensors)")
    ap.add_argument("--prompt", default="", help="prompt text (empty = start from EOT)")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.8, help="<=0 is greedy/deterministic")
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n_samples", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model, encoding, cfg = load_model(args.ckpt, device=args.device, dtype=dtype)
    tok = get_tokenizer(encoding)
    print(f"[loaded] {args.ckpt} | dim {cfg.dim} | {cfg.n_experts} experts | "
          f"{model.num_parameters() / 1e6:.1f}M params | tokenizer {encoding} | device {args.device}")

    for i in range(args.n_samples):
        text = generate(model, tok, args.prompt, max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                        device=args.device, seed=args.seed)
        if args.n_samples > 1:
            print(f"\n--- sample {i + 1}/{args.n_samples} " + "-" * 40)
        print(text)


if __name__ == "__main__":
    main()
