"""
launch.py

Submit the Arcus training run to AWS SageMaker as a (managed-spot) training job.

This is "make it spot": the estimator's `use_spot_instances` + `max_wait` + `checkpoint_s3_uri`,
with `HF_TOKEN` forwarded from YOUR shell (never hardcoded, never committed). The job runs
`scripts/train_arcus.py` in SageMaker's prebuilt PyTorch container (Script Mode) — no Docker,
no ECR. On a spot interruption SageMaker restarts and `train_arcus.py --resume` continues from
the S3-synced checkpoint dir.

  # token stays in your shell only:
  $env:HF_TOKEN = "<your hf token>"
  python launch.py --preset 1b --max_tokens 12000000000 --seq_len 512 \
      --instance ml.g5.xlarge --spot --hf_repo Islanderintel/arcus-alpha-1b

Edit BUCKET / ROLE below (or set ARCUS_S3_BUCKET / ARCUS_SM_ROLE) before the first run.
"""

import argparse
import os
import shutil

# ---- your AWS setup (edit, or set via env) ----
BUCKET = os.environ.get("ARCUS_S3_BUCKET", "arcus-training-wmckeon")
ROLE = os.environ.get("ARCUS_SM_ROLE", "arn:aws:iam::862473913080:role/ArcusSageMakerRole")
REGION = os.environ.get("AWS_REGION", "us-east-1")

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_S3 = f"s3://{BUCKET}/alpha-dataset"
CKPT_S3 = f"s3://{BUCKET}/checkpoints"


def stage_source():
    """Build a clean SageMaker source dir: the `arcus` package + the entry script + a
    TORCH-FREE `requirements.txt` (the container already ships torch/numpy — pinning torch
    here would fight its CUDA build). SageMaker auto-installs `source_dir/requirements.txt`,
    so we copy `requirements-sm.txt` into place under that name. Returns the staging dir."""
    src = os.path.join(ROOT, ".sm_src")
    if os.path.isdir(src):
        shutil.rmtree(src)
    os.makedirs(src)
    shutil.copytree(os.path.join(ROOT, "arcus"), os.path.join(src, "arcus"))
    shutil.copy(os.path.join(ROOT, "scripts", "train_arcus.py"), os.path.join(src, "train_arcus.py"))
    shutil.copy(os.path.join(ROOT, "requirements-sm.txt"), os.path.join(src, "requirements.txt"))
    return src


def main():
    ap = argparse.ArgumentParser(description="Launch Arcus training on SageMaker (spot-capable)")
    ap.add_argument("--preset", default="1b")
    ap.add_argument("--max_tokens", type=int, default=12_000_000_000)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--max_seq_len", type=int, default=131072)
    ap.add_argument("--batch_size", type=int, default=2)    # g5 A10G 24 GB fits the 1B fp32 at batch 2 (~18.6 GB); g6e L40S 48 GB fits ~8-10
    ap.add_argument("--grad_accum", type=int, default=16)   # eff. batch 2x16x512 ~= 16k tokens/step
    ap.add_argument("--instance", default="ml.g5.xlarge")   # A10G 24 GB: cheaper, faster-approved, likely already in your quota; override with ml.g6e.xlarge (L40S 48 GB, faster, scarce)
    ap.add_argument("--spot", action="store_true", help="managed spot training (~50-70%% off)")
    ap.add_argument("--no_resume", action="store_true", help="start fresh (default resumes)")
    ap.add_argument("--hf_repo", default=None)
    ap.add_argument("--max_days", type=float, default=12.0, help="max_run; max_wait = 1.2x for spot")
    # footprint levers (specs/0007) — L40S-supported; default off so a bare launch is fp32-safe
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "adamw8bit"],
                    help="adamw8bit = bitsandbytes 8-bit optimizer states (~7 GB less VRAM; L40S)")
    ap.add_argument("--fused_ce", action="store_true", help="fused linear+cross-entropy (L40S)")
    ap.add_argument("--ckpt_moment_dtype", default="fp32", choices=["fp32", "bf16-v"],
                    help="bf16-v halves the Adam v-moment in the S3 resume checkpoint")
    args = ap.parse_args()

    if "<ACCOUNT_ID>" in ROLE:
        raise SystemExit("Set ROLE/BUCKET at the top of launch.py (or ARCUS_SM_ROLE / ARCUS_S3_BUCKET).")
    # HF_TOKEN is only needed when a run actually uploads (--hf_repo); a probe skips it.
    if args.hf_repo and not os.environ.get("HF_TOKEN"):
        raise SystemExit("--hf_repo needs HF_TOKEN in your shell (forwarded to the job, never stored).")

    from sagemaker.pytorch import PyTorch

    # store_true flags reach the script as `--key value`, so pass them as "true" (train_arcus
    # parses them with nargs="?").
    hp = {
        "preset": args.preset, "max_tokens": args.max_tokens, "seq_len": args.seq_len,
        "max_seq_len": args.max_seq_len, "batch_size": args.batch_size,
        "grad_accum": args.grad_accum, "run_name": f"arcus_{args.preset}_sm",
        "stream": "true",
        "resume": "false" if args.no_resume else "true",
        "optimizer": args.optimizer,
        "fused_ce": "true" if args.fused_ce else "false",
        "ckpt_moment_dtype": args.ckpt_moment_dtype,
    }
    if args.hf_repo:
        hp["hf_repo"] = args.hf_repo

    max_run = int(args.max_days * 24 * 3600)
    est = PyTorch(
        entry_point="train_arcus.py",
        source_dir=stage_source(),
        role=ROLE,
        instance_type=args.instance,
        instance_count=1,
        framework_version="2.4",
        py_version="py311",
        hyperparameters=hp,
        environment=({"HF_TOKEN": os.environ["HF_TOKEN"]} if os.environ.get("HF_TOKEN") else {}),  # forwarded, not stored
        use_spot_instances=bool(args.spot),
        max_run=max_run,
        max_wait=int(max_run * 1.2) if args.spot else None,
        checkpoint_s3_uri=CKPT_S3 if args.spot else None,
        checkpoint_local_path="/opt/ml/checkpoints",
        # FastFile = stream shards from S3 on demand (the streaming loader reads only what it
        # touches), instead of downloading the whole ~20 GB to the instance before training.
        input_mode="FastFile",
        base_job_name=f"arcus-{args.preset}",
    )
    print(f"launching {args.instance} {'(spot)' if args.spot else '(on-demand)'} | preset {args.preset} "
          f"| {args.max_tokens:,} tokens | data {DATA_S3}")
    est.fit({"train": DATA_S3})


if __name__ == "__main__":
    main()
