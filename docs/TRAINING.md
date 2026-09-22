# Arcus — Training Runbook

## Baby Arcus training boundary

For current status use [the handoff](ARCUS_CURRENT_STATUS.md). The approved fresh
Test 2 experiment has native/Linux smoke training at capacity .25, with shared
losses, acknowledged hearing and graph-controlled interactions. Use its
[dedicated runbook](ARCUS_TEST2_RUNBOOK.md); scientific acceptance remains open.
The commands below describe separate historical training paths.

The [Baby simulation trainer](../specs/0028-baby-arcus-learning.md) now has a tested
native PPO pipeline and viewer. The commands below train the original text model;
use the [Baby runbook](BABY_ARCUS_RUNBOOK.md) for separate simulation commands, artifacts,
and training snapshots. Overnight endurance is not yet qualified. Each proposed run includes all work
within 12 hours, has no fixed experiment endpoint, and requires resource preflight.
Follow [the phase plan](BABY_ARCUS_PHASES.md) before attempting a Baby run.

> How to actually train an Arcus model: the CLI, the presets, will-it-fit, checkpoints,
> and the laptop-vs-cloud reality. Design/contract live in
> [ARCUS_MODEL_DESIGN.md](ARCUS_MODEL_DESIGN.md) / [DATASHEET.md](DATASHEET.md); this is the
> how-to. Existing commands are **Track A only**. Track B remains documentation-only until
> its specifications are accepted and implemented.

---

## The driver

`scripts/train_arcus.py` trains a preset from scratch on the alpha dataset. It seeds
everything (torch/numpy/random) before model + data, so runs reproduce and the MoDE-vs-dense
pair is seed-matched.

It does not train a pretrained donor. Do not point this command at Step, Qwen, GLM, or DeepSeek
weights; Track B requires a separate adapter-aware trainer specified below.

```powershell
.\.venv\Scripts\python.exe scripts\train_arcus.py --preset 1b `
    --seq_len 512 --batch_size 2 --grad_accum 16 --epochs 1 --max_tokens 50000000 `
    --seed 0 --run_name arcus_1b --save_every_steps 1000 --hf_repo Islanderintel/arcus-alpha-1b
```

> PowerShell line-continuation: the backtick must be the **last** character on the line — no
> trailing comment, no trailing space — or paste it as one line.

### Flags that matter

| Flag | Purpose |
|---|---|
| `--preset` | `tiny` / `boenet-medium` / `0.5b` / `0.9b` / `1b` / `alpha-0.1`… (preset values win unless overridden) |
| `--max_seq_len` | context window / RoPE cache (e.g. `131072` for GPT-OSS-120B parity), **independent of** `--seq_len` |
| `--seq_len` | the training window actually backpropagated (keep small — attention is O(T²)) |
| `--batch_size` / `--grad_accum` | micro-batch × accumulation = effective batch |
| `--n_experts` / `--expert_hidden` / `--capacity` / `--lb_loss_weight` | override the preset's mechanism (else preset wins) |
| `--max_tokens` / `--epochs` | data budget and passes |
| `--save_every_steps` | checkpoint + HF upload every N optimizer steps (0 = end-of-epoch only) |
| `--hf_repo` | HuggingFace repo to back up to (e.g. `Islanderintel/arcus-alpha-1b`) |
| `--stream` | **cloud-scale**: stream batches from the shards (no in-RAM token cap); driven by `--max_tokens` as a *step budget*, not epochs |
| `--resume` | restore the full training state (model + optimizer + scheduler + step + RNG) from the checkpoint dir — **spot-safe** |
| `--optimizer` | `adamw` (fp32) or **`adamw8bit`** (8-bit states, ~7 GB less VRAM; L40S only) — [specs/0007](../specs/0007-footprint-reduction.md) |
| `--fused_ce` | fused linear+cross-entropy (no full 200k logits; Triton/L40S, pure-torch fallback) |
| `--ckpt_moment_dtype` | `fp32` or `bf16-v` (bf16 the Adam v-moment in the resume checkpoint — smaller S3 sync) |
| `--dense` | matched baseline: `n_experts=1, capacity=1.0` (MoE+MoD off) |
| `--no_amp` / `--no_grad_ckpt` | disable bf16 AMP / gradient checkpointing |

Each run writes `runs/<run_name>/epochs.csv` (per-epoch: train_loss, val_ppl,
compute_fraction, lr, seconds) and `runs/<run_name>/checkpoint/` (safetensors + config.json).

## Stage 0 — pretrain the 0.5B to fluency (local, free)

The `0.5b` is the **largest rung that trains cleanly on the 5080 (no spill)** — so "can it
talk?" is a free, local run, not a cloud job ([specs/0008](../specs/0008-fluency-pretraining.md)).
Drive it **streaming, ~1 pass** over a real token budget (many epochs of a small slice
memorizes; it doesn't generalize):

```powershell
.\.venv\Scripts\python.exe scripts\train_arcus.py --preset 0.5b --stream `
    --max_tokens 3000000000 --seq_len 512 --batch_size 4 --grad_accum 8 --fused_ce `
    --seed 0 --run_name arcus_0.5b_fluency --save_every_steps 500 `
    --hf_repo Islanderintel/arcus-alpha-0.5b
```

- **Fits 16 GB, no spill** — confirm with `memcheck.py --preset 0.5b` (unlike the 1B, the 0.5B's
  fp32 optimizer states fit); you can push `--batch_size` past the 1B's 2.
- **It's a multi-day run.** Fluency wants billions of unique tokens; on the 5080 that's days.
  Because `--save_every_steps 500 --hf_repo` uploads a sampleable checkpoint along the way, you
  can **hear it babble early** and extend `--max_tokens` (or `--resume`) rather than commit weeks
  up front. Fluency on a STEM/code corpus is fluency *in that domain* — intended (reasoning core).

**Hear it talk** — the fluency gate is qualitative; sample the checkpoint and read it:

```powershell
.\.venv\Scripts\python.exe scripts\sample_arcus.py --ckpt runs\arcus_0.5b_fluency\checkpoint `
    --prompt "The key idea behind gradient descent is" --max_new_tokens 160 --temperature 0.8 --top_k 50
```

`--temperature 0` is greedy/deterministic. `--ckpt` takes any dir with `config.json` +
`model.safetensors` (what the trainer writes each save and what `--hf_repo` uploads). Coherent,
on-domain text = fluency reached; log verbatim samples to [RESULTS.md](RESULTS.md). The full
0.5B→85B plan this seeds is [specs/0009](../specs/0009-self-improving-loop.md).

## Will it fit? — `memcheck.py`

Measure before you commit hours. It prints the VRAM breakdown with MoD active:

```powershell
.\.venv\Scripts\python.exe scripts\memcheck.py --preset 1b --batch_size 2
```

```
[1b] params=1078.0M  experts=10  MoD capacity=0.5  batch_size=2 seq_len=512
  MoD kept-token fraction      : 0.33   (~67% of tokens SKIP the experts)
  expert+model weights         :  4.38 GB   all 10 experts, always loaded
  + AdamW optimizer states     :  9.41 GB   m+v, fp32, for every param
  + grads/activations          :  7.11 GB   MoDE shrinks THIS slice
  = PEAK                       : 18.64 GB / 16  -> OVER by 2.6 GB (spills to shared RAM)
```

> Snapshot from the **cl100k** build (1078M). The current **o200k_base** tokenizer makes the `1b`
> **~1180M** (≈+0.4 GB weights / +0.8 GB AdamW), so it spills a little harder — the fit verdict is
> unchanged. Re-run `memcheck.py --preset 1b` for live figures on your card.

### The memory reality (16 GB)

- A **1B fp32** model needs ~18–22 GB. It does **not** fit 16 GB. On Windows the driver
  spills to shared system RAM (slow but runs ~10–14 hr/epoch); on Linux/CUDA it hard-OOMs.
- **MoDE is sparse compute, dense storage.** MoD skipping tokens shrinks only the
  grads/activations slice. Weights + fp32 AdamW states (the bulk) stay resident regardless of
  routing — so the ceiling is **parameter count**, not how aggressively MoD routes. Batch
  size barely changes it.
- To fit fp32 cleanly: a **≥24 GB** GPU, or a leaner optimizer (8-bit Adam — fragile on
  Blackwell/sm_120).

## Checkpoints + backup

`--hf_repo` uploads `model.safetensors` + `config.json` after every epoch and every
`--save_every_steps` optimizer steps. The repo is overwritten so it always holds the latest
weights — that is both the crash backup and what a vLLM server pulls. On a long single-epoch
run set `--save_every_steps` so the first backup isn't hours away (e.g. `1000` ≈ a few
backups across a ~3000-step run). Upload failures are logged and swallowed — they never kill
a run.

## Laptop vs cloud

The 5080 is the **validation bench**, not where real models pretrain.

- Throughput ≈ **110M tokens/day** for the 1B (spill-slowed). A model "learns to talk" only
  after billions of *unique* tokens (a 1B wants ~20–100B; see Chinchilla pairing in ROADMAP),
  which is weeks of laptop runtime — so it's a cloud job, not a laptop one.
- Cloud is **throughput, not a requirement**: the laptop *can* do it, just slowly. The
  ~120 GB corpus is not the bottleneck; time is.
- For laptop runs, prefer **more unique tokens, ~1 pass** over many epochs of a tiny slice
  (10 epochs over 50M tokens memorizes, it doesn't generalize). Multi-epoch on a fixed slice
  is the boenet *architecture-validation* experiment (watch `val_ppl`), not a fluency run.

## Footprint levers (specs/0007) — same model, less memory/disk

All flag-gated, **fp32 defaults preserve the 5080 path**; opt in on the cloud L40S:

- **`--optimizer adamw8bit`** — 8-bit Adam states: **~7 GB off training VRAM** (9.4 → ~2.4 GB),
  so the 1B fits a **24 GB card instead of 48 GB (≈ half the $/hr)**. Only the Adam moments
  quantize; fp32 params + the bf16 forward are untouched, so outputs are identical. bitsandbytes
  is L40S/Ada-supported but fragile on the 5080/Blackwell — cloud-only, fp32 fallback local.
- **`--fused_ce`** — fused linear+cross-entropy: drops the full `[B·T, 200019]` logits (~1 GB at
  the default batch, more at larger batch). Triton kernel on the L40S; pure-torch chunked
  fallback everywhere else (verified numerically identical to `F.cross_entropy`).
- **`--ckpt_moment_dtype bf16-v`** — halves the Adam v-moment in the resume checkpoint (smaller
  S3 sync each save); re-upcast to fp32 on load, so the training math is unchanged.
- **bf16 serving** is **on by default** in `save_checkpoint` — the served artifact is ~half
  (4.7 → 2.4 GB), lossless because the forward already runs bf16.

`memcheck.py` prints a projected `--optimizer adamw8bit` peak so you can see the fit before the run.

## Cloud training — RunPod (the current path)

Real runs go on **RunPod** — an L40S (48 GB) or A100 by the hour, **no quota wall** (AWS SageMaker
spot is quota-blocked for new accounts; see below). The trainer is portable: only `launch.py` is
SageMaker-specific — `train_arcus.py` + the streaming loader run on any GPU box.

> **Only *training* needs the pod.** A 1B's ~20 GB of weights + fp32 AdamW states + gradients overflow the
> 16 GB 5080 (it spills to shared RAM and crawls), so training goes cloud. *Inference / testing*
> (`eval_ppl.py`, `sample_arcus.py`) is **weights-only (~2 GB for the 1B)** and runs fine on the 5080 — do
> all testing locally, and **stop the pod when a run finishes** (it bills by the hour).

**Pod setup** (on the pod, via its JupyterLab terminal or SSH). The clone + data land on the
persistent `/workspace` volume (one-time); the **deps do NOT persist** — repeat step 2 after every
restart (see the recovery note below):

```bash
# 1. code — clone the PRIVATE repo (fine-grained PAT, read-only, this repo)
cd /workspace && git clone https://<GITHUB_PAT>@github.com/william-mckeon/Alpha-base.git
cd Alpha-base
# 2. deps — the RunPod image already ships a CUDA-matched torch; do NOT install or upgrade torch on
#    the pod (a mismatched CUDA build makes torch fall back to device=cpu). requirements-pod.txt is
#    torch-free (direct AND transitive); bitsandbytes goes in with --no-deps so pip can't touch the
#    image torch. NEVER `pip install torch` here, or `-r` a file that pins torch.
pip install -r requirements-pod.txt              # base deps, torch-free
pip install --no-deps bitsandbytes>=0.43.0       # 8-bit AdamW lever; --no-deps = pip can't touch torch
pip install -e . --no-deps                       # register the arcus package (no deps; torch stays)
# CUDA gate — MUST print "cuda OK ...". If it asserts/fails, torch is clobbered: fix it before training.
python -c "import torch; assert torch.cuda.is_available(), 'torch cannot see the GPU'; print('cuda OK', torch.__version__)"
# 3. data — pull the 47 GB corpus S3 -> volume (RunPod Cloud Sync is cleanest; or aws s3 sync)
aws s3 sync s3://arcus-training-wmckeon/alpha-dataset "alpha dataset"
```

Keep everything on the **persistent `/workspace` volume** (a network mount) so a pod stop/interruption
is recoverable via `--resume`. It's **region-locked** — replace a dead pod in the *same* region to
reattach the volume (the HF checkpoint is your cross-region fallback).

**After a pod stop/restart — rebuild the environment.** RunPod persists only the `/workspace` volume;
the Python environment lives on the **ephemeral container disk** and is **wiped on every restart**
(that's why `huggingface_hub`/`pytest` go missing and imports fail after a restart — while `torch`
comes back, because it's baked into the base image). So each time the pod returns:

```bash
cd /workspace/Alpha-base
pip install -r requirements-pod.txt              # reinstall the wiped deps — torch left untouched
pip install --no-deps bitsandbytes>=0.43.0
pip install -e . --no-deps
python -c "import torch; assert torch.cuda.is_available(), 'no GPU'; print(torch.__version__, torch.cuda.is_available())"
```

**If that assert fails** (torch can't see the GPU — usually a stray `pip install torch` clobbered the
image build), the simplest fix is to **restart the pod**: the ephemeral disk resets to the base image,
which wipes the bad install and restores the correct torch *for free* (then rebuild deps as above).
Only if you can't restart, reinstall the exact build **your image originally shipped** — check it with
`python -c "import torch; print(torch.__version__)"` *before* it gets clobbered. For the current L40S
template that is `torch 2.4.1+cu124`:
`pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124`
— a different template ships a different torch, so match *its* version, not this one. The code on
`/workspace` survives; only the environment needs rebuilding.

**Run it — inside `tmux`** so it survives a disconnect. These are **pod (bash)** commands: the trailing
`\` is bash line-continuation and will fail on a local Windows PowerShell prompt — there, put each command
on **one line** (PowerShell's continuation character is a backtick, not `\`).

```bash
export HF_TOKEN="<your token>"
tmux new -s arcus     # detach: Ctrl-B then D  |  reattach: tmux attach -t arcus

# 0.5B fluency (Stage 0) — the L40S fits a big batch
python scripts/train_arcus.py --preset 0.5b --stream --shards "alpha dataset" \
    --max_tokens 12000000000 --seq_len 512 --batch_size 16 --grad_accum 4 --fused_ce \
    --run_name arcus_0.5b_fluency --save_every_steps 500 --resume \
    --hf_repo Islanderintel/arcus-alpha-v0.05-0.5b

# 1B seed — same card, batch 8 (add --optimizer adamw8bit to push batch 12-16)
python scripts/train_arcus.py --preset 1b --stream --shards "alpha dataset" \
    --max_tokens 12000000000 --seq_len 512 --batch_size 8 --grad_accum 8 --fused_ce \
    --run_name arcus_1b_seed --save_every_steps 500 --resume \
    --hf_repo Islanderintel/arcus-alpha-v0.1-1b
```

**Cost + throughput** (measured: ~17.5k tokens/s for the 1B on an L40S):

| Run | Time | ~Cost (L40S @ ~$0.79–0.99/hr) |
|---|---|---|
| 0.5B fluency, 12B tokens | ~7 days | ~$130–165 |
| 1B seed, 12B tokens | ~8 days | ~$150–190 |

≈ **$13–16 per billion tokens** on the L40S; faster cards (A100/H100) finish sooner at ~the same
total (compute-bound). No multi-GPU yet — the trainer is single-GPU (no FSDP).

## Growth — climb a rung (Stage 1)

Grow a *trained* checkpoint into a bigger one by **adding experts** (near-lossless@grow;
[specs/0010](../specs/0010-growth-operator.md)), then continue training — how the ladder climbs
without retraining from scratch. On the same backbone, `0.5b` (4 experts) → `1b` (10 experts) is
just `+6` experts:

```bash
# grow the finished 0.5B into the 1B architecture
python scripts/grow_arcus.py --ckpt runs/arcus_0.5b_fluency/checkpoint \
    --out runs/arcus_grown_1b/checkpoint --to_experts 10

# continue training the grown model (warm-started; fresh optimizer)
python scripts/train_arcus.py --preset 1b --stream --shards "alpha dataset" \
    --init_from runs/arcus_grown_1b/checkpoint --max_tokens 12000000000 --seq_len 512 \
    --batch_size 8 --grad_accum 8 --fused_ce --run_name arcus_grown_1b --save_every_steps 500 --resume \
    --hf_repo Islanderintel/arcus-alpha-grown-1b
```

`--dormant_margin` (default 8) trades init-losslessness (high) against how fast the new experts
differentiate (low). **Calibration:** compare the grown 1B against the *from-scratch* 1B
(`arcus_1b_seed`) with **`scripts/eval_ppl.py`**, not the training-time `val_ppl` — the latter is
domain-confounded (single-domain val slice; see [docs/RESULTS.md](RESULTS.md)). `eval_ppl` scores
both checkpoints on the same diverse, held-out val set (all domains), so the gap it reports is the
operator's true quality cost, measured while a from-scratch control is still affordable
([specs/0009](../specs/0009-self-improving-loop.md) Stage 1):

```bash
python scripts/eval_ppl.py --ckpt runs/arcus_grown_1b/checkpoint   # or --repo <hf-id>
python scripts/eval_ppl.py --ckpt runs/arcus_1b_seed/checkpoint    # the from-scratch control
```

## SageMaker spot run (the AWS alternative — currently quota-walled)

> **AWS is the *for-later* path.** New accounts have **0** g5/g6e spot quota, and AWS may *deny* a
> spot increase until you have usage history (see the note below). Use **RunPod** (above) now;
> switch here once the quota clears.

The seed can also run as a **managed-spot training job** via `launch.py` — SageMaker's prebuilt
PyTorch container (Script Mode), **no Docker / ECR**. `launch.py` forwards `HF_TOKEN` from your
shell (never hardcoded), streams the corpus from S3, checkpoints the full training state to
`/opt/ml/checkpoints` (S3-synced), and `--resume` continues after a spot interruption.

**One-time AWS setup**

```powershell
pip install -e ".[launch]"                         # SageMaker SDK v2 (pinned <3) + boto3 to submit
#   (AWS CLI v2 is a separate MSI install; SDK v3 breaks launch.py's `sagemaker.pytorch.PyTorch`)
aws configure                                      # real IAM AKIA... keys (NOT a Bedrock key), us-east-1
aws s3 mb s3://arcus-training-<you> --region us-east-1
#   IAM role (console): ArcusSageMakerRole = AmazonSageMakerFullAccess + AmazonS3FullAccess
aws s3 sync "alpha dataset" s3://arcus-training-<you>/alpha-dataset/ --exclude "*" --include "*.jsonl.zst"
```

Set `BUCKET` / `ROLE` at the top of `launch.py` (or `ARCUS_S3_BUCKET` / `ARCUS_SM_ROLE`).

> **The instance Service Quota is the real gate — and it's a new-account catch-22.** New AWS
> accounts often have **0** for the scarce cards (`ml.g6e.xlarge for spot training job usage`),
> so `CreateTrainingJob` fails with `ResourceLimitExceeded`, and AWS may **deny** a spot increase
> until you have usage history ("utilize 90% of your current quota first"). Two ways through:
> **(1)** `ml.g5.xlarge` (A10G 24 GB) usually has a small non-zero spot quota already **and fits
> the 1B** — so it is the `launch.py` default; **try it first** (a different instance = a
> different quota, so g6e being 0 says nothing about g5). **(2)** If both are 0, use **on-demand**
> (`launch.py` without `--spot`) to build history, then re-request spot. Read the actual value in
> **Service Quotas → Amazon SageMaker** (us-east-1) as **root/admin** — a plain IAM user with
> SageMaker+S3 access still can't read quotas (it needs `ServiceQuotasReadOnlyAccess`).

**Smoke test (cheap — prove IAM/S3/spot/resume for ~$1)**

```powershell
$env:HF_TOKEN = "<your hf token — stays in this shell only>"
python launch.py --preset tiny --max_tokens 2000000 --instance ml.g5.xlarge --spot
```

**The real seed (1B, ~12B tokens — g5 A10G on spot ≈ $0.35/hr)**

```powershell
python launch.py --preset 1b --max_tokens 12000000000 --seq_len 512 `
    --instance ml.g5.xlarge --spot --hf_repo Islanderintel/arcus-alpha-1b
```

`ml.g5.xlarge` is the `launch.py` default (cheaper, likelier already in quota); pass
`--instance ml.g6e.xlarge` for the L40S (~2–3× faster, if you have its quota). `launch.py`
defaults to `--stream` and `--resume` on; a reclaimed spot instance restarts and continues from
the S3 checkpoint with no action from you. Watch progress in the SageMaker console → Training
jobs → **View logs**. Memory note: the **g5 has 24 GB** — the 1B fp32 fits at **batch 2**
(~18.6 GB, the `launch.py` default); go higher with `--fused_ce` / `--optimizer adamw8bit`, or on
the 48 GB L40S. `memcheck` is the 16 GB laptop story, not the cloud one.

## Track B — planned donor training (not implemented)

Track B begins only after donor selection and capacity-1 conversion pass. The initial full-weight
cycle runs on rented multi-GPU infrastructure:

1. Load the immutable official donor checkpoint and Alpha adapter.
2. Verify donor hashes and capacity-1 equivalence.
3. Freeze all donor parameters and train only Alpha depth routers.
4. Reduce capacity conservatively and rerun the pinned coding/tool subset at every rung.
5. Continue training by explicit token budgets with donor-tokenizer manifests and replay.
6. Apply native-format coding/tool SFT; begin RLVR only after supervised gates pass.

Sparse activation lowers executed compute but does not remove the need to store all donor weights.
The laptop is for tiny donor-shaped configurations, adapter tests, data preparation, and small router
experiments—not full donor training. No executable Track-B command belongs here until the adapter,
trainer, precision, parallelism, checkpoint, and recovery paths are implemented and verified.

Contracts: [0018](../specs/0018-lossless-donor-conversion.md),
[0020](../specs/0020-depth-router-training.md),
[0021](../specs/0021-donor-continued-training.md), and
[0022](../specs/0022-agentic-sft-rlvr.md).

---

*Arcus — see [specs/0005-scale-and-training.md](../specs/0005-scale-and-training.md) for Track A,
[DONOR_FOUNDATION_STRATEGY.md](DONOR_FOUNDATION_STRATEGY.md) for Track B, and
[ROADMAP.md](../ROADMAP.md) for build order.*
