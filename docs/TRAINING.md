# Arcus — Training Runbook

> How to actually train an Arcus model: the CLI, the presets, will-it-fit, checkpoints,
> and the laptop-vs-cloud reality. Design/contract live in
> [ARCUS_MODEL_DESIGN.md](ARCUS_MODEL_DESIGN.md) / [DATASHEET.md](DATASHEET.md); this is the
> how-to.

---

## The driver

`scripts/train_arcus.py` trains a preset from scratch on the alpha dataset. It seeds
everything (torch/numpy/random) before model + data, so runs reproduce and the MoDE-vs-dense
pair is seed-matched.

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
| `--dense` | matched baseline: `n_experts=1, capacity=1.0` (MoE+MoD off) |
| `--no_amp` / `--no_grad_ckpt` | disable bf16 AMP / gradient checkpointing |

Each run writes `runs/<run_name>/epochs.csv` (per-epoch: train_loss, val_ppl,
compute_fraction, lr, seconds) and `runs/<run_name>/checkpoint/` (safetensors + config.json).

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

---

*Arcus — see [specs/0005-scale-and-training.md](../specs/0005-scale-and-training.md) for the
build and [ROADMAP.md](../ROADMAP.md) for the scaling plan.*
