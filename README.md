# arcus

> A from-scratch **MoDE** foundation model — Mixture-of-Depths + Mixture-of-Experts
> on a modern transformer backbone, with a tiktoken tokenizer. Built **on top of**
> BoeNet's validated mechanism — for **efficiency**: foundation-model quality at a
> fraction of the compute, on hardware you actually have.

**Maintainer:** William McKeon · **Status:** v0.1 — model built & validated (tiny preset); cloud runs ahead · Apache 2.0 License © 2026 William McKeon

---

## What this is

Arcus is an original, **from-scratch** MoDE language model:

- **MoD** (Mixture-of-Depths) skips the expensive FFN for easy tokens — "only turn on
  the compute you need."
- **MoE** (Mixture-of-Experts) routes the kept tokens to specialized experts.

The mechanism comes from the **BoeNet** research project (validated at toy scale: MoDE
*matches* dense quality at ~half the per-token compute). Arcus modernizes the substrate
(**RoPE · RMSNorm · GQA+QK-norm · SwiGLU**), tokenizes with tiktoken **`cl100k_base`**,
and chases boenet's *efficiency* thesis — foundation-model quality at a fraction of the
dense compute, on accessible hardware. It trains from scratch on the
**alpha dataset** (a ~120 GB STEM/code corpus).

The earlier Qwen upcycle/wrapper exploration is archived under [`legacy/`](legacy/).

---

## Architecture

```
input_ids → token embed → [ dense GQA/RoPE attention + MoD-gated MoE FFN ] × N → RMSNorm → tied head → logits
```

MoD selects ~`capacity` of tokens per block; the MoE runs on the kept tokens only;
skipped tokens take the residual. Lossless at `capacity = 1.0` (MoD becomes a no-op →
pure MoE). Full design: [docs/ARCUS_MODEL_DESIGN.md](docs/ARCUS_MODEL_DESIGN.md).

---

## Repo layout

```
arcus/
  tokenizer.py     tiktoken cl100k_base (default; o200k optional)
  backbone.py      RoPE / RMSNorm / GQA+QK-norm / SwiGLU (dense attention)
  moe.py           the E — 4 experts, top-1, grow-params, Switch lb-loss
  mod_core.py      the D — fixed-K causal selection + straight-through gate
  model.py         the assembled MoDE model + scale dispatch
  model_config.py  Alpha presets: tiny / alpha-0.1 / 0.5 / 1.0
  train.py · optim.py · eval.py · data.py · config.py
scripts/train_arcus.py   from-scratch training driver (+ --dense baseline)
docs/ · specs/ · tests/ · legacy/ (archived Qwen path)
```

---

## Quickstart

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e .

python -m pytest -q                                  # 27 tests (model, MoE, MoD, tokenizer, trainer)
python scripts/train_arcus.py --preset tiny --max_tokens 2000000   # pipeline run on the 5080
```

> Make sure your prompt shows `(.venv)` before `pip install`. If pip prints
> "Defaulting to user installation," the venv isn't active — activate it first.

The `tiny` preset is the 5080 / pipeline check. Real Alpha models (~1.3B+) are
from-scratch and run on **cloud** — see the ladder below.

---

## Scale ladder (boenet Phase-4 §7)

| Preset | Size | Where | Role |
|---|---|---|---|
| `tiny` | few M | 5080 / CPU | pipeline validation |
| `alpha-0.1` | ~1.3B | cloud | first real quality finding |
| `alpha-0.5` | ~7–13B | cloud | "this competes" |
| `alpha-1.0` | ~70–86B | cluster | optional far end — *not* the goal |

The point isn't the big rungs — it's **quality-per-compute**. MoDE's efficiency lets the
small, accessible rungs punch above their weight; the cluster sizes are optional.

---

## Status & honest gaps

- Tiny preset **built and runtime-validated** (27 tests: lossless@cap=1, causal,
  gradient to both routers + every expert, end-to-end training).
- Quality is **unproven** — that needs the cloud runs on the alpha dataset.
- Win condition (boenet): MoDE **matches** dense at lower compute, not beats — read
  every run against the `--dense` matched baseline.

---

## License & provenance

Apache 2.0 © 2026 William McKeon (see [LICENSE](LICENSE)). Arcus is an **original,
from-scratch** model (not a derivative). The archived [`legacy/`](legacy/) code wraps
Qwen3 (Apache 2.0); see [NOTICE](NOTICE).

---

*arcus — part of the OpenAgent family*
