# Stage 0 — pretrain the 0.5B to fluency (Phase 0 of the self-improving loop)

> **Status: Verified with the documented generation limitation · Track A — from scratch.**
> Donor-derived Alpha begins from an already pretrained model and does not repeat this fluency stage.

> Take the `0.5b` preset from "the pipeline runs" to "the model can talk." Fluency is the
> prerequisite for everything downstream — SFT ([0006](0006-distillation-student.md)) and the
> whole loop ([0009](0009-self-improving-loop.md)) build on a base that already generates
> coherent language. This spec is Stage 0 of that loop.

## Goal

Train `0.5b` (dim 1024 · 12 layers · 4×2560 experts · ~614M params, o200k) to **fluency**:
coherent generation, not gibberish. The `0.5b` is the largest rung that trains **cleanly on
the 5080 with no spill** (fp32 fits ~14–15 GB of 16 GB — unlike the 1B, which spills), so
Stage 0 is a **free, local** run — though in practice it runs on **RunPod L40S** (~2.5× faster, no laptop upload crashes; [TRAINING.md](../docs/TRAINING.md)), with a `1b` seed alongside. This is **private** (Arcus Code — [0012](0012-arcus-code-boundary.md)). Its output is a saved, uploaded checkpoint
that becomes the seed the loop grows and teaches.

`val_ppl` says the model is *learning*; it does not say the text is *coherent*. So Stage 0
also ships the missing capability to **hear the model talk** — an autoregressive sampler —
because fluency is a qualitative gate you can only check by reading samples.

## Concepts

- **Fluency vs. quality.** Fluency = coherent, grammatical, on-distribution text. It is NOT a
  capability verdict (that comes later, at scale). A ~614M model on a reasoning-dense corpus
  becomes fluent **in that corpus's domain** (STEM/code) — which is intended: Arcus is a
  reasoning core, not a trivia store (see [0009](0009-self-improving-loop.md)).
- **The sampler** (`arcus/generate.py`). `generate(model, tokenizer, prompt, ...)` — a plain
  no-KV-cache decode loop (greedy / temperature / top-k / top-p), fine for the short samples a
  fluency check needs. `load_model(ckpt_dir)` rebuilds an `ArcusMoDE` from the serving
  checkpoint (`config.json` + `model.safetensors`) and **re-ties the head** (dropped from the
  bf16 file), so the artifact you *uploaded* is exactly the one you *sample*. CLI:
  `scripts/sample_arcus.py`.
- **~1 pass, real token budget.** Fluency wants **more unique tokens, ~1 pass** — not many
  epochs over a small slice (that memorizes). Drive it streaming (`--stream`, token budget),
  not epoch-based. Chinchilla floor for 614M ≈ **~12B tokens**; babble emerges earlier.
- **Free local, existing stack.** Training, streaming, checkpointing, resume, and HF upload are
  already built ([0005](0005-scale-and-training.md)); Stage 0 adds only the sampler and the
  documented fluency run — it does not add training code.

## Acceptance (checkable)

- [x] A sampler exists and is tested: `arcus/generate.py` (`generate` + `load_model`),
      `scripts/sample_arcus.py`, `tests/test_generate.py` — runs, greedy is deterministic, EOT
      stops it, and a saved checkpoint round-trips (head re-tied). Public API exports `generate`.
- [x] `scripts/memcheck.py --preset 0.5b` reports the 0.5B **fits 16 GB without spill** (bench-verified;
      the real run then moved to the RunPod L40S for throughput).
- [x] A fluency run trains `0.5b` streaming over a real token budget (12B tokens on the L40S);
      `val_ppl` falls to a final **56.58** on the diverse held-out set ([RESULTS.md](../docs/RESULTS.md)).
- [~] `scripts/sample_arcus.py` produces **grammatical** samples — but they **loop / lose coherence**
      (see [RESULTS.md](../docs/RESULTS.md) "Generation quality"). PARTIAL: the model forms language but
      is not a coherent generator — and the grown 1B at 15.3 ppl still loops. **Full coherence is an SFT
      goal (Stage 2/4), not a Stage-0 pass** — the honest boundary of pretraining alone.
- [x] The trained 0.5B is uploaded to HuggingFace (`Islanderintel/arcus-alpha-v0.5-0.5b`) and reloads
      via `load_model` for sampling (upload → download → sample round-trip verified).

## Non-goals (this pass)

- **A quality verdict.** Fluency is "it can talk," not "it competes." Capability is a scale
  question — see [ROADMAP.md](../ROADMAP.md) and [0009](0009-self-improving-loop.md).
- **SFT / tool-use / chat template.** Distillation, the o200k chat template, and tool tokens
  are Stage 2+ ([0006](0006-distillation-student.md), [0009](0009-self-improving-loop.md)).
  Stage 0 is pure pretraining; keep the corpus pretraining-only.
- **The 1B locally.** The 1B spills on 16 GB — its real fluency run is cloud
  ([0005](0005-scale-and-training.md)); the 0.5B is the local fluency rung.
- **Growth.** Growing 0.5B→1B is Stage 1 ([0009](0009-self-improving-loop.md)); it needs a
  growth operator that does not exist yet.

## Notes

- **Domain-shaped fluency is a feature, not a bug.** A STEM/code corpus yields a model fluent
  in STEM/code. That aligns with the reasoning-core thesis (offload facts to retrieval; spend
  params on reasoning). See [0009](0009-self-improving-loop.md) § the reasoning core.
- **The sampler has no KV cache** (re-feeds the growing sequence, O(T²)). Correct and fine for
  fluency checks; a cached decoder is a serving concern (Stage 3, the vLLM shim).
- **Corpus hygiene for Stage 0.** The streaming loader globs `**/*.jsonl.zst`, so it will pick
  up any rendered-SFT shard mixed into the corpus. For a *pure* fluency run, point it at a
  pretraining-only path (the SFT ramp starts Stage 2, not here).
