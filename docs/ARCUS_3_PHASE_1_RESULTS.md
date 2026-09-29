# Arcus 3.0 Phase 1 — unchanged donor validation

Status: Phase 1 passed, September 28, 2026 local time (September 29 UTC).
Pinned acquisition, production CUDA probes and reload verification completed. No training,
expert expansion, RL, context extension, cloud job or publication is authorized here.

## Implemented

Isolated `arcus3` package; explicit pinned acquisition and offline verification;
safe path/hash/metadata checks; unchanged BF16 donor loading with original tokenizer
and chat template; bounded greedy greeting/Python-loop probes; full reload output
comparison; cache next-token logit comparison; exclusive GPU lock; inference deadline
and own-container cleanup. Historical Alpha state is separate.

Donor: `HuggingFaceTB/SmolLM2-1.7B-Instruct`, revision
`31b70e2e869a7173562077fd711b654946d38674`. Measured unique parameters: 1,711,376,384;
the original tied input/output embeddings were verified.
Configured context: 8,192. Two short probes cannot establish full-context competence.

## Tests and operational repairs

- Thirteen CPU fixtures passed on Windows and Linux: eleven donor/runtime checks
  plus two orchestration checks. Windows donor/runtime and final Linux logs are saved
  under `runs/diagnostics/arcus3-phase1/`; the Windows bridge checks passed separately.
- BF16 matrix multiplication passed on the actual RTX 5080 Laptop GPU (compute
  capability 12.0), in controlled Docker CUDA under the shared GPU lock.
- Actual PowerShell parser and expired-deadline rejection passed.
- Actual launcher control-flow fixture passed: a two-second deadline kills exactly
  its own mocked container, records exit 137/OOMKilled=false, and retains Alpha's pause.
  Evidence: `runs/arcus3/donor-probe-deadline-fixture-20260928202746/fixture-result.json`.
- BuildKit rejected `FROM sha256:<local-id>` as a registry reference. Build now uses
  the existing local tag after verifying its full ID against the pinned base.
- Hugging Face temporary filenames exceeded Windows MAX_PATH beneath OneDrive.
  Metadata acquisition now uses Windows extended paths.
- Standard large-file HTTPS transfer stalled. Weight acquisition now uses bounded
  resumable HTTP ranges, up to eight in flight, with exact Content-Range/length checks
  and upstream LFS SHA-256 verification before accepting the file. Fixture tests cover
  malformed ranges and partial resume. No unverified partial weights are loaded.
- The first production launch exited before model load because Python 3.10 rejected
  PowerShell's seven-digit fractional timestamp. The launcher now emits six digits;
  the parser also normalizes fractions, with a cross-platform regression fixture.
- The original fixed BF16 cache-logit cutoff of 0.25 flagged observed deltas of
  0.28125 and 0.3125 despite identical next-token choices. This is retained as a
  diagnostic flag, not concealed by increasing its cutoff. Acceptance now separately
  requires identical BF16 next-token choices, exact repeated generation tokens, and
  an independent FP32 cached/full check below 0.001. The production FP32 errors were
  0.0000648499 and 0.0000362396, repeated across both loads. FP32 checks run only after
  BF16 generation; a fresh donor reload preserves original rotary-buffer precision.
  These are short-prompt checks, not exhaustive cache equivalence.

Runtime: Torch 2.11.0+cu128, Transformers 4.46.3, Tokenizers 0.20.3,
Hugging Face Hub 0.36.2, Safetensors 0.6.2, Accelerate 1.1.1.
The additional Accelerate dependency psutil is pinned at the tested 7.2.2.
Base image: `sha256:b0977c166da170df299aad52e15701254a5dfa2ade3bba37a19aaba85ad07411`.
The built image is pinned by ID in `configs/arcus3/local_runtime.json` before probing.

## Verified production evidence

- All 14 donor files verified. Weight file: 3,422,777,952 bytes; SHA-256
  `f55217be716b6a997b97b9d8d7eb6fad02e00858f5010ec24f64603c3a98a0e8`, matching upstream LFS.
- Donor manifest SHA-256: `910a430561a4488acf7bbc371665a9e3895a5bf0c414f6ecc39cd91e606007c5`.
- Runtime image: `sha256:2bf2aa27f861c6e010af89718d0be4c4cc8dcaeb13686b570714b2b3ab1a18b1`.
  The authoritative actual image ID is also recorded in the probe runtime receipt.
- Evidence root: `runs/arcus3/donor-probe-phase1-003/`; full `report.json` includes
  serialized prompts, input/output token IDs, raw responses, dtype, dependencies,
  cache measurements and direct-versus-orchestrated paths. `runtime.json`,
  `worker.log` and `container-state.json` preserve execution evidence.
- Container `arcus3-donor-20260928-205415` exited 0, OOMKilled=false; no active containers
  remained after the test. Earlier failed probes are preserved as `phase1-001/002`.
- Peak BF16 CUDA allocation: 3,452,385,280 bytes (about 3.22 GiB).
- Peak process RSS: 5,179,547,648 bytes (about 4.82 GiB), including FP32 diagnostics.
- Probe core took 101.2 seconds; total container lifetime was about 139.9 seconds,
  including verification/startup. Observed generation times were 0.98–1.79 seconds
  per response; two short prompts are not a throughput benchmark.
- Both responses ended naturally below 128 generated tokens. The original donor
  system instruction was `You are a helpful AI assistant.`; no identity finetuning.

## Prompt and response transcripts

Prompt: `Hi, how are you?`

> I'm doing well, thank you for asking. I'm here to assist you with any questions or information you might need. How can I help you today?

Prompt: `Arcus, can you please write me a simple Python for loop?`

> Sure, here's a simple Python for loop:

```python
# Initialize a list
fruits = ['apple', 'banana', 'cherry']

# Loop through the list
for fruit in fruits:
    print(fruit)
```

> This will print each fruit in the list on a new line.

The full responses, including punctuation and token sequences, matched after fresh
reload through a LangChain Runnable inside LangGraph. Pinned versions are LangChain
1.0.7, langchain-core 1.0.7 and LangGraph 1.0.5. This proves the local integration
path works; it does not yet validate `bind_tools`, durable graph persistence or
autonomous agent capability. Generated Python was retained as text, not executed
in Phase 1. Restricted code execution and full scoring belong to Phase 2.

## Reproduction

From the workspace, acquire with `.venv/Scripts/python.exe scripts/inspect_arcus3_donor.py download`
or verify with the same command ending in `verify`. The project manifest explicitly
authorizes only download and inference. The donor path is under `artifacts/arcus3/donor/`.

Before building, compare `docker image inspect arcus-foundation:nanotron-v1 --format '{{.Id}}'`
against `base_image_id` in the runtime configuration. Build with
`docker build -f docker/baby-arcus/Dockerfile.arcus3 -t arcus3:donor-v1 .`, then record
the resulting immutable image ID in `local_runtime.json`.

Run CPU fixtures with `.venv/Scripts/python.exe -m unittest discover -s tests/arcus3 -v`.
Run Linux fixtures with `docker run --rm --network none arcus3:donor-v1 -m unittest discover -s tests/arcus3 -v`.
The PowerShell fixture is `tests/arcus3/test_launcher.ps1` (mocked Docker, no model).

After verified acquisition, invoke `scripts/start_arcus3.ps1` with `-StopAt` set to
an explicit timezone-aware deadline within 30 minutes and a fresh
`-Root runs/arcus3/donor-probe-<identifier>`. It uses Docker CUDA only, network disabled,
8 GiB RAM, 2 CPUs, 128 PIDs, the shared GPU lock and 70% CUDA allocator cap. The
removed free-memory watchdog stays disabled. The output includes full prompts,
responses, token IDs, manifest hash, resource measurements and container exit state.

## Next phase

[Phase 2](ARCUS_3_PHASE_2_FILE_PLAN.md) is ready for implementation: six existing
files to update, nine to add and none to delete. Phase 1 established local fit for
these short inference probes only. Full baseline scores, training fit and long-context
capability remain unmeasured.
