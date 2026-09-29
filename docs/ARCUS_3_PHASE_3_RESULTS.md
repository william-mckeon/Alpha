# Arcus 3 Phase 3 results — September 29, 2026

Status: the bounded application integration is implemented and live-tested.
The unchanged 1,711,376,384-parameter SmolLM2 donor runs locally in Docker CUDA.
LangChain BaseChatModel and LangGraph provide the application interface and tool loop.
No training updates, expert expansion, RL, publication or Alpha resume occurred.

## Implemented

- Text-only LangChain adapter with invoke, exact allowlisted bind_tools, tool-call
  IDs, token usage, truncation metadata and original donor chat serialization.
- LangGraph model → tool → model loop, bounded to three model calls and two tool
  calls per turn, with explicit cancellation/error/budget status.
- Real local echo and validated arithmetic functions, without shell/network/file
  access or arbitrary Python execution.
- Complete-turn memory, session isolation, history limits, explicit opt-in disk
  persistence, loading/deletion API and rejection of path traversal.
- Application launcher mode, read-only request files, fresh output roots,
  deadline/own-container cleanup and preserved historical pauses.
- Separate application evidence; the frozen Phase 2 baseline is unchanged.

## Actual live donor run

Successful run: `runs/arcus3/application-phase3-002/`.
Full report: `application-report.md`; raw model inputs/token IDs, usage and message
records: `application-report.json`; per-request records: `application-transcripts.json`.
`runtime.json`, `container-state.json`, `worker.log` and `preservation.json` retain
the runtime and integrity evidence. Four session JSON files were written because
this run explicitly selected persistence; two requests shared the memory session.

Image: `sha256:5245e90a27eb6fcc53f385c678d71f71ad595d54f87d64c46b2fb12bec3b644e`.
Donor revision: `31b70e2e869a7173562077fd711b654946d38674`.

| Live request | Outcome |
|---|---|
| Hi, how are you? | Coherent greeting; complete response. |
| Remember favorite color violet | Acknowledged violet. |
| What is my favorite color? | Recalled violet from the prior turn. |
| Echo hello, then report the result | Executed echo once, received real observation, answered in a second model turn. |
| Calculate 17 + 25, then report result | Executed calculate once, received 42, answered in a second model turn. |

All five requests reached complete status. Seven model calls, two real tool
executions, zero truncated generations. These are application diagnostics, not a
broad capability or reliability benchmark. Greeting quality is a qualitative
observation, not a new numeric fluency score. The model was not trained on feedback.

### Tool evidence

Echo returned `{"ok":true,"result":"hello"}`. The model's subsequent answer:

> The tool returned: "hello"

Calculate returned `{"ok":true,"result":42}`. The model's subsequent answer:

> The result of adding 17 and 25 is 42.

These observations came from actual local functions, not Phase 2 canned fixtures.
The initial model emission included call JSON plus prose. That prose is retained
with a protocol warning and is not counted as the final answer or a tool result.

## Problems found and fixed

1. The first CPU tests caught a callback field named generate shadowing LangChain's
   BaseChatModel.generate method. Renaming the injected callback to backend fixed
   normal invoke/bind_tools behavior.
2. Live run `application-phase3-001` passed greeting and memory but rejected both
   tool requests: a strict whole-response JSON parser rejected valid leading JSON
   followed by prose. The repaired parser accepts one complete leading object,
   preserves the raw response and warning, rejects ambiguous multiple leading
   calls and tools not bound for the request, executes the real function, then
   requires another model turn. The same five prompts passed after the repair.
3. Raw generation diagnostics are kept in reports, not copied into reusable chat
   history. Truncated non-tool responses have explicit status and are not memorized.

The failed run is preserved. No prompt, donor weight or frozen baseline was altered
to conceal the failure.

## Verification and resources

- **32 tests passed in Docker**, including the tiny CUDA regression test.
- Windows CPU tests passed with the CUDA-only test intentionally skipped.
- Controlled graph fixtures covered malformed/unauthorized calls, invalid argument
  types, division-by-zero error observations, loop budgets and cancellation. These
  injected-error tests validate infrastructure; they are not scored donor responses.
- Memory tests covered persistence reload, deletion, session isolation, whole-turn
  trimming and oversized turns. The live custom-request mount and persistence worked.
- The application-mode actual-launcher deadline cleanup fixture passed and only
  killed its own mocked container. Runtime pause tests passed.
- Successful production container exited 0, OOMKilled false.
- Model section: 51.09 seconds. Container lifetime including preparation: 103.23 seconds.
- Peak CUDA allocation: 3,559,527,424 bytes, about 3.32 GiB.
- Peak process RSS: 4,184,993,792 bytes, about 3.90 GiB.
- Docker 8 GiB, CPU 2, PID 128; allocator cap 70%; removed memory watchdog stayed off.
- Original donor files were mounted read-only; all 14 files verified. Historical
  Alpha 53,192 and foundation pilot 2-update checkpoint hashes match, Alpha's pause
  remains present, and Phase 2 frozen fixture hashes pass verification.

The interface is bounded request-file chat, not a hosted chat UI or unrestricted
agent. Supported tools are echo and arithmetic. The restricted Python evaluator
continues to operate separately through baseline mode. Persisted session files
can be loaded through the memory API; the launcher does not reuse old paused roots.

Phase 4's complete file inventory and acceptance sequence are in
`ARCUS_3_PHASE_4_FILE_PLAN.md`. Phase 4 is the dense supervised LoRA control,
before approximately 2B expert expansion. No files need deletion.
