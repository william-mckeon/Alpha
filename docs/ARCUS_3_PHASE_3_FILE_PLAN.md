# Phase 3: application integration file inventory

Phase 3 builds an application interface around the unchanged donor. It is not
training, RL, expert expansion or a release. LangChain and LangGraph are required.
Implementation should preserve the frozen Phase 2 baseline and add a separately
versioned integration test track; do not rewrite baseline prompts after seeing scores.

## Update

| File | Change |
|---|---|
| `arcus3/orchestration.py` | Typed conversation state and bounded tool round trips, retaining the existing baseline path. |
| `arcus3/config.py` | Validate application budgets, storage scope and explicit supported tools. |
| `configs/arcus3/project.json` | Record Phase 3 scope and validation evidence; keep training disabled. |
| `configs/arcus3/local_runtime.json` | Application inference budgets and pinned tested image. |
| `scripts/start_arcus3.ps1` | Explicit application mode and owned-process cleanup. |
| `docker/baby-arcus/Dockerfile.arcus3` | Package the application adapter and new tests. |
| `tests/arcus3/test_orchestration.py` | Message/tool-result round trips, errors, cancellation and bounded loops. |
| `tests/arcus3/test_runtime.py` | Application mode, pause, deadline and resource validation. |
| `tests/arcus3/test_launcher.ps1` | Application mode cleanup fixture. |
| `README.md` | Local chat/application commands and limits. |
| `docs/ARCUS_3_LOCAL_FIRST_PLAN.md` | Record the implemented application boundary and phase status. |

## Add

| File | Purpose |
|---|---|
| `arcus3/chat_model.py` | LangChain BaseChatModel adapter preserving donor serialization, tool-call IDs and usage metadata. |
| `arcus3/tools.py` | Strict tool schemas, allowlisted dispatch and explicit separation of fixture tools from real actions. |
| `arcus3/memory.py` | Bounded conversation history; opt-in local persistence, session isolation and deletion controls. |
| `configs/arcus3/application.json` | Approved tools, per-turn limits, history limits and persistence defaults. |
| `scripts/chat_arcus3.py` | Local Docker CUDA conversational entry point using LangGraph. |
| `tests/arcus3/test_chat_model.py` | LangChain invoke/bind_tools compatibility, serialization and response conversion. |
| `tests/arcus3/test_tools.py` | Schema rejection, tool errors, restricted execution and no unauthorized actions. |
| `tests/arcus3/test_memory.py` | Session separation, bounded history and persistence controls. |
| `docs/ARCUS_3_APPLICATION_PROTOCOL.md` | Interface contracts, supported operations and reproducible integration tests. |
| `docs/ARCUS_3_PHASE_3_RESULTS.md` | Actual live integration outcomes and limitations. |

## Delete

None. Preserve donor weights, historical checkpoints, Alpha's pause, existing
evaluators, the original baseline and all previous release records.

Live verification must cover greeting, multi-turn context, one successful tool
round trip, invalid arguments, executor failure, bounded retries and pause/deadline.
Do not infer successful integration merely from the model emitting valid JSON.
