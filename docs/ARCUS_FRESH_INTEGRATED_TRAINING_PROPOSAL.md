# Fresh integrated Arcus training — discussion record

Date: September 21, 2026. **Status: accepted experiment; implementation and smoke
training underway; scientific acceptance remains open.** This discussion record
explains the hypothesis. Current implementation is governed by
[spec 0048](../specs/0048-fresh-integrated-arcus.md), with measured results in
[Test 2 results](ARCUS_TEST2_RESULTS.md). The existing learner was not reset.

The requested branch `baby-arcus-test-2` now exists. Its
[file-level implementation plan](ARCUS_TEST2_FILE_PLAN.md) maps this proposal to
existing and proposed files; consolidated implementation boundaries are recorded
in [the implementation inventory](ARCUS_TEST2_IMPLEMENTATION.md).

## Caregiver hypothesis

Initialize a new Arcus with capacity 0.25 and the complete sensory/body/tooling
design available from the beginning. Train shared representations across these
activities and use ReAct-style interaction throughout development. The hypothesis
is that this will produce a more efficient, integrated learner closer to the
long-term goal of learning through experience, communication and exploration.

This is plausible but unproven. A compute constraint can encourage useful
allocation or prevent learning. Shared channels can support transfer or cause
interference. Phase 1 did not establish beneficial reuse across multiple tasks;
its instrumentation is a measurement tool, not a newly proven shared-neuron
training objective. More overlap is not itself a success criterion.

The current 0.25 setting restricts MoD expert-token routing while attention stays
dense. It is not 25% of model parameters, 25% of all layers, or a guaranteed 75%
reduction in end-to-end cost. Starting afresh at that setting remains an experiment.

## Proposed learning design

Use one shared learner with modality encoders and appropriate output heads, with
coordinated checkpoints and optimizer state. Body, RGB, symbolic hearing/text,
internal state and action outcomes should contribute to training from the start.
Do not mistake several connected independently trained policies for demonstrated
joint learning. Any staged freezing or separate objectives must be explicit and
their effects on the shared representation measured.

Begin with **observe → predict → act → observe the outcome → record → learn**.
Use demonstrations or curriculum supervision only if included in the eventual
experiment contract, with their provenance recorded. As language competence
develops, add short plans, questions and ReAct trajectories that alternate model
decisions and tool observations. Routine motor control should not require verbose
language reasoning. A from-scratch model does not gain language or valid tool
selection merely because a framework is attached.

The teacher/reviewer may be a larger model plus caregiver review, as discussed
earlier. Distinguish teacher suggestions, scripted curriculum decisions and
Arcus-generated choices in the records. Judge Arcus independently so that the
teacher or workflow does not silently solve the task being attributed to him.

Prioritize correctness and retention, then lower total cost per successful task.
Measure inference, reasoning tokens, tool calls, retries and elapsed resources,
not just the routing fraction. Avoid rewarding inactivity or short failed answers
as efficient behavior. Decisions about exact rewards, schedules and weights remain
open. Growth remains a later, separately evaluated intervention.

## LangGraph and LangChain recommendation

Use **LangGraph first**, adding selected LangChain interfaces when useful.
LangGraph can be used without LangChain. Neither framework is a neural optimizer
or an automatic implementation of continual learning.

| Component | Proposed responsibility | Boundary |
|---|---|---|
| Arcus model service | Predictions, learned decisions, action/tool selection | The same shared learner; no hidden replacement by a stronger model |
| LangGraph | High-level interaction flow, state, interruptions, recovery and budgets | Nodes are workflow steps, not separate brains or necessarily separate agents |
| LangChain components | Tool schemas, adapters and selected agent-loop utilities | A custom adapter is needed for Arcus's native outputs; valid tool use still must be learned |
| Existing simulator/body service | Fast body stepping, action execution and sensory feedback | Do not route every joint update through a text-agent cycle |
| Replay and training services | Provenance, held-out partitions, candidate updates and retention tests | Graph checkpoints do not replace neural/optimizer checkpoints or training receipts |
| Viewer and caregiver interface | Observation, communication, intervention and review | Preserve human priority and distinguish exposure from actual learning |

Proposed interaction flow:

```text
sensory observation -> Arcus predicts/chooses -> validate and execute tool
        ^                                      |
        +--------- fresh outcome --------------+
                           |
                    durable experience
                           |
               bounded candidate training
                           |
               evaluation and qualification
                           |
                   reviewed promotion
```

The graph should control delivery and execution, not hard-code the choices whose
learning is being tested. If the graph always chooses to approach a caller, that
is a scripted behavior, not evidence that Arcus learned to approach. Early
nonverbal actions can use structured outputs instead of fabricated reasoning text.

Persistence must distinguish graph execution state, sensory memory, replay and
model weights. Interrupted/replayed graph steps must not execute a body action or
consume a dataset passage twice: retain request IDs, acknowledgements and durable
deduplication. Graph recovery alone does not establish exactly-once external
effects. Dataset delivery must still meet the pending Phase 2 contract.

Keep services independently deployable for eventual cloud placement. Do not infer
cloud portability or faster execution just from adopting these libraries. Pin and
test dependencies against Windows and the required Ubuntu 22.04/Python 3.10 runtime
before adoption; the compatible package versions and integration design are not
selected yet. No dependency installation or runtime migration was performed.

## Proposed comparisons

| Candidate | Question |
|---|---|
| Current Arcus, continued training | Does a restart beat continued investment in the preserved learner? |
| Fresh integrated Arcus at 0.25 | Does training the complete system together produce useful transfer? |
| Same fresh integrated design with explicit ReAct training | What does ReAct add beyond the shared interaction loop? |

Use multiple initialization seeds, fixed held-out environments/documents and
declared stopping criteria. Compare both equal training-data exposure and equal
measured compute in separately defined protocols; those budgets are not
automatically interchangeable. Include training cost as well as inference cost.
The current model has a different history, so its comparison is a practical
restart decision, not by itself a causal proof of joint-training benefit. A fresh
staged-training control would be needed to isolate that claim more cleanly.
These controls remain required comparative research work. The completed smoke
runs do not substitute for matched data/compute budgets or independent seeds.

Measure skill acquisition, unseen-task success, cross-task transfer, forgetting,
rest behavior, tool validity, recovery, and total cost per successful outcome.
Separate ReAct model training from the framework used to execute the same policy.
Library adoption, extra traces or more shared channels alone are not improvements.

The accepted first implementation retains 151,946,954 parameters, fixes capacity
at .25, and introduces the graph from initialization. Configs pin framework
versions, bounded mixed lessons, quotas and learning gates. Quiet-time delivery
and learning are implemented inside this isolated experiment; production release,
larger comparative budgets and growth remain unqualified. Preserve all positive,
negative and inconclusive evidence.

## Research and framework references

These support the design discussion, not a prediction that Arcus will achieve
frontier capability. References were consulted in the September 21 discussion.

- [ReAct paper](https://arxiv.org/abs/2210.03629): interleaves reasoning and actions
  in existing language models; not a complete from-scratch learning recipe.
- [Mixture-of-Depths paper](https://arxiv.org/abs/2404.02258): learned allocation
  of computation; does not establish Arcus's end-to-end savings at capacity 0.25.
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview):
  orchestration, persistent state, human interaction and standalone use.
- [LangChain overview](https://docs.langchain.com/oss/python/langchain/overview):
  model/tool interfaces and agent components.
- [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence):
  saved workflow state; distinct from learned neural weights.
