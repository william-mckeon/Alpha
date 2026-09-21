# Shared output policy — revision 6

Historical snapshot. Revision 8 retains the output ceiling but replaces false-
blocking UTF-8 screening with nonblocking uncertainty metadata, and adds scoped
accounting/source-bound readiness. See `PHASE1_SCOPED_VALIDATION.md`.

Approved on 2026-09-14: every candidate receives the same 32,768-token output
ceiling. Other generation limits, tasks, provider pins and budget caps are unchanged.
This supersedes revision 5's 65,536-token allowance, not its retained historical evidence.

Known endpoint context limits are screened before reservation/payment using the
serialized messages/tools UTF-8 byte count plus 1,024 overhead and the full output
allowance. This is deliberately conservative, **not an exact tokenizer count**;
it can exclude requests that would actually fit. No prompts are truncated. Such
stops are capacity exclusions, never scored model failures. Provider context
rejections still remain possible when metadata is absent or the screening bound
does not capture provider-specific formatting.

The full smoke readiness check `phase1-policy6-readiness-20260914-01` blocked
before paid requests: current-contract BFCL, MCP and repository runner evidence
must be regenerated. Old revision-5 evidence cannot satisfy revision-6 gates.
Worker images must also be rebuilt with the updated capacity checks.

Full regression validation: 198 passed and 10 subtests passed. The first run
identified one obsolete 65,536-token protocol assertion, which was updated to the
approved limit before the successful rerun. No paid calls occurred in this pass.

No full sweep has started or been scheduled. Launch timing was requested from the
user. Step's cumulative $5 local cap has only about $0.038 available after the
retained unknown reservation. A shared $48 cap does not override per-model caps;
budget reallocation requires authorization before a meaningful full sweep.
