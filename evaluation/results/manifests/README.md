# Retained-run manifests

Commit one manifest per completed evaluation job. Each manifest records the run IDs, task IDs,
upstream revisions, storage locations, SHA-256 hashes, and whether the artifact contains benchmark
test material. Raw requests, responses, tool events, and verifier outputs remain in ignored or
access-controlled storage and must never be reused as training data.

Invalid and interrupted runs also receive manifests. Their artifacts and costs remain available
for diagnosis, but they are never included in model scores. Raw invalid-run artifacts remain in
ignored storage and are not overwritten by corrected runs.
