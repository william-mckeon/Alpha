"""Retain a just-observed terminal diagnostic, never promote it to benchmark evidence."""
import argparse
import hashlib
import json
from pathlib import Path

import eval_foundations as cli
from evaluation.control import validate_result, validate_artifact_hashes
from evaluation.execution import source_fingerprint
from evaluation.ingest import ingest_harbor_job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--expected-source", required=True)
    args = parser.parse_args()
    if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.run_id):
        raise ValueError("invalid run id")
    if source_fingerprint(cli.ROOT) != args.expected_source:
        raise ValueError("source changed since this live observation; do not relabel historical proof")
    protocol, harness, _ = cli._harbor_inputs(args.candidate)
    record = ingest_harbor_job(
        cli.ROOT / "evaluation/runs/harbor" / args.run_id,
        cli.ROOT / "evaluation/runs/proxy" / args.run_id / "proxy-exchanges.jsonl",
        candidate_id=args.candidate, attempt=1,
        registry=cli.load_json(cli.ROOT / "evaluation/candidates.json"),
        protocol=protocol, providers=cli.load_json(cli.ROOT / "evaluation/providers.json"),
        harness=harness,
    )
    record["run"].update(evidence_kind="diagnostic", execution_fingerprint=args.expected_source,
                         launcher_sha256=hashlib.sha256(Path(cli.__file__).read_bytes()).hexdigest())
    errors = validate_result(record) + validate_artifact_hashes(record, cli.ROOT)
    if errors:
        raise ValueError("; ".join(errors))
    path = cli.ROOT / "evaluation/runs/normalized" / f"{args.run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"outcome": record["outcome"], "qualification_eligible": False}, indent=2))


if __name__ == "__main__":
    main()
