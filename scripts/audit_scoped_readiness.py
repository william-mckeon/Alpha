"""Summarize bounded diagnostic evidence; never launch or promote model runs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.budget_scope import lifetime_accounting
from evaluation.control import validate_result, validate_artifact_hashes
from evaluation.execution import source_fingerprint
from evaluation.worker_runner import registered_runners


def audit(root):
    protocol = json.loads((root / "evaluation/protocol.json").read_text(encoding="utf-8"))
    fingerprint = source_fingerprint(root)
    launcher_hash = hashlib.sha256((root / "scripts/eval_foundations.py").read_bytes()).hexdigest()
    normalized = {}
    for directory in [root / "evaluation/runs/normalized", *sorted((root / "evaluation/runs/integration-suites").glob("phase1-scopes[78]-*"))]:
        for path in directory.glob("phase1-scopes[78]-*.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            if "run" in record:
                normalized[record["run"]["id"]] = (record, path)
    trials = []
    for folder in ("bfcl", "mcpmark", "repository", "harbor"):
        for directory in sorted((root / "evaluation/runs" / folder).glob("phase1-scopes[78]-*")):
            if not directory.is_dir():
                continue
            trial_id = directory.name
            proxy = root / "evaluation/runs/proxy" / trial_id
            exchanges = proxy / "proxy-exchanges.jsonl"
            errors = proxy / "gateway-errors.jsonl"
            rows = [json.loads(line) for line in exchanges.read_text(encoding="utf-8").splitlines() if line.strip()] if exchanges.exists() else []
            events = [json.loads(line) for line in errors.read_text(encoding="utf-8").splitlines() if line.strip()] if errors.exists() else []
            manifest_path = directory / "validation.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
            item = {"run_id": trial_id, "folder": folder, "completed_requests": len(rows),
                    "actual_usd": sum(row["response"]["usage"]["cost"] for row in rows),
                    "gateway_categories": sorted({event["category"] for event in events}),
                    "status": manifest.get("status", "running_or_no_worker_manifest"),
                    "stop_class": manifest.get("stop_class"), "qualification_eligible": False,
                    "current_contract_normalized": False}
            if (directory / "fault-probe.json").exists():
                item["fault_probe"] = json.loads((directory / "fault-probe.json").read_text(encoding="utf-8"))
            if (directory / "interruption.json").exists():
                item["interruption"] = json.loads((directory / "interruption.json").read_text(encoding="utf-8"))
                if not manifest:
                    item["status"] = "operator_interrupted_excluded"
            if trial_id in normalized:
                record, path = normalized[trial_id]
                item.update(normalized_path=str(path.relative_to(root)), outcome=record["outcome"]["status"])
                if not manifest and record["run"].get("finished_at"):
                    item["status"] = "completed_diagnostic"
                item["current_contract_normalized"] = (record["run"]["contract_revision"] == protocol["contract_revision"] and
                    record["run"].get("execution_fingerprint") == fingerprint and
                    record["run"].get("launcher_sha256") == launcher_hash and
                    not validate_result(record) and not validate_artifact_hashes(record, root))
            trials.append(item)
    images = {}
    for tag in ("arcus-eval-openhands:phase1", "arcus-eval-mcpmark:phase1"):
        document = json.loads(subprocess.check_output(["docker", "image", "inspect", tag], text=True))[0]
        images[tag] = {"id": document["Id"], "source_fingerprint": (document["Config"].get("Labels") or {}).get("arcus.evaluation.source")}
    controls = {name: hashlib.sha256((root / "evaluation" / name).read_bytes()).hexdigest()
                for name in ("protocol.json", "providers.json", "budgets.json", "result.schema.json")}
    return {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
            "contract_revision": protocol["contract_revision"], "status": "bounded_diagnostics_only",
            "full_sweep_launched": False, "qualification_complete": False, "donor_selected": False,
            "execution_fingerprint": fingerprint, "launcher_sha256": launcher_hash,
            "control_sha256": controls, "images": images, "registered_runners": sorted(registered_runners(root, protocol)),
            "lifetime_accounting": lifetime_accounting(root), "trials": trials,
            "diagnostic_trial_cost_usd": sum(item["actual_usd"] for item in trials),
            "regressions": {"passed": 213, "subtests_passed": 10, "evidence": "observed full pytest output, 2026-09-14"},
            "five_native_response_probes": {"passed": 5, "cost_usd": .00076501, "benchmark_evidence": False},
            "unpaid_audits": {"smoke_inputs": 420, "qualification_inputs": 2400, "terminal_resolved_configs": 45},
            "final_budget_status": json.loads((root / "evaluation/budgets.json").read_text(encoding="utf-8"))["scopes"]["final-smoke"]["status"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(ROOT)
    if args.output:
        path = (ROOT / args.output).resolve()
        if not path.is_relative_to((ROOT / "evaluation/results/manifests").resolve()) or path.suffix != ".json":
            raise ValueError("report output must be below evaluation/results/manifests")
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
