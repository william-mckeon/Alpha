"""Fresh official grading of retained smoke responses; no model generation."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.bfcl import frozen_case
from evaluation.worker_adapters import isolated_worker_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not args.run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.run_id):
        raise ValueError("invalid run identity")
    protocol = json.loads((ROOT / "evaluation/protocol.json").read_text())
    environment = isolated_worker_environment(os.environ)
    environment.update(ARCUS_WORKER_BASE_URL="http://127.0.0.1:8010/v1", ARCUS_WORKER_TOKEN="regrade-no-generation", ARCUS_WORKER_TRIAL_ID=args.run_id)
    outcomes = []
    for index in (46, 47, 48, 58, 59, 60):
        old = f"phase1-smoke-full-20260913-01-u{index:04d}"
        task = "parallel_multiple_181" if index < 50 else "web_search_37"
        output = ROOT / "evaluation/runs/bfcl" / f"{args.run_id}-u{index:04d}"
        if output.exists():
            raise ValueError("regrade evidence cannot be overwritten")
        output.mkdir()
        shutil.copytree(ROOT / "evaluation/runs/bfcl" / old / "result", output / "result")
        command = [str(ROOT / "evaluation/runs/bfcl-venv/Scripts/python.exe"), "-m", "evaluation.bfcl_worker", "--candidate", "step-3.5-flash", "--task", task, "--phase", "evaluate", "--output-dir", str(output)]
        with (output / "regrade.log").open("x", encoding="utf-8") as log:
            subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=protocol["generation"]["verifier_timeout_seconds"], check=True)
        category, official = frozen_case(task, protocol)
        score = next((output / "score").rglob(f"BFCL_v4_{category}_score.json"))
        rows = [json.loads(l) for l in score.read_text(encoding="utf-8").splitlines() if l.strip()]
        outcomes.append({"source_run": old, "official_task_id": official, "score": rows[0], "score_path": str(score), "contract_revision": protocol["contract_revision"], "paid_calls": 0, "evidence_kind": "diagnostic_regrade", "qualification_eligible": False})
    target = ROOT / "evaluation/runs/reviews" / args.run_id
    target.mkdir()
    with (target / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(outcomes, stream, indent=2)
    print(json.dumps(outcomes, indent=2))


if __name__ == "__main__":
    main()
