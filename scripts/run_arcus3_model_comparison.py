"""Run the frozen three-arm comparison one GPU job at a time."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.report_arcus3_model_comparison import (
    ARM_IDS, PROTOCOL_IDS, build_report, digest, read, render_markdown, validate_plan,
)


def atomic_json(path, value):
    """Publish state without importing the training/data dependency stack."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=".%s." % path.name, suffix=".pending")
    pending = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def _result_file(path, protocol):
    return path / ("scores.json" if protocol == "developmental" else "donor-scores.json")


def _complete_output(path, protocol):
    file = _result_file(path, protocol)
    if not file.is_file():
        return False
    value = read(file)
    return bool(value.get("execution_complete") if protocol == "developmental" else value.get("complete"))


def _container_succeeded(output):
    receipt = output / "container-state.json"
    if not receipt.is_file():
        return False
    state = read(receipt)
    return state.get("ExitCode") == 0 and state.get("Running") is False


def _finish_developmental(workspace, output, stop_at):
    """Finish the same completed generation when the legacy venv launcher cannot.

    This is not an evaluation retry: it is allowed only after a zero-exit GPU
    container has produced the raw scores and transcripts for this output.
    """
    if not _container_succeeded(output):
        return None
    scores = output / "scores.json"
    transcripts = output / "transcripts.json"
    if not scores.is_file() or not transcripts.is_file():
        return None
    raw = read(scores)
    if raw.get("execution_complete"):
        return 0
    python = Path(r"C:\Python314\python.exe")
    dependencies = workspace / ".python-deps"
    if not python.is_file() or not dependencies.is_dir():
        return None
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(dependencies) + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    result = subprocess.run(
        [str(python), "scripts/report_arcus3.py", "--root", str(output), "--deadline", stop_at],
        cwd=workspace, env=environment,
    )
    return result.returncode


def _command(workspace, plan, validated, arm_id, protocol_id, output, stop_at):
    protocol = plan["protocols"][protocol_id]
    arm = validated["arms"][arm_id]
    command = [
        "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "scripts/start_arcus3.ps1",
        "-Mode", protocol["mode"], "-Root", output.relative_to(workspace).as_posix(),
        "-StopAt", stop_at, "-RuntimeConfig", validated["evaluation_runtime"], "-EvaluationTier", "full",
    ]
    command += ["-ConvertedPath", validated["converted"], "-ExpandedPath", arm["checkpoint"]]
    if protocol_id == "benchmark":
        benchmark_root = Path(validated["storage_root"]) / plan["benchmark_data_relative"]
        if not (benchmark_root / "manifest.json").is_file():
            raise ValueError("Prepared benchmark data is missing")
        command += ["-BenchmarksPath", str(benchmark_root)]
    return command


def run(args):
    workspace = Path(__file__).resolve().parents[1]
    config_path = (workspace / args.config).resolve()
    plan = read(config_path)
    validated = validate_plan(plan, workspace)
    root = (workspace / args.root).resolve()
    owned = (workspace / "runs" / "arcus3").resolve()
    if root == owned or owned not in root.parents:
        raise ValueError("Comparison output must be an owned runs/arcus3 child")
    if root.exists() and not args.resume:
        raise ValueError("Comparison root already exists; use --resume only after reviewing partial evidence")
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "execution.json"
    state = read(state_path) if state_path.exists() else {
        "schema": "arcus3-three-model-execution-v1",
        "plan_sha256": validated["plan_sha256"],
        "complete": False,
        "status": "ready",
        "executions": [],
        "automatic_retry": False,
        "automatic_promotion": False,
    }
    if state.get("plan_sha256") != validated["plan_sha256"]:
        raise ValueError("Refusing to resume with a changed comparison plan")
    if state.get("status") == "failed":
        raise ValueError("Previous execution failed; diagnose it before creating a new comparison root")
    expected = [(arm, protocol) for arm in ARM_IDS for protocol in PROTOCOL_IDS]
    for index, (arm_id, protocol_id) in enumerate(expected):
        if index < len(state["executions"]):
            saved = state["executions"][index]
            if (saved.get("arm"), saved.get("protocol")) != (arm_id, protocol_id):
                raise ValueError("Saved execution order differs from plan")
            output = Path(saved["output"])
            result_file = _result_file(output, protocol_id)
            if saved.get("exit_code") != 0 or not _complete_output(output, protocol_id):
                raise ValueError("Incomplete prior execution requires review; no automatic retry")
            if saved.get("result_sha256") != digest(result_file):
                raise ValueError("Prior evaluation result changed; refusing automatic resume")
            continue
        reused = validated["reuse"].get((arm_id, protocol_id))
        if reused:
            output = Path(reused["output"])
            if not _complete_output(output, protocol_id):
                raise ValueError("Pinned reused evaluation is incomplete")
            entry = {"arm": arm_id, "protocol": protocol_id, "output": str(output),
                     "started_at": None, "completed_at": "preexisting", "exit_code": 0,
                     "reused": True, "result_sha256": reused["result_sha256"]}
            state["executions"].append(entry)
            atomic_json(state_path, state)
            continue
        protocol = plan["protocols"][protocol_id]
        output = owned / ("baseline-comparison-%s-%s-%s" % (arm_id.replace(".", ""), protocol_id, uuid.uuid4().hex))
        deadline = datetime.now(timezone.utc) + timedelta(minutes=protocol["max_minutes"])
        stop_at = deadline.isoformat(timespec="microseconds")
        command = _command(workspace, plan, validated, arm_id, protocol_id, output, stop_at)
        entry = {"arm": arm_id, "protocol": protocol_id, "output": str(output),
                 "started_at": datetime.now(timezone.utc).isoformat(), "completed_at": None, "exit_code": None}
        state["executions"].append(entry)
        state.update(status="running", active_index=index)
        atomic_json(state_path, state)
        result = subprocess.run(command, cwd=workspace)
        entry["launcher_exit_code"] = result.returncode
        finalizer_code = None
        if protocol_id == "developmental" and not _complete_output(output, protocol_id):
            finalizer_code = _finish_developmental(workspace, output, stop_at)
        if finalizer_code is not None:
            entry["host_finalizer_exit_code"] = finalizer_code
            entry["host_finalizer"] = "C:\\Python314\\python.exe scripts/report_arcus3.py"
        entry["exit_code"] = 0 if _complete_output(output, protocol_id) and (
            result.returncode == 0 or finalizer_code == 0) else result.returncode or finalizer_code
        entry["completed_at"] = datetime.now(timezone.utc).isoformat()
        if entry["exit_code"] != 0 or not _complete_output(output, protocol_id):
            state.update(status="failed", complete=False)
            atomic_json(state_path, state)
            raise RuntimeError("Comparison execution failed; preserved at " + str(output))
        entry["result_sha256"] = digest(_result_file(output, protocol_id))
        atomic_json(state_path, state)
    state.update(status="complete", complete=True, active_index=None, completed_at=datetime.now(timezone.utc).isoformat())
    atomic_json(state_path, state)
    report = build_report(plan, validated, state)
    atomic_json(root / "comparison.json", report)
    (root / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"complete": True, "report": str(root / "comparison.json"), "automatic_promotion": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/arcus3/alpha321_post7m_comparison.json")
    parser.add_argument("--root", required=True)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
