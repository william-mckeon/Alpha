"""Pinned local dataset validation for the official OpenHands SWE-bench runner."""
from __future__ import annotations

import json
import hashlib
import re
import urllib.request
from pathlib import Path


SDK_REVISION = "43376f1868ffd702746080714a59c16d3f69ec12"
EXTENSIONS_REVISION = "1bad294d4b9648b14ad335f516edf6f0a6622305"
JSONL_SHA256 = "f61cd55ceb35b61ad592f645abcbfc8ea4d294c6c9f3c8f15e83211a8e8db98c"
DATASET_SHA256 = "a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd"
ITERATION_BUDGET_ERROR = "trial exceeded global agent-iteration budget"


def check_gateway_stop(proxy_dir: Path, trial_id: str):
    """A host-audited financial stop aborts before verification, never grades zero."""
    errors = proxy_dir / "gateway-errors.jsonl"
    if not errors.is_file():
        return
    rows = [json.loads(line) for line in errors.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(row.get("trial_id") != trial_id for row in rows):
        raise ValueError("gateway stop belongs to another trial")
    for row in rows:
        if row.get("category") in {"provider", "timeout", "infrastructure", "context_capacity", "token_budget"}:
            from evaluation.provider import EvaluationBlocked
            exc = EvaluationBlocked("gateway " + row["category"] + " stopped inference before verification: " + str(row.get("error", "see retained evidence")))
            exc.stop_class = "gateway_" + row["category"]
            exc.artifact_references = [str(errors.resolve()), str(proxy_dir / "proxy-exchanges.jsonl")]
            raise exc
        if row.get("category") == "financial_budget":
            from evaluation.provider import EvaluationBlocked
            exc = EvaluationBlocked("financial budget stopped inference before verification")
            exc.stop_class = "financial_budget"
            exc.artifact_references = [str(errors.resolve()), str(proxy_dir / "proxy-exchanges.jsonl")]
            raise exc


def check_prediction_before_verify(directory: Path, task_id: str):
    """Never run a grader on upstream error-only/empty inference output."""
    from evaluation.provider import EvaluationBlocked
    predictions = list((directory / "inference").rglob("output.jsonl"))
    rows = [json.loads(l) for l in predictions[0].read_text(encoding="utf-8").splitlines() if l.strip()] if len(predictions) == 1 else []
    if len(rows) == 1 and rows[0].get("instance_id") == task_id:
        return
    errors = list((directory / "inference").rglob("output_errors.jsonl"))
    detail = "inference did not retain exactly one matching prediction"
    stop_class = "inference_error"
    if len(errors) == 1:
        failures = [json.loads(l) for l in errors[0].read_text(encoding="utf-8").splitlines() if l.strip()]
        if len(failures) == 1 and failures[0].get("instance_id") == task_id:
            detail = str(failures[0].get("error", detail))
            if "Remote conversation got stuck" in detail:
                stop_class = "sdk_stuck"
    exc = EvaluationBlocked("inference stopped before verification: " + detail)
    exc.stop_class = stop_class
    exc.artifact_references = [str(p.resolve()) for p in predictions + errors]
    raise exc


def gateway_iteration_stop(gateway_url: str, token: str, trial_id: str, max_iterations: int) -> bool:
    """Read causal stop state from the authenticated local gateway, never the agent."""
    from evaluation.worker_adapters import validate_gateway
    validate_gateway(gateway_url, token)
    request = urllib.request.Request(gateway_url.rstrip("/")[:-3] + "/arcus/control",
                                     headers={"Authorization": "Bearer " + token})
    with urllib.request.urlopen(request, timeout=10) as response:
        state = json.loads(response.read(65537).decode("utf-8"))
    return (state.get("trial_id") == trial_id
            and type(state.get("model_requests")) is int
            and state["model_requests"] == max_iterations
            and type(state.get("max_agent_iterations")) is int
            and state["max_agent_iterations"] == max_iterations
            and state.get("iteration_budget_exhausted") is True
            and state.get("terminal_error") is None)


def run_bounded_conversation(run, conversation, *, run_error, stop_path: Path, trial_id: str, max_iterations: int, stop_check=None):
    """Keep the patch on an explicit gateway budget stop; never retry the model."""
    try:
        return run(conversation)
    except run_error as exc:
        # Remote SDK exceptions can erase the originating HTTP error message.
        # The trusted worker queries independent gateway state in that case.
        proven = stop_check() if stop_check is not None else ITERATION_BUDGET_ERROR in str(exc)
        if not proven:
            raise
        with stop_path.open("x", encoding="utf-8") as stream:
            json.dump({"trial_id": trial_id, "reason": "model_iteration_budget_exhausted", "max_agent_iterations": max_iterations}, stream)
        # Upstream can now collect the existing patch/history and officially grade it.
        # Import requires independent gateway proof and forces model failure.
        return None


def disposable_conversation(factory, *args, **kwargs):
    """Let the owned disposable container release sessions, not a late HTTP delete."""
    kwargs["delete_on_close"] = False
    return factory(*args, **kwargs)


def guarded_workspace_command(command: list[str], *, trial_id: str, image_id: str) -> list[str]:
    """Pin and bound the SDK terminal container without granting host access."""
    if command[:2] != ["docker", "run"] or not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) or not re.fullmatch(r"[A-Za-z0-9_-]+", trial_id):
        raise ValueError("invalid owned workspace command or identity")
    index = command.index("--host") - 1
    forbidden = ("--mount", "--volume", "--privileged", "--cap-add", "--device", "--pid=host", "--network=host", "--ipc=host", "--uts=host", "--userns=host", "--cgroupns=host")
    options = command[2:index]
    host_namespace = any(flag in {"--pid", "--network", "--ipc", "--uts", "--userns", "--cgroupns"} and position + 1 < len(options) and options[position + 1] == "host" for position, flag in enumerate(options))
    if index < 2 or host_namespace or any(flag.startswith("-v") or flag.startswith(forbidden) for flag in options):
        raise ValueError("model workspace cannot mount host volumes or elevate privileges")
    resolved = list(command)
    resolved[index] = image_id
    resolved[2:2] = ["--label", "arcus.trial_id=" + trial_id, "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pids-limit=512", "--memory=8g", "--cpus=2"]
    return resolved


def export_snapshot(parquet_path: Path, output_path: Path) -> dict:
    """Lossless export of the retained pinned HF file; no mutable network lookup."""
    if output_path.exists():
        raise ValueError("snapshot output already exists; will not overwrite evidence")
    if hashlib.sha256(parquet_path.read_bytes()).hexdigest() != DATASET_SHA256:
        raise ValueError("SWE-bench parquet hash differs from the pinned snapshot")
    from importlib import import_module
    rows = import_module("pyarrow.parquet").read_table(parquet_path).to_pylist()
    ids = [row.get("instance_id") for row in rows]
    if len(ids) != 500 or not all(isinstance(value, str) and value for value in ids) or len(set(ids)) != 500:
        raise ValueError("SWE-bench Verified snapshot must contain 500 unique instances")
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as output:
        output.write(text)
    return {"source_sha256": DATASET_SHA256, "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(), "instances": len(rows)}


def load_task_catalog(dataset_path: Path) -> list[str]:
    """Require local revision-snapshotted JSONL, never mutable HF latest at runtime."""
    tasks = []
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not all(isinstance(row.get(field), str) and row[field] for field in ("instance_id", "repo", "base_commit", "problem_statement")):
            raise ValueError("SWE-bench row is missing official inference fields")
        tasks.append(row["instance_id"])
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("SWE-bench catalog must be nonempty with unique IDs")
    return sorted(tasks)


def build_commands(python: Path, *, dataset_path: Path, selection_path: Path, llm_config_path: Path, output_dir: Path, prediction_path: Path, run_id: str, generation: dict) -> dict[str, list[str]]:
    """Official local commands; prerequisites and provenance must be audited before launch."""
    if dataset_path.suffix != ".jsonl":
        raise ValueError("repository inference/grading requires a pinned local JSONL dataset")
    return {
        "inference": [str(python), "-m", "benchmarks.swebench.run_infer", str(llm_config_path), "--dataset", str(dataset_path), "--split", "test", "--select", str(selection_path), "--workspace", "docker", "--num-workers", "1", "--max-iterations", str(generation["max_agent_iterations"]), "--max-retries", "0", "--n-critic-runs", "1", "--disable-condenser", "--output-dir", str(output_dir)],
        "verification": [str(python), "-m", "benchmarks.swebench.eval_infer", str(prediction_path), "--dataset", str(dataset_path), "--split", "test", "--workers", "1", "--no-modal", "--run-id", run_id, "--timeout", str(generation["verifier_timeout_seconds"])],
    }
