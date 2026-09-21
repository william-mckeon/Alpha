"""Build and audit deterministic Harbor jobs for Arcus foundation evaluation."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any


def build_harbor_job_config(
    protocol: dict[str, Any],
    harness: dict[str, Any],
    *,
    candidate_id: str,
    task_id: str,
    attempt: int,
    run_id: str | None = None,
    jobs_dir: str = "evaluation/runs/harbor",
) -> dict[str, Any]:
    """Return one fully resolved, single-attempt Harbor job configuration."""
    if attempt < 1:
        raise ValueError("attempt must be at least 1")
    if attempt > int(protocol["generation"]["attempts"]):
        raise ValueError("attempt exceeds the frozen attempt count")
    allowed = set(harness["smoke_task_ids"]) | set(harness.get("qualification_task_ids", []))
    if task_id not in allowed:
        raise ValueError(f"task is not in a frozen Harbor subset: {task_id}")
    generation = protocol["generation"]
    kwargs = dict(harness["required_agent_kwargs"])
    kwargs.update(
        temperature=str(generation["temperature"]),
        top_p=str(generation["top_p"]),
        max_iterations=int(generation["max_agent_iterations"]),
        num_retries=int(generation["retry_on_model_failure"]),
        model_info={"max_output_tokens": int(generation["max_output_tokens"])},
    )
    return {
        "job_name": run_id or f"{candidate_id}-terminal-{task_id}-attempt-{attempt:02d}",
        "jobs_dir": jobs_dir,
        "n_attempts": 1,
        "n_concurrent_trials": int(generation["concurrency"]),
        "timeout_multiplier": 1.0,
        "agent_setup_timeout_multiplier": 1.0,
        "retry": {"max_retries": 0},
        "environment": {"type": "docker", "delete": True},
        "verifier": {"override_timeout_sec": float(generation["verifier_timeout_seconds"])},
        "agents": [
            {
                "name": harness["agent"],
                "model_name": f"openai/{candidate_id}",
                "override_timeout_sec": float(generation["wall_time_seconds"]),
                "override_setup_timeout_sec": float(
                    generation["agent_setup_timeout_seconds"]
                ),
                "kwargs": kwargs,
            }
        ],
        "tasks": [
            {
                "path": task_id,
                "git_url": "https://github.com/laude-institute/terminal-bench-2.git",
                "git_commit_id": harness["dataset_revision"],
                "source": "terminal-bench",
            }
        ],
    }


def audit_harbor_job_config(
    config: dict[str, Any], protocol: dict[str, Any], harness: dict[str, Any]
) -> list[str]:
    """Return every reason a resolved Harbor job is not valid Arcus evidence."""
    errors: list[str] = []
    generation = protocol["generation"]
    agents = config.get("agents")
    tasks = config.get("tasks")
    if config.get("n_attempts", 1) != 1:
        errors.append("Harbor job must contain exactly one attempt")
    if config.get("n_concurrent_trials") != generation["concurrency"]:
        errors.append("Harbor concurrency does not match the frozen protocol")
    if not isinstance(agents, list) or len(agents) != 1:
        errors.append("Harbor job must contain exactly one agent")
    else:
        agent = agents[0]
        if agent.get("name") != harness["agent"]:
            errors.append("Harbor agent does not match the pinned harness")
        if float(agent.get("override_timeout_sec", -1)) != float(
            generation["wall_time_seconds"]
        ):
            errors.append("Harbor agent timeout does not match the frozen protocol")
        if float(agent.get("override_setup_timeout_sec", -1)) != float(
            generation["agent_setup_timeout_seconds"]
        ):
            errors.append("Harbor setup timeout does not match the frozen protocol")
        kwargs = agent.get("kwargs", {})
        expected = {
            "version": harness["agent_version"],
            "temperature": str(generation["temperature"]),
            "top_p": str(generation["top_p"]),
            "max_iterations": generation["max_agent_iterations"],
            "num_retries": generation["retry_on_model_failure"],
            "model_info": {"max_output_tokens": generation["max_output_tokens"]},
        }
        for key, value in expected.items():
            if kwargs.get(key) != value:
                errors.append(f"Harbor agent kwarg {key} does not match the frozen protocol")
    verifier = config.get("verifier", {})
    if float(verifier.get("override_timeout_sec", -1)) != float(
        generation["verifier_timeout_seconds"]
    ):
        errors.append("Harbor verifier timeout does not match the frozen protocol")
    if not isinstance(tasks, list) or len(tasks) != 1:
        errors.append("Harbor job must contain exactly one revision-pinned task")
    else:
        task = tasks[0]
        allowed = set(harness["smoke_task_ids"]) | set(harness.get("qualification_task_ids", []))
        if task.get("path") not in allowed:
            errors.append("Harbor task is outside the frozen subsets")
        if task.get("git_commit_id") != harness["dataset_revision"]:
            errors.append("Harbor task revision does not match the frozen dataset")
        if task.get("git_url") != "https://github.com/laude-institute/terminal-bench-2.git":
            errors.append("Harbor task repository does not match the frozen dataset")
    if config.get("retry", {}).get("max_retries", 0) != 0:
        errors.append("Harbor model/harness retries must be zero")
    return errors


def load_harbor_result_state(job_dir: Path) -> dict[str, Any]:
    """Describe completion without treating partial Harbor output as a score."""
    job_config = json.loads((job_dir / "config.json").read_text(encoding="utf-8"))
    task_count = len(job_config.get("tasks", [])) + sum(
        len(dataset.get("task_names", [])) for dataset in job_config.get("datasets", [])
    )
    expected = int(job_config.get("n_attempts", 1)) * task_count
    trials = [path for path in job_dir.iterdir() if path.is_dir()]
    completed = [path for path in trials if (path / "result.json").exists()]
    aggregate = job_dir / "result.json"
    aggregate_result = (
        json.loads(aggregate.read_text(encoding="utf-8")) if aggregate.exists() else {}
    )
    aggregate_finished = bool(aggregate_result.get("finished_at"))
    return {
        "expected_trials": expected,
        "created_trials": len(trials),
        "completed_trials": len(completed),
        "aggregate_result_present": aggregate.exists(),
        "aggregate_finished": aggregate_finished,
        "complete": expected > 0 and len(completed) == expected and aggregate_finished,
    }


def stop_harbor_containers(job_dir: Path) -> list[str]:
    """Stop only Docker Compose projects named by this retained job's trial directories."""
    stopped: list[str] = []
    if not job_dir.exists():
        return stopped
    for trial in job_dir.iterdir():
        if not trial.is_dir() or not re.fullmatch(r"[A-Za-z0-9_-]+__[A-Za-z0-9]+", trial.name):
            continue
        project = trial.name.lower() + "__env"
        found = subprocess.run(
            ["docker", "ps", "-q", "--filter", f"label=com.docker.compose.project={project}"],
            capture_output=True, text=True, check=True,
        )
        for container_id in found.stdout.splitlines():
            if not re.fullmatch(r"[0-9a-f]{12,64}", container_id):
                raise ValueError("Docker returned an unexpected container ID")
            subprocess.run(["docker", "stop", container_id], check=True)
            stopped.append(container_id)
    return stopped
