"""Fail-closed planning for candidate-independent benchmark coverage.

Planning is deliberately separate from execution: an unfrozen catalog must never
start paid work or qualify a candidate.
"""

from __future__ import annotations

from typing import Any
import json
from pathlib import Path
from collections.abc import Callable
from datetime import datetime, timezone


def expected_coverage(suites: dict[str, Any], suite_name: str, attempts: int) -> set[tuple[str, str, int]]:
    if type(attempts) is not int or attempts < 1:
        raise ValueError("attempts must be a positive integer")
    suite = suites.get("suites", {}).get(suite_name)
    if not isinstance(suite, dict):
        raise ValueError(f"unknown suite: {suite_name}")
    coverage = set()
    for harness, config in suite.items():
        if harness == "seed":
            continue
        tasks = config.get("task_ids")
        if not isinstance(tasks, list) or not tasks or len(tasks) != config.get("count"):
            raise ValueError(f"{suite_name}.{harness}: task catalog is not frozen")
        if not all(isinstance(task, str) and task for task in tasks) or len(set(tasks)) != len(tasks):
            raise ValueError(f"{suite_name}.{harness}: task IDs must be unique nonempty strings")
        revision = config.get("source_revision", "")
        if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
            raise ValueError(f"{suite_name}.{harness}: immutable source revision is required")
        required = config.get("required_ids", [])
        if not set(required).issubset(tasks):
            raise ValueError(f"{suite_name}.{harness}: required tasks are missing")
        coverage.update((harness, task, attempt) for task in tasks for attempt in range(1, attempts + 1))
    return coverage


def build_suite_plan(suites: dict[str, Any], suite_name: str, attempts: int, candidate_ids: list[str]) -> list[dict[str, Any]]:
    """Return sequential one-task/one-attempt units; never launch a subprocess."""
    if not candidate_ids or not all(isinstance(candidate, str) and candidate.strip() for candidate in candidate_ids) or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate IDs must be nonempty and unique")
    coverage = expected_coverage(suites, suite_name, attempts)
    return [
        {"candidate_id": candidate, "harness_id": harness, "task_id": task, "attempt": attempt}
        for candidate in candidate_ids
        for harness, task, attempt in sorted(coverage)
    ]


def execute_suite_plan(plan: list[dict[str, Any]], runners: dict[str, Callable], directory: Path, *, contract_revision: int, evidence_kind: str | None = None, budget_scope: dict | None = None, continue_on_candidate_error: bool = False) -> list[dict[str, Any]]:
    """Sequentially supervise registered one-attempt runners; never skip unsupported units.

    Each runner owns its process/container cleanup and budget authorization. An
    By default any infrastructure/provider/invalid failure stops the suite.
    Opt-in candidate failover skips explicitly classified candidate failures,
    retaining exclusions without retries. Invalid evidence remains fatal.
    Existing directories are rejected rather than silently resumed or overwritten.
    """
    if type(contract_revision) is not int or contract_revision < 1:
        raise ValueError("contract revision must be a positive integer")
    if evidence_kind not in {None, "diagnostic", "benchmark"}:
        raise ValueError("invalid suite evidence kind")
    if not plan:
        raise ValueError("execution plan must not be empty")
    identities = set()
    for unit in plan:
        if not isinstance(unit, dict) or any(
            not isinstance(unit.get(key), str) or not unit[key].strip()
            for key in ("candidate_id", "harness_id", "task_id")
        ) or type(unit.get("attempt")) is not int or unit["attempt"] < 1:
            raise ValueError("invalid execution unit")
        identity = tuple(unit[key] for key in ("candidate_id", "harness_id", "task_id", "attempt"))
        if identity in identities:
            raise ValueError("duplicate execution unit")
        identities.add(identity)
    missing = sorted({unit["harness_id"] for unit in plan} - set(runners))
    if missing:
        raise ValueError("live runners not verified/registered: " + ", ".join(missing))
    if any(not callable(runners[unit["harness_id"]]) for unit in plan):
        raise ValueError("registered runners must be callable")
    if directory.exists():
        raise ValueError("suite artifacts already exist; choose a new run-id")
    directory.mkdir(parents=True)
    manifest = {"contract_revision": contract_revision, "expected_units": len(plan), "plan": [dict(unit) for unit in plan], "completed_units": 0, "status": "running", "units": []}
    manifest.update(started_at=datetime.now(timezone.utc).isoformat(), finished_at=None)
    if budget_scope is not None:
        manifest["budget_scope"] = budget_scope
    if evidence_kind is not None:
        manifest["evidence_kind"] = evidence_kind
    def retain():
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        (directory / "suite-state.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    retain()
    results = []
    excluded_candidates = set()
    manifest["candidate_error_policy"] = "skip_candidate" if continue_on_candidate_error else "stop_suite"
    manifest["candidate_errors"] = []
    manifest["skipped_units"] = []
    active = None
    try:
        for index, unit in enumerate(plan, 1):
            if unit["candidate_id"] in excluded_candidates:
                manifest["skipped_units"].append({**unit, "index": index, "reason": "prior_candidate_error"})
                retain()
                continue
            active = {**unit, "index": index}
            manifest["active_unit"] = active
            retain()
            try:
                record = runners[unit["harness_id"]](unit, index)
            except Exception as exc:
                if not continue_on_candidate_error or not getattr(exc, "candidate_failure", False):
                    raise
                from evaluation.secrets import redact
                manifest["candidate_errors"].append({**active, "error": redact(str(exc)), "stop_class": getattr(exc, "stop_class", "execution_error"), "artifact_references": getattr(exc, "artifact_references", [])})
                excluded_candidates.add(unit["candidate_id"])
                retain()
                continue
            if not isinstance(record, dict) or not isinstance(record.get("run"), dict) or not isinstance(record.get("outcome"), dict):
                raise ValueError("runner returned malformed evidence")
            run_id = record["run"].get("id")
            status = record["outcome"].get("status")
            if not isinstance(run_id, str) or not run_id.strip() or not isinstance(status, str) or not status:
                raise ValueError("runner returned malformed evidence")
            if (
                record.get("model", {}).get("candidate_id") != unit["candidate_id"]
                or record.get("harness", {}).get("id") != unit["harness_id"]
                or record.get("harness", {}).get("task_subset") != [unit["task_id"]]
                or record["outcome"].get("attempt") != unit["attempt"]
                or record["run"].get("contract_revision") != contract_revision
            ):
                raise ValueError("runner evidence does not match execution unit or contract")
            if any(previous["run_id"] == run_id for previous in manifest["units"]):
                raise ValueError("runner reused a trial run ID")
            if evidence_kind is not None and record["run"].get("evidence_kind") != evidence_kind:
                raise ValueError("runner evidence kind differs from the suite")
            results.append(record)
            manifest["units"].append({**unit, "run_id": record["run"]["id"], "status": status})
            manifest["completed_units"] += 1
            retain()
            if status not in {"passed", "failed"}:
                if continue_on_candidate_error and status == "provider_error":
                    excluded_candidates.add(unit["candidate_id"])
                    manifest["candidate_errors"].append({**active, "stop_class": status, "run_id": run_id})
                    retain()
                    continue
                manifest["status"] = "stopped_on_non_model_error"
                return results
        manifest["status"] = "completed_with_exclusions" if excluded_candidates else "completed"
        manifest["active_unit"] = None
        return results
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        raise
    except Exception as exc:
        manifest["status"] = "execution_error"
        from evaluation.secrets import redact
        manifest["failed_unit"] = active
        manifest["error"] = redact(str(exc))
        manifest["stop_class"] = getattr(exc, "stop_class", "execution_error")
        manifest["artifact_references"] = getattr(exc, "artifact_references", [])
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        retain()
