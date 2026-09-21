"""Contracts and small utilities for donor-foundation evaluation.

The benchmark engines stay in their upstream repositories. This module only validates Arcus's
frozen inputs and produces a common result skeleton.
"""

from __future__ import annotations

import json
import hashlib
import re
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from evaluation.runtime import validate_lock_index


FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
ALLOWED_LICENSES = {"Apache-2.0", "MIT"}


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object, rejecting a non-object top level."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def validate_candidates(registry: dict[str, Any]) -> list[str]:
    """Return human-readable violations of the candidate-registry contract."""
    errors: list[str] = []
    if registry.get("schema_version") != 1:
        errors.append("candidates: schema_version must be 1")
    candidates = registry.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return errors + ["candidates: candidates must be a non-empty list"]

    seen: set[str] = set()
    controls = 0
    for index, candidate in enumerate(candidates):
        prefix = f"candidates[{index}]"
        if not isinstance(candidate, dict):
            errors.append(f"{prefix}: must be an object")
            continue
        candidate_id = candidate.get("id")
        if not isinstance(candidate_id, str) or not candidate_id:
            errors.append(f"{prefix}.id: must be a non-empty string")
        elif candidate_id in seen:
            errors.append(f"{prefix}.id: duplicate {candidate_id!r}")
        else:
            seen.add(candidate_id)
        if candidate.get("role") == "api_comparator":
            if candidate.get("eligible") is not False or candidate.get("revision") != "api-unversioned" or candidate.get("huggingface_repo") != "":
                errors.append(f"{prefix}: API comparators must be ineligible and explicitly unversioned")
            if "/" not in str(candidate.get("openrouter_model", "")):
                errors.append(f"{prefix}.openrouter_model: must be provider/model")
            continue
        if not FULL_SHA.fullmatch(str(candidate.get("revision", ""))):
            errors.append(f"{prefix}.revision: must be a full 40-character commit SHA")
        if candidate.get("license") not in ALLOWED_LICENSES:
            errors.append(f"{prefix}.license: must be Apache-2.0 or MIT")
        if "/" not in str(candidate.get("huggingface_repo", "")):
            errors.append(f"{prefix}.huggingface_repo: must be owner/repository")
        if "/" not in str(candidate.get("openrouter_model", "")):
            errors.append(f"{prefix}.openrouter_model: must be provider/model")
        if candidate.get("role") == "conversion_control":
            controls += 1
        for field in ("total_parameters_billions", "active_parameters_billions"):
            value = candidate.get(field)
            if not isinstance(value, (int, float)) or value <= 0:
                errors.append(f"{prefix}.{field}: must be positive")
        total = candidate.get("total_parameters_billions")
        active = candidate.get("active_parameters_billions")
        if isinstance(total, (int, float)) and isinstance(active, (int, float)) and active > total:
            errors.append(f"{prefix}: active parameters cannot exceed total parameters")
    if controls != 1:
        errors.append("candidates: exactly one conversion_control is required")
    return errors


def validate_protocol(protocol: dict[str, Any]) -> list[str]:
    """Return violations of the shared evaluation-protocol contract."""
    errors: list[str] = []
    if protocol.get("schema_version") != 1:
        errors.append("protocol: schema_version must be 1")
    revision = protocol.get("contract_revision")
    if type(revision) is not int or revision < 1:
        errors.append("protocol.contract_revision must be a positive integer")
    elif revision >= 4 and protocol.get("search", {}).get("bfcl_web_search_variant") != "web_search_base":
        errors.append("protocol.search.bfcl_web_search_variant must match the approved web_search_base contract")
    if type(revision) is int and revision >= 9 and protocol.get("search", {}).get("page_error_policy") != "http_3xx_4xx_except_408_429_tool_error":
        errors.append("protocol.search.page_error_policy must match the revision-9 fetch contract")
    provider = protocol.get("provider", {})
    if provider.get("allow_fallbacks") is not False:
        errors.append("protocol.provider.allow_fallbacks must be false")
    if provider.get("require_exact_upstream_provider") is not True:
        errors.append("protocol.provider.require_exact_upstream_provider must be true")
    generation = protocol.get("generation", {})
    for field, lower, upper in (("temperature", 0.0, 2.0), ("top_p", 0.0, 1.0)):
        value = generation.get(field)
        if not isinstance(value, (int, float)) or not lower <= value <= upper:
            errors.append(f"protocol.generation.{field} is missing or out of range")
    required_positive = (
        "max_output_tokens",
        "tool_call_budget",
        "max_agent_iterations",
        "wall_time_seconds",
        "agent_setup_timeout_seconds",
        "verifier_timeout_seconds",
        "trial_timeout_seconds",
        "concurrency",
        "attempts",
    )
    for field in required_positive:
        value = generation.get(field)
        if not isinstance(value, (int, float)) or value <= 0:
            errors.append(f"protocol.generation.{field} must be positive")
    if generation.get("concurrency") != 1:
        errors.append("protocol.generation.concurrency must be 1 for isolated scored trials")
    if generation.get("retry_on_model_failure") != 0:
        errors.append("protocol.generation.retry_on_model_failure must be 0")
    if generation.get("retry_on_provider_error") != 1:
        errors.append("protocol.generation.retry_on_provider_error must be 1")
    if generation.get("trial_timeout_seconds", 0) < sum(
        float(generation.get(field, 0))
        for field in (
            "wall_time_seconds",
            "agent_setup_timeout_seconds",
            "verifier_timeout_seconds",
        )
    ):
        errors.append("protocol.generation.trial_timeout_seconds is too small")

    harnesses = protocol.get("harnesses")
    if not isinstance(harnesses, list) or not harnesses:
        return errors + ["protocol.harnesses must be a non-empty list"]
    seen: set[str] = set()
    weight = 0.0
    for index, harness in enumerate(harnesses):
        prefix = f"protocol.harnesses[{index}]"
        if not isinstance(harness, dict):
            errors.append(f"{prefix}: must be an object")
            continue
        harness_id = harness.get("id")
        if harness_id in seen:
            errors.append(f"{prefix}.id: duplicate {harness_id!r}")
        seen.add(harness_id)
        revision = str(harness.get("revision", ""))
        if harness.get("repository") != "local" and not FULL_SHA.fullmatch(revision):
            errors.append(f"{prefix}.revision: must be a full immutable commit SHA")
        value = harness.get("weight")
        if not isinstance(value, (int, float)) or value < 0:
            errors.append(f"{prefix}.weight: must be non-negative")
        else:
            weight += float(value)
    if abs(weight - 1.0) > 1e-9:
        errors.append(f"protocol.harnesses: weights must sum to 1.0, got {weight:g}")
    return errors


def validate_phase1_links(
    candidates: dict[str, Any],
    protocol: dict[str, Any],
    providers: dict[str, Any],
    parsers: dict[str, Any],
    suites: dict[str, Any],
) -> list[str]:
    """Validate cross-file identity, budget, and suite invariants."""
    errors: list[str] = []
    candidate_ids = {item["id"] for item in candidates.get("candidates", [])}
    provider_items = providers.get("providers", [])
    provider_ids = {item.get("candidate_id") for item in provider_items}
    parser_ids = {item.get("candidate_id") for item in parsers.get("parsers", [])}
    if provider_ids != candidate_ids:
        errors.append("providers: candidate IDs must exactly match candidates.json")
    if parser_ids != candidate_ids:
        errors.append("parsers: candidate IDs must exactly match candidates.json")
    candidate_models = {item["id"]: item["openrouter_model"] for item in candidates["candidates"]}
    for item in provider_items:
        if candidate_models.get(item.get("candidate_id")) != item.get("openrouter_model"):
            errors.append(f"providers: model mismatch for {item.get('candidate_id')}")
        for field in ("prompt_per_million", "completion_per_million", "smoke_cap"):
            if type(item.get(field)) not in (int, float) or not math.isfinite(item[field]) or item[field] < 0:
                errors.append(f"providers: {field} must be non-negative for {item.get('candidate_id')}")
        for field in ("context_length", "max_completion_tokens"):
            if field in item and (type(item[field]) is not int or item[field] < 1):
                errors.append(f"providers: {field} must be a positive integer")
        for tier in item.get("pricing_tiers", []):
            if type(tier.get("min_prompt_tokens")) is not int or tier["min_prompt_tokens"] < 0 or any(type(tier.get(field)) not in (int, float) or not math.isfinite(tier[field]) or tier[field] < 0 for field in ("prompt_per_million", "completion_per_million")):
                errors.append("providers: malformed pricing tier")
        if not str(item.get("provider_slug", "")).strip():
            errors.append(f"providers: missing provider_slug for {item.get('candidate_id')}")
    cap_sum = sum(float(item.get("smoke_cap", 0)) for item in provider_items)
    aggregate = float(providers.get("aggregate_smoke_cap", -1))
    if abs(cap_sum - aggregate) > 1e-9 or abs(aggregate - 48.0) > 1e-9:
        errors.append("providers: approved per-model caps and aggregate cap must sum to $48")

    harness_ids = {item["id"] for item in protocol.get("harnesses", []) if item["id"] != "efficiency"}
    suite_map = suites.get("suites", {})
    for suite_name in ("smoke", "qualification"):
        suite = suite_map.get(suite_name)
        if not isinstance(suite, dict):
            errors.append(f"suites: missing {suite_name}")
            continue
        suite_ids = set(suite) - {"seed"}
        if suite_ids != harness_ids:
            errors.append(f"suites.{suite_name}: harness IDs must match protocol")
        for harness_id in suite_ids:
            if not isinstance(suite[harness_id].get("count"), int) or suite[harness_id]["count"] <= 0:
                errors.append(f"suites.{suite_name}.{harness_id}.count must be positive")
            task_ids = suite[harness_id].get("task_ids")
            if task_ids is None:
                errors.append(f"suites.{suite_name}.{harness_id}: task IDs must be frozen")
            if not FULL_SHA.fullmatch(str(suite[harness_id].get("source_revision", ""))):
                errors.append(f"suites.{suite_name}.{harness_id}: immutable source revision required")
            if task_ids is not None:
                if not isinstance(task_ids, list) or len(task_ids) != suite[harness_id].get("count"):
                    errors.append(
                        f"suites.{suite_name}.{harness_id}.task_ids must match count"
                    )
                elif not all(
                    isinstance(task_id, str) and task_id for task_id in task_ids
                ) or len(set(task_ids)) != len(task_ids):
                    errors.append(
                        f"suites.{suite_name}.{harness_id}.task_ids must be unique strings"
                    )
    return errors


def select_task_ids(task_ids: list[str], *, seed: str, count: int) -> list[str]:
    """Select a candidate-independent deterministic subset from upstream task IDs."""
    if count <= 0:
        raise ValueError("count must be positive")
    unique = sorted(set(task_ids))
    if count > len(unique):
        raise ValueError(f"requested {count} tasks, but upstream exposed only {len(unique)}")
    return sorted(
        unique,
        key=lambda task_id: hashlib.sha256(f"{seed}\0{task_id}".encode()).hexdigest(),
    )[:count]


def validate_result(record: dict[str, Any]) -> list[str]:
    """Validate the fields required before a generated template becomes evidence."""
    errors: list[str] = []
    for section in ("run", "model", "provider", "harness", "generation", "outcome", "artifacts"):
        if not isinstance(record.get(section), dict):
            errors.append(f"result: missing object {section}")
    if errors:
        return errors
    if record.get("schema_version") != 1:
        errors.append("result: schema_version must be 1")
    if "budget_scope" in record["run"] and (record["run"]["budget_scope"] not in {"diagnostic", "final-smoke", "six-model-token-restart"} or "budget_scope" not in record["artifacts"]):
        errors.append("result.run.budget_scope requires a known identity and retained artifact")
    if "execution_fingerprint" in record["run"]:
        fingerprint = record["run"]["execution_fingerprint"]
        if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint):
            errors.append("result.run.execution_fingerprint must be a SHA256")
    outcome = record["outcome"]
    if outcome.get("status") not in {"passed", "failed", "infrastructure_error", "provider_error", "parser_error", "harness_error", "invalid_run", "interrupted"}:
        errors.append("result.outcome.status is unknown")
    for field in ("score", "cost_usd", "latency_seconds"):
        value = outcome.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            errors.append(f"result.outcome.{field} must be finite and nonnegative")
    if isinstance(outcome.get("score"), (int, float)) and outcome["score"] > 1:
        errors.append("result.outcome.score cannot exceed 1")
    if type(outcome.get("attempt")) is not int or outcome["attempt"] < 1:
        errors.append("result.outcome.attempt must be a positive integer")
    if outcome.get("status") in {"passed", "failed"} and outcome.get("passed") is not (outcome["status"] == "passed"):
        errors.append("result.outcome.passed conflicts with status")
    if not record["run"].get("finished_at"):
        errors.append("result.run.finished_at is required")
    if not record["provider"].get("price_snapshot_at"):
        errors.append("result.provider.price_snapshot_at is required")
    if not record["harness"].get("task_subset"):
        errors.append("result.harness.task_subset cannot be empty")
    if record["outcome"].get("status") in {"passed", "failed"}:
        expected_trials = record["run"].get("expected_trials")
        if not isinstance(expected_trials, int) or expected_trials < 1:
            errors.append("result.run.expected_trials must be a positive integer")
        if record["run"].get("completed_trials") != record["run"].get("expected_trials"):
            errors.append("result.run completed trial count does not match expected trial count")
        reason = record["run"].get("termination_reason")
        verified_budget_failure = reason == "model_iteration_budget_exhausted" and record["outcome"].get("status") == "failed" and record["outcome"].get("failure_class") == "model" and all(record["artifacts"].get(key) for key in ("budget_stop", "gateway_errors"))
        if reason is not None and not verified_budget_failure:
            errors.append("result.run.termination_reason must be null for scored evidence")
        if record["provider"].get("verified") is not True:
            errors.append("result.provider.verified must be true")
        if record["generation"].get("contract_verified") is not True:
            errors.append("result.generation.contract_verified must be true")
    if record["outcome"].get("failure_class") == "unrun_template":
        errors.append("result is still an unrun template")
    for key in ("raw_requests", "raw_responses", "tool_events", "verifier_results"):
        if not record["artifacts"].get(key):
            errors.append(f"result.artifacts.{key} is required")
    if not record["artifacts"].get("sha256"):
        errors.append("result.artifacts.sha256 cannot be empty")
    return errors


def validate_artifact_hashes(record: dict[str, Any], root: Path) -> list[str]:
    """Verify retained evidence without accepting paths outside evaluation/runs."""
    errors = []
    artifacts = record.get("artifacts", {})
    hashes = artifacts.get("sha256", {})
    runs = (root / "evaluation" / "runs").resolve()
    supplemental = [key for key in ("search_discovery", "search_events", "fetch_events", "budget_stop", "gateway_errors", "budget_scope") if key in artifacts]
    for key in ("raw_requests", "raw_responses", "tool_events", "verifier_results", *supplemental):
        label = artifacts.get(key)
        if not isinstance(label, str) or not label:
            errors.append(f"artifact {key}: missing path")
            continue
        path = Path(label)
        path = (path if path.is_absolute() else root / path).resolve()
        if runs not in path.parents or not path.is_file():
            errors.append(f"artifact {key}: must be a retained file below evaluation/runs")
            continue
        digest = hashes.get(key, hashes.get(label))
        if digest is None:
            # Importers historically key the map by resolved artifact path.
            digest = hashes.get(str(path))
        if digest != hashlib.sha256(path.read_bytes()).hexdigest():
            errors.append(f"artifact {key}: SHA256 missing or mismatched")
    return errors


def validate_control_files(root: Path) -> list[str]:
    """Validate all committed Phase 1 control files under *root*."""
    evaluation = root / "evaluation"
    candidates = load_json(evaluation / "candidates.json")
    protocol = load_json(evaluation / "protocol.json")
    providers = load_json(evaluation / "providers.json")
    parsers = load_json(evaluation / "parsers.json")
    suites = load_json(evaluation / "suites.json")
    from evaluation.tavily import validate_search_config
    search = load_json(evaluation / "search.json")
    schema = load_json(evaluation / "result.schema.json")
    errors = validate_candidates(candidates) + validate_protocol(protocol)
    from evaluation.budget_scope import load_scope
    from evaluation.provider import EvaluationBlocked
    try:
        load_scope(root, "diagnostic")
        load_scope(root, "final-smoke", require_authorized=False)
    except (ValueError, OSError, KeyError, EvaluationBlocked) as exc:
        errors.append("budget scope control: " + str(exc))
    from evaluation.bfcl import load_answer_overrides
    try:
        load_answer_overrides(root, protocol)
    except (ValueError, OSError, KeyError) as exc:
        errors.append("BFCL correction control: " + str(exc))
    errors += validate_search_config(search)
    if protocol.get("search", {}).get("contract_id") != search.get("contract_id"):
        errors.append("protocol search contract differs from search.json")
    if protocol.get("contract_revision", 0) >= 9 and protocol["search"].get("page_error_policy") != search.get("page_error_policy"):
        errors.append("protocol page-error policy differs from search.json")
    errors += validate_phase1_links(candidates, protocol, providers, parsers, suites)
    pinned = {item["id"]: item for item in protocol["harnesses"] if item["id"] != "efficiency"}
    errors += validate_lock_index(root, load_json(evaluation / "requirements-harnesses.lock"), {key: value["revision"] for key, value in pinned.items()})
    harness_files = {
        item["id"]: item
        for path in (evaluation / "harnesses").glob("*.json")
        for item in [load_json(path)]
    }
    if set(harness_files) != set(pinned):
        errors.append("harnesses: config IDs must exactly match non-derived protocol harnesses")
    for harness_id, item in harness_files.items():
        expected = pinned.get(harness_id)
        if expected and item.get("revision") != expected.get("revision"):
            errors.append(f"harnesses: revision mismatch for {harness_id}")
        if harness_id == "terminal":
            for suite_name, field in (("smoke", "smoke_task_ids"), ("qualification", "qualification_task_ids")):
                frozen = suites.get("suites", {}).get(suite_name, {}).get("terminal", {}).get("task_ids", [])
                if set(item.get(field, [])) != set(frozen):
                    errors.append(f"harnesses: Harbor {suite_name} task IDs differ from frozen suite")
            required_kwargs = item.get("required_agent_kwargs", {})
            generation = protocol["generation"]
            expected_kwargs = {
                "version": item.get("agent_version"),
                "temperature": str(generation["temperature"]),
                "top_p": str(generation["top_p"]),
                "max_iterations": generation["max_agent_iterations"],
                "num_retries": generation["retry_on_model_failure"],
            }
            if required_kwargs != expected_kwargs:
                errors.append("harnesses: Harbor agent kwargs must match protocol exactly")
            if item.get("execution_unit") != "one task attempt per isolated proxy":
                errors.append("harnesses: Harbor execution must isolate each task attempt")
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        errors.append("result schema: expected JSON Schema draft 2020-12")
    if schema.get("additionalProperties") is not False:
        errors.append("result schema: top-level additionalProperties must be false")
    return errors


def build_result_skeleton(
    registry: dict[str, Any],
    protocol: dict[str, Any],
    *,
    candidate_id: str,
    harness_id: str,
    upstream_provider: str,
    precision: str,
    parser: str,
    run_id: str,
    seed: int = 0,
) -> dict[str, Any]:
    """Build a normalized, revision-pinned template for one external-harness attempt."""
    candidates = {item["id"]: item for item in registry["candidates"]}
    harnesses = {item["id"]: item for item in protocol["harnesses"]}
    if candidate_id not in candidates:
        raise ValueError(f"unknown candidate: {candidate_id}")
    if harness_id not in harnesses:
        raise ValueError(f"unknown harness: {harness_id}")
    for label, value in (
        ("upstream_provider", upstream_provider),
        ("precision", precision),
        ("parser", parser),
        ("run_id", run_id),
    ):
        if not value.strip():
            raise ValueError(f"{label} must not be empty")

    candidate = candidates[candidate_id]
    harness = harnesses[harness_id]
    now = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": 1,
        "run": {
            "id": run_id,
            "contract_revision": protocol.get("contract_revision", 1),
            "started_at": now,
            "finished_at": "",
            "seed": seed,
            "expected_trials": 1,
            "completed_trials": 0,
            "termination_reason": "unrun_template",
        },
        "model": {
            "candidate_id": candidate_id,
            "donor_eligible": candidate.get("eligible", False),
            "huggingface_repo": candidate["huggingface_repo"],
            "revision": candidate["revision"],
            "precision": precision,
            "parser": parser,
        },
        "provider": {
            "gateway": protocol["provider"]["gateway"],
            "upstream": upstream_provider,
            "provider_slug": "unresolved",
            "model": candidate["openrouter_model"],
            "quantization": precision,
            "price_snapshot_at": "",
            "verified": False,
        },
        "search": {
            "provider": "tavily",
            "contract_id": protocol.get("search", {}).get("contract_id", "unconfigured"),
            "used": False,
            "tool_calls": 0,
            "credits_upper_bound": 0,
            "official_bfcl_comparable": False,
        },
        "harness": {
            "id": harness_id,
            "revision": harness["revision"],
            "dataset": harness["dataset"],
            "task_subset": [],
        },
        "generation": {**protocol["generation"], "contract_verified": False},
        "outcome": {
            "status": "infrastructure_error",
            "passed": False,
            "score": 0.0,
            "failure_class": "unrun_template",
            "tokens": 0,
            "tool_calls": 0,
            "latency_seconds": 0.0,
            "cost_usd": 0.0,
            "attempt": 1,
            "valid_tool_calls": 0,
            "wrong_tool_calls": 0,
            "missed_tool_calls": 0,
            "unnecessary_tool_calls": 0,
        },
        "artifacts": {
            "raw_requests": "",
            "raw_responses": "",
            "tool_events": "",
            "verifier_results": "",
            "sha256": {},
        },
    }
