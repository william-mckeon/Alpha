"""Deterministic candidate scorecard calculation."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Iterable


NON_MODEL_ERRORS = {
    "infrastructure_error",
    "provider_error",
    "parser_error",
    "harness_error",
    "invalid_run",
    "interrupted",
}


def score_candidates(
    records: Iterable[dict[str, Any]], weights: dict[str, float],
    *, expected: set[tuple[str, str, int]] | None = None,
) -> list[dict[str, Any]]:
    """Rank candidates while excluding non-model failures from score denominators."""
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    errors: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    cost: dict[str, float] = defaultdict(float)
    observed: dict[str, set[tuple[str, str, int]]] = defaultdict(set)
    successes: dict[str, int] = defaultdict(int)
    run_ids: set[tuple[str, str]] = set()
    contracts: set[tuple[Any, Any]] = set()
    search_credits: dict[str, int] = defaultdict(int)
    for record in records:
        candidate = record["model"]["candidate_id"]
        harness = record["harness"]["id"]
        outcome = record["outcome"]
        amount = float(outcome.get("cost_usd", 0.0))
        score = float(outcome.get("score", 0.0))
        if not math.isfinite(amount) or amount < 0 or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("cost and score must be finite; cost nonnegative and score within [0,1]")
        run_id = record.get("run", {}).get("id")
        if run_id:
            identity = (candidate, run_id)
            if identity in run_ids:
                raise ValueError(f"duplicate run: {identity}")
            run_ids.add(identity)
        cost[candidate] += amount
        search_credits[candidate] += int(record.get("search", {}).get("credits_upper_bound", 0))
        if record.get("run", {}).get("evidence_kind") == "diagnostic":
            errors[candidate]["diagnostic_only"] += 1
            continue
        status = outcome["status"]
        if status in NON_MODEL_ERRORS:
            errors[candidate][status] += 1
            continue
        if status not in {"passed", "failed"}:
            raise ValueError(f"unknown outcome status cannot be scored: {status}")
        contracts.add((record.get("run", {}).get("contract_revision"), record.get("search", {}).get("contract_id")))
        if len(contracts) > 1:
            raise ValueError("cannot mix evaluation/search contract versions")
        if expected is not None:
            tasks = record["harness"].get("task_subset", [])
            if len(tasks) != 1:
                raise ValueError("scoring requires one task per normalized attempt")
            key = (harness, tasks[0], outcome.get("attempt"))
            if key not in expected:
                raise ValueError(f"unexpected task/attempt: {key}")
            if key in observed[candidate]:
                raise ValueError(f"duplicate scored task/attempt: {key}")
            observed[candidate].add(key)
        grouped[candidate][harness].append(score)
        successes[candidate] += int(status == "passed" and score == 1.0)

    rows = []
    task_weights = {key: value for key, value in weights.items() if key != "efficiency"}
    for candidate in sorted(set(grouped) | set(errors)):
        missing = sorted(harness for harness in task_weights if not grouped[candidate].get(harness))
        components = {
            harness: sum(values) / len(values)
            for harness, values in grouped[candidate].items()
            if values
        }
        missing_attempts = sorted(expected - observed[candidate]) if expected is not None else []
        rows.append(
            {
                "candidate_id": candidate,
                "qualification_complete": expected is not None and not missing and not missing_attempts,
                "weighted_score": None,
                "components": components,
                "missing_harnesses": missing,
                "non_model_errors": dict(errors[candidate]),
                "cost_usd": cost[candidate],
                "search_credits_upper_bound": search_credits[candidate],
                "verified_successes": successes[candidate],
                "missing_task_attempts": [list(key) for key in missing_attempts],
                "coverage_verified": expected is not None,
            }
        )
    efficiencies = {
        row["candidate_id"]: (
            row["verified_successes"] / row["cost_usd"] if row["cost_usd"] > 0 else 0.0
        )
        for row in rows
    }
    best_efficiency = max(efficiencies.values(), default=0.0)
    for row in rows:
        efficiency = efficiencies[row["candidate_id"]] / best_efficiency if best_efficiency else 0.0
        row["components"]["efficiency"] = efficiency
        if row["qualification_complete"]:
            row["weighted_score"] = sum(
                row["components"].get(harness, 0.0) * weight
                for harness, weight in weights.items()
            )
    rows.sort(
        key=lambda row: (
            row["weighted_score"] is not None,
            row["weighted_score"] if row["weighted_score"] is not None else -1.0,
        ),
        reverse=True,
    )
    return rows
