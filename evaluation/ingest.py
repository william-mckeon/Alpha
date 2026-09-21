"""Normalize objective upstream-harness outcomes without replacing their verifiers."""

from __future__ import annotations

from typing import Any
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from evaluation.control import build_result_skeleton
from evaluation.bfcl import frozen_case, load_answer_overrides
from evaluation.harbor import audit_harbor_job_config, load_harbor_result_state
from evaluation.budget_scope import attach_scope
from evaluation.execution import source_fingerprint
import math


FAILURE_STATUSES = {
    "model": "failed",
    "provider": "provider_error",
    "parser": "parser_error",
    "infrastructure": "infrastructure_error",
    "harness": "harness_error",
    "protocol": "invalid_run",
    "interrupted": "interrupted",
}


def read_swebench_verifier(report_path: Path, task_id: str) -> dict[str, Any]:
    """Extract one official verdict; this alone is not eligible normalized evidence."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    groups = {name: report.get(name, []) for name in ("resolved_ids", "unresolved_ids", "empty_patch_ids", "error_ids", "incomplete_ids")}
    if not all(isinstance(values, list) and all(isinstance(value, str) for value in values) for values in groups.values()):
        raise ValueError("SWE-bench report contains malformed instance groups")
    if task_id in groups["error_ids"] or task_id in groups["incomplete_ids"]:
        return {"status": "harness_error", "passed": False, "score": 0.0}
    if task_id in groups["resolved_ids"]:
        if task_id in groups["unresolved_ids"] or task_id in groups["empty_patch_ids"]:
            raise ValueError("SWE-bench report contains conflicting verdicts")
        return {"status": "passed", "passed": True, "score": 1.0}
    if task_id in groups["unresolved_ids"] or task_id in groups["empty_patch_ids"]:
        return {"status": "failed", "passed": False, "score": 0.0}
    raise ValueError("SWE-bench report does not contain the requested task verdict")


def read_bfcl_verifier(score_path: Path, responses_path: Path, task_id: str) -> dict[str, Any]:
    responses = [json.loads(line) for line in responses_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(responses) != 1 or responses[0].get("id") != task_id:
        raise ValueError("BFCL import requires exactly one matching official response ID")
    scores = [json.loads(line) for line in score_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not scores:
        raise ValueError("BFCL official score file is empty")
    header = scores[0]
    accuracy = header.get("accuracy")
    correct = header.get("correct_count")
    if type(header.get("total_count")) is not int or header.get("total_count") != 1 or type(correct) is not int or correct not in (0, 1) or type(accuracy) not in (int, float) or not math.isfinite(accuracy) or accuracy != correct:
        raise ValueError("BFCL score header does not describe one complete attempt")
    failures = scores[1:]
    if len(failures) != 1 - correct or any(row.get("id") != task_id for row in failures):
        raise ValueError("BFCL score header and failure IDs disagree")
    if failures and "inference_error" in str(failures[0].get("error", failures[0])):
        return {"status": "harness_error", "passed": False, "score": 0.0}
    return {"status": "passed" if correct else "failed", "passed": bool(correct), "score": float(correct)}


def read_mcpmark_verifier(meta_path: Path, task_id: str) -> dict[str, Any]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    parts = task_id.split("/")
    if len(parts) != 3 or parts[0] != "filesystem" or meta.get("mcp") != "filesystem" or meta.get("task_name") != f"{parts[1]}__{parts[2]}":
        raise ValueError("MCPMark metadata does not match the frozen filesystem task")
    execution = meta.get("execution_result", {})
    success = execution.get("success")
    if type(success) is not bool:
        raise ValueError("MCPMark metadata omitted its official verifier result")
    error = execution.get("error_message")
    if error and "Max turns" not in error:
        return {"status": "harness_error", "passed": False, "score": 0.0}
    if not success and execution.get("verification_error") and execution.get("verification_output") is None:
        return {"status": "harness_error", "passed": False, "score": 0.0}
    return {"status": "passed" if success else "failed", "passed": success, "score": float(success)}


def classify_failure(
    *,
    provider_error: bool,
    parser_error: bool,
    infrastructure_error: bool,
    harness_error: bool = False,
    protocol_error: bool = False,
    interrupted: bool = False,
) -> str:
    """Classify failures in a stable precedence order separate from model failure."""
    if protocol_error:
        return "invalid_run"
    if interrupted:
        return "interrupted"
    if infrastructure_error:
        return "infrastructure_error"
    if harness_error:
        return "harness_error"
    if provider_error:
        return "provider_error"
    if parser_error:
        return "parser_error"
    return "failed"


def normalized_outcome(
    *,
    passed: bool,
    score: float,
    tokens: int,
    tool_calls: int,
    latency_seconds: float,
    cost_usd: float,
    failure_source: str | None = None,
) -> dict[str, Any]:
    """Create the common outcome section from an upstream objective verifier result."""
    if type(passed) is not bool or type(score) not in (int, float) or not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("score must be in [0, 1]")
    for name, value in (("tokens", tokens), ("tool_calls", tool_calls)):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} cannot be negative")
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in (latency_seconds, cost_usd)):
        raise ValueError("latency and cost cannot be negative")
    if failure_source is not None and failure_source != "model":
        passed, score = False, 0.0
    status = "passed" if passed else FAILURE_STATUSES.get(failure_source or "model", "failed")
    return {
        "status": status,
        "passed": passed,
        "score": score,
        "failure_class": None if passed else (failure_source or "model"),
        "tokens": tokens,
        "tool_calls": tool_calls,
        "latency_seconds": latency_seconds,
        "cost_usd": cost_usd,
    }


def audit_proxy_log(proxy_log: Path, *, trial_id: str, provider: dict, generation: dict) -> dict:
    """Audit retained gateway identity/settings/accounting, never default missing cost to zero."""
    exchanges = [json.loads(line) for line in proxy_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not exchanges:
        raise ValueError("proxy log contains no completed exchanges")
    tokens, tools, cost = 0, 0, 0.0
    for exchange in exchanges:
        request, response = exchange["request"], exchange["response"]
        if exchange.get("trial_id") != trial_id or request.get("model") != provider["openrouter_model"] or response.get("model") != provider["openrouter_model"]:
            raise ValueError("gateway trial or model identity differs from pin")
        if response.get("provider") != provider["upstream"]:
            raise ValueError("gateway upstream provider differs from pin")
        for field, expected in (("temperature", generation["temperature"]), ("top_p", generation["top_p"]), ("max_tokens", generation["max_output_tokens"]), ("stream", False)):
            if request.get(field) != expected or (field != "stream" and isinstance(request.get(field), bool)):
                raise ValueError("gateway generation settings differ from protocol")
        route = request.get("provider", {})
        if route.get("only") != [provider["provider_slug"]] or route.get("allow_fallbacks") is not False or route.get("require_parameters") is not True:
            raise ValueError("gateway provider route differs from pin")
        usage = response.get("usage", {})
        amount = usage.get("cost")
        if type(amount) not in (int, float) or not math.isfinite(amount) or amount < 0:
            raise ValueError("gateway omitted finite nonnegative actual cost")
        counts = [usage.get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")]
        if any(type(count) is not int or count < 0 for count in counts) or counts[0] + counts[1] != counts[2]:
            raise ValueError("gateway token accounting is missing or inconsistent")
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("gateway requires one completion per request")
        tools += len(choices[0].get("message", {}).get("tool_calls") or [])
        if exchange.get("tool_calls_total") != tools:
            raise ValueError("gateway retained tool count is inconsistent")
        tokens += counts[2]
        cost += amount
    if tools > generation["tool_call_budget"]:
        raise ValueError("gateway tool budget exceeded")
    errors_path = proxy_log.with_name("gateway-errors.jsonl")
    errors = [json.loads(line) for line in errors_path.read_text(encoding="utf-8").splitlines() if line.strip()] if errors_path.exists() else []
    if any(error.get("trial_id") != trial_id for error in errors):
        raise ValueError("gateway error evidence belongs to another trial")
    return {"tokens": tokens, "tool_calls": tools, "cost_usd": cost, "errors": errors, "exchanges": len(exchanges)}


def ingest_worker_verdict(*, root: Path, manifest_path: Path, proxy_log: Path, verifier_paths: list[Path], registry: dict, protocol: dict, providers: dict) -> dict:
    """Normalize a controlled worker's official verdict and independent gateway audit.

    Diagnostics remain diagnostic-only even when their objective verifier passes.
    Historical runs without a retained execution contract cannot be promoted.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "integration_passed" or manifest.get("protocol_snapshot") != protocol:
        raise ValueError("worker execution is incomplete or lacks the exact retained contract")
    if "execution_fingerprint" in manifest and manifest["execution_fingerprint"] != source_fingerprint(root):
        raise ValueError("worker evidence execution sources differ from current sources")
    candidate, task, harness_id = manifest["candidate_id"], manifest["task_id"], manifest["harness_id"]
    suites = json.loads((root / "evaluation/suites.json").read_text(encoding="utf-8"))["suites"]
    frozen = {value for suite in suites.values() for value in suite.get(harness_id, {}).get("task_ids", [])}
    if task not in frozen:
        raise ValueError("worker task is not in a frozen suite")
    harness = next(item for item in protocol["harnesses"] if item["id"] == harness_id)
    if manifest.get("harness_revision") != harness["revision"]:
        raise ValueError("worker source revision differs from frozen harness")
    runs = (root / "evaluation/runs").resolve()
    paths = [manifest_path, proxy_log, *verifier_paths]
    for path in paths:
        if not path.resolve().is_relative_to(runs) or not path.is_file():
            raise ValueError("worker evidence must remain below evaluation/runs")
    for label, digest in manifest.get("artifact_sha256", {}).items():
        path = (root / label).resolve()
        if not path.is_relative_to(runs) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("retained worker artifact hash is missing or mismatched")
    error_log = proxy_log.with_name("gateway-errors.jsonl")
    scope_log = proxy_log.with_name("budget-scope.json")
    required_paths = [proxy_log, *verifier_paths, *([error_log] if error_log.exists() else []), *([scope_log] if scope_log.exists() else [])]
    for path in required_paths:
        if str(path.resolve().relative_to(root.resolve())) not in manifest.get("artifact_sha256", {}):
            raise ValueError("official verdict or gateway evidence has no manifest hash")
    if manifest.get("worker_control_contract_required") is True:
        name = "worker-provenance.json" if harness_id == "mcp" else "controller-provenance.json"
        provenance_path = manifest_path.parent / name
        label = str(provenance_path.resolve().relative_to(root.resolve()))
        if label not in manifest.get("artifact_sha256", {}):
            raise ValueError("worker-loaded control contract lacks retained provenance")
        provenance = json.loads(provenance_path.read_text())
        if provenance.get("protocol_snapshot") != protocol:
            raise ValueError("worker-loaded controls differ from the retained host contract")
        if "execution_fingerprint" in manifest and provenance.get("execution_fingerprint") != manifest["execution_fingerprint"]:
            raise ValueError("worker-loaded execution sources differ from the host manifest")
    provider = next(item for item in providers["providers"] if item["candidate_id"] == candidate)
    audit = audit_proxy_log(proxy_log, trial_id=manifest["run_id"], provider=provider, generation=protocol["generation"])
    if audit["exchanges"] > protocol["generation"]["max_agent_iterations"]:
        raise ValueError("gateway model iteration ceiling exceeded")
    if harness_id == "function_calling" and len(verifier_paths) == 2:
        load_answer_overrides(root, protocol)
        if protocol.get("search", {}).get("answer_override"):
            policy_path = manifest_path.parent / "bfcl-policy-evaluate.json"
            label = str(policy_path.resolve().relative_to(root.resolve()))
            if label not in manifest.get("artifact_sha256", {}):
                raise ValueError("BFCL grading policy lacks hashed provenance")
            if json.loads(policy_path.read_text(encoding="utf-8")) != {"contract_revision": protocol["contract_revision"], "underscore_to_dot": True, "answer_override": protocol["search"]["answer_override"]}:
                raise ValueError("BFCL grading policy differs from frozen contract")
        official_task = manifest.get("official_task_id", task)
        _, expected_task = frozen_case(task, protocol)
        if official_task != expected_task:
            raise ValueError("official BFCL task differs from frozen variant mapping")
        verdict = read_bfcl_verifier(verifier_paths[0], verifier_paths[1], official_task)
    elif harness_id == "mcp" and len(verifier_paths) == 1:
        provenance_path = manifest_path.parent / "worker-provenance.json"
        label = str(provenance_path.resolve().relative_to(root.resolve()))
        if label not in manifest.get("artifact_sha256", {}):
            raise ValueError("MCP worker provenance is missing a retained hash")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("verifier_overlay", {}).get("timeout_seconds") != protocol["generation"]["verifier_timeout_seconds"]:
            raise ValueError("MCP worker lacks the frozen verifier deadline; retain raw diagnostic, not normalized contract evidence")
        if provenance.get("archive_sha256") != manifest.get("fixture_snapshot", {}).get("archive_sha256") or provenance.get("loop_overlay", {}).get("source_sha256") != "f2264f1af21cf6d616de38a87561b1b4c0f61539f369202305ee09b6dcab5736":
            raise ValueError("MCP fixture or policy-overlay provenance differs from pin")
        verdict = read_mcpmark_verifier(verifier_paths[0], task)
    elif harness_id == "repository" and len(verifier_paths) == 1:
        verdict = read_swebench_verifier(verifier_paths[0], task)
    else:
        raise ValueError("unsupported official verifier evidence")
    start, finish = (datetime.fromisoformat(manifest[key]) for key in ("started_at", "finished_at"))
    latency = (finish - start).total_seconds()
    if latency < 0 or latency > protocol["generation"]["trial_timeout_seconds"]:
        raise ValueError("retained trial duration violates the execution ceiling")
    record = build_result_skeleton(registry, protocol, candidate_id=candidate, harness_id=harness_id, upstream_provider=provider["upstream"], precision=provider["quantization"], parser="native-openai-tools", run_id=manifest["run_id"])
    diagnostic = manifest.get("benchmark_scored") is not True
    failure = "harness" if verdict["status"] == "harness_error" else None
    budget_errors = [error for error in audit["errors"] if error.get("category") == "model_budget"]
    if harness_id == "repository" and (manifest_path.parent / "budget-stop.json").exists() and not budget_errors:
        raise ValueError("worker budget termination lacks independent gateway evidence")
    if budget_errors:
        stop_path = manifest_path.parent / "budget-stop.json"
        label = str(stop_path.resolve().relative_to(root.resolve()))
        if harness_id != "repository" or label not in manifest.get("artifact_sha256", {}):
            raise ValueError("model budget stop requires retained repository termination proof")
        stop = json.loads(stop_path.read_text())
        if (
            stop != {"trial_id": manifest["run_id"], "reason": "model_iteration_budget_exhausted", "max_agent_iterations": protocol["generation"]["max_agent_iterations"]}
            or audit["exchanges"] != protocol["generation"]["max_agent_iterations"]
            or any(error.get("error") != "trial exceeded global agent-iteration budget" for error in budget_errors)
        ):
            raise ValueError("model budget stop differs from independently audited gateway ceiling")
        failure = "model"
    other_errors = [error for error in audit["errors"] if error.get("category") != "model_budget"]
    if other_errors:
        categories = {error.get("category") for error in other_errors}
        failure = "protocol" if "protocol" in categories else "provider" if "provider" in categories else "infrastructure"
    record["run"].update(started_at=manifest["started_at"], finished_at=manifest["finished_at"], completed_trials=1, termination_reason=None, evidence_kind="diagnostic" if diagnostic else "benchmark")
    if budget_errors:
        record["run"]["termination_reason"] = "model_iteration_budget_exhausted"
    record["provider"].update(provider_slug=provider["provider_slug"], verified=True, price_snapshot_at=providers["price_snapshot_at"])
    record["harness"]["task_subset"] = [task]
    record["generation"]["contract_verified"] = True
    supplemental = {}
    if harness_id == "function_calling" and task.startswith("web_search_"):
        search_dir = proxy_log.parent / "search"
        discovery = search_dir / "mcp-discovery.json"
        if manifest.get("benchmark_scored") is True and not discovery.is_file():
            raise ValueError("scored web search lacks retained MCP discovery")
        for key, name in (("search_discovery", "mcp-discovery.json"), ("search_events", "search-exchanges.jsonl"), ("fetch_events", "fetch-exchanges.jsonl")):
            path = search_dir / name
            if path.is_file():
                label = str(path.resolve().relative_to(root.resolve()))
                if label not in manifest.get("artifact_sha256", {}):
                    raise ValueError("search evidence lacks retained manifest hash")
                supplemental[key] = path
        queries = []
        if "search_events" in supplemental:
            queries = [json.loads(line) for line in supplemental["search_events"].read_text(encoding="utf-8").splitlines() if line.strip()]
        config = json.loads((root / "evaluation/search.json").read_text())
        selected_tool = None
        if "search_discovery" in supplemental:
            discovered = json.loads(supplemental["search_discovery"].read_text(encoding="utf-8"))
            selected_tool = discovered.get("selected_tool")
            if discovered.get("contract_id") != protocol["search"]["contract_id"] or selected_tool not in config["tool_name_aliases"]:
                raise ValueError("retained MCP discovery differs from the search contract")
        for query in queries:
            if query.get("trial_id") != manifest["run_id"] or query.get("contract_id") != protocol["search"]["contract_id"] or query.get("status") != "completed" or query.get("tool") != selected_tool or type(query.get("credits_upper_bound")) is not int or query["credits_upper_bound"] != config["credit_upper_bound_per_call"]:
                raise ValueError("search event identity, outcome or credit accounting is invalid")
        if "fetch_events" in supplemental:
            fetches = [json.loads(line) for line in supplemental["fetch_events"].read_text(encoding="utf-8").splitlines() if line.strip()]
            from evaluation.web_fetch import valid_fetch_error
            def accepted_fetch(event):
                if event.get("trial_id") != manifest["run_id"]:
                    return False
                if event.get("status") == "completed":
                    return True
                detail = event.get("error")
                return (protocol.get("contract_revision", 0) >= 9
                        and protocol["search"].get("page_error_policy") == "http_3xx_4xx_except_408_429_tool_error"
                        and config.get("page_error_policy") == protocol["search"]["page_error_policy"]
                        and event.get("status") == "tool_error" and valid_fetch_error(detail)
                        and detail["recoverable"] is True and event.get("response") == {"error": detail})
            if not all(accepted_fetch(event) for event in fetches):
                raise ValueError("fetch evidence contains a tool-provider error or wrong trial")
        record["search"].update(used=bool(queries), tool_calls=len(queries), credits_upper_bound=sum(query["credits_upper_bound"] for query in queries))
    record["outcome"] = {**normalized_outcome(passed=verdict["passed"] and failure is None, score=verdict["score"] if failure is None else 0.0, tokens=audit["tokens"], tool_calls=audit["tool_calls"], latency_seconds=latency, cost_usd=audit["cost_usd"], failure_source=failure), "attempt": manifest.get("attempt", 1)}
    artifacts = {"raw_requests": proxy_log, "raw_responses": proxy_log, "tool_events": verifier_paths[-1], "verifier_results": verifier_paths[0]}
    artifacts.update(supplemental)
    if error_log.exists():
        artifacts["gateway_errors"] = error_log
    if budget_errors:
        artifacts.update(budget_stop=manifest_path.parent / "budget-stop.json", gateway_errors=error_log)
    record["artifacts"] = {**{key: str(path.resolve()) for key, path in artifacts.items()}, "sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in artifacts.items()}}
    if "execution_fingerprint" in manifest:
        record["run"]["execution_fingerprint"] = manifest["execution_fingerprint"]
        record["run"]["launcher_sha256"] = manifest["launcher_sha256"]
        if not proxy_log.with_name("budget-scope.json").is_file():
            raise ValueError("current execution evidence lacks retained budget scope")
    attach_scope(record, proxy_log, root)
    if protocol.get("execution_policy", {}).get("budget_scope_required") and "budget_scope" not in record["run"]:
        raise ValueError("current contract requires retained budget scope")
    return record


def ingest_harbor_job(
    job_dir: Path,
    proxy_log: Path,
    *,
    candidate_id: str,
    attempt: int = 1,
    registry: dict[str, Any],
    protocol: dict[str, Any],
    providers: dict[str, Any],
    harness: dict[str, Any],
) -> dict[str, Any]:
    """Normalize one isolated Harbor attempt, refusing mismatched evidence as a score."""
    config = json.loads((job_dir / "config.json").read_text(encoding="utf-8"))
    state = load_harbor_result_state(job_dir)
    errors = audit_harbor_job_config(config, protocol, harness)
    trial_dirs = [path for path in job_dir.iterdir() if path.is_dir()]
    if len(trial_dirs) != 1 or not (trial_dirs[0] / "result.json").exists():
        raise ValueError("Harbor ingestion requires exactly one retained trial result")
    trial = trial_dirs[0]
    result_path = trial / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    provider = next(
        item for item in providers["providers"] if item["candidate_id"] == candidate_id
    )
    exchanges = [
        json.loads(line)
        for line in proxy_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not exchanges:
        errors.append("proxy exchange log is empty")
    generation = protocol["generation"]
    tools = 0
    cost = 0.0
    provider_verified = True
    for exchange in exchanges:
        request = exchange["request"]
        response = exchange["response"]
        if exchange.get("trial_id") != config["job_name"]:
            errors.append("proxy trial identity differs from Harbor job identity")
        for key, expected in (
            ("temperature", generation["temperature"]),
            ("top_p", generation["top_p"]),
            ("max_tokens", generation["max_output_tokens"]),
        ):
            if request.get(key) != expected:
                errors.append(f"retained request {key} differs from protocol")
        route = request.get("provider", {})
        if route.get("only") != [provider["provider_slug"]] or route.get("allow_fallbacks") is not False:
            errors.append("retained request provider route differs from protocol")
        if response.get("provider") != provider["upstream"]:
            provider_verified = False
            errors.append("retained response upstream provider differs from pin")
        cost += float((response.get("usage") or {}).get("cost", 0.0))
        tools += sum(
            len((choice.get("message") or {}).get("tool_calls") or [])
            for choice in response.get("choices", [])
        )
    if tools > generation["tool_call_budget"]:
        errors.append("retained tool-call count exceeds protocol")
    gateway_audit = None
    try:
        gateway_audit = audit_proxy_log(proxy_log, trial_id=config["job_name"], provider=provider, generation=generation)
        cost, tools = gateway_audit["cost_usd"], gateway_audit["tool_calls"]
    except ValueError as exc:
        errors.append("gateway audit: " + str(exc))
    gateway_errors_path = proxy_log.with_name("gateway-errors.jsonl")
    gateway_errors = [json.loads(line) for line in gateway_errors_path.read_text().splitlines() if line.strip()] if gateway_errors_path.exists() else []
    if any(row.get("trial_id") != config["job_name"] for row in gateway_errors):
        errors.append("gateway error evidence belongs to another trial")
    task = (result.get("config") or {}).get("task", {})
    if task.get("git_commit_id") != harness["dataset_revision"]:
        errors.append("actual task revision differs from frozen dataset")
    errors = sorted(set(errors))
    exception = (result.get("exception_info") or {}).get("exception_type")
    reward = float(((result.get("verifier_result") or {}).get("rewards") or {}).get("reward", 0.0))
    failure_source = None
    if errors:
        failure_source = "protocol"
    elif not state["complete"]:
        failure_source = "interrupted"
    elif exception:
        failure_source = "harness"
    elif not result.get("verifier_result"):
        failure_source = "harness"
    categories = {row.get("category") for row in gateway_errors}
    if categories:
        failure_source = "protocol" if "protocol" in categories else "provider" if "provider" in categories else "infrastructure"
    passed = reward == 1.0 and failure_source is None and exception is None
    start = result.get("started_at")
    finish = result.get("finished_at") or datetime.now(timezone.utc).isoformat()
    latency = max(
        0.0,
        (datetime.fromisoformat(finish.replace("Z", "+00:00"))
         - datetime.fromisoformat(start.replace("Z", "+00:00"))).total_seconds(),
    )
    record = build_result_skeleton(
        registry,
        protocol,
        candidate_id=candidate_id,
        harness_id="terminal",
        upstream_provider=provider["upstream"],
        precision=provider["quantization"],
        parser="native-openai-tools",
        run_id=config["job_name"],
    )
    record["run"].update(
        started_at=start,
        finished_at=finish,
        expected_trials=state["expected_trials"],
        completed_trials=state["completed_trials"],
        termination_reason="; ".join(errors) if errors else ("interrupted" if not state["complete"] else None),
    )
    record["provider"].update(
        provider_slug=provider["provider_slug"],
        price_snapshot_at=providers["price_snapshot_at"],
        verified=provider_verified and bool(exchanges),
    )
    record["harness"]["task_subset"] = [result["task_name"]]
    record["generation"]["contract_verified"] = not errors
    agent = result.get("agent_result") or {}
    record["outcome"] = {
        **normalized_outcome(
            passed=passed,
            score=reward,
            tokens=gateway_audit["tokens"] if gateway_audit is not None else int(agent.get("n_input_tokens") or 0) + int(agent.get("n_output_tokens") or 0),
            tool_calls=tools,
            latency_seconds=latency,
            cost_usd=cost,
            failure_source=failure_source,
        ),
        "attempt": attempt,
    }
    trajectory = trial / "agent" / "trajectory.json"
    verifier = trial / "verifier" / "ctrf.json"
    paths = {
        "raw_requests": proxy_log,
        "raw_responses": proxy_log,
        "tool_events": trajectory if trajectory.exists() else result_path,
        "verifier_results": verifier if verifier.exists() else result_path,
    }
    record["artifacts"] = {
        **{key: str(path.resolve()) for key, path in paths.items()},
        "sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()},
    }
    if gateway_errors_path.exists():
        record["artifacts"]["gateway_errors"] = str(gateway_errors_path.resolve())
        record["artifacts"]["sha256"]["gateway_errors"] = hashlib.sha256(gateway_errors_path.read_bytes()).hexdigest()
    attach_scope(record, proxy_log, Path(__file__).resolve().parents[1])
    return record
