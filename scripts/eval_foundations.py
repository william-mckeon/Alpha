"""Validate Phase 1 controls or initialize a normalized evaluation record."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import uuid
import time
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.control import (
    build_result_skeleton,
    load_json,
    validate_control_files,
    validate_result,
    validate_artifact_hashes,
)
from evaluation.provider import (
    BudgetLedger,
    EvaluationBlocked,
    OpenRouterClient,
    ProviderRequestError,
    TrialControlStopped,
    load_local_api_key,
    load_provider_configs,
)
from evaluation.harbor import (
    audit_harbor_job_config,
    build_harbor_job_config,
    load_harbor_result_state,
    stop_harbor_containers,
)
from evaluation.scoring import score_candidates
from evaluation.proxy import ArcusProxyServer
from evaluation.ingest import ingest_harbor_job, read_bfcl_verifier, read_mcpmark_verifier, ingest_worker_verdict
from evaluation.suite_runner import build_suite_plan, expected_coverage, execute_suite_plan
from evaluation.secrets import load_credentials, redact
from evaluation.mcp_client import MCPClient, MCPFailure, fetch_tavily_usage
from evaluation.tavily import TavilySearch, SearchBudget, free_credits_remaining
from evaluation.search_adapter import BFCLSearchAdapter
from evaluation.search_service import SearchService
from evaluation.bfcl import frozen_case, category_group
from evaluation.fixtures import snapshot_archive, sha256
from evaluation.supervision import run_worker
from evaluation.worker_adapters import isolated_worker_environment
from evaluation.worker_runner import control_mounts, validate_worker_inputs, worker_verifier_paths, registered_runners
from evaluation.budget_scope import load_scope, lifetime_accounting
from evaluation.execution import source_fingerprint, inspect_worker_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("validate", help="validate committed Phase 1 control files")
    subcommands.add_parser("preflight", help="check controls, API credential, and Docker")
    subcommands.add_parser("budget-report", help="read actual and unknown spending across budget scopes")

    init = subcommands.add_parser("init-result", help="create a pinned normalized result template")
    init.add_argument("--candidate", required=True)
    init.add_argument("--harness", required=True)
    init.add_argument("--upstream-provider", required=True)
    init.add_argument("--precision", required=True)
    init.add_argument("--parser", required=True)
    init.add_argument("--run-id")
    init.add_argument("--seed", type=int, default=0)
    init.add_argument("--output", type=Path, required=True)

    validate_run = subcommands.add_parser("validate-result", help="validate a completed result")
    validate_run.add_argument("path", type=Path)

    live = subcommands.add_parser("live-tool-smoke", help="run one paid native tool-call probe")
    live.add_argument("--candidate", default="all")
    live.add_argument("--confirm-budget", type=float, required=True)

    prepare = subcommands.add_parser(
        "prepare-harbor", help="write one isolated, protocol-pinned Harbor job"
    )
    prepare.add_argument("--candidate", required=True)
    prepare.add_argument("--task", required=True)
    prepare.add_argument("--attempt", type=int, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--run-id")

    launch = subcommands.add_parser(
        "run-harbor", help="dry-run audit then launch one paid isolated Harbor attempt"
    )
    launch.add_argument("--candidate", required=True)
    launch.add_argument("--task", required=True)
    launch.add_argument("--attempt", type=int, required=True)
    launch.add_argument("--run-id", required=True)
    launch.add_argument("--confirm-budget", type=float, required=True)
    launch.add_argument("--port", type=int, default=8010)
    launch.add_argument("--dry-run", action="store_true")

    audit_config = subcommands.add_parser(
        "audit-harbor-config", help="reject a Harbor config that differs from protocol"
    )
    audit_config.add_argument("path", type=Path)

    audit_run = subcommands.add_parser(
        "audit-harbor-run", help="report whether a Harbor job is complete"
    )
    audit_run.add_argument("path", type=Path)
    stop = subcommands.add_parser("stop-harbor-containers", help="stop only one job's exact trial projects")
    stop.add_argument("path", type=Path)

    ingest = subcommands.add_parser("ingest-harbor", help="normalize one isolated Harbor attempt")
    ingest.add_argument("--candidate", required=True)
    ingest.add_argument("--attempt", type=int, required=True)
    ingest.add_argument("--job-dir", type=Path, required=True)
    ingest.add_argument("--proxy-log", type=Path, required=True)
    ingest.add_argument("--output", type=Path, required=True)

    score = subcommands.add_parser("score", help="score completed normalized result files")
    score.add_argument("paths", nargs="*", type=Path)
    score.add_argument("--results-dir", type=Path, help="load suite records without exceeding Windows argument-length limits")
    score.add_argument("--output", type=Path)
    score.add_argument("--suite", choices=("smoke", "qualification"), default="qualification")
    plan = subcommands.add_parser("plan-suite", help="validate frozen coverage without starting paid work")
    plan.add_argument("--suite", choices=("smoke", "qualification"), default="smoke")
    plan.add_argument("--candidate", default="all")
    tavily = subcommands.add_parser("tavily-preflight", help="check Tavily auth, free allowance and MCP tool discovery without search calls")
    tavily.add_argument("--run-id", required=True)
    search = subcommands.add_parser("live-search-smoke", help="one Tavily MCP search and adapted BFCL result mapping")
    search.add_argument("--run-id", required=True)
    search.add_argument("--query", default="Python official documentation pathlib")
    search.add_argument("--confirm-search-credits", type=int, required=True)
    model_search = subcommands.add_parser("live-tavily-model-smoke", help="native model tool call delegated to Tavily MCP; diagnostic only")
    model_search.add_argument("--candidate", default="all")
    model_search.add_argument("--run-id", required=True)
    model_search.add_argument("--confirm-budget", type=float, required=True)
    model_search.add_argument("--confirm-search-credits", type=int, required=True)
    bfcl = subcommands.add_parser("run-bfcl-diagnostic", help="one frozen BFCL task through official native generation and grading; unscored")
    bfcl.add_argument("--candidate", required=True)
    bfcl.add_argument("--task", default="simple_python_272")
    bfcl.add_argument("--run-id", required=True)
    bfcl.add_argument("--confirm-budget", type=float, required=True)
    bfcl.add_argument("--port", type=int, default=8010)
    freeze = subcommands.add_parser("freeze-mcp-fixture", help="retain a downloaded official archive once; no trial-time refetch")
    freeze.add_argument("--source", type=Path, required=True)
    freeze.add_argument("--category", required=True)
    mcp = subcommands.add_parser("run-mcp-diagnostic", help="one frozen filesystem task in the isolated MCPMark image; unscored")
    mcp.add_argument("--candidate", required=True)
    mcp.add_argument("--task", required=True)
    mcp.add_argument("--run-id", required=True)
    mcp.add_argument("--confirm-budget", type=float, required=True)
    mcp.add_argument("--port", type=int, default=8010)
    repository = subcommands.add_parser("run-repository-diagnostic", help="one frozen SWE-bench task through pinned trusted Linux orchestration; unscored")
    repository.add_argument("--candidate", required=True)
    repository.add_argument("--task", required=True)
    repository.add_argument("--run-id", required=True)
    repository.add_argument("--confirm-budget", type=float, required=True)
    repository.add_argument("--port", type=int, default=8011)
    worker_import = subcommands.add_parser("ingest-worker", help="normalize controlled official worker evidence; diagnostic runs stay unscored")
    worker_import.add_argument("--manifest", type=Path, required=True)
    worker_import.add_argument("--proxy-log", type=Path, required=True)
    worker_import.add_argument("--verifier", type=Path, action="append", required=True)
    worker_import.add_argument("--output", type=Path, required=True)
    for name in ("run-suite", "run-integration-suite"):
        suite = subcommands.add_parser(name, help="sequential qualification-gated runners" if name == "run-suite" else "diagnostic-only cross-harness integration; never qualification")
        suite.add_argument("--suite", choices=("smoke", "qualification"), default="smoke")
        suite.add_argument("--candidate", default="all")
        suite.add_argument("--continue-on-candidate-error", action="store_true", help="skip remaining tasks for a candidate after a classified provider failure; retain exclusions")
        suite.add_argument("--harness", choices=("all", "terminal", "repository", "function_calling", "mcp"), default="all")
        suite.add_argument("--run-id", required=True)
        suite.add_argument("--confirm-budget", type=float, required=True)
        suite.add_argument("--limit", type=int)
        suite.add_argument("--port", type=int, default=8010)
        suite.add_argument("--dry-run", action="store_true")
    for name, command in subcommands.choices.items():
        if any(action.dest == "confirm_budget" for action in command._actions):
            command.add_argument("--budget-scope", default="final-smoke" if name == "run-suite" else "diagnostic")
    return parser.parse_args()


def _budget(args):
    return load_scope(ROOT, getattr(args, "budget_scope", "diagnostic"), confirm=args.confirm_budget, require_authorized=not getattr(args, "dry_run", False))


def _execution_evidence(server):
    return {"execution_fingerprint": source_fingerprint(ROOT), "launcher_sha256": sha256(Path(__file__)), "budget_scope": server.budget_scope.snapshot()}


def _preflight() -> list[str]:
    failures = validate_control_files(ROOT)
    if not os.environ.get("OPENROUTER_API_KEY", "").strip():
        failures.append("OPENROUTER_API_KEY is not set")
    if shutil.which("docker") is None:
        failures.append("Docker command is not installed")
    else:
        check = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if check.returncode != 0:
            detail = f"{check.stdout}\n{check.stderr}".lower()
            permission_markers = ("access is denied", "permission denied", "docker_engine")
            if any(marker in detail for marker in permission_markers):
                failures.append(
                    "Docker daemon is inaccessible from this process; verify it outside the "
                    "restricted environment before concluding that Docker is stopped"
                )
            else:
                failures.append("Docker daemon did not answer docker info")
    return failures


def _tool_probe(candidate_id: str, budget_scope="diagnostic") -> int:
    providers, aggregate_cap = load_provider_configs(ROOT / "evaluation" / "providers.json")
    if candidate_id not in providers:
        raise ValueError(f"unknown candidate: {candidate_id}")
    config = providers[candidate_id]
    ledger = load_scope(ROOT, budget_scope).ledger()
    tool = {
        "type": "function",
        "function": {
            "name": "lookup_symbol",
            "description": "Look up a symbol in a source repository.",
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"],
                "additionalProperties": False,
            },
        },
    }
    client = OpenRouterClient()
    try:
        exchange = client.chat(
            config,
            ledger,
            messages=[{"role": "user", "content": "Use the tool to look up the symbol ArcusMoDE."}],
            tools=[tool],
            max_tokens=256,
            provider_retries=1,
        )
    except ProviderRequestError as exc:
        print(f"{candidate_id}: PROVIDER_ERROR ({exc})")
        return 1
    message = exchange["response"]["choices"][0]["message"]
    calls = message.get("tool_calls") or []
    passed = False
    if calls:
        function = calls[0].get("function", {})
        try:
            arguments = json.loads(function.get("arguments", "{}"))
        except json.JSONDecodeError:
            arguments = {}
        passed = function.get("name") == "lookup_symbol" and arguments == {"symbol": "ArcusMoDE"}
    output = ROOT / "evaluation" / "runs" / "tool-smoke" / uuid.uuid4().hex / f"{candidate_id}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"candidate_id": candidate_id, "passed": passed, **exchange}, indent=2) + "\n",
        encoding="utf-8",
    )
    actual_cost = float(exchange["response"]["usage"]["cost"])
    print(f"{candidate_id}: {'PASS' if passed else 'FAIL'} (${actual_cost:.6f})")
    print(f"Retained isolated probe: {output}")
    return 0 if passed else 1


def _harbor_inputs(candidate_id: str) -> tuple[dict, dict, dict]:
    protocol = load_json(ROOT / "evaluation" / "protocol.json")
    harness = load_json(ROOT / "evaluation" / "harnesses" / "harbor.json")
    registry = load_json(ROOT / "evaluation" / "candidates.json")
    candidates = {item["id"]: item for item in registry["candidates"]}
    if candidate_id not in candidates:
        raise ValueError(f"unknown candidate: {candidate_id}")
    return protocol, harness, candidates[candidate_id]


def _run_harbor(args: argparse.Namespace) -> int:
    """Own the gateway and Harbor process lifecycle for exactly one isolated attempt."""
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in args.run_id):
        raise ValueError("run-id must contain only letters, digits, hyphens, and underscores")
    failures = validate_control_files(ROOT)
    if failures:
        raise EvaluationBlocked("; ".join(failures))
    providers, aggregate_cap = load_provider_configs(ROOT / "evaluation" / "providers.json")
    scope = _budget(args)
    protocol, harness, _ = _harbor_inputs(args.candidate)
    config = build_harbor_job_config(
        protocol,
        harness,
        candidate_id=args.candidate,
        task_id=args.task,
        attempt=args.attempt,
        run_id=args.run_id,
    )
    config_path = ROOT / "evaluation" / "runs" / "configs" / f"{args.run_id}.json"
    job_dir = ROOT / "evaluation" / "runs" / "harbor" / args.run_id
    proxy_dir = ROOT / "evaluation" / "runs" / "proxy" / args.run_id
    if job_dir.exists() or (proxy_dir / "proxy-exchanges.jsonl").exists():
        raise EvaluationBlocked("run-id already has artifacts; use a new run-id")
    executable = ROOT / "evaluation" / "vendor" / "harbor" / ".venv" / "Scripts" / "harbor.exe"
    if not executable.exists():
        raise EvaluationBlocked("the pinned Harbor executable is not installed")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    child_env = isolated_worker_environment(os.environ)
    resolved = subprocess.run(
        [str(executable), "run", "--config", str(config_path), "--print-config"],
        cwd=ROOT,
        env=child_env,
        capture_output=True,
        text=True,
        check=False,
    )
    if resolved.returncode:
        raise EvaluationBlocked(f"Harbor config resolution failed: {resolved.stderr.strip()}")
    resolved_config = json.loads(resolved.stdout)
    errors = audit_harbor_job_config(resolved_config, protocol, harness)
    if errors:
        raise EvaluationBlocked("Harbor resolved-config audit failed: " + "; ".join(errors))
    (config_path.parent / f"{args.run_id}.resolved.json").write_text(
        json.dumps(resolved_config, indent=2) + "\n", encoding="utf-8"
    )
    print("Harbor resolved configuration matches the frozen protocol.", flush=True)
    if args.dry_run:
        print("Dry run complete: no gateway started and no paid request attempted.")
        return 0
    preflight = _preflight()
    if preflight:
        raise EvaluationBlocked("; ".join(preflight))
    token = uuid.uuid4().hex + uuid.uuid4().hex
    server = ArcusProxyServer(
        ("0.0.0.0", args.port),
        args.candidate,
        trial_id=args.run_id,
        artifact_dir=proxy_dir,
        proxy_token=token,
        budget_scope=scope.identity,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    child_env["LLM_API_KEY"] = token
    child_env["LLM_BASE_URL"] = f"http://host.docker.internal:{args.port}/v1"
    process = None
    try:
        process = subprocess.Popen(
            [str(executable), "run", "--config", str(config_path), "--yes"],
            cwd=ROOT,
            env=child_env,
        )
        started = time.monotonic()
        while True:
            server.raise_if_stopped()
            try:
                returncode = process.wait(timeout=1)
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() - started > protocol["generation"]["trial_timeout_seconds"]:
                    raise
        state = load_harbor_result_state(job_dir)
        print(json.dumps(state, indent=2), flush=True)
        return returncode or int(not state["complete"])
    except (KeyboardInterrupt, subprocess.TimeoutExpired, TrialControlStopped) as exc:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        print("Harbor attempt interrupted; its artifacts are not scored.", file=sys.stderr)
        stop_harbor_containers(job_dir)
        if isinstance(exc, TrialControlStopped):
            job_dir.mkdir(parents=True, exist_ok=True)
            (job_dir / "validation.json").write_text(json.dumps({"status": "integration_error", "stop_class": exc.stop_class, "error": redact(str(exc)), "budget_scope": scope.snapshot(), "finished_at": datetime.now(timezone.utc).isoformat()}, indent=2))
            return 1
        return 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        print("Isolated evaluation gateway stopped.", flush=True)


def _run_bfcl_diagnostic(args: argparse.Namespace) -> int:
    errors = validate_control_files(ROOT)
    if errors:
        raise EvaluationBlocked("; ".join(errors))
    providers, cap = load_provider_configs(ROOT / "evaluation/providers.json")
    scope = _budget(args)
    if args.candidate not in providers:
        raise EvaluationBlocked("unknown candidate or incorrect approved budget")
    if not args.run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.run_id):
        raise EvaluationBlocked("invalid run-id")
    suites = load_json(ROOT / "evaluation/suites.json")["suites"]
    frozen = set(suites["smoke"]["function_calling"]["task_ids"]) | set(suites["qualification"]["function_calling"]["task_ids"])
    category, official_task = frozen_case(args.task, load_json(ROOT / "evaluation/protocol.json"))
    if args.task not in frozen:
        raise EvaluationBlocked("BFCL worker requires a frozen task ID")
    python = ROOT / "evaluation/runs/bfcl-venv/Scripts/python.exe"
    if not python.exists():
        raise EvaluationBlocked("isolated BFCL runtime is not installed")
    directory = ROOT / "evaluation/runs/bfcl" / args.run_id
    proxy_dir = ROOT / "evaluation/runs/proxy" / args.run_id
    if directory.exists() or proxy_dir.exists():
        raise EvaluationBlocked("run-id already has artifacts")
    token = uuid.uuid4().hex + uuid.uuid4().hex
    search_service = None
    if category == "web_search_base":
        config = load_json(ROOT / "evaluation/search.json")
        key = os.environ.get(config["credential_env"])
        if not key:
            raise EvaluationBlocked("Tavily credential is missing")
        search_service = SearchService(config, SearchBudget(ROOT / "evaluation/runs/search-budget.sqlite3", cap=config["approved_credit_cap"], contract_id=config["contract_id"]), proxy_dir / "search", trial_id=args.run_id, api_key=key, usage_cache_path=ROOT / "evaluation/runs/tavily-usage-cache.json")
    server = ArcusProxyServer(("127.0.0.1", args.port), args.candidate, trial_id=args.run_id, artifact_dir=proxy_dir, proxy_token=token, search_service=search_service, budget_scope=scope.identity)
    directory.mkdir(parents=True)
    environment = isolated_worker_environment(os.environ)
    environment.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", ARCUS_WORKER_TRIAL_ID=args.run_id, ARCUS_WORKER_TOKEN=token, ARCUS_WORKER_BASE_URL=f"http://127.0.0.1:{args.port}/v1", PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / "evaluation/vendor/bfcl/berkeley-function-call-leaderboard"))
    evidence = {"run_id": args.run_id, "candidate_id": args.candidate, "task_id": args.task, "benchmark_scored": False, "status": "running"}
    evidence.update(_execution_evidence(server))
    evidence.update(harness_id="function_calling", harness_revision=load_json(ROOT / "evaluation/harnesses/bfcl.json")["revision"], protocol_snapshot=load_json(ROOT / "evaluation/protocol.json"), started_at=datetime.now(timezone.utc).isoformat(), attempt=getattr(args, "attempt", 1))
    evidence.update(official_task_id=official_task, official_category=category, benchmark_variant="BFCL V4 / Arcus Tavily MCP adaptation" if search_service else "BFCL V4 native official diagnostic")
    evidence["benchmark_scored"] = getattr(args, "benchmark_scored", False) is True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        if search_service:
            search_service.start()
        for phase in ("generate", "evaluate"):
            if phase == "evaluate":
                from evaluation.openhands import check_gateway_stop
                check_gateway_stop(proxy_dir, args.run_id)
                if search_service and (search_service.failed_calls or search_service.failure):
                    raise EvaluationBlocked("search provider failed; diagnostic is excluded before grading")
            command = [str(python), "-m", "evaluation.bfcl_worker", "--candidate", args.candidate, "--task", args.task, "--phase", phase, "--output-dir", str(directory)]
            with (directory / f"{phase}.log").open("x", encoding="utf-8") as log:
                if phase == "generate":
                    code = run_worker(command, cwd=ROOT, env=environment, log=log, server=server, generation=load_json(ROOT / "evaluation/protocol.json")["generation"])
                else:
                    code = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=load_json(ROOT / "evaluation/protocol.json")["generation"]["verifier_timeout_seconds"]).returncode
            if phase == "generate":
                from evaluation.openhands import check_gateway_stop
                check_gateway_stop(proxy_dir, args.run_id)
            if code:
                raise EvaluationBlocked(f"official BFCL {phase} exited {code}; see retained log")
        alias = "arcus-" + args.candidate + "-FC"
        if search_service and (search_service.failed_calls or search_service.failure):
            raise EvaluationBlocked("search provider failed; diagnostic is excluded from model scoring")
        if search_service:
            search_service.close()
        group = category_group(category)
        response = directory / "result" / alias / group / f"BFCL_v4_{category}_result.json"
        score = directory / "score" / alias / group / f"BFCL_v4_{category}_score.json"
        evidence["official_verdict"] = read_bfcl_verifier(score, response, official_task)
        if evidence["official_verdict"]["status"] == "harness_error":
            raise EvaluationBlocked("official BFCL reported an inference/harness error; diagnostic is unscored")
        evidence["status"] = "integration_passed"
        print("Official BFCL pipeline completed; task outcome: " + evidence["official_verdict"]["status"])
        return 0
    except Exception as exc:
        evidence.update(status="integration_error", error=redact(str(exc)), stop_class=getattr(exc, "stop_class", "execution_error"))
        print("BFCL diagnostic: " + redact(str(exc)), file=sys.stderr)
        return 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        if search_service:
            try:
                search_service.close()
            except MCPFailure as exc:
                evidence.update(status="integration_error", error=redact(str(exc)))
        paths = [path for path in directory.rglob("*") if path.is_file()]
        paths.extend(path for path in proxy_dir.rglob("*") if path.is_file())
        evidence["artifact_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        evidence["finished_at"] = datetime.now(timezone.utc).isoformat()
        (directory / "validation.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")


def _run_mcp_diagnostic(args: argparse.Namespace) -> int:
    errors = validate_control_files(ROOT)
    providers, cap = load_provider_configs(ROOT / "evaluation/providers.json")
    scope = _budget(args)
    if errors or args.candidate not in providers:
        raise EvaluationBlocked("invalid controls, candidate or approved budget")
    if not args.run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.run_id):
        raise EvaluationBlocked("invalid run-id")
    suites = load_json(ROOT / "evaluation/suites.json")["suites"]
    frozen = set(suites["smoke"]["mcp"]["task_ids"]) | set(suites["qualification"]["mcp"]["task_ids"])
    if args.task not in frozen:
        raise EvaluationBlocked("MCP diagnostic requires one frozen filesystem task")
    category = args.task.split("/")[1]
    archive = ROOT / "evaluation/runs/fixture-snapshots" / (category + ".zip")
    snapshot = load_json(archive.with_suffix(".manifest.json"))
    if sha256(archive) != snapshot["archive_sha256"]:
        raise EvaluationBlocked("fixture differs from its frozen snapshot")
    image = inspect_worker_image(ROOT, "arcus-eval-mcpmark:phase1")
    directory = ROOT / "evaluation/runs/mcpmark" / args.run_id
    proxy_dir = ROOT / "evaluation/runs/proxy" / args.run_id
    if directory.exists() or proxy_dir.exists():
        raise EvaluationBlocked("trial artifacts cannot be reused")
    token = uuid.uuid4().hex + uuid.uuid4().hex
    server = ArcusProxyServer(("0.0.0.0", args.port), args.candidate, trial_id=args.run_id, artifact_dir=proxy_dir, proxy_token=token, budget_scope=scope.identity)
    directory.mkdir(parents=True)
    name = "arcus-mcp-" + args.run_id
    if subprocess.check_output(["docker", "ps", "-aq", "--filter", "name=^/" + name + "$"], text=True).strip():
        server.server_close()
        raise EvaluationBlocked("trial container already exists")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    evidence = {"run_id": args.run_id, "candidate_id": args.candidate, "task_id": args.task, "benchmark_scored": False, "image_id": image, "fixture_snapshot": snapshot}
    evidence.update(_execution_evidence(server))
    evidence.update(harness_id="mcp", harness_revision=load_json(ROOT / "evaluation/harnesses/mcpmark.json")["revision"], protocol_snapshot=load_json(ROOT / "evaluation/protocol.json"), started_at=datetime.now(timezone.utc).isoformat(), attempt=getattr(args, "attempt", 1))
    evidence["worker_control_contract_required"] = True
    evidence["benchmark_scored"] = getattr(args, "benchmark_scored", False) is True
    try:
        command = ["docker", "run", *control_mounts(ROOT, "mcpmark"), "--name", name, "--init", "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pids-limit=256", "--memory=4g", "--cpus=2", "--mount", f"type=bind,source={directory},target=/work", "--mount", f"type=bind,source={archive},target=/fixture/archive.zip,readonly", "--env", "ARCUS_WORKER_TOKEN=" + token, "--env", f"ARCUS_WORKER_BASE_URL=http://host.docker.internal:{args.port}/v1", image, "--candidate", args.candidate, "--task", args.task, "--archive", "/fixture/archive.zip", "--archive-sha256", snapshot["archive_sha256"]]
        with (directory / "worker.log").open("x", encoding="utf-8") as log:
            code = run_worker(command, log=log, server=server, generation=load_json(ROOT / "evaluation/protocol.json")["generation"], model_completed=lambda: (directory / "model-completed.json").is_file())
        from evaluation.openhands import check_gateway_stop
        check_gateway_stop(proxy_dir, args.run_id)
        if code:
            raise EvaluationBlocked(f"MCP worker exited {code}; see retained worker.log")
        metas = list((directory / "results").rglob("meta.json"))
        if len(metas) != 1:
            raise EvaluationBlocked("official MCP worker did not retain exactly one task verdict")
        evidence["official_verdict"] = read_mcpmark_verifier(metas[0], args.task)
        if evidence["official_verdict"]["status"] == "harness_error":
            raise EvaluationBlocked("official MCPMark reported a harness error; diagnostic is unscored")
        evidence["status"] = "integration_passed"
        print("Official isolated MCPMark task completed: " + evidence["official_verdict"]["status"])
        return 0
    except Exception as exc:
        evidence.update(status="integration_error", error=redact(str(exc), extra_secrets=(token,)), stop_class=getattr(exc, "stop_class", "execution_error"))
        print(evidence["error"], file=sys.stderr)
        return 1
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        paths = [p for base in (directory, proxy_dir) for p in base.rglob("*") if p.is_file()]
        evidence["artifact_sha256"] = {str(p.relative_to(ROOT)): sha256(p) for p in paths}
        evidence["finished_at"] = datetime.now(timezone.utc).isoformat()
        (directory / "validation.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")


def _run_repository_diagnostic(args: argparse.Namespace) -> int:
    errors = validate_control_files(ROOT)
    providers, cap = load_provider_configs(ROOT / "evaluation/providers.json")
    scope = _budget(args)
    if errors or args.candidate not in providers:
        raise EvaluationBlocked("invalid controls, candidate or approved budget")
    if not args.run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.run_id):
        raise EvaluationBlocked("invalid repository run-id")
    suites = load_json(ROOT / "evaluation/suites.json")["suites"]
    if args.task not in {task for suite in suites.values() for task in suite["repository"]["task_ids"]}:
        raise EvaluationBlocked("repository diagnostic requires one frozen task")
    from evaluation.openhands import JSONL_SHA256
    dataset = ROOT / "evaluation/runs/swebench-verified-c104f840.jsonl"
    if sha256(dataset) != JSONL_SHA256:
        raise EvaluationBlocked("repository snapshot differs from frozen dataset")
    image = inspect_worker_image(ROOT, "arcus-eval-openhands:phase1")
    directory = ROOT / "evaluation/runs/repository" / args.run_id
    proxy_dir = ROOT / "evaluation/runs/proxy" / args.run_id
    if directory.exists() or proxy_dir.exists():
        raise EvaluationBlocked("repository artifacts cannot be reused")
    # Retain a unique image reference across infer/verify even if the shared
    # development tag is rebuilt while this trial is running.
    trial_image = "arcus-eval-openhands:trial-" + args.run_id.lower()
    if subprocess.run(["docker", "image", "inspect", trial_image], capture_output=True).returncode == 0:
        raise EvaluationBlocked("repository trial image reference already exists")
    subprocess.run(["docker", "tag", image, trial_image], check=True)
    token = uuid.uuid4().hex + uuid.uuid4().hex
    server = ArcusProxyServer(("0.0.0.0", args.port), args.candidate, trial_id=args.run_id, artifact_dir=proxy_dir, proxy_token=token, budget_scope=scope.identity)
    directory.mkdir(parents=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    evidence = {"run_id": args.run_id, "candidate_id": args.candidate, "task_id": args.task, "harness_id": "repository", "harness_revision": load_json(ROOT / "evaluation/harnesses/openhands.json")["revision"], "protocol_snapshot": load_json(ROOT / "evaluation/protocol.json"), "benchmark_scored": False, "attempt": getattr(args, "attempt", 1), "image_id": image, "dataset_sha256": JSONL_SHA256, "started_at": datetime.now(timezone.utc).isoformat()}
    evidence.update(_execution_evidence(server))
    names = []
    evidence["worker_control_contract_required"] = True
    evidence["benchmark_scored"] = getattr(args, "benchmark_scored", False) is True
    verifier_name = "sweb.eval." + args.task.lower() + "." + args.run_id
    verifier_owned = False
    try:
        if subprocess.check_output(["docker", "ps", "-aq", "--filter", "name=^/" + verifier_name + "$"], text=True).strip():
            raise EvaluationBlocked("official verifier container already exists")
        verifier_owned = True
        for phase in ("infer", "verify"):
            name = "arcus-repository-" + args.run_id + "-" + phase
            if subprocess.check_output(["docker", "ps", "-aq", "--filter", "name=^/" + name + "$"], text=True).strip():
                raise EvaluationBlocked("owned repository controller name already exists")
            names.append(name)
            command = ["docker", "run", *control_mounts(ROOT, "openhands"), "--name", name, "--init", "--cap-drop=ALL", "--security-opt=no-new-privileges", "--mount", "type=bind,source=/var/run/docker.sock,target=/var/run/docker.sock", "--mount", f"type=bind,source={directory},target=/work", "--mount", f"type=bind,source={dataset},target=/data/swebench.jsonl,readonly", "--env", "IMAGE_TAG_PREFIX=43376f1", "--env", "ARCUS_WORKER_TOKEN=" + token, "--env", f"ARCUS_WORKER_BASE_URL=http://host.docker.internal:{args.port}/v1", trial_image, "--candidate", args.candidate, "--task", args.task, "--trial-id", args.run_id, "--dataset", "/data/swebench.jsonl", "--phase", phase]
            with (directory / (phase + ".log")).open("x", encoding="utf-8") as log:
                timeout = evidence["protocol_snapshot"]["generation"]["wall_time_seconds"] if phase == "infer" else evidence["protocol_snapshot"]["generation"]["verifier_timeout_seconds"]
                if phase == "infer":
                    code = run_worker(command, log=log, server=server, generation=evidence["protocol_snapshot"]["generation"])
                else:
                    code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout).returncode
            if phase == "infer":
                from evaluation.openhands import check_gateway_stop, check_prediction_before_verify
                check_gateway_stop(proxy_dir, args.run_id)
                check_prediction_before_verify(directory, args.task)
            if code:
                raise EvaluationBlocked(f"official repository {phase} exited {code}; see retained log")
        reports = list((directory / "inference").rglob("*.report.json"))
        if len(reports) != 1:
            raise EvaluationBlocked("repository worker did not retain exactly one official report")
        from evaluation.ingest import read_swebench_verifier
        evidence["official_verdict"] = read_swebench_verifier(reports[0], args.task)
        if evidence["official_verdict"]["status"] == "harness_error":
            raise EvaluationBlocked("official SWE-bench verifier reported a harness error")
        evidence["status"] = "integration_passed"
        print("Official repository diagnostic completed: " + evidence["official_verdict"]["status"])
        return 0
    except Exception as exc:
        evidence.update(status="integration_error", error=redact(str(exc), extra_secrets=(token,)), stop_class=getattr(exc, "stop_class", "execution_error"))
        print(evidence["error"], file=sys.stderr)
        return 1
    finally:
        # Only SDK workspaces carrying our exact trial label and owned controllers.
        for name in names:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)
        ids = subprocess.check_output(["docker", "ps", "-aq", "--filter", "label=arcus.trial_id=" + args.run_id], text=True).split()
        for name in ids:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)
        if verifier_owned and evidence.get("status") != "integration_passed":
            # Upstream uses this exact task+run name; never remove preexisting targets.
            subprocess.run(["docker", "rm", "-f", verifier_name], capture_output=True, check=False)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        evidence["finished_at"] = datetime.now(timezone.utc).isoformat()
        paths = [p for base in (directory, proxy_dir) for p in base.rglob("*") if p.is_file()]
        evidence["artifact_sha256"] = {str(p.relative_to(ROOT)): sha256(p) for p in paths}
        (directory / "validation.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")


def _run_integration_suite(args: argparse.Namespace, *, benchmark: bool = False) -> int:
    """Exercise all diagnostic callbacks without granting qualification eligibility."""
    errors = validate_control_files(ROOT)
    providers, cap = load_provider_configs(ROOT / "evaluation/providers.json")
    scope = _budget(args)
    if benchmark and not args.dry_run and scope.identity not in {"final-smoke", "six-model-token-restart"}:
        raise EvaluationBlocked("benchmark launch requires explicit authorized final-run accounting")
    if errors:
        raise EvaluationBlocked("invalid controls or approved budget")
    if not args.run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.run_id):
        raise EvaluationBlocked("invalid integration run-id")
    candidates = list(scope.definition["caps"]) if args.candidate == "all" else [args.candidate]
    if any(candidate not in providers for candidate in candidates):
        raise EvaluationBlocked("unknown candidate")
    protocol = load_json(ROOT / "evaluation/protocol.json")
    plan = build_suite_plan(load_json(ROOT / "evaluation/suites.json"), args.suite, protocol["generation"]["attempts"], candidates)
    if args.harness != "all":
        plan = [unit for unit in plan if unit["harness_id"] == args.harness]
    if benchmark:
        missing = sorted({unit["harness_id"] for unit in plan} - registered_runners(ROOT, protocol))
        if missing:
            raise EvaluationBlocked("live runners not verified: " + ", ".join(missing))
    errors = validate_worker_inputs(ROOT, plan, protocol)
    if errors:
        raise EvaluationBlocked("integration prerequisites failed before paid work: " + "; ".join(errors))
    selected_harnesses = {unit["harness_id"] for unit in plan}
    if args.limit is not None:
        if args.limit < 1:
            raise EvaluationBlocked("limit must be positive")
        plan = plan[:args.limit]
    directory = ROOT / "evaluation/runs" / ("suites" if benchmark else "integration-suites") / args.run_id
    if directory.exists():
        raise EvaluationBlocked("integration artifacts already exist")
    folders = {"terminal": "harbor", "repository": "repository", "function_calling": "bfcl", "mcp": "mcpmark"}
    for index, unit in enumerate(plan, 1):
        run_id = f"{args.run_id}-u{index:04d}"
        if any((ROOT / "evaluation/runs" / folder / run_id).exists() for folder in (folders[unit["harness_id"]], "proxy")):
            raise EvaluationBlocked("integration child artifacts already exist")
    if args.dry_run:
        if benchmark:
            for index, unit in enumerate(plan, 1):
                if unit["harness_id"] == "terminal":
                    child = argparse.Namespace(candidate=unit["candidate_id"], task=unit["task_id"], attempt=unit["attempt"], run_id=f"{args.run_id}-u{index:04d}", confirm_budget=args.confirm_budget, port=args.port, dry_run=True, budget_scope=scope.identity)
                    if _run_harbor(child):
                        raise EvaluationBlocked("terminal resolved-config audit failed")
        audit = {"status": "local_prerequisites_passed", "paid_requests": 0, "qualification_complete": False, "contract_revision": protocol["contract_revision"], "units": plan}
        audit["budget_scope"] = scope.snapshot()
        directory.mkdir(parents=True)
        with (directory / "dry-run.json").open("x", encoding="utf-8") as stream:
            json.dump(audit, stream, indent=2)
        print(f"Local prerequisites passed for {len(plan)} integration units; no paid requests. Retained: {directory / 'dry-run.json'}")
        return 0
    failures = _preflight()
    if failures:
        raise EvaluationBlocked("; ".join(failures))
    for harness, tag in (("mcp", "arcus-eval-mcpmark:phase1"), ("repository", "arcus-eval-openhands:phase1")):
        if harness in selected_harnesses:
            inspect_worker_image(ROOT, tag)
    launchers = {"terminal": _run_harbor, "repository": _run_repository_diagnostic, "function_calling": _run_bfcl_diagnostic, "mcp": _run_mcp_diagnostic}
    def runner(unit, index):
        run_id = f"{args.run_id}-u{index:04d}"
        child = argparse.Namespace(candidate=unit["candidate_id"], task=unit["task_id"], attempt=unit["attempt"], run_id=run_id, confirm_budget=args.confirm_budget, port=args.port, dry_run=False, benchmark_scored=benchmark, budget_scope=scope.identity)
        code = launchers[unit["harness_id"]](child)
        task_dir = ROOT / "evaluation/runs" / folders[unit["harness_id"]] / run_id
        proxy_log = ROOT / "evaluation/runs/proxy" / run_id / "proxy-exchanges.jsonl"
        if code:
            manifest_file = task_dir / "validation.json"
            cause = load_json(manifest_file) if manifest_file.is_file() else {}
            exc = EvaluationBlocked(cause.get("error", "integration worker failed; raw artifacts retained, no retry"))
            exc.stop_class = cause.get("stop_class", "execution_error")
            exc.candidate_failure = exc.stop_class in {"provider_error", "gateway_provider_error", "gateway_provider", "gateway_timeout", "harness_timeout", "gateway_token_budget", "token_budget"}
            exc.artifact_references = [str(task_dir), str(proxy_log.parent)]
            gateway_errors = proxy_log.with_name("gateway-errors.jsonl")
            if gateway_errors.is_file():
                retained_errors = [json.loads(line) for line in gateway_errors.read_text(encoding="utf-8").splitlines() if line.strip()]
                categories = {item.get("category") for item in retained_errors}
                # Only candidate-local failures may advance. A shared protocol,
                # accounting or infrastructure error must remain a hard stop.
                if categories and categories <= {"provider", "timeout", "token_budget", "control_stop"}:
                    exc.candidate_failure = True
                    exc.stop_class = "gateway_" + next((item["category"] for item in retained_errors if item.get("category") != "control_stop"), "control_stop")
            raise exc
        if unit["harness_id"] == "terminal":
            p, h, _ = _harbor_inputs(unit["candidate_id"])
            record = ingest_harbor_job(task_dir, proxy_log, candidate_id=unit["candidate_id"], attempt=unit["attempt"], registry=load_json(ROOT / "evaluation/candidates.json"), protocol=p, providers=load_json(ROOT / "evaluation/providers.json"), harness=h)
            record["run"]["evidence_kind"] = "benchmark" if benchmark else "diagnostic"
            record["run"]["execution_fingerprint"] = source_fingerprint(ROOT)
            record["run"]["launcher_sha256"] = sha256(Path(__file__))
        else:
            record = ingest_worker_verdict(root=ROOT, manifest_path=task_dir / "validation.json", proxy_log=proxy_log, verifier_paths=worker_verifier_paths(task_dir, unit["harness_id"], unit["candidate_id"], unit["task_id"], protocol), registry=load_json(ROOT / "evaluation/candidates.json"), protocol=protocol, providers=load_json(ROOT / "evaluation/providers.json"))
        if record["run"].get("evidence_kind") != ("benchmark" if benchmark else "diagnostic"):
            raise EvaluationBlocked("callback returned the wrong evidence kind")
        errors = validate_result(record) + validate_artifact_hashes(record, ROOT)
        if errors:
            raise EvaluationBlocked("integration evidence rejected: " + "; ".join(errors))
        with (directory / f"{run_id}.json").open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
        return record
    records = execute_suite_plan(plan, {harness: runner for harness in launchers}, directory, contract_revision=protocol["contract_revision"], evidence_kind="benchmark" if benchmark else "diagnostic", budget_scope=scope.snapshot(), continue_on_candidate_error=getattr(args, "continue_on_candidate_error", False))
    print(f"Retained {len(records)} {'benchmark' if benchmark else 'diagnostic integration'} outcomes; no automatic qualification scoring.")
    return int(any(record["outcome"]["status"] not in {"passed", "failed"} for record in records))


def _run_suite(args: argparse.Namespace) -> int:
    """Use shared callbacks only after current-contract live proof is retained."""
    return _run_integration_suite(args, benchmark=True)


async def _tavily_check(args: argparse.Namespace) -> int:
    if not args.run_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in args.run_id):
        raise EvaluationBlocked("search run-id must contain only letters, digits, hyphens and underscores")
    config = load_json(ROOT / "evaluation" / "search.json")
    if args.command != "tavily-preflight" and args.confirm_search_credits != config["approved_credit_cap"]:
        raise EvaluationBlocked("confirm-search-credits must match the approved free-only cap")
    directory = ROOT / "evaluation" / "runs" / "search" / args.run_id
    if directory.exists():
        raise EvaluationBlocked("search run-id already exists; previous evidence will not be overwritten")
    directory.mkdir(parents=True)
    evidence = {"run_id": args.run_id, "contract_id": config["contract_id"], "status": "pending", "benchmark_scored": False}
    budget = SearchBudget(ROOT / "evaluation" / "runs" / "search-budget.sqlite3", cap=config["approved_credit_cap"], contract_id=config["contract_id"])
    try:
        key = os.environ.get("TAVILY_API_KEY", "")
        usage_cache = ROOT / "evaluation" / "runs" / "tavily-usage-cache.json"
        usage = await fetch_tavily_usage(key, cache_path=usage_cache)
        evidence["usage_before"] = usage
        evidence["free_credits_remaining_before"] = free_credits_remaining(usage)
        async with MCPClient(config["endpoint"], api_key=key, timeout_seconds=config["timeout_seconds"], usage_cache_path=usage_cache) as client:
            client.usage_snapshot = usage
            client.usage_checked_monotonic = time.monotonic()
            search = TavilySearch(client, config, budget, directory, trial_id=args.run_id)
            discovery = await search.discover()
            evidence["selected_tool"] = discovery["selected_tool"]
            if args.command == "live-tavily-model-smoke":
                providers, aggregate = load_provider_configs(ROOT / "evaluation" / "providers.json")
                scope = _budget(args)
                candidate_ids = list(providers) if args.candidate == "all" else [args.candidate]
                if any(candidate not in providers for candidate in candidate_ids):
                    raise EvaluationBlocked("unknown candidate")
                tool = next(tool for tool in discovery["tools"] if tool["name"] == discovery["selected_tool"])
                functions = [{"type": "function", "function": {"name": tool["name"], "description": tool["description"], "parameters": tool["inputSchema"]}}]
                model_budget = scope.ledger()
                evidence["budget_scope"] = scope.snapshot()
                model_client = OpenRouterClient()
                evidence["candidates"] = []
                for candidate in candidate_ids:
                    exchange = await asyncio.to_thread(model_client.forward, providers[candidate], model_budget, {
                        "messages": [{"role": "user", "content": "Call tavily_search once to find Python's official pathlib documentation. Use basic search_depth, max_results 3 and no extra options. Do not answer without calling the tool."}],
                        "tools": functions, "tool_choice": "auto", "temperature": 0, "top_p": 1, "max_tokens": 256,
                    })
                    (directory / f"{candidate}-model-exchange.json").write_text(json.dumps(redact(exchange), indent=2) + "\n", encoding="utf-8")
                    calls = exchange["response"]["choices"][0]["message"].get("tool_calls", [])
                    if len(calls) != 1 or calls[0].get("function", {}).get("name") != tool["name"]:
                        raise MCPFailure(f"{candidate}: native search tool emission failed")
                    arguments = json.loads(calls[0]["function"]["arguments"])
                    if arguments.get("search_depth", "basic") != "basic" or arguments.get("max_results", 3) != 3:
                        raise MCPFailure(f"{candidate}: emitted search settings differ from probe instructions")
                    from importlib import import_module
                    import_module("jsonschema").validate(arguments, tool["inputSchema"])
                    mapped = await BFCLSearchAdapter(search).search_engine_query(arguments["query"], max_results=3)
                    if not mapped:
                        raise MCPFailure(f"{candidate}: delegated Tavily search returned no results")
                    evidence["candidates"].append({"candidate_id": candidate, "native_tool_call_passed": True, "mcp_execution_passed": True, "result_count": len(mapped), "openrouter_cost_usd": exchange["response"]["usage"]["cost"]})
                    print(f"{candidate}: native tool call -> Tavily MCP -> result mapping PASS", flush=True)
            elif args.command == "live-search-smoke":
                mapped = await BFCLSearchAdapter(search).search_engine_query(args.query, max_results=3)
                if not mapped:
                    raise MCPFailure("live search returned no results for the integration probe")
                evidence.update(result_count=len(mapped), mapped_results=mapped, credits_reserved_upper_bound=config["credit_upper_bound_per_call"])
        evidence["usage_after"] = await fetch_tavily_usage(key, cache_path=usage_cache)
        evidence["status"] = "passed"
        label = {"live-search-smoke": "MCP search", "live-tavily-model-smoke": "five-model delegated search" if getattr(args, "candidate", None) == "all" else "model delegated search", "tavily-preflight": "authentication and tool discovery"}[args.command]
        print(f"Tavily {label}: PASS")
        return 0
    except Exception as exc:
        evidence.update(status="integration_error", error=redact(str(exc)))
        print(f"Tavily integration: BLOCKED ({redact(str(exc))})", file=sys.stderr)
        return 2
    finally:
        evidence["search_credits_reserved_cumulative_upper_bound"] = budget.spent_upper_bound()
        evidence["search_config_sha256"] = hashlib.sha256((ROOT / "evaluation" / "search.json").read_bytes()).hexdigest()
        (directory / "validation.json").write_text(json.dumps(redact(evidence), indent=2) + "\n", encoding="utf-8")
        print(f"Retained search evidence: {directory}")


def main() -> int:
    load_credentials(ROOT / ".env")
    args = parse_args()
    if args.command == "run-integration-suite":
        return _run_integration_suite(args)
    if args.command == "run-bfcl-diagnostic":
        return _run_bfcl_diagnostic(args)
    if args.command == "run-mcp-diagnostic":
        return _run_mcp_diagnostic(args)
    if args.command == "run-repository-diagnostic":
        return _run_repository_diagnostic(args)
    if args.command == "ingest-worker":
        output = args.output.resolve()
        if not output.is_relative_to((ROOT / "evaluation/runs").resolve()) or output.exists():
            raise EvaluationBlocked("normalized output must be a new retained file below evaluation/runs")
        record = ingest_worker_verdict(root=ROOT, manifest_path=args.manifest, proxy_log=args.proxy_log, verifier_paths=args.verifier, registry=load_json(ROOT / "evaluation/candidates.json"), protocol=load_json(ROOT / "evaluation/protocol.json"), providers=load_json(ROOT / "evaluation/providers.json"))
        errors = validate_result(record) + validate_artifact_hashes(record, ROOT)
        if errors:
            raise EvaluationBlocked("; ".join(errors))
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
        print("Official worker result normalized; evidence kind: " + record["run"]["evidence_kind"])
        return 0
    if args.command == "freeze-mcp-fixture":
        suites = load_json(ROOT / "evaluation/suites.json")["suites"]
        categories = {task.split("/")[1] for suite in suites.values() for task in suite["mcp"]["task_ids"]}
        if args.category not in categories:
            raise EvaluationBlocked("fixture category is not in the frozen suites")
        snapshot_archive(args.source, ROOT / "evaluation/runs/fixture-snapshots" / (args.category + ".zip"), source_url=f"https://storage.mcpmark.ai/filesystem/{args.category}.zip", source_revision=load_json(ROOT / "evaluation/harnesses/mcpmark.json")["revision"])
        print("Fixture archive retained with its initial SHA-256; trial-time downloads are disabled.")
        return 0
    if args.command == "run-suite":
        return _run_suite(args)
    if args.command in {"tavily-preflight", "live-search-smoke", "live-tavily-model-smoke"}:
        return asyncio.run(_tavily_check(args))
    if args.command == "budget-report":
        print(json.dumps(lifetime_accounting(ROOT), indent=2))
        return 0
    if args.command == "validate":
        errors = validate_control_files(ROOT)
        if errors:
            for error in errors:
                print(f"ERROR: {error}", file=sys.stderr)
            return 1
        print("Phase 1 evaluation controls are valid.")
        return 0

    if args.command == "preflight":
        failures = _preflight()
        if failures:
            for failure in failures:
                print(f"BLOCKED: {failure}")
            return 1
        print("Phase 1 live-evaluation preflight passed.")
        return 0

    if args.command == "plan-suite":
        registry = load_json(ROOT / "evaluation" / "candidates.json")
        candidates = [item["id"] for item in registry["candidates"]]
        if args.candidate != "all":
            if args.candidate not in candidates:
                raise EvaluationBlocked("unknown candidate")
            candidates = [args.candidate]
        try:
            plan = build_suite_plan(
                load_json(ROOT / "evaluation" / "suites.json"), args.suite,
                load_json(ROOT / "evaluation" / "protocol.json")["generation"]["attempts"], candidates,
            )
        except ValueError as exc:
            raise EvaluationBlocked(str(exc)) from exc
        print(json.dumps({"suite": args.suite, "execution_units": plan}, indent=2))
        return 0

    if args.command == "validate-result":
        path = args.path if args.path.is_absolute() else ROOT / args.path
        errors = validate_result(load_json(path))
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return int(bool(errors))

    if args.command == "live-tool-smoke":
        providers, aggregate_cap = load_provider_configs(ROOT / "evaluation" / "providers.json")
        scope = _budget(args)
        candidates = list(providers) if args.candidate == "all" else [args.candidate]
        failed = False
        for candidate_id in candidates:
            failed = bool(_tool_probe(candidate_id, scope.identity)) or failed
        return int(failed)

    if args.command == "prepare-harbor":
        protocol, harness, candidate = _harbor_inputs(args.candidate)
        config = build_harbor_job_config(
            protocol,
            harness,
            candidate_id=args.candidate,
            task_id=args.task,
            attempt=args.attempt,
            run_id=args.run_id,
        )
        errors = audit_harbor_job_config(config, protocol, harness)
        if errors:
            for error in errors:
                print(f"ERROR: {error}", file=sys.stderr)
            return 1
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote audited Harbor config: {output}")
        return 0

    if args.command == "run-harbor":
        return _run_harbor(args)

    if args.command == "audit-harbor-config":
        path = args.path if args.path.is_absolute() else ROOT / args.path
        config = load_json(path)
        metadata = config.get("metadata", {})
        candidate_id = metadata.get("arcus_candidate_id")
        if not candidate_id:
            agent_model = config.get("agents", [{}])[0].get("model_name", "")
            candidate_id = agent_model.removeprefix("openai/")
        protocol, harness, _ = _harbor_inputs(candidate_id)
        errors = audit_harbor_job_config(config, protocol, harness)
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        if not errors:
            print("Harbor configuration matches the frozen Arcus protocol.")
        return int(bool(errors))

    if args.command == "audit-harbor-run":
        path = args.path if args.path.is_absolute() else ROOT / args.path
        state = load_harbor_result_state(path)
        print(json.dumps(state, indent=2))
        return int(not state["complete"])

    if args.command == "stop-harbor-containers":
        path = args.path if args.path.is_absolute() else ROOT / args.path
        runs = (ROOT / "evaluation" / "runs" / "harbor").resolve()
        if runs not in path.resolve().parents:
            raise ValueError("container cleanup requires one job directory below evaluation/runs/harbor")
        print(json.dumps({"stopped_container_ids": stop_harbor_containers(path)}))
        return 0

    if args.command == "ingest-harbor":
        protocol, harness, _ = _harbor_inputs(args.candidate)
        registry = load_json(ROOT / "evaluation" / "candidates.json")
        providers = load_json(ROOT / "evaluation" / "providers.json")
        job_dir = args.job_dir if args.job_dir.is_absolute() else ROOT / args.job_dir
        proxy_log = args.proxy_log if args.proxy_log.is_absolute() else ROOT / args.proxy_log
        record = ingest_harbor_job(
            job_dir,
            proxy_log,
            candidate_id=args.candidate,
            attempt=args.attempt,
            registry=registry,
            protocol=protocol,
            providers=providers,
            harness=harness,
        )
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"Retained normalized outcome: {record['outcome']['status']} ({output})")
        return 0

    if args.command == "score":
        protocol = load_json(ROOT / "evaluation" / "protocol.json")
        weights = {item["id"]: float(item["weight"]) for item in protocol["harnesses"]}
        if bool(args.paths) == bool(args.results_dir):
            raise EvaluationBlocked("provide result paths or --results-dir, not both or neither")
        paths = args.paths
        if args.results_dir:
            directory = (args.results_dir if args.results_dir.is_absolute() else ROOT / args.results_dir).resolve()
            if not directory.is_relative_to((ROOT / "evaluation/runs").resolve()) or not directory.is_dir():
                raise EvaluationBlocked("results directory must remain below evaluation/runs")
            paths = sorted(path for path in directory.glob("*.json") if path.name not in {"suite-state.json", "dry-run.json"})
            if not paths:
                raise EvaluationBlocked("results directory contains no normalized outcomes")
        records = [load_json(path if path.is_absolute() else ROOT / path) for path in paths]
        registry = load_json(ROOT / "evaluation" / "candidates.json")
        revisions = {item["id"]: item["revision"] for item in registry["candidates"]}
        provider_configs, _ = load_provider_configs(ROOT / "evaluation" / "providers.json")
        harness_revisions = {item["id"]: item["revision"] for item in protocol["harnesses"]}
        for record in records:
            errors = validate_result(record)
            errors += validate_artifact_hashes(record, ROOT)
            if record["outcome"].get("status") in {"passed", "failed"}:
                if record.get("run", {}).get("contract_revision") != protocol["contract_revision"]:
                    errors.append("evaluation contract revision differs from frozen protocol")
                if record.get("search", {}).get("contract_id") != protocol["search"]["contract_id"]:
                    errors.append("search contract differs from frozen protocol")
                for key, expected in protocol["generation"].items():
                    if record["generation"].get(key) != expected:
                        errors.append(f"generation.{key} differs from frozen protocol")
                candidate_id = record["model"].get("candidate_id")
                provider_config = provider_configs.get(candidate_id)
                if record["model"].get("revision") != revisions.get(candidate_id):
                    errors.append("model revision differs from frozen candidate")
                if record["harness"].get("revision") != harness_revisions.get(record["harness"].get("id")):
                    errors.append("harness revision differs from frozen protocol")
                if provider_config is None or any(
                    record["provider"].get(key) != expected
                    for key, expected in (
                        ("upstream", provider_config.upstream if provider_config else None),
                        ("provider_slug", provider_config.provider_slug if provider_config else None),
                        ("model", provider_config.model if provider_config else None),
                    )
                ):
                    errors.append("provider identity differs from frozen pin")
            if errors:
                raise EvaluationBlocked("result cannot enter scorecard: " + "; ".join(errors))
        try:
            expected = expected_coverage(load_json(ROOT / "evaluation" / "suites.json"), args.suite, protocol["generation"]["attempts"])
            rows = score_candidates(records, weights, expected=expected)
        except ValueError as exc:
            raise EvaluationBlocked(str(exc)) from exc
        if args.suite == "smoke":
            for row in rows:
                row["smoke_score"] = row["weighted_score"]
                row["weighted_score"] = None
                row["qualification_complete"] = False
        rendered = json.dumps({"schema_version": 1, "suite": args.suite, "scorecard": rows}, indent=2) + "\n"
        if args.output:
            output = args.output if args.output.is_absolute() else ROOT / args.output
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
        return 0

    registry = load_json(ROOT / "evaluation" / "candidates.json")
    protocol = load_json(ROOT / "evaluation" / "protocol.json")
    run_id = args.run_id or f"{args.candidate}-{args.harness}-seed{args.seed}"
    record = build_result_skeleton(
        registry,
        protocol,
        candidate_id=args.candidate,
        harness_id=args.harness,
        upstream_provider=args.upstream_provider,
        precision=args.precision,
        parser=args.parser,
        run_id=run_id,
        seed=args.seed,
    )
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote unrun result template: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvaluationBlocked as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    except (subprocess.CalledProcessError, OSError, ValueError) as exc:
        print("Evaluation error: " + redact(str(exc)), file=sys.stderr)
        raise SystemExit(1) from None
