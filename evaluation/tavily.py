"""Tavily MCP search with conservative, persistent credit authorization."""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.mcp_client import MCPFailure
from evaluation.provider import EvaluationBlocked
from evaluation.secrets import redact


def validate_search_config(config: dict[str, Any]) -> list[str]:
    errors = []
    fixed = {"schema_version": 1, "provider": "tavily", "transport": "streamable_http", "endpoint": "https://mcp.tavily.com/mcp/", "credential_env": "TAVILY_API_KEY", "search_depth": "basic", "tool_call_retries": 0, "official_bfcl_comparable": False, "region_policy": "reject_non_default"}
    for key, expected in fixed.items():
        if config.get(key) != expected:
            errors.append(f"search.{key} differs from the supported contract")
    for key in ("approved_credit_cap", "calls_per_trial", "credit_upper_bound_per_call", "max_results"):
        if type(config.get(key)) is not int or config[key] < (0 if key == "approved_credit_cap" else 1):
            errors.append(f"search.{key} must be a bounded nonnegative integer")
    if config.get("max_results", 100) > 10 or config.get("credit_upper_bound_per_call", 0) < 2:
        errors.append("search result/credit bounds are unsafe")
    if not isinstance(config.get("timeout_seconds"), (int, float)) or not 0 < config["timeout_seconds"] <= 60:
        errors.append("search timeout must be within (0,60]")
    if config.get("include_raw_content") is not False or config.get("include_answer") is not False:
        errors.append("search cannot enable additional answer/content features")
    if not config.get("contract_id"):
        errors.append("search.contract_id is required")
    return errors


def free_credits_remaining(usage: dict[str, Any]) -> int:
    account, key = usage.get("account", {}), usage.get("key", {})
    if str(account.get("current_plan", "")).casefold() not in {"free", "researcher"}:
        raise EvaluationBlocked("Tavily account is not confirmed as a free plan")
    if account.get("paygo_usage") not in (0, None) or account.get("paygo_limit") not in (0, None):
        raise EvaluationBlocked("Tavily pay-as-you-go is enabled; free-only testing requires it disabled")
    for field in ("plan_usage", "plan_limit"):
        if type(account.get(field)) is not int or account[field] < 0:
            raise EvaluationBlocked("Tavily account omitted usable credit limits")
    left = min(1000, account["plan_limit"]) - account["plan_usage"]
    if key.get("limit") is not None:
        if type(key.get("limit")) is not int or type(key.get("usage")) is not int:
            raise EvaluationBlocked("Tavily key omitted usable credit limits")
        left = min(left, key["limit"] - key["usage"])
    return max(0, left)


class SearchBudget:
    """Debit the conservative maximum BEFORE sending a call, including uncertain failures.

    SQLite transactions make this cap shared across processes. A timeout is never
    refunded automatically: the upstream call may have executed and been billed.
    """
    def __init__(self, path: Path, *, cap: int, contract_id: str):
        if type(cap) is not int or cap < 0:
            raise ValueError("search credit cap must be a nonnegative integer")
        self.path, self.cap, self.contract_id = path, cap, contract_id

    def reserve(self, amount: int, *, free_remaining: int | None = None) -> None:
        if type(amount) is not int or amount < 1:
            raise ValueError("credit reservation must be positive")
        if not self.cap:
            raise EvaluationBlocked("Tavily credit cap is zero; no tool call was attempted")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=5)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("CREATE TABLE IF NOT EXISTS debits (id INTEGER PRIMARY KEY, contract TEXT NOT NULL, credits INTEGER NOT NULL, timestamp TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS allowance (id INTEGER PRIMARY KEY CHECK(id=1), ceiling INTEGER NOT NULL)")
            initial_ceiling = min(self.cap, free_remaining) if free_remaining is not None else self.cap
            db.execute("INSERT OR IGNORE INTO allowance(id,ceiling) VALUES(1,?)", (initial_ceiling,))
            ceiling = min(self.cap, db.execute("SELECT ceiling FROM allowance WHERE id=1").fetchone()[0])
            spent = db.execute("SELECT COALESCE(SUM(credits),0) FROM debits").fetchone()[0]
            if spent + amount > ceiling:
                raise EvaluationBlocked("Tavily call would exceed the approved credit cap")
            db.execute("INSERT INTO debits(contract,credits,timestamp) VALUES(?,?,?)", (self.contract_id, amount, datetime.now(timezone.utc).isoformat()))

    def spent_upper_bound(self) -> int:
        if not self.path.exists():
            return 0
        with closing(sqlite3.connect(self.path, timeout=5)) as db:
            return db.execute("SELECT COALESCE(SUM(credits),0) FROM debits").fetchone()[0]


def parse_search_result(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("isError"):
        raise MCPFailure("Tavily returned an MCP tool error; retained as a search-provider failure")
    structured = result.get("structuredContent")
    if isinstance(structured, dict) and isinstance(structured.get("results"), list):
        return structured
    for content in result.get("content", []):
        if content.get("type") == "text":
            try:
                data = json.loads(content.get("text", ""))
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(data, dict) and isinstance(data.get("results"), list):
                return data
    raise MCPFailure("Tavily response omitted structured search results")


class TavilySearch:
    def __init__(self, client: Any, config: dict[str, Any], budget: SearchBudget, artifact_dir: Path, *, trial_id: str):
        errors = validate_search_config(config)
        if errors:
            raise ValueError("; ".join(errors))
        if not trial_id:
            raise ValueError("search trial_id is required")
        self.client, self.config, self.budget, self.artifact_dir, self.trial_id = client, config, budget, artifact_dir, trial_id
        self.tool_name = None
        self.calls = 0
        self.properties: dict[str, Any] = {}
        if (artifact_dir / "search-exchanges.jsonl").exists():
            raise EvaluationBlocked("search artifacts already exist; use an isolated new trial")

    async def discover(self) -> dict[str, Any]:
        tools = await self.client.list_tools()
        aliases = self.config["tool_name_aliases"]
        matches = [tool for tool in tools if tool.get("name") in aliases]
        if len(matches) != 1:
            raise MCPFailure("Tavily search tool is missing or ambiguous")
        tool = matches[0]
        properties = tool.get("inputSchema", {}).get("properties", {})
        if not all(key in properties for key in ("query", "search_depth", "max_results")):
            raise MCPFailure("Tavily search schema cannot enforce the frozen settings")
        self.tool_name = tool["name"]
        self.properties = properties
        evidence = redact({"initialization": self.client.initialization, "tools": tools, "selected_tool": self.tool_name, "contract_id": self.config["contract_id"]})
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        (self.artifact_dir / "mcp-discovery.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        return evidence

    async def search(self, query: str, *, max_results: int = 10) -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip() or len(query) > 4096:
            raise ValueError("search query must be nonempty and at most 4096 characters")
        if type(max_results) is not int or not 1 <= max_results <= self.config["max_results"]:
            raise ValueError("search max_results is outside the frozen bound")
        if self.tool_name is None:
            raise MCPFailure("discover the Tavily search tool before calling it")
        if self.calls >= self.config["calls_per_trial"]:
            raise EvaluationBlocked("Tavily trial tool-call limit reached")
        arguments = {"query": query, "search_depth": self.config["search_depth"], "max_results": max_results, "include_raw_content": False}
        # Hosted Tavily MCP does not advertise include_answer; never send extras
        # against additionalProperties:false. Pin it only if exposed by the server.
        if "include_answer" in self.properties:
            arguments["include_answer"] = False
        credits = self.config["credit_upper_bound_per_call"]
        free_remaining = None
        if self.config.get("require_usage_preflight"):
            usage = await self.client.usage()
            free_remaining = free_credits_remaining(usage)
            if free_remaining < credits:
                raise EvaluationBlocked("Tavily free-plan allowance is insufficient for this call")
        self.budget.reserve(credits, free_remaining=free_remaining)
        self.calls += 1
        exchange: dict[str, Any] = {"trial_id": self.trial_id, "contract_id": self.config["contract_id"], "tool": self.tool_name, "arguments": arguments, "credits_upper_bound": credits, "timestamp": datetime.now(timezone.utc).isoformat()}
        try:
            response = await self.client.call_tool(self.tool_name, arguments)
            exchange["response"] = response
            parsed = parse_search_result(response)
            exchange["status"] = "completed"
            return parsed
        except Exception as exc:
            exchange.update(status="search_provider_error", error=redact(str(exc)))
            raise
        finally:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            with (self.artifact_dir / "search-exchanges.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(redact(exchange), ensure_ascii=False) + "\n")
