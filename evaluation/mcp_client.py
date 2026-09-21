"""Thin lifecycle wrapper around the pinned official MCP Python SDK."""
from __future__ import annotations

from contextlib import AsyncExitStack
import asyncio
import time
import hashlib
import json
from pathlib import Path
from datetime import timedelta
from importlib import import_module
from typing import Any
from urllib.parse import urlsplit

from evaluation.secrets import redact


class MCPFailure(RuntimeError):
    """Sanitized transport, protocol, or tool failure; not a model failure."""


def error_summary(exc: BaseException) -> str:
    children = getattr(exc, "exceptions", None)
    if children:
        return "; ".join(error_summary(child) for child in children)
    return f"{type(exc).__name__}: {exc}"


async def fetch_tavily_usage(api_key: str, *, cache_path: Path | None = None, max_age_seconds: int = 600) -> dict[str, Any]:
    if not api_key:
        raise MCPFailure("TAVILY_API_KEY is not configured")
    fingerprint = hashlib.sha256(api_key.encode()).hexdigest()
    if cache_path is not None and cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        age = time.time() - cached.get("checked_at_epoch", 0)
        if cached.get("credential_fingerprint") == fingerprint and 0 <= age < max_age_seconds:
            return {**cached["usage"], "usage_snapshot_age_seconds": round(age), "usage_snapshot_cached": True}
    try:
        httpx = import_module("httpx")
        async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
            response = await client.get("https://api.tavily.com/usage", headers={"Authorization": f"Bearer {api_key}"})
            if response.status_code == 429:
                try:
                    delay = max(1.0, float(response.headers.get("Retry-After", 60)))
                except ValueError:
                    delay = 60.0
                if delay > 10:
                    raise MCPFailure(f"Tavily usage endpoint is rate-limited; retry after {delay:.0f} seconds. No early retry attempted.")
                await asyncio.sleep(delay)
                response = await client.get("https://api.tavily.com/usage", headers={"Authorization": f"Bearer {api_key}"})
            response.raise_for_status()
            result = redact(response.json(), extra_secrets=(api_key,))
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps({"credential_fingerprint": fingerprint, "checked_at_epoch": time.time(), "usage": result}) + "\n", encoding="utf-8")
            return result
    except Exception as exc:
        raise MCPFailure(redact(f"Tavily usage check failed: {exc}", extra_secrets=(api_key,))) from None


class MCPClient:
    def __init__(self, endpoint: str, *, api_key: str, timeout_seconds: float = 45, usage_cache_path: Path | None = None):
        parsed = urlsplit(endpoint)
        if parsed.scheme != "https" or parsed.query or parsed.fragment or parsed.username or parsed.password:
            raise ValueError("MCP endpoint must use HTTPS without credentials or query parameters")
        if not api_key or not 0 < timeout_seconds <= 60:
            raise ValueError("credential and bounded timeout are required")
        self.endpoint = endpoint
        self._api_key = api_key
        self.timeout = timeout_seconds
        self.stack = AsyncExitStack()
        self.session = None
        self.initialization: dict[str, Any] = {}
        self.usage_snapshot: dict[str, Any] | None = None
        self.usage_checked_monotonic = 0.0
        self.usage_cache_path = usage_cache_path

    async def __aenter__(self):
        try:
            httpx = import_module("httpx")
            transport = import_module("mcp.client.streamable_http").streamable_http_client
            session_class = import_module("mcp").ClientSession
            http = await self.stack.enter_async_context(httpx.AsyncClient(
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=httpx.Timeout(self.timeout), follow_redirects=False,
            ))
            read, write, _ = await self.stack.enter_async_context(transport(self.endpoint, http_client=http))
            self.session = await self.stack.enter_async_context(session_class(read, write, read_timeout_seconds=timedelta(seconds=self.timeout)))
            result = await self.session.initialize()
            self.initialization = redact(result.model_dump(mode="json", by_alias=True), extra_secrets=(self._api_key,))
            return self
        except Exception as exc:
            await self.stack.aclose()
            self.session = None
            raise MCPFailure(redact(f"MCP initialization failed: {exc}", extra_secrets=(self._api_key,))) from None

    async def __aexit__(self, exc_type, exc, traceback):
        try:
            return await self.stack.__aexit__(exc_type, exc, traceback)
        except Exception as failure:
            raise MCPFailure(redact(error_summary(failure), extra_secrets=(self._api_key,))) from None
        finally:
            self.session = None

    async def list_tools(self) -> list[dict[str, Any]]:
        if self.session is None:
            raise MCPFailure("MCP session is not initialized")
        tools = []
        cursor = None
        seen = set()
        for _ in range(10):
            result = await self.session.list_tools(cursor=cursor)
            tools.extend(tool.model_dump(mode="json", by_alias=True) for tool in result.tools)
            cursor = result.nextCursor
            if not cursor:
                return redact(tools, extra_secrets=(self._api_key,))
            if cursor in seen:
                break
            seen.add(cursor)
        raise MCPFailure("MCP tool discovery pagination exceeded its bound")

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.session is None:
            raise MCPFailure("MCP session is not initialized")
        try:
            result = await self.session.call_tool(name, arguments, read_timeout_seconds=timedelta(seconds=self.timeout))
            return redact(result.model_dump(mode="json", by_alias=True), extra_secrets=(self._api_key,))
        except Exception as exc:
            # No application-level tools/call retry: a timed-out call may be billed.
            raise MCPFailure(redact(f"MCP tool call failed: {exc}", extra_secrets=(self._api_key,))) from None

    async def usage(self) -> dict[str, Any]:
        if self.usage_snapshot is None or time.monotonic() - self.usage_checked_monotonic > 600:
            self.usage_snapshot = await fetch_tavily_usage(self._api_key, cache_path=self.usage_cache_path)
            self.usage_checked_monotonic = time.monotonic()
        return self.usage_snapshot
