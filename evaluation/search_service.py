"""Host-owned serialized Tavily MCP lifecycle and credential-free worker RPC."""
from __future__ import annotations

import asyncio
from concurrent.futures import TimeoutError as FutureTimeout
from importlib import import_module
import json
from pathlib import Path
import threading
from typing import Any

from evaluation.mcp_client import MCPClient, MCPFailure
from evaluation.search_adapter import BFCLSearchAdapter
from evaluation.secrets import redact
from evaluation.tavily import SearchBudget, TavilySearch
from evaluation.web_fetch import bounded_fetch, FetchFailure
from evaluation.worker_adapters import validate_gateway


class SearchService:
    def __init__(self, config: dict, budget: SearchBudget, artifact_dir: Path, *, trial_id: str, api_key: str, usage_cache_path: Path | None = None, client_factory=MCPClient):
        self.config, self.budget, self.artifact_dir = config, budget, artifact_dir
        self.trial_id, self.api_key = trial_id, api_key
        self.usage_cache_path, self.client_factory = usage_cache_path, client_factory
        self.ready = threading.Event()
        self.thread = None
        self.loop = None
        self.queue = None
        self.task = None
        self.failure = None
        self.failed_calls = 0
        self.fetch_calls = 0

    async def _serve(self):
        self.loop = asyncio.get_running_loop()
        self.task = asyncio.current_task()
        self.queue = asyncio.Queue()
        try:
            async with self.client_factory(self.config["endpoint"], api_key=self.api_key, timeout_seconds=self.config["timeout_seconds"], usage_cache_path=self.usage_cache_path) as client:
                search = TavilySearch(client, self.config, self.budget, self.artifact_dir, trial_id=self.trial_id)
                await search.discover()
                adapter = BFCLSearchAdapter(search)
                self.ready.set()
                while True:
                    item = await self.queue.get()
                    if item is None:
                        return
                    method, arguments, future = item
                    if future.cancelled():
                        continue
                    try:
                        if method == "search_engine_query":
                            allowed = {"keywords", "max_results", "region", "show_snippet"}
                            if set(arguments) - allowed or type(arguments.get("show_snippet", True)) is not bool:
                                raise ValueError("unsupported search arguments")
                            adapter.show_snippet = arguments.get("show_snippet", True)
                            result = await adapter.search_engine_query(arguments["keywords"], arguments.get("max_results", 10), arguments.get("region", "wt-wt"))
                        elif method == "fetch_url_content":
                            if set(arguments) - {"url", "mode"} or self.fetch_calls >= self.config["calls_per_trial"]:
                                raise ValueError("unsupported fetch arguments or trial fetch limit reached")
                            self.fetch_calls += 1
                            event = {"trial_id": self.trial_id, "method": method, "arguments": arguments}
                            try:
                                result = await asyncio.to_thread(bounded_fetch, arguments["url"], arguments.get("mode", "raw"))
                                event.update(status="completed", response=result)
                            except FetchFailure as exc:
                                event.update(status="fetch_error", error=exc.details)
                                if self.config.get("page_error_policy") != "http_3xx_4xx_except_408_429_tool_error" or not exc.details["recoverable"]:
                                    raise
                                result = {"error": exc.details}
                                event.update(status="tool_error", response=result)
                            except Exception as exc:
                                event.update(status="fetch_error", error=redact(str(exc)))
                                raise
                            finally:
                                with (self.artifact_dir / "fetch-exchanges.jsonl").open("a", encoding="utf-8") as log:
                                    log.write(json.dumps(redact(event), ensure_ascii=False) + "\n")
                        else:
                            raise ValueError("unsupported search-service method")
                        if not future.done():
                            future.set_result(result)
                    except Exception as exc:
                        self.failed_calls += 1
                        error = MCPFailure(redact(str(exc), extra_secrets=(self.api_key,)))
                        if not future.done():
                            future.set_exception(error)
                        # Failed calls remain visible even if the official tool executor catches them.
                        with (self.artifact_dir / "service-errors.jsonl").open("a", encoding="utf-8") as log:
                            log.write(json.dumps({"trial_id": self.trial_id, "method": method, "error": str(error)}) + "\n")
        except BaseException as exc:
            self.failure = redact(f"{type(exc).__name__}: {exc}", extra_secrets=(self.api_key,))
            self.ready.set()

    def start(self):
        if self.thread is not None:
            raise ValueError("search service cannot be restarted or reused")
        self.thread = threading.Thread(target=lambda: asyncio.run(self._serve()), daemon=True)
        self.thread.start()
        if not self.ready.wait(60) or self.failure:
            self.close()
            raise MCPFailure(self.failure or "search service initialization timed out")
        return self

    async def _submit(self, method, arguments):
        future = asyncio.get_running_loop().create_future()
        await self.queue.put((method, arguments, future))
        return await future

    def call(self, method: str, arguments: dict, *, trial_id: str):
        if trial_id != self.trial_id or not isinstance(arguments, dict):
            raise ValueError("search RPC trial identity or arguments invalid")
        if self.thread is None or not self.thread.is_alive() or self.failure:
            raise MCPFailure(self.failure or "search service is not running")
        pending = asyncio.run_coroutine_threadsafe(self._submit(method, arguments), self.loop)
        try:
            return pending.result(timeout=55)
        except FutureTimeout:
            self.failed_calls += 1
            self.failure = "search RPC timed out; no execution retry is allowed"
            pending.cancel()
            self.loop.call_soon_threadsafe(self.task.cancel)
            raise MCPFailure(self.failure) from None

    def close(self):
        if self.thread is None:
            return
        if self.thread.is_alive() and self.loop is not None:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, None)
            self.thread.join(timeout=55)
            if self.thread.is_alive():
                self.loop.call_soon_threadsafe(self.task.cancel)
                raise MCPFailure("search service cleanup exceeded its deadline")


class RemoteSearchAPI:
    """BFCL's synchronous tool interface; only the local trial token enters the worker."""
    def __init__(self, gateway_url: str, proxy_token: str, trial_id: str):
        validate_gateway(gateway_url, proxy_token)
        self.url = gateway_url.rstrip("/")[:-3] + "/arcus/tools"
        self.token, self.trial_id = proxy_token, trial_id
        self.show_snippet = True
        self._api_description = "Arcus Tavily-adapted web search and public HTTPS fetching."

    def _load_scenario(self, initial_config: dict, long_context: bool = False):
        if type(initial_config.get("show_snippet")) is not bool:
            raise ValueError("invalid BFCL search scenario")
        self.show_snippet = initial_config["show_snippet"]

    def _call(self, method: str, arguments: dict) -> Any:
        httpx = import_module("httpx")
        with httpx.Client(timeout=60, follow_redirects=False, trust_env=False) as client:
            response = client.post(self.url, headers={"Authorization": "Bearer " + self.token}, json={"trial_id": self.trial_id, "method": method, "arguments": arguments})
            data = response.json()
            if response.status_code != 200:
                raise MCPFailure(redact(data.get("error", {}).get("message", "search RPC failed"), extra_secrets=(self.token,)))
            return data["result"]

    def search_engine_query(self, keywords: str, max_results: int = 10, region: str = "wt-wt"):
        return self._call("search_engine_query", {"keywords": keywords, "max_results": max_results, "region": region, "show_snippet": self.show_snippet})

    def fetch_url_content(self, url: str, mode: str = "raw"):
        return self._call("fetch_url_content", {"url": url, "mode": mode})
