"""BFCL-shaped search over Tavily MCP; this is explicitly an adapted benchmark."""
from __future__ import annotations

from typing import Any
from collections.abc import Callable
from urllib.parse import urlsplit

from evaluation.mcp_client import MCPFailure
from evaluation.web_fetch import fetch_url_content


class BFCLSearchAdapter:
    def __init__(self, search: Any):
        self.search = search
        self.show_snippet = True

    def _load_scenario(self, initial_config: dict[str, Any], long_context: bool = False):
        self.show_snippet = initial_config["show_snippet"]

    async def search_engine_query(self, keywords: str, max_results: int = 10, region: str = "wt-wt") -> list[dict[str, str]]:
        if region != "wt-wt":
            raise ValueError("Tavily adaptation cannot reproduce BFCL region semantics; non-default region rejected")
        response = await self.search.search(keywords, max_results=max_results)
        mapped = []
        for row in response["results"][:max_results]:
            if not isinstance(row, dict) or not isinstance(row.get("title"), str) or not isinstance(row.get("url"), str):
                raise MCPFailure("Tavily result is missing a title or URL")
            if urlsplit(row["url"]).scheme not in {"https", "http"}:
                raise MCPFailure("Tavily result URL has an unsupported scheme")
            result = {"title": row["title"], "href": row["url"]}
            if self.show_snippet:
                content = row.get("content")
                if not isinstance(content, str):
                    raise MCPFailure("Tavily result is missing its snippet")
                result["body"] = content
            mapped.append(result)
        return mapped


class SyncBFCLSearchAdapter:
    """Official synchronous tool surface; caller owns the persistent MCP event loop."""
    def __init__(self, adapter: BFCLSearchAdapter, run_coroutine: Callable):
        self.adapter = adapter
        self.run_coroutine = run_coroutine
        self._api_description = "Web search and public HTTPS page fetching (Arcus Tavily adaptation)."

    def _load_scenario(self, initial_config: dict[str, Any], long_context: bool = False):
        self.adapter._load_scenario(initial_config, long_context)

    def search_engine_query(self, keywords: str, max_results: int = 10, region: str = "wt-wt") -> list[dict[str, str]]:
        return self.run_coroutine(self.adapter.search_engine_query(keywords, max_results, region))

    def fetch_url_content(self, url: str, mode: str = "raw") -> dict[str, str]:
        return fetch_url_content(url, mode)
