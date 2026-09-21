from contextlib import asynccontextmanager
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from evaluation.web_fetch import FetchFailure
from evaluation.mcp_client import MCPFailure
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from evaluation.proxy import ArcusProxyServer
from evaluation.search_service import SearchService
from evaluation.tavily import SearchBudget


ROOT = Path(__file__).resolve().parents[2]


class FakeClient:
    initialization = {"server": "fake"}
    calls = 0
    async def list_tools(self):
        return [{"name": "tavily_search", "inputSchema": {"properties": {"query": {}, "search_depth": {}, "max_results": {}}}}]
    async def usage(self):
        return {"account": {"current_plan": "Researcher", "plan_usage": 0, "plan_limit": 1000, "paygo_usage": 0, "paygo_limit": None}, "key": {"limit": None}}
    async def call_tool(self, name, arguments):
        self.calls += 1
        return {"structuredContent": {"results": [{"title": "test", "url": "https://example.com", "content": "snippet"}]}}


class SearchServiceTests(unittest.TestCase):
    def test_page_error_allows_another_source_but_timeout_is_fatal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            @asynccontextmanager
            async def factory(*args, **kwargs):
                yield FakeClient()
            config = json.loads((ROOT / "evaluation/search.json").read_text())
            service = SearchService(config, SearchBudget(root / "budget.sqlite3", cap=1000,
                contract_id=config["contract_id"]), root / "search", trial_id="trial",
                api_key="fake", client_factory=factory).start()
            try:
                with patch("evaluation.search_service.bounded_fetch", side_effect=[FetchFailure("http_status", 403), {"content": "alternative"}, FetchFailure("timeout")]) as fetch:
                    result = service.call("fetch_url_content", {"url": "https://example.com/blocked"}, trial_id="trial")
                    self.assertEqual(result, {"error": FetchFailure("http_status", 403).details})
                    self.assertEqual(service.failed_calls, 0)
                    result = service.call("fetch_url_content", {"url": "https://example.com/alternative"}, trial_id="trial")
                    self.assertEqual(result, {"content": "alternative"})
                    with self.assertRaises(MCPFailure):
                        service.call("fetch_url_content", {"url": "https://example.com/slow"}, trial_id="trial")
                    self.assertEqual(fetch.call_count, 3)
                self.assertEqual(service.failed_calls, 1)
                self.assertEqual(service.fetch_calls, 3)
                events = [json.loads(line) for line in (root / "search/fetch-exchanges.jsonl").read_text().splitlines()]
                self.assertEqual([e["status"] for e in events], ["tool_error", "completed", "fetch_error"])
            finally:
                service.close()

    def test_persistent_session_mapping_identity_and_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = FakeClient()
            @asynccontextmanager
            async def factory(*args, **kwargs):
                yield client
            config = json.loads((ROOT / "evaluation/search.json").read_text())
            budget = SearchBudget(root / "budget.sqlite3", cap=1000, contract_id=config["contract_id"])
            service = SearchService(config, budget, root / "search", trial_id="trial", api_key="fake", client_factory=factory).start()
            try:
                for _ in range(2):
                    result = service.call("search_engine_query", {"keywords": "test", "show_snippet": False}, trial_id="trial")
                    self.assertNotIn("body", result[0])
                self.assertEqual(client.calls, 2)
                self.assertEqual(budget.spent_upper_bound(), 4)
                with self.assertRaises(ValueError):
                    service.call("search_engine_query", {"keywords": "test"}, trial_id="different")
            finally:
                service.close()
            self.assertFalse(service.thread.is_alive())

    def test_http_rpc_requires_authentication_and_correct_trial(self):
        class Backend:
            def call(self, method, arguments, *, trial_id):
                if trial_id != "trial":
                    raise ValueError("wrong trial")
                return ["mapped"]
        server = ArcusProxyServer(("127.0.0.1", 0), "step-3.5-flash", trial_id="trial", proxy_token="test-token", search_service=Backend())
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}/arcus/tools"
        try:
            body = json.dumps({"trial_id": "trial", "method": "search_engine_query", "arguments": {"keywords": "test"}}).encode()
            with self.assertRaises(HTTPError) as denied:
                urlopen(Request(url, data=body), timeout=5)
            self.assertEqual(denied.exception.code, 401)
            request = Request(url, data=body, headers={"Authorization": "Bearer test-token"})
            with urlopen(request, timeout=5) as response:
                self.assertEqual(json.load(response)["result"], ["mapped"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
