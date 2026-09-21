import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from evaluation.provider import EvaluationBlocked
from evaluation.mcp_client import MCPFailure
from evaluation.tavily import SearchBudget, TavilySearch, free_credits_remaining, parse_search_result


CONFIG = json.loads((Path(__file__).resolve().parents[2] / "evaluation" / "search.json").read_text())
USAGE = {"account": {"current_plan": "Researcher", "plan_usage": 0, "plan_limit": 1000, "paygo_usage": 0, "paygo_limit": None}, "key": {"usage": 0, "limit": None}}


class SearchTests(unittest.IsolatedAsyncioTestCase):
    def client(self):
        tool = {"name": "tavily_search", "inputSchema": {"properties": {name: {} for name in ("query", "search_depth", "max_results", "include_raw_content")}}}
        return SimpleNamespace(initialization={}, list_tools=AsyncMock(return_value=[tool]), usage=AsyncMock(return_value=USAGE), call_tool=AsyncMock(return_value={"structuredContent": {"results": [{"title": "Python", "url": "https://python.org", "content": "docs"}]}}))

    async def test_advertised_schema_omits_unsupported_argument(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = self.client()
            search = TavilySearch(client, CONFIG, SearchBudget(root / "credits.sqlite", cap=10, contract_id=CONFIG["contract_id"]), root / "trial", trial_id="trial")
            await search.discover()
            result = await search.search("Python")
            self.assertEqual(len(result["results"]), 1)
            self.assertNotIn("include_answer", client.call_tool.call_args.args[1])
            self.assertTrue((root / "trial" / "search-exchanges.jsonl").exists())

    async def test_unknown_failure_keeps_credit_reservation_and_is_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = self.client()
            client.call_tool.side_effect = MCPFailure("timeout")
            budget = SearchBudget(root / "credits.sqlite", cap=10, contract_id=CONFIG["contract_id"])
            search = TavilySearch(client, CONFIG, budget, root / "trial", trial_id="trial")
            await search.discover()
            with self.assertRaises(MCPFailure):
                await search.search("Python")
            self.assertEqual(budget.spent_upper_bound(), 2)
            self.assertEqual(client.call_tool.await_count, 1)

    async def test_zero_cap_blocks_before_network_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = self.client()
            search = TavilySearch(client, CONFIG, SearchBudget(root / "credits.sqlite", cap=0, contract_id="test"), root / "trial", trial_id="trial")
            await search.discover()
            with self.assertRaises(EvaluationBlocked):
                await search.search("Python")
            client.call_tool.assert_not_awaited()

    def test_shared_budget_and_initial_allowance_survive_new_instances(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "credits.sqlite"
            first = SearchBudget(path, cap=1000, contract_id="test")
            first.reserve(2, free_remaining=2)
            second = SearchBudget(path, cap=1000, contract_id="test")
            with self.assertRaises(EvaluationBlocked):
                second.reserve(2, free_remaining=1000)

    def test_free_policy_rejects_paid_or_paygo_accounts(self):
        for account in ({**USAGE["account"], "current_plan": "Bootstrap"}, {**USAGE["account"], "paygo_limit": 10}):
            with self.assertRaises(EvaluationBlocked):
                free_credits_remaining({"account": account, "key": USAGE["key"]})

    def test_structured_and_json_text_results(self):
        self.assertEqual(parse_search_result({"content": [{"type": "text", "text": '{"results":[]}'}]}), {"results": []})
        with self.assertRaises(MCPFailure):
            parse_search_result({"isError": True})
