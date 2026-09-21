import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from evaluation.search_adapter import BFCLSearchAdapter


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_bfcl_shape_and_snippet_visibility(self):
        search = SimpleNamespace(search=AsyncMock(return_value={"results": [{"title": "docs", "url": "https://example.com", "content": "snippet"}]}))
        adapter = BFCLSearchAdapter(search)
        self.assertEqual(await adapter.search_engine_query("query"), [{"title": "docs", "href": "https://example.com", "body": "snippet"}])
        adapter._load_scenario({"show_snippet": False})
        self.assertNotIn("body", (await adapter.search_engine_query("query"))[0])

    async def test_region_is_not_silently_ignored(self):
        search = SimpleNamespace(search=AsyncMock())
        with self.assertRaisesRegex(ValueError, "region"):
            await BFCLSearchAdapter(search).search_engine_query("query", region="us-en")
        search.search.assert_not_awaited()
