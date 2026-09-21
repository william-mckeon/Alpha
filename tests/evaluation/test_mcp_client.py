import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

from evaluation.mcp_client import MCPClient, MCPFailure


class MCPClientTests(unittest.IsolatedAsyncioTestCase):
    def test_authenticated_urls_are_rejected(self):
        for url in ("http://example.com", "https://example.com/?api_key=secret", "https://user:password@example.com"):
            with self.assertRaises(ValueError):
                MCPClient(url, api_key="test")

    async def test_tool_call_requires_initialized_session(self):
        client = MCPClient("https://example.com/mcp", api_key="test")
        with self.assertRaises(MCPFailure):
            await client.call_tool("search", {})

    async def test_exception_is_sanitized_and_never_retried(self):
        client = MCPClient("https://example.com/mcp", api_key="secret-key")
        call = AsyncMock(side_effect=RuntimeError("Bearer secret-key failed"))
        client.session = SimpleNamespace(call_tool=call)
        with self.assertRaises(MCPFailure) as failure:
            await client.call_tool("search", {})
        self.assertNotIn("secret-key", str(failure.exception))
        self.assertEqual(call.await_count, 1)

    async def test_pagination_is_bounded(self):
        client = MCPClient("https://example.com/mcp", api_key="test")
        client.session = SimpleNamespace(list_tools=AsyncMock(return_value=SimpleNamespace(tools=[], nextCursor="same")))
        with self.assertRaisesRegex(MCPFailure, "pagination"):
            await client.list_tools()
