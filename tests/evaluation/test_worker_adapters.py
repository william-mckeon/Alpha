import ast
import unittest
from unittest.mock import Mock

from evaluation.worker_adapters import native_bfcl_handler, adapted_mcp_loop, register_mcpmark_route, singleton_stdev, isolated_worker_environment


POLICY = {"temperature": 0, "top_p": 1, "max_output_tokens": 65536, "max_agent_iterations": 100}
SOURCE = "async def _execute_litellm_tool_loop(self):\n    max_consecutive_failures = 3\n    completion_kwargs = {'model': self.model}\n    return completion_kwargs, max_consecutive_failures\n"


class WorkerAdapterTests(unittest.TestCase):
    def test_workers_strip_credentials_and_disable_implicit_dotenv_reload(self):
        parent = {"PATH": "runtime", "OPENROUTER_API_KEY": "secret", "TAVILY_API_KEY": "secret", "OPENAI_API_KEY": "secret", "LLM_API_KEY": "secret", "ARCUS_WORKER_TOKEN": "old-token"}
        environment = isolated_worker_environment(parent)
        self.assertEqual(environment["PATH"], "runtime")
        self.assertEqual(environment["PYTHON_DOTENV_DISABLED"], "1")
        self.assertNotIn("secret", environment.values())
        self.assertNotIn("ARCUS_WORKER_TOKEN", environment)
        self.assertEqual(parent["OPENAI_API_KEY"], "secret")
    def test_singleton_reporting_does_not_change_multi_sample_statistics(self):
        import statistics
        self.assertEqual(singleton_stdev([1.2]), 0)
        self.assertEqual(singleton_stdev([1, 2, 3]), statistics.stdev([1, 2, 3]))
        with self.assertRaises(statistics.StatisticsError):
            singleton_stdev([])
    def test_bfcl_transport_preserves_native_call_and_does_not_retry(self):
        handler = native_bfcl_handler(object, gateway_url="http://127.0.0.1:8010/v1", proxy_token="test-token", generation=POLICY)()
        handler.client = Mock()
        handler.client.chat.completions.create.side_effect = RuntimeError("provider failure")
        with self.assertRaises(RuntimeError):
            handler.generate_with_backoff(model="test", messages=[])
        handler.client.chat.completions.create.assert_called_once()
        self.assertEqual(handler._build_client_kwargs()["max_retries"], 0)
        self.assertEqual(handler.client.chat.completions.create.call_args.kwargs["max_tokens"], 65536)

    def test_external_gateway_rejected(self):
        with self.assertRaises(ValueError):
            native_bfcl_handler(object, gateway_url="https://api.openai.com/v1", proxy_token="token", generation=POLICY)

    def test_mcpmark_overlay_changes_only_known_retry_and_request_policy(self):
        tree, evidence = adapted_mcp_loop(SOURCE, POLICY)
        rendered = ast.unparse(tree)
        self.assertIn("max_consecutive_failures = 1", rendered)
        self.assertIn("'num_retries': 0", rendered)
        self.assertEqual(evidence["changes"], {"failure_limit": 1, "completion_policy": 1})
        with self.assertRaises(ValueError):
            adapted_mcp_loop(SOURCE.replace("= 3", "= 4"), POLICY)

    def test_mcpmark_route_explicitly_uses_proxy_not_external_keys(self):
        class Config:
            MODEL_CONFIGS = {}
        alias = register_mcpmark_route(Config, candidate_id="step-3.5-flash")
        self.assertEqual(Config.MODEL_CONFIGS[alias]["base_url_var"], "ARCUS_WORKER_BASE_URL")
        self.assertEqual(Config.MODEL_CONFIGS[alias]["api_key_var"], "ARCUS_WORKER_TOKEN")
