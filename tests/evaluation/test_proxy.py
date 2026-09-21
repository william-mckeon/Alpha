import tempfile
import json
import threading
import unittest
from unittest.mock import Mock, MagicMock, patch
import http.client
import os
import urllib.error
import urllib.request
from pathlib import Path

from evaluation.provider import EvaluationBlocked
from evaluation.proxy import ArcusProxyServer


class ProxyAccountingTests(unittest.TestCase):
    def test_ledger_replace_recovery_never_replays_paid_request(self):
        from evaluation.provider import BudgetLedger, OpenRouterClient
        original_replace = os.replace
        for permanent in (False, True):
            with self.subTest(permanent=permanent), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                server = ArcusProxyServer(("127.0.0.1", 0), "deepseek-v4-flash",
                    trial_id="ledger-fault", proxy_token="test-only", artifact_dir=root / "artifacts")
                server.ledger = BudgetLedger(root / "ledger.json", {"deepseek-v4-flash": 5}, 5)
                server.client = OpenRouterClient(api_key="test-only")
                upstream = MagicMock()
                upstream.__enter__.return_value.read.return_value = json.dumps({
                    "provider": server.config.upstream, "usage": {"cost": .001},
                    "choices": [{"message": {"content": "done"}}]}).encode()
                error = PermissionError("Windows sharing denial")
                error.winerror = 5
                settlement_attempts = []
                def replace(source, target):
                    snapshot = json.loads(source.read_text())
                    if snapshot["spent"]["deepseek-v4-flash"]:
                        settlement_attempts.append(source)
                        if permanent or len(settlement_attempts) < 3:
                            raise error
                    original_replace(source, target)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
                try:
                    with patch("evaluation.provider.urllib.request.urlopen", return_value=upstream) as paid, patch("evaluation.provider.os.replace", side_effect=replace), patch("evaluation.provider.time.sleep"):
                        connection.request("POST", "/v1/chat/completions", b'{"messages":[]}',
                            {"Authorization": "Bearer test-only", "Content-Type": "application/json"})
                        response = connection.getresponse()
                        response.read()
                        self.assertEqual(response.status, 502 if permanent else 200)
                        if permanent:
                            connection.request("POST", "/v1/chat/completions", b'{"messages":[]}',
                                {"Authorization": "Bearer test-only", "Content-Type": "application/json"})
                            response = connection.getresponse()
                            response.read()
                            self.assertGreaterEqual(response.status, 400)
                        paid.assert_called_once()
                    self.assertEqual(len(settlement_attempts), 10 if permanent else 3)
                    self.assertEqual(len(set(settlement_attempts)), 1)
                    restored = BudgetLedger(server.ledger.path, {"deepseek-v4-flash": 5}, 5)
                    if permanent:
                        self.assertEqual(server.terminal_error["category"], "infrastructure")
                        self.assertEqual(len(restored._reserved), 1)
                        self.assertEqual(restored.spent["deepseek-v4-flash"], 0)
                    else:
                        self.assertIsNone(server.terminal_error)
                        self.assertEqual(restored._reserved, {})
                        self.assertEqual(restored.spent["deepseek-v4-flash"], .001)
                finally:
                    connection.close()
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=5)

    def test_100_forwards_then_wrapped_stop_preserves_patch_path(self):
        from evaluation.openhands import gateway_iteration_stop, run_bounded_conversation
        with tempfile.TemporaryDirectory() as directory:
            server = ArcusProxyServer(("127.0.0.1", 0), "step-3.5-flash", trial_id="hundred",
                                      proxy_token="test-only", artifact_dir=Path(directory))
            server.client = Mock()
            server.client.forward.return_value = {"request": {}, "response": {"choices": []}}
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}/v1"
            def request():
                req = urllib.request.Request(base + "/chat/completions", data=b'{"messages":[]}',
                    headers={"Authorization": "Bearer test-only", "Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=5) as response:
                    response.read()
            try:
                self.assertFalse(gateway_iteration_stop(base, "test-only", "hundred", 100))
                for _ in range(100):
                    request()
                self.assertFalse(gateway_iteration_stop(base, "test-only", "hundred", 100))
                def wrapped_run(conversation):
                    try:
                        request()
                    except urllib.error.HTTPError:
                        raise RuntimeError("Remote conversation ended with error") from None
                stop = Path(directory) / "budget-stop.json"
                run_bounded_conversation(wrapped_run, None, run_error=RuntimeError, stop_path=stop,
                    trial_id="hundred", max_iterations=100,
                    stop_check=lambda: gateway_iteration_stop(base, "test-only", "hundred", 100))
                self.assertTrue(stop.exists())
                self.assertEqual(server.client.forward.call_count, 100)
                self.assertEqual(server.model_requests, 100)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
    def test_fatal_provider_error_blocks_next_forward_without_payment(self):
        with tempfile.TemporaryDirectory() as directory:
            server = ArcusProxyServer(("127.0.0.1", 0), "deepseek-v4-flash", trial_id="fatal", proxy_token="test-only", artifact_dir=Path(directory))
            from evaluation.provider import TrialControlStopped
            server.record_error("provider", "upstream engine overloaded")
            with self.assertRaises(TrialControlStopped):
                server.raise_if_stopped()
            self.assertEqual(server.terminal_error["category"], "provider")
            server.server_close()
    def test_iteration_ceiling_is_explicit_model_budget_without_provider_call(self):
        with tempfile.TemporaryDirectory() as directory:
            server = ArcusProxyServer(("127.0.0.1", 0), "step-3.5-flash", trial_id="ceiling", proxy_token="test-only", artifact_dir=Path(directory))
            server.model_requests = server.generation["max_agent_iterations"]
            server.client = Mock()
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                request = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/v1/chat/completions", data=b'{"messages":[]}', headers={"Content-Type": "application/json", "Authorization": "Bearer test-only"})
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    urllib.request.urlopen(request, timeout=5)
                self.assertEqual(raised.exception.code, 400)
                server.client.forward.assert_not_called()
                error = json.loads((Path(directory) / "gateway-errors.jsonl").read_text())
                self.assertEqual(error["category"], "model_budget")
                from evaluation.openhands import gateway_iteration_stop
                base = f"http://127.0.0.1:{server.server_port}/v1"
                self.assertTrue(gateway_iteration_stop(base, "test-only", "ceiling", 100))
                self.assertFalse(gateway_iteration_stop(base, "test-only", "other-trial", 100))
                with self.assertRaises(urllib.error.HTTPError):
                    gateway_iteration_stop(base, "wrong-token", "ceiling", 100)
                server.record_error("provider", "fatal provider failure")
                self.assertFalse(gateway_iteration_stop(base, "test-only", "ceiling", 100))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
    def test_shutdown_drains_forward_before_sealing_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            server = ArcusProxyServer(("127.0.0.1", 0), "step-3.5-flash", trial_id="drain", proxy_token="test-only", artifact_dir=Path(directory))
            entered, release, closed = threading.Event(), threading.Event(), threading.Event()

            def forward(*args, **kwargs):
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("test did not release request")
                return {"request": {"messages": []}, "response": {"choices": [{"message": {"content": "done"}}]}}

            def request():
                payload = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/v1/chat/completions", data=b'{"messages":[]}', headers={"Authorization": "Bearer test-only", "Content-Type": "application/json"})
                with urllib.request.urlopen(payload, timeout=5) as response:
                    response.read()

            def close():
                server.shutdown()
                server.server_close()
                closed.set()

            server.client = Mock()
            server.client.forward.side_effect = forward
            serving = threading.Thread(target=server.serve_forever, daemon=True)
            client = threading.Thread(target=request)
            closing = threading.Thread(target=close)
            serving.start()
            try:
                client.start()
                self.assertTrue(entered.wait(3))
                closing.start()
                self.assertFalse(closed.wait(0.1))
                release.set()
                client.join(timeout=3)
                closing.join(timeout=3)
                self.assertTrue(closed.is_set())
                self.assertTrue((Path(directory) / "proxy-exchanges.jsonl").is_file())
            finally:
                release.set()
                if closing.ident is None:
                    close()
                else:
                    closing.join(timeout=5)
                serving.join(timeout=5)
                if client.ident is not None:
                    client.join(timeout=5)

    def test_global_iteration_cap_blocks_before_paid_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            server = ArcusProxyServer(("127.0.0.1", 0), "step-3.5-flash", trial_id="iterations", proxy_token="test-only", artifact_dir=Path(directory))
            server.model_requests = server.generation["max_agent_iterations"]
            server.client = Mock()
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                request = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/v1/chat/completions", data=b'{"messages":[]}', headers={"Authorization": "Bearer test-only", "Content-Type": "application/json"})
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    urllib.request.urlopen(request, timeout=5)
                self.assertEqual(raised.exception.code, 400)
                server.client.forward.assert_not_called()
                self.assertIn("agent-iteration", (Path(directory) / "gateway-errors.jsonl").read_text())
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def server(self, directory: str):
        server = ArcusProxyServer.__new__(ArcusProxyServer)
        server.generation = {"tool_call_budget": 1}
        server.trial_id = "trial-1"
        server.artifact_dir = Path(directory)
        server.tool_calls = 0
        server._state_lock = threading.Lock()
        return server

    def test_exchange_is_retained_and_tools_are_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            server = self.server(directory)
            server.record_exchange(
                {
                    "request": {"messages": []},
                    "response": {"choices": [{"message": {"tool_calls": [{"id": "1"}]}}]},
                }
            )
            self.assertEqual(server.tool_calls, 1)
            self.assertTrue((Path(directory) / "proxy-exchanges.jsonl").exists())

    def test_tool_budget_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            server = self.server(directory)
            exchange = {
                "request": {"messages": []},
                "response": {"choices": [{"message": {"tool_calls": [{"id": "1"}]}}]},
            }
            server.record_exchange(exchange)
            with self.assertRaises(EvaluationBlocked):
                server.record_exchange(exchange)

    def test_constructor_refuses_existing_trial_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "proxy-exchanges.jsonl").write_text("already used\n", encoding="utf-8")
            with self.assertRaises(EvaluationBlocked):
                ArcusProxyServer(
                    ("127.0.0.1", 0),
                    "step-3.5-flash",
                    trial_id="x",
                    artifact_dir=path,
                    proxy_token="test-only",
                )

    def test_unauthenticated_request_is_rejected_before_provider(self):
        server = ArcusProxyServer(
            ("127.0.0.1", 0), "step-3.5-flash", trial_id="auth-test", proxy_token="test-only"
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            request = urllib.request.Request(
                f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                data=b'{"messages":[]}',
                headers={"Content-Type": "application/json"},
            )
            with self.assertRaises(urllib.error.HTTPError) as raised:
                urllib.request.urlopen(request, timeout=5)
            self.assertEqual(raised.exception.code, 401)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
