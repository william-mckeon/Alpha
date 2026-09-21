import tempfile
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import io
import os
import json
import threading
import ctypes
import urllib.error
from evaluation.provider import ProviderRequestError

from evaluation.provider import (
    BudgetLedger,
    EvaluationBlocked,
    ProviderConfig,
    estimate_worst_case_cost,
    enforce_generation_contract,
    is_retryable_http_status,
    OpenRouterClient,
)


class FoundationProviderTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "requires a real Windows file-sharing lock")
    def test_real_windows_reader_lock_recovers(self):
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
            wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        with tempfile.TemporaryDirectory() as directory:
            ledger = BudgetLedger(Path(directory) / "ledger.json", {"alpha": 1.0}, 1.0)
            ledger.record("alpha", .1)
            # Allow reads/writes but deny deletion/replacement until the reader closes.
            handle = kernel.CreateFileW(str(ledger.path), 0x80000000, 3, None, 3, 0x80, None)
            self.assertNotEqual(handle, ctypes.c_void_p(-1).value)
            release = threading.Timer(.3, kernel.CloseHandle, args=(handle,))
            release.start()
            try:
                reservation = ledger.reserve("alpha", .5)
                ledger.settle(reservation, .2)
                restored = BudgetLedger(ledger.path, {"alpha": 1.0}, 1.0)
                self.assertAlmostEqual(restored.spent["alpha"], .3)
                self.assertAlmostEqual(restored.remaining("alpha"), .7)
            finally:
                release.join()

    def test_windows_replace_retries_same_snapshot_for_reserve_and_settle(self):
        original_replace = os.replace
        for code in (5, 32, 33):
            with self.subTest(winerror=code), tempfile.TemporaryDirectory() as directory:
                ledger = BudgetLedger(Path(directory) / "ledger.json", {"alpha": 1.0}, 1.0)
                error = PermissionError("temporary Windows sharing denial")
                error.winerror = code
                for operation in ("reserve", "settle"):
                    calls = []
                    def replace(source, target):
                        calls.append((source, target, source.read_bytes()))
                        if len(calls) < 3:
                            raise error
                        original_replace(source, target)
                    with patch("evaluation.provider.os.replace", side_effect=replace), patch("evaluation.provider.time.sleep"):
                        if operation == "reserve":
                            reservation = ledger.reserve("alpha", .75)
                        else:
                            ledger.settle(reservation, .25)
                    self.assertEqual(len(calls), 3)
                    self.assertTrue(all(call == calls[0] for call in calls))
                restored = BudgetLedger(ledger.path, {"alpha": 1.0}, 1.0)
                self.assertEqual(restored.spent["alpha"], .25)
                self.assertEqual(restored.remaining("alpha"), .75)
                self.assertEqual(restored._reserved, {})
                self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_permanent_replace_failure_preserves_durable_reservation(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = BudgetLedger(Path(directory) / "ledger.json", {"alpha": 1.0}, 1.0)
            reservation = ledger.reserve("alpha", .75)
            before = ledger.path.read_bytes()
            error = PermissionError("persistent Windows denial")
            error.winerror = 5
            with patch("evaluation.provider.os.replace", side_effect=error) as replace, patch("evaluation.provider.time.sleep") as sleep:
                with self.assertRaises(PermissionError):
                    ledger.settle(reservation, .25)
            self.assertEqual(replace.call_count, 10)
            self.assertEqual(sleep.call_count, 9)
            self.assertEqual(ledger.path.read_bytes(), before)
            restored = BudgetLedger(ledger.path, {"alpha": 1.0}, 1.0)
            self.assertEqual(restored.remaining("alpha"), .25)
            self.assertEqual(restored.spent["alpha"], 0)
            retained = list(Path(directory).glob("*.tmp"))
            self.assertEqual(len(retained), 1)
            self.assertEqual(json.loads(retained[0].read_text())["spent"]["alpha"], .25)

    def test_nonsharing_replace_errors_fail_immediately(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = BudgetLedger(Path(directory) / "ledger.json", {"alpha": 1.0}, 1.0)
            with patch("evaluation.provider.os.replace", side_effect=OSError("disk failure")) as replace, patch("evaluation.provider.time.sleep") as sleep:
                with self.assertRaises(OSError):
                    ledger.reserve("alpha", .5)
            replace.assert_called_once()
            sleep.assert_not_called()
            self.assertFalse(ledger.path.exists())

    def test_retry_does_not_reset_absolute_request_deadline(self):
        config = ProviderConfig("alpha", "model", "provider", "slug", "unknown", .1, .3, 5)
        error = urllib.error.HTTPError("https://example", 429, "rate limit", {}, io.BytesIO(b'{"error":{"message":"rate limit"}}'))
        generation = {"temperature": 0, "top_p": 1, "max_output_tokens": 4, "wall_time_seconds": 5}
        with tempfile.TemporaryDirectory() as directory:
            ledger = BudgetLedger(Path(directory) / "ledger.json", {"alpha": 5}, 5)
            with patch("evaluation.provider.time.monotonic", side_effect=[0, 0, 1, 6]), patch("evaluation.provider.time.sleep"), patch("evaluation.provider.urllib.request.urlopen", side_effect=error) as request:
                with self.assertRaises(ProviderRequestError):
                    OpenRouterClient(api_key="test").forward(config, ledger, {"messages": []}, generation_contract=generation, provider_retries=1)
            self.assertEqual(request.call_count, 1)
            self.assertEqual(ledger.remaining("alpha"), 5)
    def test_separate_ledger_instances_do_not_overwrite_spend(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.json"
            first = BudgetLedger(path, {"alpha": 1.0}, 1.0)
            second = BudgetLedger(path, {"alpha": 1.0}, 1.0)
            first.record("alpha", 0.2)
            second.record("alpha", 0.3)
            self.assertAlmostEqual(first.remaining("alpha"), 0.5)
            self.assertAlmostEqual(second.remaining("alpha"), 0.5)

    def test_restart_preserves_inflight_reservations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.json"
            first = BudgetLedger(path, {"alpha": 1.0}, 1.0)
            reservation = first.reserve("alpha", 0.75)
            restarted = BudgetLedger(path, {"alpha": 1.0}, 1.0)
            with self.assertRaises(EvaluationBlocked):
                restarted.reserve("alpha", 0.3)
            self.assertAlmostEqual(restarted.remaining("alpha"), 0.25)
            restarted.settle(reservation, 0.5)
            self.assertAlmostEqual(first.remaining("alpha"), 0.5)

    def test_uncertain_network_failure_keeps_budget_reservation(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = BudgetLedger(Path(directory) / "ledger.json", {"alpha": 1.0}, 1.0)
            with patch("evaluation.provider.urllib.request.urlopen", side_effect=TimeoutError("uncertain")):
                with self.assertRaises(TimeoutError):
                    OpenRouterClient(api_key="test-only").forward(self.config, ledger, {"messages": []})
            restored = BudgetLedger(ledger.path, {"alpha": 1.0}, 1.0)
            self.assertLess(restored.remaining("alpha"), 1.0)

    def test_unreconciled_response_keeps_budget_reservation(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"usage":{"cost":true}}'
        with tempfile.TemporaryDirectory() as directory:
            ledger = BudgetLedger(Path(directory) / "ledger.json", {"alpha": 1.0}, 1.0)
            with patch("evaluation.provider.urllib.request.urlopen", return_value=response):
                with self.assertRaises(RuntimeError):
                    OpenRouterClient(api_key="test-only").forward(self.config, ledger, {"messages": []})
            self.assertLess(ledger.remaining("alpha"), 1.0)

    def test_nonfinite_or_boolean_spend_cannot_bypass_caps(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = BudgetLedger(Path(directory) / "ledger.json", {"alpha": 1.0}, 1.0)
            for amount in (float("nan"), float("inf"), True, -1):
                with self.assertRaises(ValueError):
                    ledger.reserve("alpha", amount)
                with self.assertRaises(ValueError):
                    ledger.record("alpha", amount)
            self.assertEqual(ledger.remaining("alpha"), 1.0)

    def setUp(self):
        self.config = ProviderConfig("alpha", "vendor/model", "Vendor", "vendor", "fp8", 1.0, 2.0, 0.01)

    def test_estimate_and_ledger_round_trip(self):
        estimate = estimate_worst_case_cost(
            self.config, {"messages": [{"role": "user", "content": "hello"}]}, 100
        )
        self.assertGreater(estimate, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.json"
            ledger = BudgetLedger(path, {"alpha": 0.01}, 0.01)
            ledger.authorize("alpha", estimate)
            ledger.record("alpha", 0.001)
            restored = BudgetLedger(path, {"alpha": 0.01}, 0.01)
            self.assertAlmostEqual(restored.remaining("alpha"), 0.009)

    def test_budget_overrun_is_blocked_before_request(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = BudgetLedger(Path(directory) / "ledger.json", {"alpha": 0.01}, 0.01)
            with self.assertRaises(EvaluationBlocked):
                ledger.authorize("alpha", 0.02)

    def test_only_transient_http_errors_are_retried(self):
        self.assertTrue(is_retryable_http_status(429))
        self.assertTrue(is_retryable_http_status(503))
        self.assertFalse(is_retryable_http_status(400))
        self.assertFalse(is_retryable_http_status(401))

    def test_forward_rejects_streaming_before_network(self):
        client = OpenRouterClient(api_key="test-only")
        with tempfile.TemporaryDirectory() as directory:
            ledger = BudgetLedger(Path(directory) / "ledger.json", {"alpha": 0.01}, 0.01)
            with self.assertRaises(EvaluationBlocked):
                client.forward(
                    self.config,
                    ledger,
                    {"messages": [{"role": "user", "content": "hello"}], "stream": True},
                )

    def test_generation_contract_injects_every_frozen_parameter(self):
        generation = {"temperature": 0.0, "top_p": 1.0, "max_output_tokens": 65536}
        resolved = enforce_generation_contract(
            {"messages": [{"role": "user", "content": "hello"}]}, generation
        )
        self.assertEqual(resolved["temperature"], 0.0)
        self.assertEqual(resolved["top_p"], 1.0)
        self.assertEqual(resolved["max_tokens"], 65536)
        self.assertFalse(resolved["stream"])

    def test_generation_contract_rejects_conflict(self):
        generation = {"temperature": 0.0, "top_p": 1.0, "max_output_tokens": 65536}
        with self.assertRaises(EvaluationBlocked):
            enforce_generation_contract(
                {"messages": [], "temperature": 0.7}, generation
            )

    def test_inflight_reservation_reduces_remaining_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = BudgetLedger(Path(directory) / "ledger.json", {"alpha": 1.0}, 1.0)
            reservation = ledger.reserve("alpha", 0.75)
            self.assertAlmostEqual(ledger.remaining("alpha"), 0.25)
            ledger.settle(reservation, 0.5)
            self.assertAlmostEqual(ledger.remaining("alpha"), 0.5)

    def test_actual_overrun_is_persisted_before_blocking(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.json"
            ledger = BudgetLedger(path, {"alpha": 1.0}, 1.0)
            reservation = ledger.reserve("alpha", 0.1)
            with self.assertRaises(EvaluationBlocked):
                ledger.settle(reservation, 0.2)
            restored = BudgetLedger(path, {"alpha": 1.0}, 1.0)
            self.assertAlmostEqual(restored.spent["alpha"], 0.2)


if __name__ == "__main__":
    unittest.main()
