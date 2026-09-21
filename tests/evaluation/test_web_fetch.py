import unittest
from unittest.mock import patch, Mock

from evaluation.web_fetch import public_target, fetch_url_content, bounded_fetch, FetchFailure, valid_fetch_error, safe_fetch_failure
import json
import subprocess


class WebFetchTests(unittest.TestCase):
    def test_structured_subprocess_errors_preserve_safe_classification(self):
        for status in (302, 403, 404, 408, 429, 503):
            detail = FetchFailure("http_status", status).details
            with patch("evaluation.web_fetch.subprocess.run", return_value=Mock(returncode=2, stdout=json.dumps({"error": detail}))):
                with self.assertRaises(FetchFailure) as error:
                    bounded_fetch("https://example.com")
            self.assertEqual(error.exception.details, detail)
            self.assertEqual(detail["recoverable"], status in (302, 403, 404))
        with patch("evaluation.web_fetch.subprocess.run", side_effect=subprocess.TimeoutExpired("secret", 45)):
            with self.assertRaises(FetchFailure) as error:
                bounded_fetch("https://example.com")
        self.assertEqual(error.exception.details["code"], "timeout")
        self.assertNotIn("secret", str(error.exception))

    def test_malformed_error_cannot_claim_recoverability(self):
        detail = {"code": "timeout", "http_status": None, "recoverable": True}
        self.assertFalse(valid_fetch_error(detail))
        with patch("evaluation.web_fetch.subprocess.run", return_value=Mock(returncode=2, stdout=json.dumps({"error": detail}))):
            with self.assertRaises(FetchFailure) as error:
                bounded_fetch("https://example.com")
        self.assertEqual(error.exception.details["code"], "internal")
        self.assertNotIn("secret", str(safe_fetch_failure(RuntimeError("secret"))))
    def test_subprocess_has_wall_deadline_and_no_provider_credentials(self):
        completed = Mock(returncode=0, stdout='{"content":"safe"}')
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "secret", "TAVILY_API_KEY": "secret", "ARCUS_WORKER_TOKEN": "token"}), patch("evaluation.web_fetch.subprocess.run", return_value=completed) as run:
            self.assertEqual(bounded_fetch("https://example.com"), {"content": "safe"})
        self.assertEqual(run.call_args.kwargs["timeout"], 45)
        self.assertEqual(run.call_args.kwargs["encoding"], "utf-8")
        self.assertEqual(run.call_args.kwargs["env"]["PYTHONIOENCODING"], "utf-8")
        for key in ("OPENROUTER_API_KEY", "TAVILY_API_KEY", "ARCUS_WORKER_TOKEN"):
            self.assertNotIn(key, run.call_args.kwargs["env"])

    def test_unicode_fetch_content_preserved(self):
        completed = Mock(returncode=0, stdout='{"content":"漢字 🧰 café \\u0081"}')
        with patch("evaluation.web_fetch.subprocess.run", return_value=completed):
            self.assertEqual(bounded_fetch("https://example.com")["content"], "漢字 🧰 café \u0081")

    def test_private_mixed_dns_and_unsafe_urls_rejected(self):
        for url in ("http://example.com", "https://user:secret@example.com", "https://example.com:8443"):
            with self.assertRaises(ValueError):
                public_target(url)
        with patch("evaluation.web_fetch.socket.getaddrinfo", return_value=[(0, 0, 0, "", ("8.8.8.8", 443)), (0, 0, 0, "", ("127.0.0.1", 443))]):
            with self.assertRaisesRegex(ValueError, "security"):
                public_target("https://example.com")

    def test_connection_pins_ip_and_tls_hostname_without_redirect_retry(self):
        library = Mock()
        pool = library.HTTPSConnectionPool.return_value
        response = pool.request.return_value
        response.status = 200
        response.read.return_value = b"safe"
        with patch("evaluation.web_fetch.public_target", return_value=("example.com", "8.8.8.8", "/page")), patch("evaluation.web_fetch.import_module", return_value=library):
            self.assertEqual(fetch_url_content("https://example.com/page"), {"content": "safe"})
        self.assertEqual(library.HTTPSConnectionPool.call_args.args[0], "8.8.8.8")
        self.assertEqual(library.HTTPSConnectionPool.call_args.kwargs["assert_hostname"], "example.com")
        self.assertFalse(pool.request.call_args.kwargs["redirect"])
        self.assertFalse(pool.request.call_args.kwargs["retries"])
        pool.close.assert_called_once()

    def test_size_limit_and_redirect_fail_closed(self):
        for status, data in ((302, b""), (200, b"12345")):
            library = Mock()
            response = library.HTTPSConnectionPool.return_value.request.return_value
            response.status = status
            response.read.return_value = data
            with patch("evaluation.web_fetch.public_target", return_value=("example.com", "8.8.8.8", "/")), patch("evaluation.web_fetch.import_module", return_value=library):
                with self.assertRaises(ValueError):
                    fetch_url_content("https://example.com", max_bytes=4)
            response.close.assert_called_once()
