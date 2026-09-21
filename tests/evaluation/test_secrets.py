import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.secrets import load_credentials, redact


class SecretTests(unittest.TestCase):
    def test_loader_allowlist_and_host_precedence(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"TAVILY_API_KEY": "host-key"}, clear=True):
            path = Path(directory) / ".env"
            path.write_text('TAVILY_API_KEY=file-key\nOPENROUTER_API_KEY="router-key"\nUNRELATED_SECRET=ignored\n')
            load_credentials(path)
            self.assertEqual(os.environ["TAVILY_API_KEY"], "host-key")
            self.assertEqual(os.environ["OPENROUTER_API_KEY"], "router-key")
            self.assertNotIn("UNRELATED_SECRET", os.environ)

    def test_recursive_redaction_in_fields_text_and_urls(self):
        data = {"Authorization": "Bearer arbitrary", "nested": ["https://example.com/?tavilyApiKey=tvly-test-secret", "Bearer another-secret", "sk-or-v1-test-secret"], "usage": 1}
        rendered = str(redact(data))
        for secret in ("arbitrary", "another-secret", "tvly-test-secret", "sk-or-v1-test-secret"):
            self.assertNotIn(secret, rendered)
        self.assertEqual(redact(data)["usage"], 1)
