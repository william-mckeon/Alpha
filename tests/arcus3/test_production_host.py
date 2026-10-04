"""Host readiness checks must fail before a production worker is created."""
import os
import unittest
from unittest.mock import patch

from scripts.check_arcus3_host import REQUIRED_MODULES, preflight


class ProductionHostTests(unittest.TestCase):
    def test_missing_module_is_named(self):
        with patch('scripts.check_arcus3_host.importlib.import_module',
                   side_effect=ModuleNotFoundError('test-only')):
            with self.assertRaisesRegex(RuntimeError, 'Production host Python lacks usable modules') as caught:
                preflight(check_remote=False)
        self.assertIn(REQUIRED_MODULES[0], str(caught.exception))

    def test_offline_check_does_not_require_a_token(self):
        with patch.dict(os.environ, {'HF_TOKEN': ''}):
            result = preflight(check_remote=False)
        self.assertTrue(result['modules_ok'])
        self.assertEqual(result['pinned_datasets_checked'], 0)

    def test_remote_error_names_repo_but_not_token(self):
        from scripts.prepare_arcus3_production import SOURCES
        with patch.dict(os.environ, {'HF_TOKEN': 'test-only-secret'}), \
                patch('dotenv.load_dotenv'), \
                patch('huggingface_hub.HfApi') as api:
            api.return_value.list_repo_files.side_effect = RuntimeError('test-only-secret')
            with self.assertRaises(RuntimeError) as caught:
                preflight(check_remote=True)
        self.assertIn(SOURCES[0][1], str(caught.exception))
        self.assertNotIn('test-only-secret', str(caught.exception))


if __name__ == '__main__':
    unittest.main()
