import json
import tempfile
import unittest
from pathlib import Path
from arcus3.checkpoint import digest
from scripts.finalize_arcus3_phase8_stage import finalize, SHARES
from scripts.prepare_arcus3_phase8_data import seal


class StageFinalizationTests(unittest.TestCase):
    def part(self, root):
        root.mkdir()
        (root / 'exclusions.json').write_text('["held-out"]')
        rows = [{'sha256': name, 'input_ids': [1] * int(100 * share),
                 'labels': [1] * int(100 * share), 'source': name}
                for name, share in SHARES.items()]
        (root / 'train-000.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        provenance = {'reviewed': True, 'scope': 'combined-phase8', 'tokenizer_sha256': 'donor',
            'sources_sha256': 'source', 'evaluation_exclusions_sha256': digest(root / 'exclusions.json')}
        (root / 'provenance.json').write_text(json.dumps(provenance))
        seal(root, root / 'provenance.json', True)
        return root

    def test_seals_exact_mixture_and_deduplicates_repairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.part(root / 'part')
            result = finalize([source, source], root / 'final', budget=100)
            self.assertEqual(result['records'], 5)
            self.assertEqual(result['input_tokens_per_pass'], 100)
            self.assertFalse(result['qualification_only'])

    def test_rejects_changed_exclusions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.part(root / 'part')
            (source / 'exclusions.json').write_text('[]')
            with self.assertRaisesRegex(ValueError, 'Exclusions changed'):
                finalize([source], root / 'final', budget=100)
