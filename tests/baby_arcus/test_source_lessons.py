import json
from pathlib import Path
import tempfile
import unittest
from arcus.tokenizer import get_tokenizer
from baby_arcus.data_staging import StagingStore
from baby_arcus.sft_source_lessons import extract
from scripts.prepare_alpha_source_lessons import prepare


def row(session='one', old='return 0', new='return 1'):
    return {'meta': {'session_id': session}, 'messages': [], 'completion': {
        'tool_calls': [{'type': 'function', 'function': {'name': 'edit_file',
            'arguments': {'path': 'foreign/repo/module.py', 'old_string': old, 'new_string': new}}}]}}


class SourceLessonTests(unittest.TestCase):
    def test_real_tools_exact_text_review_and_provenance(self):
        base = Path('runs/test2'); base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            root = Path(tmp); source = root / 'source.jsonl'
            source.write_text(json.dumps(row()) + '\n' + json.dumps(row('two')), encoding='utf-8')
            result = prepare(source, root/'output', get_tokenizer('o200k_base'), fixture=True)
            self.assertEqual(result['accepted_lessons'], 1, result)
            self.assertEqual(result['duplicates'], 1)
            self.assertEqual(result['assistant_targets'], 6)
            self.assertFalse(result['model_loaded'])
            evidence = json.loads((root/'output/lesson-0.json').read_text())
            self.assertTrue(evidence['readback_verified'])
            self.assertEqual(evidence['events'][-1]['result']['content'], 'return 1')
            self.assertEqual(evidence['records'][0]['provenance']['lesson']['teacher'], 'deterministic')
            self.assertFalse(evidence['records'][0]['provenance']['lesson']['original_task_solved'])
            store = StagingStore(root/'output/staging.sqlite', fixture=True)
            try:
                batch = result['batches'][0]['id']
                self.assertIsNone(store.get(batch)['decision'])
                with self.assertRaises(ValueError): store.approved([batch], allow_fixture=True)
            finally: store.close()

    def test_unsafe_semantics_secrets_and_noop_rejected(self):
        samples = [row(old='', new='x'), row(old='x', new='x'), row(new='hf_'+'a'*30)]
        unsafe = row(); unsafe['completion']['tool_calls'][0]['function']['arguments']['replace_all'] = True
        samples.append(unsafe)
        for sample in samples:
            with self.assertRaises(ValueError): extract(sample)

    def test_oversized_context_is_not_silently_truncated(self):
        base = Path('runs/test2'); base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            root = Path(tmp); source = root/'source.jsonl'
            source.write_text(json.dumps(row(old=' old'*1200, new=' new'*1200)), encoding='utf-8')
            report = prepare(source, root/'output', get_tokenizer('o200k_base'), fixture=True)
            self.assertEqual(report['accepted_lessons'], 0)
            self.assertEqual(report['quarantined'], 1)

    def test_duplicate_connected_sessions_cannot_cross_splits(self):
        base = Path('runs/test2'); base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            root = Path(tmp); source = root/'source.jsonl'
            rows = [row('one'), row('two'), row('two','return 2','return 3'),
                    row('three','return 4','return 5')]
            source.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
            report = prepare(source, root/'output', get_tokenizer('o200k_base'), fixture=True)
            self.assertEqual(report['accepted_lessons'], 3, report)
            first, connected, other = report['batches']
            self.assertEqual(first['group'], connected['group'])
            self.assertEqual(first['split'], connected['split'])
            self.assertNotEqual(first['split'], other['split'])
            self.assertGreater(report['split_counts']['validation'], 0)


if __name__ == '__main__': unittest.main()
