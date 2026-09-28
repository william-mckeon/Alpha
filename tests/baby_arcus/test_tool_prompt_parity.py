import unittest
from baby_arcus.tool_discovery_curriculum import action_records,tasks
from baby_arcus.sft_dataset import windows


class PromptTests(unittest.TestCase):
    def test_real_tokenizer_action_targets_and_prompt_parity(self):
        from arcus.tokenizer import get_tokenizer
        tokenizer=get_tokenizer('o200k_base')
        records=list(action_records('validation',tokenizer))
        self.assertEqual(len(records),48)
        for record in records:
            window=next(windows(record,tokenizer,16384))
            target=tokenizer.decode([i for i,m in zip(window['ids'],window['mask']) if m])
            self.assertTrue(target.endswith('\n</assistant>\n'))
            self.assertIn('"arguments"',target)
            self.assertEqual(sum(m['role']=='assistant' and m.get('train',True) for m in record['messages']),1)

    def test_names_disjoint(self):
        sets=[{t['tool']['name'] for t in tasks(split)} for split in ('training','validation','test')]
        self.assertFalse(sets[0]&sets[1] or sets[1]&sets[2] or sets[0]&sets[2])

    def test_bootstrap_prompts_do_not_leak_across_splits(self):
        from arcus.tokenizer import get_tokenizer
        from baby_arcus.contracts import digest
        tokenizer=get_tokenizer('o200k_base')
        sets=[{digest(r['messages']) for r in action_records(split,tokenizer)} for split in ('training','validation','test')]
        self.assertFalse(sets[0]&sets[1] or sets[1]&sets[2] or sets[0]&sets[2])
