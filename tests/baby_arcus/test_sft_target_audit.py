import json
from pathlib import Path
import tempfile
import unittest
from baby_arcus.sft_target_contract import classify, annotate
from baby_arcus.sft_validation import validate
from scripts.audit_alpha_sft_targets import audit


class TargetTests(unittest.TestCase):
    def record(self,text):
        return {'version':1,'source':'fixture:a','group':'a','split':'training',
                'messages':[{'role':'user','content':'hello'},{'role':'assistant','content':text}]}

    def test_external_transcript_is_quarantined_not_executed(self):
        for text in ('[external_agent_tool_call: Edit] file: a','[external_agent_tool_result] Success'):
            self.assertEqual(classify(text),'external_transcript')
            with self.assertRaisesRegex(ValueError,'Quarantine'):validate(self.record(text))
        validate(self.record('Use tool_search to find a tool.'))

    def test_native_action_and_malformed_action(self):
        action=json.dumps({'name':'tool_search','version':1,'arguments':{'query':'read'}})
        self.assertEqual(annotate(self.record(action))['messages'][-1]['target_kind'],'action_prediction')
        with self.assertRaises(ValueError):validate(self.record('{"name":"read_file"}'))

    def test_historical_observation_remains_untrained(self):
        record=self.record('Hello')
        record['messages'].insert(1,{'role':'assistant','content':'[external_agent_tool_call: Edit]','train':False})
        validate(record)

    def test_audit_reads_legacy_without_admitting_it(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'records.jsonl'
            path.write_text(json.dumps(self.record('[external_agent_tool_result] done'))+'\n')
            self.assertEqual(audit(path)['targets'],{'external_transcript':1})
