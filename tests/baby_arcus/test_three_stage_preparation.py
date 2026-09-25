import json
import tempfile
import unittest
from pathlib import Path
from baby_arcus.data_manifest import file_digest
from baby_arcus.data_staging import StagingStore
from baby_arcus.sft_dataset import packing_report
from scripts.prepare_alpha_training_data import stage_structured


class Tokenizer:
    def encode(self,text): return list(text.encode())


class PreparationTests(unittest.TestCase):
    def test_hash_cancellation_between_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'large'; path.write_bytes(b'x'*(2*1024*1024))
            calls=[]
            def cancelled(): calls.append(True); return len(calls)>1
            with self.assertRaises(InterruptedError): file_digest(path,cancelled)
            self.assertEqual(len(calls),2)

    def test_disk_import_keeps_latest_session_and_quarantines_unknown_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); store=StagingStore(root/'review.sqlite',fixture=True)
            rows=[{'session_id':'one','meta':{'step':step},'messages':[
                {'role':'user','content':'Hello'},{'role':'assistant','content':str(step)}]} for step in (2,1)]
            rows.append({'session_id':'bad','messages':[{'role':'user','content':'Do it'},
                {'role':'assistant','tool_calls':[{'id':'x','type':'function','function':{'name':'foreign_shell','arguments':'{}'}}]}]})
            path=root/'sft.jsonl'; path.write_text('\n'.join(json.dumps(row) for row in rows))
            report={}
            try:
                batches=stage_structured(path,store,'fixture:disk',report=report)
                self.assertEqual(report['input_records'],3); self.assertEqual(report['quarantined'],1)
                records=[record for batch in batches for record in store.get(batch)['records']]
                self.assertEqual(records[0]['messages'][-1]['content'],'2')
                self.assertIsNone(store.get(batches[0])['decision'])
                with self.assertRaises(InterruptedError): stage_structured(path,store,'fixture:disk',cancelled=lambda:True)
            finally: store.close()

    def test_packing_report_quarantines_whole_oversize_record(self):
        good={'version':1,'source':'fixture:packing','group':'one','split':'training',
              'messages':[{'role':'user','content':'Question'},{'role':'assistant','content':'Answer'}]}
        bad=dict(good,messages=[good['messages'][0],{'role':'assistant','content':'x'*600}])
        result=packing_report([good,bad],Tokenizer())
        self.assertEqual(len(result['accepted']),1); self.assertEqual(len(result['quarantined']),1)
        self.assertGreater(result['target_tokens'],0)

    def test_storage_quota_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=StagingStore(Path(tmp)/'review.sqlite',fixture=True,max_bytes=1048576)
            try:
                with self.assertRaises(ValueError): store.check_budget(1048576)
            finally: store.close()
