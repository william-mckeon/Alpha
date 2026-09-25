import json
import tempfile
import unittest
from pathlib import Path
from baby_arcus.sft_importers import import_messages
from baby_arcus.sft_dataset import windows
from baby_arcus.data_staging import StagingStore
from baby_arcus.services.data_review import Application


class Tokenizer:
    def encode(self,text): return list(text.encode())


class ReviewControls(unittest.TestCase):
    def test_known_tool_normalized_and_result_matched(self):
        rows=[{'role':'user','content':'Inspect solution.py'},
              {'role':'assistant','content':None,'tool_calls':[{'id':'one','type':'function',
                'function':{'name':'read_file','arguments':'{"path":"solution.py"}'}}]},
              {'role':'tool','tool_call_id':'one','content':'source'}]
        record=import_messages(rows,'fixture:import','one')
        self.assertEqual(json.loads(record['messages'][1]['content'])['version'],1)
        rows[2]['tool_call_id']='different'
        with self.assertRaises(ValueError): import_messages(rows,'fixture:import','one')
        rows[2]['tool_call_id']='one'; rows[1]['tool_calls'][0]['function']['name']='arbitrary_shell'
        with self.assertRaises(ValueError): import_messages(rows,'fixture:import','one')

    def test_failed_assistant_can_remain_context_without_targets(self):
        record={'version':1,'source':'fixture:test','group':'one','split':'training',
                'messages':[{'role':'user','content':'Fix it'},
                  {'role':'assistant','content':'wrong','train':False},
                  {'role':'tool','content':'failed'}, {'role':'assistant','content':'correct'}]}
        chunks=list(windows(record,Tokenizer()))
        targets=bytes(t for chunk in chunks for t,m in zip(chunk['ids'],chunk['mask']) if m).decode()
        self.assertEqual(targets,'correct\n</assistant>\n')
        self.assertIn(b'wrong',bytes(chunks[0]['ids']))

    def test_queue_and_revocation_preserve_original_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            secret='human-credential-fixture-12345'; machine='machine-credential-fixture-12345'
            store=StagingStore(Path(tmp)/'review.sqlite',secret,True)
            try:
                app=Application(store,machine)
                record=import_messages([{'role':'user','content':'Hello'},{'role':'assistant','content':'Hello'}],'fixture:test','one')
                batch=store.stage([record])
                self.assertEqual(app('POST','/queue',{'credential':machine})[1]['batches'][0]['id'],batch)
                store.review(batch,'approved','human',secret)
                body={'credential':machine,'batch_id':batch,'reviewer':'machine','reason':'no'}
                self.assertEqual(app('POST','/revoke',body)[0],403)
                body.update(credential=secret,reviewer='human')
                status,result=app('POST','/revoke',body)
                self.assertEqual(status,200); self.assertEqual(result['decision'],'approved')
                with self.assertRaises(ValueError): store.approved([batch],True)
            finally: store.close()
