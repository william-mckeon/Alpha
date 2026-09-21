import copy,json,tempfile,unittest
from pathlib import Path
from baby_arcus.visual_replay import transition,load,split_key,ReplayIndex

class ReplayTests(unittest.TestCase):
    def row(self):
        a={'id':'a','session':'s','entity_id':'e','scope_id':'p','tick':2}
        b=dict(a,id='b',tick=3)
        return transition(a,{'id':'a'},None,b,True,1.)
    def test_pairing_and_scope(self):
        row=self.row();after=dict(row['after'],scope_id='other')
        with self.assertRaises(ValueError):transition(row['before'],row['decision'],None,after,True,1)
        with self.assertRaises(ValueError):transition(row['before'],{'id':'wrong'},None,row['after'],True,1)
    def test_deduplication_and_interrupted_records(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'log';row=self.row()
            p.write_text(json.dumps(row)+'\n'+json.dumps(row)+'\n{"partial":')
            self.assertEqual(load([p]),[row])
            other=copy.deepcopy(row);other['after']['tick']=4
            p.write_text(json.dumps(row)+'\n'+json.dumps(other))
            with self.assertRaises(ValueError):load([p])
    def test_whole_session_split(self):
        row=self.row();other=copy.deepcopy(row);other['id']='different'
        self.assertEqual(split_key(row),split_key(other))

    def test_index_restart_fingerprint_and_deduplication(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source.jsonl';row=self.row()
            source.write_text(json.dumps(row)+'\n{"partial":',encoding='utf-8')
            index=ReplayIndex(root/'index.db')
            self.assertEqual(index.ingest([source]),1)
            manifest=index.manifest()
            reopened=ReplayIndex(root/'index.db')
            self.assertEqual(reopened.ingest([source]),0)
            self.assertEqual(reopened.manifest(),manifest)
            self.assertEqual(list(reopened.rows([split_key(row)])),[row])
            self.assertEqual(list(reopened.rows([(split_key(row)+1)%10])),[])

    def test_atomic_quota_and_conflict_rollback(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';row=self.row()
            other=copy.deepcopy(row);other['id']='c';other['before']['id']='c';other['decision']['id']='c'
            source.write_text(json.dumps(row)+'\n'+json.dumps(other)+'\n')
            index=ReplayIndex(root/'index.db',max_rows=1)
            with self.assertRaisesRegex(ValueError,'quota'):index.ingest([source])
            self.assertEqual(index.manifest()['rows'],0)
            self.assertTrue(source.exists())
            source.write_text(json.dumps(row)+'\n');index.ingest([source])
            changed=copy.deepcopy(row);changed['after']['tick']=5
            source.write_text(json.dumps(changed)+'\n')
            with self.assertRaisesRegex(ValueError,'Conflicting'):index.ingest([source])
            self.assertEqual(list(index.rows()),[row])

    def test_corrupt_terminated_tail_and_forged_id_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source';row=self.row()
            source.write_text(json.dumps(row)+'\n{"broken":\n')
            with self.assertRaises(ValueError):load([source])
            row['id']='forged';source.write_text(json.dumps(row)+'\n')
            with self.assertRaisesRegex(ValueError,'ID mismatch'):load([source])

    def test_byte_quota_and_corruption_rollback(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';row=self.row()
            source.write_text(json.dumps(row)+'\n')
            small=ReplayIndex(root/'small.db',max_bytes=1)
            with self.assertRaisesRegex(ValueError,'quota'):small.ingest([source])
            self.assertEqual(small.manifest()['rows'],0)
            source.write_text(json.dumps(row)+'\n{corrupt}\n')
            index=ReplayIndex(root/'index.db')
            with self.assertRaises(ValueError):index.ingest([source])
            self.assertEqual(index.manifest()['rows'],0)

    def test_fingerprint_independent_of_import_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);a=root/'a';b=root/'b';row=self.row()
            other=copy.deepcopy(row)
            other['id']='c';other['before']['id']='c';other['decision']['id']='c'
            a.write_text(json.dumps(row)+'\n');b.write_text(json.dumps(other)+'\n')
            first=ReplayIndex(root/'first.db');second=ReplayIndex(root/'second.db')
            first.ingest([a,b]);second.ingest([b,a])
            self.assertEqual(first.manifest()['sha256'],second.manifest()['sha256'])

if __name__=='__main__':unittest.main()
