import tempfile,unittest,json
from copy import deepcopy
from pathlib import Path
from baby_arcus.shared_replay import Replay,partition
from tests.baby_arcus.test_shared_learning import fixture,Tokenizer

class ReplayTests(unittest.TestCase):
    def test_language_learns_without_movement_and_recovers_once(self):
        _,row=fixture()
        row['lesson_provenance']={'split':'training'}
        exposure={'source_id':'caregiver-1','prefix':[10,11],'target':12}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);replay=Replay(root/'queue.db')
            try:
                self.assertTrue(replay.add_language(row,exposure))
                changed=deepcopy(row);changed['tick']+=1
                self.assertFalse(replay.add_language(changed,exposure))
                item=replay.select()[0]
                self.assertEqual(item['targets'],{'text':12});self.assertFalse(item['row']['eligibility']['executed'])
                with self.assertRaises(ValueError):replay.add_language(row,dict(exposure,target=13))
                held=deepcopy(row);held['lesson_provenance']={'split':'confirmation'}
                replay.add_language(held,dict(exposure,source_id='held'))
                self.assertEqual(len(replay.select()),1)
                journal=root/'journal.jsonl'
                journal.write_text('\n'.join(json.dumps(e) for e in [
                    {'phase':'proposed','experience':row,'prediction':{'language_example':exposure}},
                    {'phase':'outcome','experience_id':row['id'],'executed':False,'after':row}])+'\n')
                self.assertEqual(replay.recover(journal),0)
            finally:replay.close()

    def test_lesson_and_corpus_holdouts_cannot_be_reclassified(self):
        _,row=fixture()
        while partition(row)!='training':row['session']+='x'
        row['lesson_provenance']={'split':'confirmation'}
        self.assertEqual(partition(row),'evaluation')
        row['lesson_provenance']={'split':'training'}
        row['ambient_hearing']={'source':'dataset','document':20}
        self.assertEqual(partition(row),'evaluation')
        row['ambient_hearing']['document']=21
        self.assertEqual(partition(row),'training')

    def test_publication_failure_and_replay_resume_do_not_double_train(self):
        import torch
        from unittest.mock import patch
        from baby_arcus.shared_checkpoint import save,load
        from baby_arcus.shared_learning import train_records
        model,row=fixture()
        from baby_arcus.shared_depth import set_depth
        set_depth(model)
        while partition(row)!='training':row['session']+='x'
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);manifest=save(root,model,torch.optim.AdamW(model.parameters()),{'updates':0,'receipts':[]})
            (root/'candidate.json').write_text(json.dumps(manifest))
            cfg=root/'config.json';cfg.write_text(json.dumps({'root':str(root),'learning_rate':.001,'encoding':'test','training_updates':2}))
            replay=Replay(root/'queue.db')
            replay.add(row,{'experience_id':row['id'],'executed':True,'action':{'kind':'gaze','yaw':.25,'pitch':0},'after':deepcopy(row)})
            replay.close()
            with patch('arcus.tokenizer.get_tokenizer',return_value=Tokenizer()):
                with patch('baby_arcus.language_stream.atomic_json',side_effect=OSError('simulated pointer publication failure')):
                    with self.assertRaises(OSError):train_records(cfg,replay=root/'queue.db')
                self.assertEqual(json.loads((root/'candidate.json').read_text()),manifest)
                first=train_records(cfg,replay=root/'queue.db')
                second=train_records(cfg,replay=root/'queue.db')
            self.assertEqual(first['candidate'],second['candidate'])
            self.assertEqual(second['losses'],[])
            _,data=load(root,first['candidate'])
            self.assertEqual(data['progress']['updates'],1)
            self.assertEqual(data['progress']['consumed_experiences'],[row['id']])

    def test_journal_recovers_once_and_ignores_uncommitted_tail(self):
        _,row=fixture()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);journal=root/'journal.jsonl'
            events=[{'phase':'proposed','experience':row},{'phase':'outcome','experience_id':row['id'],'executed':True,'after':deepcopy(row)}]
            journal.write_text(''.join(json.dumps(e)+'\n' for e in events)+'{"phase":',encoding='utf-8')
            replay=Replay(root/'queue.db')
            try:
                self.assertEqual(replay.recover(journal),1)
                self.assertEqual(replay.recover(journal),0)
            finally:replay.close()

    def test_duplicate_conflict_protection_and_balancing(self):
        _,row=fixture()
        while partition(row)!='training':row['session']+='x'
        with tempfile.TemporaryDirectory() as root:
            replay=Replay(Path(root)/'replay.db')
            def outcome(r):return {'experience_id':r['id'],'executed':True,'after':deepcopy(r)}
            self.assertTrue(replay.add(row,outcome(row)))
            self.assertFalse(replay.add(row,outcome(row)))
            with self.assertRaises(ValueError):replay.add(row,outcome(row),{'body':0})
            other=deepcopy(row);other['id']='second'
            replay.add(other,outcome(other),{'body':0},task='body')
            self.assertEqual({item['task'] for item in replay.select(2)},{'body','prediction'})
            self.assertEqual(len(replay.select(consumed=[row['id']])),1)
            held=deepcopy(row);held['id']='held'
            while partition(held)!='evaluation':held['session']+='y'
            replay.add(held,outcome(held));self.assertEqual(replay.counts()['evaluation'],1)
            self.assertEqual(len(replay.select()),2)
            bad=outcome(row);bad['after']['epoch']+=1
            with self.assertRaises(ValueError):replay.add(row,bad)
            replay.close()
    def test_history_and_gaze_are_real_model_inputs(self):
        import torch
        from baby_arcus.shared_experience import validate
        from baby_arcus.contracts import ContractError
        from baby_arcus.services.shared_worker import gaze_action
        model,row=fixture();base=model([row],Tokenizer())['hidden']
        row['history']=[{key:deepcopy(row[key]) for key in ('session','entity_id','scope_id','epoch','tick','senses')}]
        changed=model([row],Tokenizer())
        self.assertFalse(torch.allclose(base,changed['hidden']))
        self.assertEqual(changed['gaze_choice'].shape[-1],7)
        row['gaze']=[0,0,1,0];self.assertEqual(gaze_action(row,2)['yaw'],1)
        self.assertEqual(gaze_action(row,6),{'kind':'eyelids','openness':0})
        row['history'][0]['epoch']+=1
        with self.assertRaises(ContractError):validate(row)
