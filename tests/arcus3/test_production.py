import copy,json,tempfile,unittest
from pathlib import Path
from arcus3.config import read
from arcus3.production import token_due,thresholds,transition,identity,validate_policy
from arcus3.corpus_stream import CorpusStream
from scripts.prepare_arcus3_phase8_data import seal
from arcus3.production_data import normalize,read_rows
from arcus3.checkpoint import digest

class ProductionTests(unittest.TestCase):
    def test_donor_result_must_match_checkpoint_protocol_and_context(self):
        from arcus3.production import accept_donor_receipt
        p=read('configs/arcus3/production.json')
        r={'complete':True,'checkpoint_sha256':'c','tier':'light','protocol':{'revision':p['donor_evaluation_revision']},
           'lighteval_revision':p['lighteval_revision'],'model_context':8192,'donor_context':8192,'results':{'x':1}}
        accept_donor_receipt(r,'c','light',p)
        for field,value in [('complete',False),('checkpoint_sha256','other'),('model_context',512),('results',{})]:
            with self.assertRaises(ValueError):accept_donor_receipt({**r,field:value},'c','light',p)
    def test_heldout_overlap(self):
        from arcus3.production_data import ProductionExclusions
        text=' '.join('word'+str(i) for i in range(20))
        f=ProductionExclusions([text])
        self.assertTrue(f.matches([{'content':'prefix '+text+' suffix'}]))
        self.assertFalse(f.matches([{'content':'unrelated words'}]))
    def test_threshold_crossing_and_priority(self):
        s={'light':1000000,'developmental':10000000,'full':100000000}
        self.assertEqual(token_due(999900,1000300,s),['light'])
        self.assertEqual(token_due(9999900,10000100,s),['developmental'])
        self.assertEqual(token_due(99999990,100000000,s),['full'])
        self.assertEqual(token_due(1000300,1000900,s),[])
        self.assertEqual(thresholds(1000300,s)['light'],2000000)
    def test_transition_preserves_counters_and_refuses_pending_eval(self):
        p=validate_policy(read('configs/arcus3/production.json'))
        state={'input_tokens':100,'updates':12,'data_sha256':'old','teacher_sha256':'teacher','accumulation_position':0,
            'evaluation_pending':[],'stream':{'offset':10},'batch_complete':True}
        receipt={'checkpoint_sha256':'checkpoint','old_data_sha256':'old','new_data_sha256':'new','new_teacher_sha256':'newteacher',
                 'policy_sha256':identity(p),'batch_id':'next'}
        result=transition(state,receipt,'checkpoint','new','newteacher',p)
        self.assertEqual(result['input_tokens'],100);self.assertEqual(result['updates'],12)
        self.assertIsNone(result['stream']);self.assertEqual(state['stream'],{'offset':10})
        state['evaluation_pending']=['light']
        with self.assertRaises(ValueError):transition(state,receipt,'checkpoint','new','newteacher',p)
    def test_pilot_migration_preserves_cursor_and_next_token_evaluation(self):
        policy=validate_policy(read('configs/arcus3/production.json'))
        state={'updates':5952,'input_tokens':4125876,'target_tokens':3531621,
               'data_sha256':'pilot','teacher_sha256':'teacher','config_sha256':'optimizer-policy',
               'accumulation_position':0,'evaluation_pending':[],
               'evaluation_completed':[{'update':5000,'nll':2.24}],
               'stream':{'offset':45764348,'records':5952,'epoch':0,'shard':0}}
        before=copy.deepcopy(state)
        receipt={'checkpoint_sha256':'checkpoint','old_data_sha256':'pilot','new_data_sha256':'pilot',
                 'new_teacher_sha256':'teacher','policy_sha256':identity(policy),'batch_id':'pilot-migration'}
        result=transition(state,receipt,'checkpoint','pilot','teacher',policy,same_data=True)
        for key in before:self.assertEqual(result[key],before[key])
        self.assertEqual(state,before)
        self.assertEqual(result['production']['next_evaluation'],
                         {'light':5000000,'developmental':10000000,'full':100000000})
    def test_exhaustion_does_not_wrap(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'train-0.jsonl').write_text(json.dumps({'input_ids':[1,2],'labels':[-100,2]})+'\n')
            (p/'provenance.json').write_text(json.dumps({'reviewed':True,'scope':'combined-phase8','tokenizer_sha256':'t','sources_sha256':'s','evaluation_exclusions_sha256':'e'}))
            seal(p,p/'provenance.json');s=CorpusStream(p,repeat=False);s.next()
            with self.assertRaises(StopIteration):s.next()
            resumed=CorpusStream(p,s.snapshot(),repeat=False)
            with self.assertRaises(StopIteration):resumed.next()
    def test_local_secrets_and_heldout_split(self):
        for row in ({'split':'test','messages':[]},{'split':'train','messages':[{'role':'assistant','content':'hf_'+'a'*30}]}):
            with self.assertRaises(ValueError):normalize(row,'local')
    def test_source_cursor_replay_and_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.jsonl';p.write_text('{"text":"first"}\n{"text":"second"}\n')
            spec={'local':True,'files':[{'path':str(p),'sha256':digest(p)}]}
            rows=list(read_rows(spec,{'row':1},None));self.assertEqual(rows[0][0]['text'],'second')
            p.write_text('changed')
            with self.assertRaises(ValueError):list(read_rows(spec,{},None))
