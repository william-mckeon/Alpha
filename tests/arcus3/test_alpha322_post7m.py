"""The 7M handoff changes evaluation cadence without resetting training state."""
import copy
import unittest

from arcus3.config import read
from arcus3.production import identity,thresholds,token_due,transition,validate_policy


class PostSevenMillionTests(unittest.TestCase):
    def setUp(self):
        self.old=read('configs/arcus3/production_alpha322.json')
        self.new=validate_policy(read('configs/arcus3/production_alpha322_post7m.json'))
        self.checkpoint=self.new['continuation_checkpoint_sha256']
        self.state={
            'updates':10344,'input_tokens':7001336,'target_tokens':5949068,
            'accumulation_position':0,'evaluation_pending':[],
            'evaluation_completed':[{'update':10344,'tier':'full','checkpoint_sha256':self.checkpoint}],
            'data_sha256':'data','teacher_sha256':'teacher','stream':{'offset':17},
            'optimizer':'preserved','python_rng':'preserved','torch_rng':'preserved','cuda_rng':'preserved',
            'production':{'campaign_id':self.old['campaign_id'],'policy_sha256':identity(self.old),
                          'batch_id':'pilot-migration','review_input_tokens':100000000,
                          'next_evaluation':{'light':14000000,'developmental':14000000,'full':14000000}},
        }
        self.receipt={'schema':'arcus3-alpha322-evaluation-cadence-v1',
            'prior_policy_path':'configs/arcus3/production_alpha322.json',
            'prior_policy_sha256':identity(self.old),'checkpoint_sha256':self.checkpoint,
            'old_data_sha256':'data','new_data_sha256':'data','new_teacher_sha256':'teacher',
            'policy_sha256':identity(self.new),'batch_id':'pilot-migration'}

    def test_exact_transition_and_next_milestones(self):
        result=transition(self.state,self.receipt,self.checkpoint,'data','teacher',self.new,same_data=True)
        for key in ('updates','input_tokens','target_tokens','stream','optimizer','python_rng','torch_rng','cuda_rng'):
            self.assertEqual(result[key],self.state[key])
        self.assertEqual(result['production']['next_evaluation'],
            {'light':10000000,'developmental':10000000,'full':10000000})
        schedule=self.new['evaluation']
        self.assertEqual(token_due(9999999,10000001,schedule),['full'])
        self.assertEqual(token_due(14999999,15000001,schedule),['light'])
        self.assertEqual(token_due(19999999,20000001,schedule),['full'])
        self.assertEqual(thresholds(95000001,schedule)['full'],100000000)

    def test_requires_accepted_full_7m_evaluation_and_exact_checkpoint(self):
        bad=copy.deepcopy(self.state);bad['evaluation_completed']=[]
        with self.assertRaisesRegex(ValueError,'policy changed'):
            transition(bad,self.receipt,self.checkpoint,'data','teacher',self.new,same_data=True)
        bad=copy.deepcopy(self.state);bad['evaluation_completed'][-1]['tier']='light'
        with self.assertRaisesRegex(ValueError,'policy changed'):
            transition(bad,self.receipt,self.checkpoint,'data','teacher',self.new,same_data=True)
        bad=copy.deepcopy(self.receipt);bad['checkpoint_sha256']='wrong'
        with self.assertRaises(ValueError):
            transition(self.state,bad,self.checkpoint,'data','teacher',self.new,same_data=True)

    def test_rejects_any_other_policy_change(self):
        changed=copy.deepcopy(self.new);changed['mixture']['code']=.3
        with self.assertRaises(ValueError):
            transition(self.state,{**self.receipt,'policy_sha256':identity(changed)},self.checkpoint,
                       'data','teacher',changed,same_data=True)


if __name__=='__main__':unittest.main()
