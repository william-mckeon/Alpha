import unittest
from baby_arcus.shared_curriculum import example,samples
from baby_arcus.shared_experience import validate

class CurriculumTests(unittest.TestCase):
    def test_counterfactual_pairs_keep_body_constant(self):
        first,_,_=example(0,family='commands',paired=True)
        second,_,_=example(1,family='commands',paired=True)
        self.assertEqual(first['senses']['joint_positions'],second['senses']['joint_positions'])
        self.assertNotEqual(first['hearing'],second['hearing'])
        first,label1,_=example(0,family='color_reference',paired=True)
        second,label2,_=example(1,family='color_reference',paired=True)
        self.assertEqual(first['vision'],second['vision'])
        self.assertNotEqual(label1['gaze_choice'],label2['gaze_choice'])

    def test_labels_and_oracle_metadata_are_not_model_inputs(self):
        for family in ('commands','color_reference'):
            row,label,meta=example(7,family=family)
            self.assertTrue(validate(row));self.assertEqual(row['objects'],[])
            self.assertNotIn('targets',row);self.assertNotIn('gaze_choice',row)
            held,_,held_meta=example(7,'confirmation',family)
            self.assertFalse(held['eligibility']['training']);self.assertNotEqual(meta['seed'],held_meta['seed'])
            self.assertNotEqual(row['session'],held['session'])
    def test_balanced_families(self):
        rows=samples(10)
        self.assertEqual(sum(m['family']=='commands' for _,_,m in rows),5)

    def test_runtime_body_history_is_present_and_scope_valid(self):
        lengths=set()
        for i in range(150):
            row,_,_=example(i,paired=True);validate(row);lengths.add(len(row['history']))
            self.assertTrue(all(prior['tick']<row['tick'] for prior in row['history']))
        self.assertEqual(lengths,{0,1,2})
