import tempfile,unittest
from copy import deepcopy
from pathlib import Path
import torch
from tests.baby_arcus.test_shared_learning import fixture,Tokenizer
from baby_arcus.shared_model import SharedModel
from baby_arcus.shared_depth import set_depth,verify_depth
from baby_arcus.shared_causal import causal_features,action_features
from baby_arcus.shared_causal_curriculum import transition
from baby_arcus.shared_temporal import sequence_targets
from baby_arcus.shared_replay import Replay

class CausalTests(unittest.TestCase):
    def test_depth_is_enforced_and_residual_starts_at_persistence(self):
        old,row=fixture()
        with self.assertRaises(ValueError):verify_depth(old)
        set_depth(old);self.assertEqual(verify_depth(old),.25)
        model=SharedModel(old.body,old.language,version=9)
        observed_depths=[]
        hooks=[block.register_forward_pre_hook(lambda block,args:observed_depths.append(block.capacity)) for block in model.core.blocks]
        row['objects']=[];row['executed_action']={'kind':'joint','joint':'front_left.hip','delta':-.15}
        out=model([row],Tokenizer(),requested=('future_body','future_rgb','future_ensemble'))
        features=causal_features(row)
        self.assertEqual(len(features),164)
        self.assertTrue(torch.allclose(out['future_body'][0],torch.tensor(features[:20])))
        self.assertEqual(tuple(out['future_ensemble'].shape),(1,3,68))
        self.assertGreaterEqual(len(observed_depths),len(model.core.blocks)*3)
        self.assertEqual(set(observed_depths),{.25})
        for hook in hooks:hook.remove()
        loss=(out['future_body']-.1).square().mean();loss.backward()
        self.assertIsNotNone(model.causal_predictors[0][-1].weight.grad)
        model.core.blocks[0].capacity=1
        with self.assertRaises(ValueError):verify_depth(model)
        with self.assertRaisesRegex(ValueError,'override'):model([row],Tokenizer(),requested=('body',))

    def test_actual_sequences_replay_and_holdouts(self):
        row,label,transitions=transition(0,'training')
        self.assertGreater(row['prediction_horizon'],3)
        self.assertEqual(len(row['executed_action']['actions']),3)
        bad=deepcopy(transitions);bad[1][0]['id']='different'
        with self.assertRaises(ValueError):sequence_targets(bad)
        with tempfile.TemporaryDirectory() as root:
            replay=Replay(Path(root)/'replay.sqlite3')
            self.assertTrue(replay.add_sequence(transitions));self.assertFalse(replay.add_sequence(transitions))
            held=transition(1,'confirmation')[2];self.assertTrue(replay.add_sequence(held))
            self.assertEqual(len(replay.select()),1);self.assertEqual(replay.counts(),{'evaluation':1,'training':1});replay.close()

    def test_memory_is_model_input_not_future_label(self):
        row,_,_=transition(15,'validation');features=causal_features(row)
        self.assertEqual(len({tuple(m['features'][20:24]) for m in row['memory']}),4)
        self.assertEqual(features[-1],1)
        row['memory']=[];self.assertEqual(causal_features(row)[-73:],[0.0]*73)
        row['prediction_horizon']=91
        with self.assertRaises(ValueError):action_features(row)
