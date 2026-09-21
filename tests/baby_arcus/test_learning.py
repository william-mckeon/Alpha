"""Reward-only primitive smoke; this is not the cooperative milestone."""
import unittest
import torch
from baby_arcus.checkpoint import seed_everything
from baby_arcus.model import BabyModel
from baby_arcus.presets import configuration
from baby_arcus.learner import Learner,LearningConfig
from baby_arcus.lessons import generate
from baby_arcus.observations import observe
from baby_arcus.actions import Action
from baby_arcus.experience import prepare
from baby_arcus.contracts import ContractError

class LearningTests(unittest.TestCase):
    def test_policy_diagnostics_have_known_values_without_changing_loss(self):
        import math
        from baby_arcus.objectives import loss
        from baby_arcus.vocabulary import targets
        model=BabyModel(configuration('tiny'))
        output=model([[1,2]]*3,[[True]*8]*3)
        output['action']=torch.zeros_like(output['action'])
        output['signal']=torch.zeros_like(output['signal'])
        target=targets(observe(generate('clue_search',12),'a'))
        batch=[{'action':0,'signal':0,'logp':-math.log(72)-math.log(ratio),
                'advantage':1.,'return':0.,'target':target} for ratio in (1.,2.,.5)]
        objective,stats=loss(output,batch,prediction_coefficient=0)
        self.assertAlmostEqual(stats['approx_kl'],(1-math.log(2)-.5-math.log(.5))/3,places=6)
        self.assertAlmostEqual(stats['clip_fraction'],2/3,places=6)
        self.assertAlmostEqual(stats['policy'],-(1+1.2+.5)/3,places=6)
        for row in batch: row['logp']=-math.log(72)
        _,unchanged=loss(output,batch)
        self.assertAlmostEqual(unchanged['approx_kl'],0,places=6)
        self.assertEqual(unchanged['clip_fraction'],0)

    def test_reward_only_improves_without_demonstrations(self):
        torch.set_num_threads(2)
        seed_everything(51)
        model=BabyModel(configuration("tiny"))
        learner=Learner(model,LearningConfig(lr=.003,epochs=2,microbatch=32,min_samples=32,prediction_coefficient=0))
        world=generate("clue_search",12,max_steps=1)
        world.advance({"a":Action(),"b":Action()})
        observation=observe(world,"a")
        context=[1,2,3,4]
        mask=[True,True]+[False]*6
        before=float(model([context],[mask])["action"].softmax(-1)[0,1].detach())
        for update in range(16):
            records=[]
            for i,sample in enumerate(model.sample([context]*32,[mask]*32)):
                records.append({**sample,"context":context,"mask":mask,"checkpoint_id":str(update),
                    "actor":"agent","split":"train","episode_id":str(i),"agent_id":"a","step":0,
                    "reward":float(sample["action"]==1),"next_value":0.0,"terminated":True,"truncated":False,
                    "next_observation":observation})
            metrics=learner.update(records,str(update),str(update))
        after=float(model([context],[mask])["action"].softmax(-1)[0,1].detach())
        print({"primitive":"two-action reward task","before":before,"after":after,"updates":16})
        self.assertGreater(after,before+.2)
        self.assertGreater(after,.85)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
        self.assertEqual(metrics['diagnostic_samples'],64)
        self.assertGreaterEqual(metrics['approx_kl'],0)
        self.assertTrue(0<=metrics['clip_fraction']<=1)
        with self.assertRaises(ContractError):
            prepare(records,"stale",32)
        records[0]["split"]="evaluation"
        with self.assertRaises(ContractError):
            prepare(records,str(update),32)
