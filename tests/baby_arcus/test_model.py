import tempfile
import unittest
from pathlib import Path
import torch
from baby_arcus.model import BabyModel
from baby_arcus.presets import configuration
from baby_arcus.memory import AgentMemory
from baby_arcus.lessons import generate
from baby_arcus.observations import observe
from baby_arcus.actions import ACTIONS
from baby_arcus.checkpoint import seed_everything,save,load
from baby_arcus.learner import Learner,LearningConfig
from baby_arcus.objectives import advantages

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_grouped_contexts_match_individual_and_masks(self):
        seed_everything(10)
        model = BabyModel(configuration("tiny")).eval()
        contexts = [[1,2,3],[1,3,4,5],[1,2,3]]
        masks = [[True]+[False]*7]*3
        combined = model(contexts,masks)
        for i,ctx in enumerate(contexts):
            torch.testing.assert_close(combined["value"][i],model([ctx],[masks[i]])["value"][0],atol=1e-6,rtol=1e-5)
        self.assertTrue(all(r["action"]==0 for r in model.sample(contexts,masks)))
        combined["value"].sum().backward()
        self.assertTrue(any(p.grad is not None for p in model.core.parameters()))

    def test_memory_separate_and_bounded(self):
        from baby_arcus.actions import Action
        world = generate("clue_search",10)
        memories = {a:AgentMemory(a,256) for a in ("a","b")}
        for _ in range(20):
            for a,memory in memories.items():
                memory.observe(observe(world,a))
                memory.choose(0,0)
                self.assertLessEqual(len(memory.context()),256)
            world.advance({"a":Action(),"b":Action()})
        self.assertNotEqual(memories["a"].context(),memories["b"].context())
        with self.assertRaises(ValueError):
            memories["a"].observe(observe(world,"b"))

    def test_routing_metrics_weight_tokens_and_capacity_probe_restores(self):
        from baby_arcus.routing_probe import compare
        model=BabyModel(configuration("tiny")).eval()
        # Force every token to expert zero: known capacity overflow, independent of weights.
        for block in model.core.blocks:
            with torch.no_grad():
                block.moe.router.weight.zero_()
                block.moe.router.bias.zero_()
                block.moe.router.bias[0]=10
        output=model([[1,2],[1,2,3,4,5,6]])
        self.assertAlmostEqual(output["overflow"],.5)
        self.assertAlmostEqual(output["routing"]["router_layer_0_expert_0"],1)
        results=compare(model,[[1,2,3,4]],(1,2))
        self.assertEqual(results[0]["last_token_drop"],1)
        self.assertEqual(results[1]["last_token_drop"],0)
        self.assertEqual(results[1]["overflow"],0)
        self.assertTrue(all(b.moe.capacity_factor==1 for b in model.core.blocks))

    def test_checkpoint_restores_weights_optimizer_rng(self):
        seed_everything(7)
        learner = Learner(BabyModel(configuration("tiny")),LearningConfig(min_samples=8))
        learner.model([[1,2,3]])["value"].sum().backward()
        learner.optimizer.step()
        learner.updates=1
        learner.processed.add("batch")
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/"state.pt"
            save(path,learner,{"curriculum":{"level":0}})
            expected=torch.rand(4)
            model,restored,extra=load(path)
            torch.testing.assert_close(torch.rand(4),expected)
            for p,q in zip(learner.model.parameters(),model.parameters()):
                torch.testing.assert_close(p,q,rtol=0,atol=0)
            self.assertEqual(restored.updates,1)
            self.assertEqual(restored.processed,{"batch"})
            self.assertTrue(restored.optimizer.state)
            original_state=learner.optimizer.state_dict()
            restored_state=restored.optimizer.state_dict()
            self.assertEqual(original_state["param_groups"],restored_state["param_groups"])
            self.assertEqual(set(original_state["state"]),set(restored_state["state"]))
            for parameter,state in original_state["state"].items():
                self.assertEqual(set(state),set(restored_state["state"][parameter]))
                for name,value in state.items():
                    torch.testing.assert_close(value,restored_state["state"][parameter][name],rtol=0,atol=0)
            self.assertEqual(extra["curriculum"]["level"],0)

    def test_terminal_and_truncation_bootstrap(self):
        a,r=advantages([1,2],[.2,.3],[.3,5],[False,True],[False,True],gamma=1,lam=1)
        self.assertEqual(r,[3,2])
        a,r=advantages([1],[.2],[5],[False],[True],gamma=1,lam=1)
        self.assertEqual(r,[6])
