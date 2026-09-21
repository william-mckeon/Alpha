import tempfile
import unittest
from baby_arcus.services.controller import ControllerApplication
from baby_arcus.curriculum import Curriculum
from baby_arcus.evaluation import EvaluationGate,summary

class EvaluationFlowTests(unittest.TestCase):
    def test_layout_does_not_reveal_balanced_clue_answer(self):
        from baby_arcus.evaluation import SPLITS,seed_for,layout_for
        from baby_arcus.lessons import generate
        from baby_arcus.observations import observe
        for split in SPLITS:
            for index in range(0,20,2):
                worlds=[generate("clue_search",seed_for(split,i),64,layout_for(split,i),split,2) for i in (index,index+1)]
                self.assertNotEqual(worlds[0].correct_marker,worlds[1].correct_marker)
                seeker=next(a for a,s in worlds[0].agents.items() if s["role"]=="seeker")
                self.assertEqual(observe(worlds[0],seeker),observe(worlds[1],seeker))

    def test_checked_in_definition(self):
        from baby_arcus.evaluation import validate_definition
        validate_definition("configs/baby_arcus/evaluation.json")

    def test_heldout_geometry_disjoint_under_rotations_and_reflections(self):
        from baby_arcus.evaluation import LAYOUTS
        from baby_arcus.lessons import generate
        def geometry(world):
            points=[("wall",p) for p in world.walls]
            points += [(name,getattr(world,name)) for name in ("door","plate","object_position","delivery","clue_position") if getattr(world,name) is not None]
            points += [("container",c["position"]) for c in world.containers]
            points += [("agent-"+a["role"],a["position"]) for a in world.agents.values()]
            forms=[]
            for mirror in (False,True):
                for rotation in range(4):
                    transformed=[]
                    for label,(x,y) in points:
                        if mirror: x=6-x
                        for _ in range(rotation): x,y=6-y,x
                        transformed.append((label,x,y))
                    forms.append(tuple(sorted(transformed)))
            return min(forms)
        for family in ("switch_delivery","clue_search"):
            populations={split:{geometry(generate(family,0,64,layout,split,difficulty)) for layout in layouts for difficulty in range(3)} for split,layouts in LAYOUTS.items()}
            for a in populations:
                for b in populations:
                    if a!=b:
                        self.assertFalse(populations[a]&populations[b],(family,a,b))

    def test_three_frozen_batches_then_reserved_confirmation(self):
        with tempfile.TemporaryDirectory() as root:
            app=ControllerApplication(root,{})
            app.state.start("run",100,"checkpoint")
            calls=[]
            def batch(lease,split,count,curriculum,checkpoint=None,index=None):
                calls.append((split,count,checkpoint))
                return {"checkpoint_id":checkpoint or "checkpoint","split":split,
                        "start_index":len(calls)*200,
                        "results":{f:summary([True]*count) for f in curriculum.state}}
            app.evaluate_batch=batch
            gate=EvaluationGate()
            app.evaluate_policy("lease",Curriculum(),gate)
            self.assertEqual([c[0] for c in calls],["practice","evaluation","evaluation","evaluation","reserved"])
            self.assertEqual(gate.state["mastery_checkpoint"],"checkpoint")
            self.assertTrue(app.state.value["evaluation"][-1]["gate"]["milestone"])
            app.close()

    def test_streak_does_not_mix_checkpoint_versions(self):
        gate=EvaluationGate()
        results={f:summary([True]*200) for f in ("switch_delivery","clue_search")}
        gate.record("one","1",results)
        gate.record("one","2",results)
        gate.record("two","3",results)
        self.assertEqual(gate.state["streak"],1)
        self.assertFalse(gate.record("two","r",results,"reserved")["milestone"])
