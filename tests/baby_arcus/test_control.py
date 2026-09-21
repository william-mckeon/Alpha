import tempfile
import time
import unittest
from pathlib import Path
from baby_arcus.artifacts import Conflict
from baby_arcus.contracts import ContractError
from baby_arcus.curriculum import Curriculum
from baby_arcus.evaluation import EvaluationGate,summary,LAYOUTS,seed_for
from baby_arcus.resource_control import ResourceManager
from baby_arcus.run_control import RunState
from baby_arcus.lessons import generate,validate_solvable
from baby_arcus.reports import write_report

class ControlTests(unittest.TestCase):
    def test_curriculum_independent_three_batch_gate(self):
        c=Curriculum()
        for _ in range(2):
            c.record("clue_search",0,40,50)
        self.assertEqual(c.state["clue_search"]["level"],0)
        restored=Curriculum(c.state)
        restored.record("clue_search",0,40,50)
        self.assertEqual(restored.state["clue_search"]["level"],1)
        self.assertEqual(restored.state["switch_delivery"]["level"],0)
        for _ in range(2):
            restored.record("clue_search",1,10,50)
        self.assertTrue(restored.state["clue_search"]["recovering"])

    def test_evaluation_exact_denominators_and_reserved_gate(self):
        gate=EvaluationGate()
        results={f:summary([True]*160+[False]*40) for f in ("switch_delivery","clue_search")}
        with self.assertRaises(ContractError):
            gate.record("p","bad",{"switch_delivery":summary([True])})
        for i in range(3):
            result=gate.record("p",str(i),results)
            self.assertFalse(result["milestone"])
        result=gate.record("p","reserved",results,"reserved")
        self.assertTrue(result["milestone"])
        with self.assertRaises(ContractError):
            gate.record("p","reserved",results,"reserved")
        lower={f:summary([True]*140+[False]*60) for f in results}
        self.assertTrue(gate.record("q","drop1",lower,reference=results)["confirm_regression"])
        self.assertTrue(gate.record("q","drop2",lower,reference=results)["pause"])

    def test_all_split_difficulty_layouts_solvable(self):
        for split,layouts in LAYOUTS.items():
            for layout in layouts:
                for difficulty in range(3):
                    for family in ("switch_delivery","clue_search"):
                        for seed in range(8):
                            world=generate(family,seed,64,layout,split,difficulty)
                            self.assertTrue(validate_solvable(world),(split,layout,difficulty,family))
                            self.assertNotEqual(world.agents["a"]["position"],world.agents["b"]["position"])

    def test_lease_restart_never_reassigns_until_unload(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/"lease.json"
            manager=ResourceManager(path)
            lease=manager.acquire("inference")
            with self.assertRaises(Conflict):
                manager.acquire("training")
            manager=ResourceManager(path)
            with self.assertRaises(Conflict):
                manager.check(lease["id"],"inference")
            with self.assertRaises(Conflict):
                manager.acquire("training")
            manager.release(lease["id"],True)
            self.assertEqual(manager.acquire("training")["generation"],2)

    def test_run_restart_budget_disk_and_report(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/"run.json"
            run=RunState(path)
            with self.assertRaises(ContractError):
                run.start("r",43201,"p")
            run.start("r",10,"p")
            resumed=RunState(path)
            self.assertEqual(resumed.value["status"],"paused")
            self.assertEqual(resumed.value["checkpoint_id"],"p")
            with self.assertRaises(RuntimeError):
                run.check_disk(root,minimum_free=10**20)
            write_report(root,resumed.value)
            self.assertIn("No transfer milestone", (Path(root)/"report.md").read_text())

