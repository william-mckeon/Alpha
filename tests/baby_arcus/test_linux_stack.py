"""Opt-in checks of the completed isolated Docker qualification run.

Set BABY_LINUX_TOKEN_FILE to its private single-line env file. This suite does
not start/stop containers or train a model; run baby_arcus.qualify first.
"""
import os
from pathlib import Path
import unittest
from baby_arcus.transport import Client

@unittest.skipUnless(os.environ.get("BABY_LINUX_TOKEN_FILE"),"Requires explicit live Docker qualification stack")
class LinuxStackTests(unittest.TestCase):
    def setUp(self):
        token=Path(os.environ["BABY_LINUX_TOKEN_FILE"]).read_text().strip().split("=",1)[1]
        self.clients={name:Client("http://127.0.0.1:"+str(port),token,timeout=10)
                      for name,port in (("simulation",8765),("artifacts",8766),("inference",8767),
                                        ("training",8768),("controller",8769),("evaluator",8770))}

    def test_settled_gpu_run_has_released_both_workers(self):
        for name,client in self.clients.items():
            ready=client.request("GET","/ready")
            self.assertTrue(ready["ready"])
            if name in ("inference","training"):
                self.assertFalse(ready["loaded"])
        state=self.clients["controller"].request("GET","/v1/status")
        self.assertIn(state["run"]["status"],("completed","paused"))
        if state["run"]["status"]=="paused":
            self.assertTrue(state["run"].get("reason"))
        self.assertIsNone(state["resource"]["lease"])
        self.assertGreaterEqual(state["run"]["cycles"],1)
        self.assertGreaterEqual(state["run"]["metrics"][-1]["samples"],1024)
        manifest=self.clients["artifacts"].request("GET","/v1/artifacts/"+state["run"]["checkpoint_id"])
        self.assertEqual(manifest["payload"]["metadata"]["purpose"],"policy")

    def test_report_and_evaluation_provenance_are_served(self):
        controller=self.clients["controller"]
        state=controller.request("GET","/v1/status")["run"]
        report=controller.request("GET","/v1/reports/"+state["run_id"])
        self.assertEqual(report["checkpoint_id"],state["checkpoint_id"])
        self.assertTrue(report["evaluation"])
        for index,compact in enumerate(report["evaluation"]):
            batch=controller.request("GET","/v1/reports/"+state["run_id"]+"/evaluations/"+str(index))
            self.assertEqual(len(batch["provenance"]),sum(r["episodes"] for r in batch["results"].values()))
            if batch["complete"]:
                self.assertTrue(all(r["episodes"]==batch["requested_per_family"] for r in batch["results"].values()))
            else:
                self.assertTrue(batch["reason"])
                self.assertTrue(all(r["episodes"]<=batch["requested_per_family"] for r in batch["results"].values()))
