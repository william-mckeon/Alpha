import json
from pathlib import Path
import tempfile
import unittest
import torch
from dataclasses import asdict
from baby_arcus.model import BabyModel
from baby_arcus.presets import configuration
from baby_arcus.vocabulary import VOCABULARY_HASH
from baby_arcus.large_body_learning import adapt
from baby_arcus.body_policy import save,load
from baby_arcus.body_controller_config import configuration as controller_config
from baby_arcus.standing_environment import StandingEnvironment

class LargeAdapterTests(unittest.TestCase):
    def test_only_action_head_changes_and_reload_preserves_freeze(self):
        torch.set_num_threads(2);torch.manual_seed(716)
        cfg=configuration("tiny");source=BabyModel(cfg)
        payload={"format":1,"vocabulary_hash":VOCABULARY_HASH,"model_config":asdict(cfg),"model":source.state_dict()}
        adapter=adapt(payload)
        self.assertEqual(adapter.cfg.capacity_factor,cfg.capacity_factor)
        optimizer=torch.optim.AdamW(adapter.actor.parameters(),lr=.01)
        before={k:v.clone() for k,v in adapter.core.state_dict().items()}
        actor=adapter.actor.weight.clone()
        loss=adapter([StandingEnvironment().observe()])[0,2]
        loss.backward();optimizer.step()
        self.assertFalse(torch.equal(actor,adapter.actor.weight))
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in adapter.core.state_dict().items()))
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/"body.pt";save(path,adapter,optimizer,1)
            restored,data=load(path)
            self.assertTrue(all(not p.requires_grad for p in restored.core.parameters()))
            resumed=torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad])
            resumed.load_state_dict(data["optimizer"])
            torch.testing.assert_close(adapter([StandingEnvironment().observe()]),restored([StandingEnvironment().observe()]))
    def test_activation_requires_matching_passed_qualification(self):
        with tempfile.TemporaryDirectory() as project:
            root=Path(project)/"runs/arcus_large_body";(root/"qualification").mkdir(parents=True)
            manifest={"checkpoint_sha256":"abc","trunk_weights_preserved":True,"parameters":125085761}
            (root/"manifest.json").write_text(json.dumps(manifest))
            report={"gate_passed":False,"checkpoint_unchanged":True,"checkpoint_sha256":"abc"}
            path=root/"qualification/report.json";path.write_text(json.dumps(report))
            self.assertIn("pilot",str(controller_config(project)["checkpoint"]))
            report["gate_passed"]=True;path.write_text(json.dumps(report))
            self.assertEqual(controller_config(project)["expected_hash"],"abc")
            report["checkpoint_sha256"]="changed";path.write_text(json.dumps(report))
            self.assertIn("pilot",str(controller_config(project)["checkpoint"]))
    def test_two_posture_activation_requires_passed_matching_report(self):
        with tempfile.TemporaryDirectory() as project:
            root=Path(project)/"runs/arcus_postures_v2";(root/"qualification").mkdir(parents=True)
            manifest={"checkpoint_sha256":"postures","trunk_weights_preserved":True,"standing_weights_preserved":True,"parameters":125098586}
            (root/"manifest.json").write_text(json.dumps(manifest))
            report={"gate_passed":False,"checkpoint_unchanged":True,"checkpoint_sha256":"postures"}
            path=root/"qualification/report.json";path.write_text(json.dumps(report))
            self.assertNotIn("goals",controller_config(project))
            report["gate_passed"]=True;path.write_text(json.dumps(report))
            self.assertEqual(controller_config(project)["goals"],("standing","lying"))
            report["checkpoint_sha256"]="changed";path.write_text(json.dumps(report))
            self.assertNotIn("goals",controller_config(project))
