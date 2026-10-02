import copy
import json
import pickle
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.report_arcus3_model_comparison import build_report, digest, render_markdown, validate_plan
from scripts.run_arcus3_model_comparison import _container_succeeded


class ThreeModelComparisonTests(unittest.TestCase):
    categories = ("conversation", "comprehension", "instructions", "reasoning", "python", "tools")
    suite_sha = "1" * 64
    settings_sha = "2" * 64
    benchmark_sha = "3" * 64

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def developmental(self, checkpoint):
        return {
            "tier": "full", "execution_complete": True,
            "expanded_manifest_sha256": checkpoint, "conversion_manifest_sha256": "converted",
            "adapter_manifest_sha256": None, "suite_sha256": self.suite_sha, "settings_sha256": self.settings_sha,
            "tokenizer_revision": "revision", "precision": "mixed", "orchestration": "graph",
            "language": {"target_tokens": 309, "nll": 2.0, "perplexity": 7.389056},
            "categories": {name: {"count": 6, "passed": 3, "measured": 0 if name == "conversation" else 6, "truncated": 0}
                           for name in self.categories},
            "tool_metrics": {"total": 6, "executed": 3}, "resources": {"seconds": 1},
        }

    def benchmark(self, checkpoint):
        tasks = {
            "extended:ifeval:0": {"strict": .1}, "custom:hellaswag:0": {"acc_norm": .2},
            "custom:arc:challenge:0": {"acc_norm": .3}, "custom:piqa:0": {"acc_norm": .4},
            "custom:mmlu_pro:0": {"acc_norm": .5}, "custom:bbh:one:3": {"em": .6},
            "custom:gsm8k:5": {"qem": .7},
        }
        return {
            "complete": True, "tier": "full", "max_samples_per_task": None,
            "checkpoint_sha256": checkpoint, "model_context": 8192, "benchmark_context": 2048,
            "lighteval_revision": "light", "benchmark_manifest_sha256": self.benchmark_sha,
            "results": {"results": tasks}, "runtime_adjustments": [],
        }

    def fixture(self, root):
        workspace = root / "workspace"; storage = root / "storage"
        self.write(workspace / "configs/arcus3/phase8_storage.json", {"ready": True, "external_root": str(storage)})
        self.write(workspace / "configs/arcus3/runtime.json", {"image_id": "sha256:test"})
        self.write(workspace / "converted/manifest.json", {"converted": True})
        self.write(workspace / "donor/manifest.json", {"revision": "revision"})
        arms = []
        specs = (("donor", 0, 0, 0, "not-applicable"),
                 ("alpha3.2.0", 10, 7_400_000, 6_000_000, "after"),
                 ("alpha3.2.1", 11, 7_900_000, 6_500_000, "after"))
        for arm_id, updates, inputs, targets, relation in specs:
            cp = storage / arm_id / "checkpoint"; cp.mkdir(parents=True)
            (cp / "delta.safetensors").write_bytes((arm_id + "delta").encode())
            state_config={"model_label": None if arm_id=="alpha3.2.0" else "alpha3.2.1"}
            # Match the production json.dumps(sort_keys=True) identity, not the file bytes.
            import hashlib
            config_sha=hashlib.sha256(json.dumps(state_config,sort_keys=True).encode()).hexdigest()
            with zipfile.ZipFile(cp/"state.pt","w") as archive:
                archive.writestr("state/data.pkl",pickle.dumps({
                    "updates":updates,"input_tokens":inputs,"target_tokens":targets,
                    "parent_sha256":"parent","config_sha256":config_sha,"config":state_config}))
            manifest = {"schema": "arcus3-expanded-delta-v1", "parent_sha256": "parent", "updates": updates,
                        "config_sha256":config_sha,
                        "files": {name: digest(cp / name) for name in ("delta.safetensors", "state.pt")}}
            self.write(cp / "manifest.json", manifest); manifest_sha = digest(cp / "manifest.json")
            evidence_root = workspace if arm_id == "donor" else storage
            evidence_rel = "evidence/%s.json" % arm_id
            evidence = {"checkpoint": str(cp), "manifest_sha256": manifest_sha, "updates": updates,
                        "input_tokens": inputs, "target_tokens": targets, "committed": True, "retention_complete": True}
            if arm_id=="donor":evidence.update(verified=True,fresh_optimizer=True,frozen_unchanged=True,model_label="alpha3.2.1")
            self.write(evidence_root / evidence_rel, evidence)
            arm = {"id": arm_id, "label": arm_id, "checkpoint_relative": "%s/checkpoint" % arm_id,
                   "evidence_relative": evidence_rel, "checkpoint_manifest_sha256": manifest_sha,
                   "evidence_sha256": digest(evidence_root / evidence_rel), "parent_sha256": "parent",
                   "updates": updates, "input_tokens": inputs, "target_tokens": targets,
                   "boundary_relation": relation}
            if arm_id == "donor":
                arm.update(control_kind="donor-derived-zero-update-initialization", evidence_scope="workspace",
                           state_model_label="alpha3.2.1",
                           donor_revision="revision", donor_manifest_relative="donor/manifest.json",
                           donor_manifest_sha256=digest(workspace / "donor/manifest.json"))
            else:
                arm["claimed_exact_boundary"] = False
                arm["state_model_label"] = arm_id
                if arm_id=="alpha3.2.0":arm["allow_legacy_missing_model_label"]=True
            arms.append(arm)
        protocols = {
            "developmental": {"mode": "baseline", "tier": "full", "max_minutes": 29, "suite_sha256": self.suite_sha,
                              "settings_sha256": self.settings_sha, "tokenizer_revision": "revision", "orchestration": "graph",
                              "language_target_tokens": 309, "categories": list(self.categories)},
            "benchmark": {"mode": "donor-baseline", "tier": "full", "max_minutes": 60,
                          "model_context": 8192, "benchmark_context": 2048, "lighteval_revision": "light",
                          "benchmark_manifest_sha256": self.benchmark_sha, "required_task_prefixes": ["extended:ifeval:",
                          "custom:hellaswag:", "custom:arc:", "custom:piqa:", "custom:mmlu_pro:", "custom:bbh:", "custom:gsm8k:"]},
        }
        plan = {"schema": "arcus3-three-model-comparison-v1", "requested_boundary_input_tokens": 7_000_000,
                "exact_boundary_checkpoint_available": False, "execution": "strictly-sequential", "automatic_promotion": False,
                "storage_config": "configs/arcus3/phase8_storage.json", "converted_relative": "converted",
                "evaluation_runtime": "configs/arcus3/runtime.json",
                "evaluation_runtime_sha256": digest(workspace / "configs/arcus3/runtime.json"),
                "converted_manifest_sha256": digest(workspace / "converted/manifest.json"), "protocols": protocols,
                "arms": arms}
        pure_donor = workspace / "runs/pure-donor"
        pure_value = self.developmental(arms[0]["checkpoint_manifest_sha256"])
        pure_value.update(schema="arcus3-baseline-v1", tier=None, unique_parameters=123,
                          adapter_manifest_sha256=None, expanded_manifest_sha256=None,
                          conversion_manifest_sha256=None)
        self.write(pure_donor / "scores.json", pure_value)
        plan["supplemental_evidence"] = {"pure_donor_developmental": {
            "label": "pure donor", "output_relative": "runs/pure-donor",
            "result_sha256": digest(pure_donor / "scores.json"), "unique_parameters": 123,
            "donor_manifest_sha256": arms[0]["donor_manifest_sha256"]}}
        # Pin two reusable control results to prove reuse cannot silently change.
        donor_dev = workspace / "runs/donor-dev"; donor_bench = workspace / "runs/donor-bench"
        self.write(donor_dev / "scores.json", self.developmental(arms[0]["checkpoint_manifest_sha256"]))
        self.write(donor_bench / "donor-scores.json", self.benchmark(arms[0]["checkpoint_manifest_sha256"]))
        arms[0]["reuse"] = {
            "developmental": {"output_relative": "runs/donor-dev", "result_sha256": digest(donor_dev / "scores.json")},
            "benchmark": {"output_relative": "runs/donor-bench", "result_sha256": digest(donor_bench / "donor-scores.json")},
        }
        return workspace, plan

    def test_plan_binds_actual_exposure_and_rejects_false_exact_7m_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace, plan = self.fixture(Path(directory))
            verified = validate_plan(plan, workspace)
            self.assertFalse(verified["exact_boundary_checkpoint_available"])
            self.assertEqual(verified["arms"]["alpha3.2.1"]["boundary_delta_input_tokens"], 900_000)
            bad = copy.deepcopy(plan); bad["arms"][2]["boundary_relation"] = "exact"; bad["arms"][2]["claimed_exact_boundary"] = True
            with self.assertRaisesRegex(ValueError, "mislabeled"):
                validate_plan(bad, workspace)
            bad = copy.deepcopy(plan); bad["exact_boundary_checkpoint_available"] = True
            with self.assertRaisesRegex(ValueError, "availability"):
                validate_plan(bad, workspace)

    def test_plan_rejects_changed_checkpoint_or_reused_result(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace, plan = self.fixture(Path(directory)); storage = Path(directory) / "storage"
            (storage / "alpha3.2.1/checkpoint/delta.safetensors").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "payload"):
                validate_plan(plan, workspace)
        with tempfile.TemporaryDirectory() as directory:
            workspace, plan = self.fixture(Path(directory))
            (workspace / "runs/donor-dev/scores.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Reused result"):
                validate_plan(plan, workspace)

    def test_plan_rejects_state_counter_mismatch_and_false_zero_control(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace, plan = self.fixture(Path(directory));storage=Path(directory)/"storage"
            arm=plan["arms"][2];cp=storage/arm["checkpoint_relative"]
            config_sha=json.loads((cp/"manifest.json").read_text())["config_sha256"]
            with zipfile.ZipFile(cp/"state.pt","w") as archive:
                archive.writestr("state/data.pkl",pickle.dumps({"updates":arm["updates"],
                    "input_tokens":arm["input_tokens"]+1,"target_tokens":arm["target_tokens"],
                    "parent_sha256":"parent","config_sha256":config_sha,
                    "config":{"model_label":"alpha3.2.1"}}))
            manifest=json.loads((cp/"manifest.json").read_text());manifest["files"]["state.pt"]=digest(cp/"state.pt")
            self.write(cp/"manifest.json",manifest);arm["checkpoint_manifest_sha256"]=digest(cp/"manifest.json")
            evidence=workspace/"unused"
            evidence=storage/arm["evidence_relative"]
            value=json.loads(evidence.read_text());value["manifest_sha256"]=arm["checkpoint_manifest_sha256"]
            self.write(evidence,value);arm["evidence_sha256"]=digest(evidence)
            with self.assertRaisesRegex(ValueError,"state input_tokens"):
                validate_plan(plan,workspace)
        with tempfile.TemporaryDirectory() as directory:
            workspace, plan = self.fixture(Path(directory));arm=plan["arms"][0]
            evidence=workspace/arm["evidence_relative"];value=json.loads(evidence.read_text())
            value["fresh_optimizer"]=False;self.write(evidence,value);arm["evidence_sha256"]=digest(evidence)
            with self.assertRaisesRegex(ValueError,"Zero-update control"):
                validate_plan(plan,workspace)

    def test_host_finalizer_guard_requires_verified_zero_exit_container(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.write(output / "container-state.json", {"Running": False, "ExitCode": 0})
            self.assertTrue(_container_succeeded(output))
            self.write(output / "container-state.json", {"Running": False, "ExitCode": 1})
            self.assertFalse(_container_succeeded(output))
            (output / "container-state.json").unlink()
            self.assertFalse(_container_succeeded(output))

    def test_full_report_keeps_all_three_arms_and_never_promotes(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace, plan = self.fixture(Path(directory)); verified = validate_plan(plan, workspace)
            executions = []
            for arm in ("donor", "alpha3.2.0", "alpha3.2.1"):
                for protocol in ("developmental", "benchmark"):
                    cp = verified["arms"][arm]["checkpoint_manifest_sha256"]
                    output = (workspace / ("runs/donor-dev" if protocol == "developmental" else "runs/donor-bench")) \
                        if arm == "donor" else workspace / ("results/%s/%s" % (arm, protocol))
                    if arm != "donor":
                        self.write(output / ("scores.json" if protocol == "developmental" else "donor-scores.json"),
                                   self.developmental(cp) if protocol == "developmental" else self.benchmark(cp))
                    executions.append({"arm": arm, "protocol": protocol, "output": str(output),
                                       "exit_code": 0, "completed_at": "done",
                                       "reused": arm == "donor",
                                       "result_sha256": digest(output / ("scores.json" if protocol == "developmental" else "donor-scores.json"))})
            execution = {"schema": "arcus3-three-model-execution-v1", "complete": True,
                         "plan_sha256": verified["plan_sha256"], "executions": executions}
            report = build_report(plan, verified, execution)
            self.assertEqual(tuple(report["arms"]), ("donor", "alpha3.2.0", "alpha3.2.1"))
            self.assertFalse(report["automatic_promotion"]); self.assertFalse(report["winner_selected"])
            self.assertFalse(report["matched_training_exposure"]); self.assertEqual(len(report["benchmark_task_coverage"]), 7)
            self.assertEqual(len(report["benchmark_task_coverage_sha256"]), 64)
            self.assertEqual(report["supplemental_pure_donor_developmental"]["unique_parameters"], 123)
            self.assertEqual(len(report["arms"]["alpha3.2.1"]["evaluation_provenance"]["benchmark"]["result_sha256"]), 64)
            rendered = render_markdown(report)
            self.assertIn("7,900,000-token", rendered); self.assertIn("No winner", rendered)

    def test_report_rejects_sampled_or_different_task_coverage_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace, plan = self.fixture(Path(directory)); verified = validate_plan(plan, workspace)
            executions = []
            for arm in ("donor", "alpha3.2.0", "alpha3.2.1"):
                for protocol in ("developmental", "benchmark"):
                    output = (workspace / ("runs/donor-dev" if protocol == "developmental" else "runs/donor-bench")) \
                        if arm == "donor" else workspace / ("results/%s/%s" % (arm, protocol))
                    cp = verified["arms"][arm]["checkpoint_manifest_sha256"]
                    value = self.developmental(cp) if protocol == "developmental" else self.benchmark(cp)
                    if arm != "donor":
                        if arm == "alpha3.2.1" and protocol == "benchmark": value["max_samples_per_task"] = 16
                        self.write(output / ("scores.json" if protocol == "developmental" else "donor-scores.json"), value)
                    executions.append({"arm": arm, "protocol": protocol, "output": str(output), "exit_code": 0,
                                       "completed_at": "done", "reused": arm == "donor",
                                       "result_sha256": digest(output / ("scores.json" if protocol == "developmental" else "donor-scores.json"))})
            execution = {"schema": "arcus3-three-model-execution-v1", "complete": True,
                         "plan_sha256": verified["plan_sha256"], "executions": executions}
            with self.assertRaisesRegex(ValueError, "sampled"):
                build_report(plan, verified, execution)
            execution["executions"][0], execution["executions"][1] = execution["executions"][1], execution["executions"][0]
            with self.assertRaisesRegex(ValueError, "sequential order"):
                build_report(plan, verified, execution)


if __name__ == "__main__":
    unittest.main()
