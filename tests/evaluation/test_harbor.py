import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

from evaluation.control import load_json
from evaluation.harbor import (
    audit_harbor_job_config,
    build_harbor_job_config,
    load_harbor_result_state,
    stop_harbor_containers,
)


ROOT = Path(__file__).resolve().parents[2]


class HarborContractTests(unittest.TestCase):
    def setUp(self):
        self.protocol = load_json(ROOT / "evaluation" / "protocol.json")
        self.harness = load_json(ROOT / "evaluation" / "harnesses" / "harbor.json")

    def config(self):
        return build_harbor_job_config(
            self.protocol,
            self.harness,
            candidate_id="step-3.5-flash",
            task_id="openssl-selfsigned-cert",
            attempt=1,
        )

    def test_config_is_single_attempt_and_fully_pinned(self):
        config = self.config()
        self.assertEqual(audit_harbor_job_config(config, self.protocol, self.harness), [])
        self.assertEqual(config["n_attempts"], 1)
        self.assertEqual(config["n_concurrent_trials"], 1)
        self.assertEqual(config["agents"][0]["override_timeout_sec"], 3600.0)
        self.assertEqual(config["verifier"]["override_timeout_sec"], 3600.0)
        self.assertEqual(config["tasks"][0]["git_commit_id"], self.harness["dataset_revision"])

    def test_timeout_mismatch_is_rejected(self):
        config = self.config()
        config["agents"][0]["override_timeout_sec"] = 1200
        self.assertTrue(
            any("agent timeout" in error for error in audit_harbor_job_config(
                config, self.protocol, self.harness
            ))
        )

    def test_output_metadata_is_explicit_and_audited(self):
        config = self.config()
        self.assertEqual(config["agents"][0]["kwargs"]["model_info"],
                         {"max_output_tokens": self.protocol["generation"]["max_output_tokens"]})
        config["agents"][0]["kwargs"].pop("model_info")
        self.assertTrue(any("model_info" in error for error in
                            audit_harbor_job_config(config, self.protocol, self.harness)))

    def test_all_frozen_qualification_tasks_are_audited(self):
        tasks = load_json(ROOT / "evaluation" / "suites.json")["suites"]["qualification"]["terminal"]["task_ids"]
        self.assertEqual(set(tasks), set(self.harness["qualification_task_ids"]))
        for task in tasks:
            config = build_harbor_job_config(self.protocol, self.harness, candidate_id="step-3.5-flash", task_id=task, attempt=1)
            self.assertEqual(audit_harbor_job_config(config, self.protocol, self.harness), [])
        with self.assertRaises(ValueError):
            build_harbor_job_config(self.protocol, self.harness, candidate_id="step-3.5-flash", task_id="not-frozen", attempt=1)

    def test_partial_job_is_not_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            (job / "config.json").write_text(json.dumps(self.config()), encoding="utf-8")
            (job / "trial-one").mkdir()
            state = load_harbor_result_state(job)
            self.assertFalse(state["complete"])
            self.assertEqual(state["completed_trials"], 0)

    def test_harbor_omitted_defaults_still_audit(self):
        config = self.config()
        config.pop("n_attempts")
        config.pop("retry")
        self.assertEqual(audit_harbor_job_config(config, self.protocol, self.harness), [])

    def test_cleanup_targets_only_exact_trial_compose_label(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            (job / "openssl-selfsigned-cert__ABC123").mkdir()
            (job / "unrelated").mkdir()
            with patch("evaluation.harbor.subprocess.run") as run:
                run.side_effect = [Mock(stdout="abcdef123456\n"), Mock()]
                self.assertEqual(stop_harbor_containers(job), ["abcdef123456"])
                self.assertEqual(
                    run.call_args_list[0].args[0],
                    ["docker", "ps", "-q", "--filter", "label=com.docker.compose.project=openssl-selfsigned-cert__abc123__env"],
                )
                self.assertEqual(run.call_args_list[1].args[0], ["docker", "stop", "abcdef123456"])


if __name__ == "__main__":
    unittest.main()
