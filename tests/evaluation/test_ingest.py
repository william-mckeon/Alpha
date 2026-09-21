import unittest
import json
import tempfile
from pathlib import Path

from evaluation.ingest import classify_failure, normalized_outcome, ingest_harbor_job
from evaluation.control import load_json, validate_result
from evaluation.harbor import build_harbor_job_config

ROOT = Path(__file__).resolve().parents[2]


class FoundationIngestTests(unittest.TestCase):
    def test_failure_precedence_and_model_default(self):
        self.assertEqual(classify_failure(provider_error=True, parser_error=True, infrastructure_error=True), "infrastructure_error")
        self.assertEqual(classify_failure(provider_error=False, parser_error=False, infrastructure_error=False), "failed")
        self.assertEqual(
            classify_failure(
                provider_error=False,
                parser_error=False,
                infrastructure_error=False,
                protocol_error=True,
            ),
            "invalid_run",
        )
        self.assertEqual(
            classify_failure(
                provider_error=False,
                parser_error=False,
                infrastructure_error=False,
                interrupted=True,
            ),
            "interrupted",
        )

    def test_normalized_outcome_rejects_invalid_score(self):
        with self.assertRaises(ValueError):
            normalized_outcome(passed=False, score=2, tokens=0, tool_calls=0, latency_seconds=0, cost_usd=0)

    def test_valid_harbor_verifier_failure_remains_model_failure(self):
        protocol = load_json(ROOT / "evaluation" / "protocol.json")
        harness = load_json(ROOT / "evaluation" / "harnesses" / "harbor.json")
        registry = load_json(ROOT / "evaluation" / "candidates.json")
        providers = load_json(ROOT / "evaluation" / "providers.json")
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            config = build_harbor_job_config(
                protocol, harness, candidate_id="step-3.5-flash",
                task_id="openssl-selfsigned-cert", attempt=1, run_id="fixture-job"
            )
            (job / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (job / "result.json").write_text(
                json.dumps({"finished_at": "2026-09-12T01:00:01+00:00"}), encoding="utf-8"
            )
            trial = job / "trial"
            trial.mkdir()
            result = {
                "task_name": "openssl-selfsigned-cert",
                "started_at": "2026-09-12T01:00:00+00:00",
                "finished_at": "2026-09-12T01:00:01+00:00",
                "config": {"task": {"git_commit_id": harness["dataset_revision"]}},
                "verifier_result": {"rewards": {"reward": 0.0}},
                "agent_result": {"n_input_tokens": 10, "n_output_tokens": 5},
            }
            (trial / "result.json").write_text(json.dumps(result), encoding="utf-8")
            log = job / "proxy.jsonl"
            exchange = {
                "trial_id": "fixture-job",
                "request": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 32768,
                            "provider": {"only": ["siliconflow"], "allow_fallbacks": False}},
                "response": {"provider": "SiliconFlow", "usage": {"cost": 0.001}, "choices": []},
            }
            model = next(row["openrouter_model"] for row in providers["providers"] if row["candidate_id"] == "step-3.5-flash")
            exchange["request"].update(model=model, stream=False)
            exchange["request"]["provider"]["require_parameters"] = True
            exchange["response"].update(model=model, choices=[{"message": {"content": "done"}}])
            exchange["response"]["usage"].update(prompt_tokens=10, completion_tokens=5, total_tokens=15)
            exchange["tool_calls_total"] = 0
            log.write_text(json.dumps(exchange) + "\n", encoding="utf-8")
            record = ingest_harbor_job(
                job, log, candidate_id="step-3.5-flash", registry=registry,
                protocol=protocol, providers=providers, harness=harness
            )
            self.assertEqual(record["outcome"]["status"], "failed")
            self.assertEqual(validate_result(record), [])
            exchange["trial_id"] = "wrong-job"
            log.write_text(json.dumps(exchange) + "\n", encoding="utf-8")
            record = ingest_harbor_job(
                job, log, candidate_id="step-3.5-flash", registry=registry,
                protocol=protocol, providers=providers, harness=harness
            )
            self.assertEqual(record["outcome"]["status"], "invalid_run")


if __name__ == "__main__":
    unittest.main()
