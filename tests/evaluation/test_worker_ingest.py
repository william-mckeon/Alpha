import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from evaluation.ingest import ingest_worker_verdict, normalized_outcome
from evaluation.control import validate_result, validate_artifact_hashes
from evaluation.scoring import score_candidates


ROOT = Path(__file__).resolve().parents[2]


class WorkerIngestTests(unittest.TestCase):
    def test_only_proven_recoverable_fetch_errors_can_be_scored(self):
        from evaluation.web_fetch import FetchFailure
        for code, status, accepted in (("http_status", 403, True), ("http_status", 503, False), ("timeout", None, False)):
            with self.subTest(code=code, status=status), tempfile.TemporaryDirectory(dir=ROOT / "evaluation/runs") as temporary:
                directory = Path(temporary)
                kwargs = self.fixture(directory)
                manifest = json.loads(kwargs["manifest_path"].read_text())
                response = kwargs["verifier_paths"][1]
                response.write_text('{"id":"web_search_base_37","result":[]}\n')
                manifest.update(task_id="web_search_37", official_task_id="web_search_base_37")
                search = directory / "search"
                search.mkdir()
                event = search / "fetch-exchanges.jsonl"
                detail = FetchFailure(code, status).details
                event.write_text(json.dumps({"trial_id": "fixture-worker", "status": "tool_error", "error": detail, "response": {"error": detail}}) + "\n")
                for path in (response, event):
                    manifest["artifact_sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
                kwargs["manifest_path"].write_text(json.dumps(manifest))
                if accepted:
                    record = ingest_worker_verdict(**kwargs)
                    self.assertEqual(record["outcome"]["status"], "passed")
                    self.assertEqual(validate_artifact_hashes(record, ROOT), [])
                else:
                    with self.assertRaisesRegex(ValueError, "fetch evidence"):
                        ingest_worker_verdict(**kwargs)

    def test_current_grading_policy_is_required_and_hash_checked(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "evaluation/runs") as temporary:
            directory = Path(temporary)
            kwargs = self.fixture(directory)
            policy = directory / "bfcl-policy-evaluate.json"
            policy.write_text('{"underscore_to_dot":false}')
            with self.assertRaisesRegex(ValueError, "hash"):
                ingest_worker_verdict(**kwargs)
            manifest = json.loads(kwargs["manifest_path"].read_text())
            manifest["artifact_sha256"].pop(str(policy.relative_to(ROOT)))
            kwargs["manifest_path"].write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "hashed provenance"):
                ingest_worker_verdict(**kwargs)
    def test_budget_failure_requires_gateway_ceiling_and_retained_stop(self):
        for count in (100, 99):
            with self.subTest(count=count), tempfile.TemporaryDirectory(dir=ROOT / "evaluation/runs") as temporary:
                directory = Path(temporary)
                kwargs = self.fixture(directory)
                manifest = json.loads(kwargs["manifest_path"].read_text())
                gateway = kwargs["proxy_log"]
                gateway.write_text(gateway.read_text() * count)
                report, stop, errors = directory / "report.json", directory / "budget-stop.json", directory / "gateway-errors.jsonl"
                report.write_text('{"resolved_ids":["django__django-14170"],"unresolved_ids":[],"error_ids":[],"incomplete_ids":[],"empty_patch_ids":[]}')
                stop.write_text('{"trial_id":"fixture-worker","reason":"model_iteration_budget_exhausted","max_agent_iterations":100}')
                errors.write_text('{"trial_id":"fixture-worker","category":"model_budget","error":"trial exceeded global agent-iteration budget"}\n')
                manifest.update(harness_id="repository", task_id="django__django-14170", harness_revision=next(h["revision"] for h in kwargs["protocol"]["harnesses"] if h["id"] == "repository"))
                manifest["artifact_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (gateway, report, stop, errors, gateway.with_name("budget-scope.json"))}
                kwargs["manifest_path"].write_text(json.dumps(manifest))
                kwargs["verifier_paths"] = [report]
                if count == 100:
                    record = ingest_worker_verdict(**kwargs)
                    self.assertEqual(record["outcome"]["status"], "failed")
                    self.assertEqual(record["outcome"]["score"], 0)
                    self.assertEqual(record["run"]["termination_reason"], "model_iteration_budget_exhausted")
                    self.assertEqual(validate_result(record) + validate_artifact_hashes(record, ROOT), [])
                    from evaluation.suite_runner import execute_suite_plan
                    import copy
                    unit = {"candidate_id": record["model"]["candidate_id"], "harness_id": "repository",
                            "task_id": "django__django-14170", "attempt": 1}
                    plan = [unit, {**unit, "attempt": 2}]
                    calls = []
                    def runner(unit, index):
                        calls.append(index)
                        result = copy.deepcopy(record)
                        result["run"]["id"] = f"budget-continuation-{index}"
                        result["outcome"]["attempt"] = unit["attempt"]
                        return result
                    result = execute_suite_plan(plan, {"repository": runner}, directory / "suite",
                        contract_revision=kwargs["protocol"]["contract_revision"], evidence_kind="diagnostic")
                    self.assertEqual(calls, [1, 2])
                    self.assertEqual(len(result), 2)
                else:
                    with self.assertRaisesRegex(ValueError, "gateway ceiling"):
                        ingest_worker_verdict(**kwargs)
    def test_required_worker_loaded_contract_cannot_be_inferred_from_host(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "evaluation/runs") as temporary:
            kwargs = self.fixture(Path(temporary))
            manifest = json.loads(kwargs["manifest_path"].read_text())
            manifest["worker_control_contract_required"] = True
            kwargs["manifest_path"].write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "retained provenance"):
                ingest_worker_verdict(**kwargs)
    def fixture(self, directory):
        protocol = json.loads((ROOT / "evaluation/protocol.json").read_text())
        registry = json.loads((ROOT / "evaluation/candidates.json").read_text())
        providers = json.loads((ROOT / "evaluation/providers.json").read_text())
        provider = providers["providers"][0]
        gateway, score, response, manifest = (directory / name for name in ("proxy-exchanges.jsonl", "score.jsonl", "response.jsonl", "validation.json"))
        exchange = {"trial_id": "fixture-worker", "tool_calls_total": 0, "request": {"model": provider["openrouter_model"], "temperature": 0.0, "top_p": 1.0, "max_tokens": 32768, "stream": False, "provider": {"only": [provider["provider_slug"]], "allow_fallbacks": False, "require_parameters": True}}, "response": {"model": provider["openrouter_model"], "provider": provider["upstream"], "choices": [{"message": {"content": "done"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "cost": 0.01}}}
        gateway.write_text(json.dumps(exchange) + "\n")
        score.write_text('{"accuracy":1,"correct_count":1,"total_count":1}\n')
        response.write_text('{"id":"simple_python_272","result":[]}\n')
        revision = next(h["revision"] for h in protocol["harnesses"] if h["id"] == "function_calling")
        policy = directory / "bfcl-policy-evaluate.json"
        policy.write_text(json.dumps({"contract_revision": protocol["contract_revision"], "underscore_to_dot": True, "answer_override": protocol["search"]["answer_override"]}))
        data = {"run_id": "fixture-worker", "candidate_id": provider["candidate_id"], "task_id": "simple_python_272", "harness_id": "function_calling", "harness_revision": revision, "protocol_snapshot": protocol, "status": "integration_passed", "benchmark_scored": False, "started_at": "2026-09-13T00:00:00+00:00", "finished_at": "2026-09-13T00:00:01+00:00", "artifact_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (gateway, score, response)}}
        manifest.write_text(json.dumps(data))
        data["artifact_sha256"][str(policy.relative_to(ROOT))] = hashlib.sha256(policy.read_bytes()).hexdigest()
        manifest.write_text(json.dumps(data))
        from evaluation.budget_scope import load_scope, retain_scope
        budget_path = retain_scope(load_scope(ROOT), directory, "fixture-worker")
        data["artifact_sha256"][str(budget_path.relative_to(ROOT))] = hashlib.sha256(budget_path.read_bytes()).hexdigest()
        manifest.write_text(json.dumps(data))
        return {"root": ROOT, "manifest_path": manifest, "proxy_log": gateway, "verifier_paths": [score, response], "registry": registry, "protocol": protocol, "providers": providers}

    def test_official_pass_normalizes_but_diagnostic_cannot_rank(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "evaluation/runs") as temporary:
            kwargs = self.fixture(Path(temporary))
            record = ingest_worker_verdict(**kwargs)
            self.assertEqual(validate_result(record) + validate_artifact_hashes(record, ROOT), [])
            self.assertEqual(record["outcome"]["status"], "passed")
            self.assertEqual(record["outcome"]["tokens"], 2)
            row = score_candidates([record], {"function_calling": 1}, expected={("function_calling", "simple_python_272", 1)})[0]
            self.assertFalse(row["qualification_complete"])
            self.assertIsNone(row["weighted_score"])

    def test_missing_contract_and_tampered_artifact_block(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "evaluation/runs") as temporary:
            kwargs = self.fixture(Path(temporary))
            kwargs["verifier_paths"][0].write_text("tampered")
            with self.assertRaisesRegex(ValueError, "hash"):
                ingest_worker_verdict(**kwargs)

    def test_approved_search_mapping_preserves_diagnostic_status(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "evaluation/runs") as temporary:
            kwargs = self.fixture(Path(temporary))
            manifest = json.loads(kwargs["manifest_path"].read_text())
            response = kwargs["verifier_paths"][1]
            response.write_text('{"id":"web_search_base_37","result":[]}\n')
            manifest.update(task_id="web_search_37", official_task_id="web_search_base_37")
            manifest["artifact_sha256"][str(response.relative_to(ROOT))] = hashlib.sha256(response.read_bytes()).hexdigest()
            kwargs["manifest_path"].write_text(json.dumps(manifest))
            record = ingest_worker_verdict(**kwargs)
            self.assertEqual(record["harness"]["task_subset"], ["web_search_37"])
            self.assertEqual(record["run"]["evidence_kind"], "diagnostic")
            manifest["official_task_id"] = "web_search_no_snippet_37"
            kwargs["manifest_path"].write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "frozen variant"):
                ingest_worker_verdict(**kwargs)

    def test_search_credits_and_supplemental_artifacts_are_retained(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "evaluation/runs") as temporary:
            directory = Path(temporary)
            kwargs = self.fixture(directory)
            manifest = json.loads(kwargs["manifest_path"].read_text())
            response = kwargs["verifier_paths"][1]
            response.write_text('{"id":"web_search_base_37","result":[]}\n')
            search = directory / "search"
            search.mkdir()
            discovery, events = search / "mcp-discovery.json", search / "search-exchanges.jsonl"
            discovery.write_text('{"contract_id":"arcus-tavily-mcp-v1","selected_tool":"tavily_search"}')
            events.write_text('{"trial_id":"fixture-worker","contract_id":"arcus-tavily-mcp-v1","status":"completed","tool":"tavily_search","credits_upper_bound":2}\n')
            manifest.update(task_id="web_search_37", official_task_id="web_search_base_37")
            for path in (response, discovery, events):
                manifest["artifact_sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            kwargs["manifest_path"].write_text(json.dumps(manifest))
            record = ingest_worker_verdict(**kwargs)
            self.assertTrue(record["search"]["used"])
            self.assertEqual(record["search"]["tool_calls"], 1)
            self.assertEqual(record["search"]["credits_upper_bound"], 2)
            self.assertEqual(validate_artifact_hashes(record, ROOT), [])
            events.write_text("tampered")
            self.assertTrue(validate_artifact_hashes(record, ROOT))

    def test_non_search_task_cannot_substitute_another_official_id(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "evaluation/runs") as temporary:
            kwargs = self.fixture(Path(temporary))
            manifest = json.loads(kwargs["manifest_path"].read_text())
            manifest["official_task_id"] = "simple_python_273"
            kwargs["manifest_path"].write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "frozen variant"):
                ingest_worker_verdict(**kwargs)

    def test_nonfinite_and_boolean_accounting_rejected(self):
        for amount in (float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                normalized_outcome(passed=False, score=0, tokens=0, tool_calls=0, latency_seconds=0, cost_usd=amount)

    def test_unhashed_gateway_error_evidence_cannot_be_imported(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "evaluation/runs") as temporary:
            kwargs = self.fixture(Path(temporary))
            kwargs["proxy_log"].with_name("gateway-errors.jsonl").write_text('{"trial_id":"fixture-worker","category":"provider"}\n')
            with self.assertRaisesRegex(ValueError, "manifest hash"):
                ingest_worker_verdict(**kwargs)

    def test_legacy_mcp_deadline_cannot_claim_verified_contract(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "evaluation/runs") as temporary:
            directory = Path(temporary)
            kwargs = self.fixture(directory)
            manifest = json.loads(kwargs["manifest_path"].read_text())
            meta, provenance = directory / "meta.json", directory / "worker-provenance.json"
            meta.write_text("{}")
            provenance.write_text('{"verifier_overlay":{"timeout_seconds":300}}')
            manifest.update(harness_id="mcp", task_id="filesystem/file_property/size_classification", harness_revision=next(h["revision"] for h in kwargs["protocol"]["harnesses"] if h["id"] == "mcp"))
            for path in (meta, provenance):
                manifest["artifact_sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            kwargs["manifest_path"].write_text(json.dumps(manifest))
            kwargs["verifier_paths"] = [meta]
            with self.assertRaisesRegex(ValueError, "verifier deadline"):
                ingest_worker_verdict(**kwargs)
