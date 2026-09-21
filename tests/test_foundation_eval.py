import json
from pathlib import Path

import pytest

from evaluation.control import (
    build_result_skeleton,
    load_json,
    validate_candidates,
    validate_control_files,
    validate_protocol,
)
from evaluation.suite_runner import build_suite_plan


ROOT = Path(__file__).resolve().parents[1]


def test_committed_phase1_controls_are_valid():
    assert validate_control_files(ROOT) == []


def test_all_committed_suite_catalogs_are_frozen():
    suites = load_json(ROOT / "evaluation" / "suites.json")
    assert len(build_suite_plan(suites, "smoke", 3, ["step-3.5-flash"])) == 84
    assert len(build_suite_plan(suites, "qualification", 3, ["step-3.5-flash"])) == 480


def test_candidate_registry_has_one_control_and_full_revisions():
    registry = load_json(ROOT / "evaluation" / "candidates.json")
    assert len(registry["candidates"]) == 7
    assert sum(item["role"] == "conversion_control" for item in registry["candidates"]) == 1
    assert all(len(item["revision"]) == 40 for item in registry["candidates"] if item["role"] != "api_comparator")
    assert all(item["eligible"] is False and item["revision"] == "api-unversioned" for item in registry["candidates"] if item["role"] == "api_comparator")


def test_candidate_validation_rejects_custom_license():
    registry = load_json(ROOT / "evaluation" / "candidates.json")
    copied = json.loads(json.dumps(registry))
    copied["candidates"][0]["license"] = "custom"
    assert any("license" in error for error in validate_candidates(copied))


def test_protocol_validation_rejects_provider_fallbacks():
    protocol = load_json(ROOT / "evaluation" / "protocol.json")
    copied = json.loads(json.dumps(protocol))
    copied["provider"]["allow_fallbacks"] = True
    assert any("allow_fallbacks" in error for error in validate_protocol(copied))


def test_protocol_pins_execution_and_generation_limits():
    generation = load_json(ROOT / "evaluation" / "protocol.json")["generation"]
    assert generation["wall_time_seconds"] == 3600
    assert generation["verifier_timeout_seconds"] == 3600
    assert generation["concurrency"] == 1
    assert generation["retry_on_model_failure"] == 0
    assert generation["max_output_tokens"] == 32768


def test_result_skeleton_carries_immutable_revisions():
    registry = load_json(ROOT / "evaluation" / "candidates.json")
    protocol = load_json(ROOT / "evaluation" / "protocol.json")
    record = build_result_skeleton(
        registry,
        protocol,
        candidate_id="step-3.5-flash",
        harness_id="function_calling",
        upstream_provider="example-provider",
        precision="fp8",
        parser="native-openai-tools",
        run_id="smoke-1",
    )
    assert record["model"]["revision"] == registry["candidates"][0]["revision"]
    assert record["harness"]["revision"] == "f7cf7359b7ac615a0b294831c5ba2bc95ee4a000"
    assert record["provider"]["upstream"] == "example-provider"
    assert record["outcome"]["failure_class"] == "unrun_template"


def test_result_skeleton_rejects_unknown_candidate():
    registry = load_json(ROOT / "evaluation" / "candidates.json")
    protocol = load_json(ROOT / "evaluation" / "protocol.json")
    with pytest.raises(ValueError, match="unknown candidate"):
        build_result_skeleton(
            registry,
            protocol,
            candidate_id="missing",
            harness_id="terminal",
            upstream_provider="provider",
            precision="fp8",
            parser="native",
            run_id="bad",
        )
