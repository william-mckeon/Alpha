"""Validate and render the read-only donor/3.2.0/3.2.1 comparison.

The comparison is deliberately descriptive.  It binds every adapted arm to a
specific checkpoint, evidence receipt and measured exposure, and it never
selects or promotes a winner.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import math
import pickle
import re
import zipfile
from pathlib import Path


HEX64 = re.compile(r"^[0-9a-f]{64}$")
ARM_IDS = ("donor", "alpha3.2.0", "alpha3.2.1")
PROTOCOL_IDS = ("developmental", "benchmark")


def read(path: Path | str):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path: Path | str):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _safe_relative(root: Path, relative: str):
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("Path escapes configured storage root")
    return candidate


def _validate_hash(value, label):
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise ValueError(label + " must be a lowercase SHA256")


def _relation(actual, boundary):
    return "exact" if actual == boundary else "before" if actual < boundary else "after"


def _tensor_stub(*_args):
    return {"tensor_payload_omitted": True}


class _StateUnpickler(pickle.Unpickler):
    """Read trusted, hash-pinned checkpoint metadata without importing Torch."""
    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _tensor_stub
        if module == "torch" and name in {"ByteStorage", "FloatStorage"}:
            return type(name, (), {})
        raise pickle.UnpicklingError("Unsupported checkpoint metadata global: %s.%s" % (module, name))

    def persistent_load(self, _identifier):
        return {"storage_payload_omitted": True}


def _checkpoint_state(path):
    with zipfile.ZipFile(path) as archive:
        members=[name for name in archive.namelist() if name.endswith("/data.pkl")]
        if len(members)!=1:raise ValueError("Unsupported checkpoint state archive")
        state=_StateUnpickler(io.BytesIO(archive.read(members[0]))).load()
    if not isinstance(state,dict):raise ValueError("Checkpoint state metadata is not a mapping")
    return state


def _checkpoint_identity(arm, storage_root, workspace):
    checkpoint = _safe_relative(storage_root, arm["checkpoint_relative"])
    evidence_root = workspace if arm.get("evidence_scope") == "workspace" else storage_root
    evidence_path = _safe_relative(evidence_root, arm["evidence_relative"])
    manifest_path = checkpoint / "manifest.json"
    for path in (manifest_path, evidence_path):
        if not path.is_file():
            raise ValueError("Missing comparison evidence: " + str(path))
    _validate_hash(arm["checkpoint_manifest_sha256"], "checkpoint manifest")
    _validate_hash(arm["evidence_sha256"], "checkpoint evidence")
    if digest(manifest_path) != arm["checkpoint_manifest_sha256"]:
        raise ValueError("Checkpoint manifest does not match comparison plan")
    if digest(evidence_path) != arm["evidence_sha256"]:
        raise ValueError("Checkpoint evidence receipt changed")
    manifest = read(manifest_path)
    if manifest.get("schema") != "arcus3-expanded-delta-v1":
        raise ValueError("Unsupported checkpoint schema")
    if manifest.get("parent_sha256") != arm["parent_sha256"]:
        raise ValueError("Checkpoint parent mismatch")
    if manifest.get("updates") != arm["updates"]:
        raise ValueError("Checkpoint update count mismatch")
    if set(manifest.get("files", {})) != {"delta.safetensors", "state.pt"}:
        raise ValueError("Incomplete comparison checkpoint")
    for name, expected in manifest["files"].items():
        _validate_hash(expected, "checkpoint payload")
        if Path(name).name != name or digest(checkpoint / name) != expected:
            raise ValueError("Checkpoint payload hash mismatch: " + name)
    state=_checkpoint_state(checkpoint/"state.pt")
    for field in ("updates","input_tokens","target_tokens"):
        if state.get(field)!=arm[field]:
            raise ValueError("Checkpoint state %s does not match comparison plan" % field)
    if state.get("parent_sha256")!=manifest["parent_sha256"] or state.get("config_sha256")!=manifest.get("config_sha256"):
        raise ValueError("Checkpoint state lineage does not match its manifest")
    state_config=state.get("config",{})
    config_sha=hashlib.sha256(json.dumps(state_config,sort_keys=True).encode()).hexdigest()
    if config_sha!=manifest.get("config_sha256"):
        raise ValueError("Checkpoint state configuration hash mismatch")
    expected_label=arm.get("state_model_label")
    actual_label=state_config.get("model_label")
    if expected_label is not None and actual_label!=expected_label:
        if not (arm.get("allow_legacy_missing_model_label") is True and actual_label is None):
            raise ValueError("Checkpoint state model label mismatch")
    evidence = read(evidence_path)
    for field in ("updates", "input_tokens", "target_tokens"):
        if evidence.get(field) != arm[field]:
            raise ValueError("Evidence %s does not match comparison plan" % field)
    if evidence.get("manifest_sha256") != arm["checkpoint_manifest_sha256"]:
        raise ValueError("Evidence points at a different checkpoint manifest")
    evidence_checkpoint = evidence.get("checkpoint")
    if evidence_checkpoint and Path(evidence_checkpoint).name != checkpoint.name:
        raise ValueError("Evidence points at a different checkpoint generation")
    if "committed" in evidence and not evidence.get("committed"):
        raise ValueError("Checkpoint evidence is not committed")
    if "retention_complete" in evidence and not evidence.get("retention_complete"):
        raise ValueError("Checkpoint retention is incomplete")
    if arm.get("control_kind")=="donor-derived-zero-update-initialization":
        required={"verified":True,"fresh_optimizer":True,"frozen_unchanged":True,
                  "model_label":expected_label}
        if any(evidence.get(key)!=value for key,value in required.items()):
            raise ValueError("Zero-update control verification is incomplete")
    return {
        "label": arm["label"],
        "checkpoint": str(checkpoint),
        "checkpoint_generation": checkpoint.name,
        "checkpoint_manifest_sha256": arm["checkpoint_manifest_sha256"],
        "checkpoint_payload_sha256": dict(manifest["files"]),
        "evidence": str(evidence_path),
        "evidence_sha256": arm["evidence_sha256"],
        "parent_sha256": manifest["parent_sha256"],
        "updates": arm["updates"],
        "input_tokens": arm["input_tokens"],
        "target_tokens": arm["target_tokens"],
    }


def validate_plan(plan, workspace):
    workspace = Path(workspace).resolve()
    if plan.get("schema") != "arcus3-three-model-comparison-v1":
        raise ValueError("Unsupported comparison plan")
    if plan.get("execution") != "strictly-sequential" or plan.get("automatic_promotion") is not False:
        raise ValueError("Comparison must be sequential and non-promotional")
    boundary = plan.get("requested_boundary_input_tokens")
    if not isinstance(boundary, int) or boundary <= 0:
        raise ValueError("Positive requested boundary required")
    storage_config = (workspace / plan["storage_config"]).resolve()
    storage = read(storage_config)
    if not storage.get("ready"):
        raise ValueError("Configured comparison storage is not ready")
    storage_root = Path(storage["external_root"]).resolve()
    evaluation_runtime = _safe_relative(workspace, plan["evaluation_runtime"])
    _validate_hash(plan.get("evaluation_runtime_sha256"), "evaluation runtime")
    if digest(evaluation_runtime) != plan["evaluation_runtime_sha256"]:
        raise ValueError("Evaluation runtime changed")
    converted = (workspace / plan["converted_relative"]).resolve()
    _validate_hash(plan.get("converted_manifest_sha256"), "converted parent manifest")
    if digest(converted / "manifest.json") != plan["converted_manifest_sha256"]:
        raise ValueError("Converted parent manifest mismatch")
    protocols = plan.get("protocols", {})
    if tuple(protocols) != PROTOCOL_IDS:
        raise ValueError("Both frozen full protocols are required in their declared order")
    if protocols["developmental"].get("mode") != "baseline" or protocols["developmental"].get("tier") != "full":
        raise ValueError("Developmental protocol must be full baseline evaluation")
    if protocols["benchmark"].get("mode") != "donor-baseline" or protocols["benchmark"].get("tier") != "full":
        raise ValueError("Benchmark protocol must be full donor protocol")
    for key in ("suite_sha256", "settings_sha256"):
        _validate_hash(protocols["developmental"].get(key), "developmental " + key)
    _validate_hash(protocols["benchmark"].get("benchmark_manifest_sha256"), "benchmark manifest")
    arms = plan.get("arms", [])
    if tuple(arm.get("id") for arm in arms) != ARM_IDS:
        raise ValueError("Comparison requires donor, alpha3.2.0 and alpha3.2.1 in order")
    validated = {}
    exact_adapted = []
    for arm in arms:
        relation = arm.get("boundary_relation")
        if arm["id"] == "donor":
            donor_manifest = workspace / arm["donor_manifest_relative"]
            if digest(donor_manifest) != arm["donor_manifest_sha256"]:
                raise ValueError("Donor manifest mismatch")
            if read(donor_manifest).get("revision")!=arm.get("donor_revision"):
                raise ValueError("Donor revision does not match its manifest")
            if relation != "not-applicable" or any(arm.get(k) != 0 for k in ("updates", "input_tokens", "target_tokens")):
                raise ValueError("Donor exposure must be zero and boundary relation not-applicable")
            if arm.get("control_kind") != "donor-derived-zero-update-initialization":
                raise ValueError("The reused control must be explicitly labeled as zero-update initialization")
            validated[arm["id"]] = _checkpoint_identity(arm, storage_root, workspace)
            validated[arm["id"]].update(
                donor_manifest_sha256=arm["donor_manifest_sha256"], donor_revision=arm["donor_revision"],
                control_kind=arm["control_kind"], boundary_relation="not-applicable")
            continue
        actual_relation = _relation(arm["input_tokens"], boundary)
        if relation != actual_relation:
            raise ValueError(
                "%s is mislabeled as %s at the %s-token boundary; actual relation is %s"
                % (arm["id"], relation, boundary, actual_relation)
            )
        if arm.get("claimed_exact_boundary") is not (actual_relation == "exact"):
            raise ValueError("Exact-boundary claim disagrees with measured checkpoint exposure")
        if actual_relation == "exact":
            exact_adapted.append(arm["id"])
        validated[arm["id"]] = _checkpoint_identity(arm, storage_root, workspace)
        validated[arm["id"]]["boundary_relation"] = actual_relation
        validated[arm["id"]]["boundary_delta_input_tokens"] = arm["input_tokens"] - boundary
    available = bool(exact_adapted)
    if plan.get("exact_boundary_checkpoint_available") is not available:
        raise ValueError("Exact-boundary availability flag does not match checkpoint evidence")
    reuse = {}
    for arm in arms:
        for protocol_id, receipt in arm.get("reuse", {}).items():
            if protocol_id not in PROTOCOL_IDS:
                raise ValueError("Unknown reused protocol")
            output = _safe_relative(workspace, receipt["output_relative"])
            result_file = output / ("scores.json" if protocol_id == "developmental" else "donor-scores.json")
            _validate_hash(receipt["result_sha256"], "reused result")
            if digest(result_file) != receipt["result_sha256"]:
                raise ValueError("Reused result changed: " + str(result_file))
            reuse[(arm["id"], protocol_id)] = {"output": str(output), "result_sha256": receipt["result_sha256"]}
    supplemental = plan.get("supplemental_evidence", {}).get("pure_donor_developmental")
    if not supplemental:
        raise ValueError("Pinned pure-donor developmental evidence is required")
    if supplemental.get("donor_manifest_sha256") != plan["arms"][0]["donor_manifest_sha256"]:
        raise ValueError("Supplemental donor evidence references a different donor")
    supplemental_output = _safe_relative(workspace, supplemental["output_relative"])
    supplemental_file = supplemental_output / "scores.json"
    _validate_hash(supplemental.get("result_sha256"), "pure donor developmental result")
    if digest(supplemental_file) != supplemental["result_sha256"]:
        raise ValueError("Pure donor developmental result changed")
    return {
        "plan_sha256": canonical_digest(plan),
        "workspace": str(workspace),
        "storage_root": str(storage_root),
        "converted": str(converted),
        "evaluation_runtime": str(evaluation_runtime),
        "evaluation_runtime_sha256": plan["evaluation_runtime_sha256"],
        "requested_boundary_input_tokens": boundary,
        "exact_boundary_checkpoint_available": available,
        "exact_boundary_arms": exact_adapted,
        "protocols": protocols,
        "arms": validated,
        "reuse": reuse,
        "supplemental": {"pure_donor_developmental": {
            "label": supplemental["label"], "output": str(supplemental_output),
            "result_sha256": supplemental["result_sha256"],
            "unique_parameters": supplemental["unique_parameters"],
            "donor_manifest_sha256": supplemental["donor_manifest_sha256"],
        }},
        "automatic_promotion": False,
    }


def _developmental(scores, arm, protocol):
    if not scores.get("execution_complete") or scores.get("tier") != "full":
        raise ValueError("Developmental evaluation is incomplete or not full")
    for key in ("suite_sha256", "settings_sha256", "tokenizer_revision", "orchestration"):
        if scores.get(key) != protocol[key]:
            raise ValueError("Developmental identity mismatch: " + key)
    if scores.get("expanded_manifest_sha256") != arm["checkpoint_manifest_sha256"]:
        raise ValueError("Developmental result checkpoint mismatch")
    categories = scores.get("categories", {})
    expected_categories = tuple(protocol["categories"])
    if tuple(categories) != expected_categories or any(categories[name].get("count") != 6 for name in expected_categories):
        raise ValueError("Developmental task coverage mismatch")
    language = scores.get("language", {})
    if language.get("target_tokens") != protocol["language_target_tokens"]:
        raise ValueError("Developmental language coverage mismatch")
    return {
        "identity": {key: scores[key] for key in ("suite_sha256", "settings_sha256", "tokenizer_revision", "precision", "orchestration")},
        "language": language,
        "categories": categories,
        "tool_metrics": scores.get("tool_metrics", {}),
        "resources": scores.get("resources", {}),
        "prompt_count": sum(v["count"] for v in categories.values()),
        "human_review_pending": sum(v.get("measured", 0) == 0 for v in categories.values()) > 0,
    }


def _pure_donor_developmental(scores, evidence, protocol):
    if scores.get("schema") != "arcus3-baseline-v1" or not scores.get("execution_complete"):
        raise ValueError("Pure donor developmental evidence is incomplete")
    for key in ("suite_sha256", "settings_sha256", "tokenizer_revision", "orchestration"):
        if scores.get(key) != protocol[key]:
            raise ValueError("Pure donor developmental identity mismatch: " + key)
    if any(scores.get(key) is not None for key in
           ("adapter_manifest_sha256", "expanded_manifest_sha256", "conversion_manifest_sha256")):
        raise ValueError("Supplemental developmental evidence is not a pure donor run")
    if scores.get("unique_parameters") != evidence["unique_parameters"]:
        raise ValueError("Pure donor parameter count mismatch")
    categories = scores.get("categories", {})
    expected_categories = tuple(protocol["categories"])
    if tuple(categories) != expected_categories or any(categories[name].get("count") != 6 for name in expected_categories):
        raise ValueError("Pure donor developmental task coverage mismatch")
    language = scores.get("language", {})
    if language.get("target_tokens") != protocol["language_target_tokens"]:
        raise ValueError("Pure donor language coverage mismatch")
    return {
        "label": evidence["label"],
        "identity": {key: scores.get(key) for key in
                     ("schema", "tier", "suite_sha256", "settings_sha256", "tokenizer_revision", "precision", "orchestration")},
        "unique_parameters": scores["unique_parameters"],
        "language": language,
        "categories": categories,
        "tool_metrics": scores.get("tool_metrics", {}),
        "resources": scores.get("resources", {}),
        "prompt_count": sum(value["count"] for value in categories.values()),
        "human_review_pending": any(value.get("measured", 0) == 0 for value in categories.values()),
        "compatibility_note": "Complete historical run with exact suite/settings/tokenizer/orchestration identity; predates the explicit tier field.",
    }


def _benchmark(scores, arm, protocol):
    if not scores.get("complete") or scores.get("tier") != "full" or scores.get("max_samples_per_task") is not None:
        raise ValueError("Donor benchmark is incomplete or sampled")
    for key in ("model_context", "benchmark_context", "lighteval_revision", "benchmark_manifest_sha256"):
        if scores.get(key) != protocol[key]:
            raise ValueError("Benchmark identity mismatch: " + key)
    if scores.get("checkpoint_sha256") != arm["checkpoint_manifest_sha256"]:
        raise ValueError("Benchmark result checkpoint mismatch")
    task_results = scores.get("results", {}).get("results", {})
    if not task_results or not all(isinstance(value, dict) for value in task_results.values()):
        raise ValueError("Benchmark contains no task results")
    task_names = sorted(task_results)
    for prefix in protocol["required_task_prefixes"]:
        if not any(name.startswith(prefix) for name in task_names):
            raise ValueError("Benchmark task family missing: " + prefix)
    return {
        "identity": {key: scores[key] for key in ("model_context", "benchmark_context", "lighteval_revision", "benchmark_manifest_sha256")},
        "max_samples_per_task": None,
        "task_count": len(task_names),
        "task_names": task_names,
        "metrics": task_results,
        "runtime_adjustments": scores.get("runtime_adjustments", []),
    }


def _primary_metrics(metrics):
    return {key: value for key, value in metrics.items()
            if isinstance(value, (int, float)) and math.isfinite(value) and not key.endswith("_stderr")}


def build_report(plan, validated, execution):
    if execution.get("schema") != "arcus3-three-model-execution-v1" or not execution.get("complete"):
        raise ValueError("Comparison execution is incomplete")
    if execution.get("plan_sha256") != validated["plan_sha256"]:
        raise ValueError("Execution used a different comparison plan")
    entries = execution.get("executions", [])
    expected = [(arm, protocol) for arm in ARM_IDS for protocol in PROTOCOL_IDS]
    if [(entry.get("arm"), entry.get("protocol")) for entry in entries] != expected:
        raise ValueError("Comparison was not executed in the declared sequential order")
    if any(entry.get("exit_code") != 0 or not entry.get("completed_at") for entry in entries):
        raise ValueError("Comparison contains an incomplete execution")
    by_key = {(entry["arm"], entry["protocol"]): entry for entry in entries}
    arms = {}
    benchmark_coverage = None
    for arm_id in ARM_IDS:
        arm_identity = validated["arms"][arm_id]
        dev_path = Path(by_key[(arm_id, "developmental")]["output"])
        bench_path = Path(by_key[(arm_id, "benchmark")]["output"])
        dev_entry = by_key[(arm_id, "developmental")]
        bench_entry = by_key[(arm_id, "benchmark")]
        dev_file = dev_path / "scores.json"
        bench_file = bench_path / "donor-scores.json"
        for protocol_id, entry, result_file in (
            ("developmental", dev_entry, dev_file), ("benchmark", bench_entry, bench_file)
        ):
            _validate_hash(entry.get("result_sha256"), protocol_id + " result")
            if digest(result_file) != entry["result_sha256"]:
                raise ValueError("Evaluation result changed after execution: " + str(result_file))
            pinned = validated["reuse"].get((arm_id, protocol_id))
            if bool(entry.get("reused")) != bool(pinned):
                raise ValueError("Evaluation reuse provenance does not match the comparison plan")
            if pinned and (str(Path(pinned["output"]).resolve()) != str(Path(entry["output"]).resolve()) or
                           pinned["result_sha256"] != entry["result_sha256"]):
                raise ValueError("Reused evaluation does not match pinned evidence")
        developmental = _developmental(read(dev_file), arm_identity, plan["protocols"]["developmental"])
        benchmark = _benchmark(read(bench_file), arm_identity, plan["protocols"]["benchmark"])
        coverage = benchmark["task_names"]
        if benchmark_coverage is None:
            benchmark_coverage = coverage
        elif coverage != benchmark_coverage:
            raise ValueError("Benchmark task coverage differs across arms")
        arms[arm_id] = {
            "checkpoint": arm_identity,
            "developmental": developmental,
            "benchmark": benchmark,
            "evaluation_provenance": {
                "developmental": {"output": str(dev_path), "reused": bool(dev_entry.get("reused")),
                                  "result_sha256": dev_entry["result_sha256"]},
                "benchmark": {"output": str(bench_path), "reused": bool(bench_entry.get("reused")),
                              "result_sha256": bench_entry["result_sha256"]},
            },
        }
    deltas = {}
    supplemental_evidence = validated["supplemental"]["pure_donor_developmental"]
    supplemental_file = Path(supplemental_evidence["output"]) / "scores.json"
    if digest(supplemental_file) != supplemental_evidence["result_sha256"]:
        raise ValueError("Pure donor developmental result changed after plan validation")
    pure_donor = _pure_donor_developmental(
        read(supplemental_file), supplemental_evidence, plan["protocols"]["developmental"])
    pure_donor["provenance"] = {
        "output": supplemental_evidence["output"],
        "result_sha256": supplemental_evidence["result_sha256"],
        "donor_manifest_sha256": supplemental_evidence["donor_manifest_sha256"],
        "reused": True,
    }
    donor_nll = arms["donor"]["developmental"]["language"]["nll"]
    pure_donor_nll = pure_donor["language"]["nll"]
    control_nll = arms["alpha3.2.0"]["developmental"]["language"]["nll"]
    for arm_id in ARM_IDS:
        nll = arms[arm_id]["developmental"]["language"]["nll"]
        deltas[arm_id] = {"nll_vs_donor_derived_zero_update": nll - donor_nll,
                          "nll_vs_pure_donor_developmental": nll - pure_donor_nll,
                          "nll_vs_alpha3.2.0": nll - control_nll}
    return {
        "schema": "arcus3-three-model-comparison-report-v1",
        "complete": True,
        "plan_sha256": validated["plan_sha256"],
        "requested_boundary_input_tokens": validated["requested_boundary_input_tokens"],
        "exact_boundary_checkpoint_available": validated["exact_boundary_checkpoint_available"],
        "matched_evaluation_protocols": True,
        "matched_training_exposure": False,
        "arms": arms,
        "supplemental_pure_donor_developmental": pure_donor,
        "deltas": deltas,
        "benchmark_task_coverage": benchmark_coverage,
        "benchmark_task_coverage_sha256": canonical_digest(benchmark_coverage),
        "protocols": plan["protocols"],
        "automatic_promotion": False,
        "winner_selected": False,
        "limitations": [
            "The control is the preserved donor-derived Alpha 3.2.1 zero-update initialization, not a newly rerun pure-donor model. Its already completed full evaluations are reused by exact result hash.",
            "A compatible frozen 1.711B pure-donor developmental run is reported separately; no matching full pure-donor benchmark result was found.",
            "No exact %s-token Alpha 3.2.1 checkpoint survived retention; results use the verified %s-token pause checkpoint."
            % (f"{validated['requested_boundary_input_tokens']:,}", f"{validated['arms']['alpha3.2.1']['input_tokens']:,}"),
            "Alpha 3.2.0 and Alpha 3.2.1 have different measured training exposures and objectives, so this is not an equal-exposure causal comparison.",
            "Human fluency, relevance and coherence fields remain pending human review.",
            "The report records measurements and does not promote a model or authorize training.",
        ],
    }


def render_markdown(report):
    boundary = report["requested_boundary_input_tokens"]
    alpha321_tokens = report["arms"]["alpha3.2.1"]["checkpoint"]["input_tokens"]
    lines = [
        "# Arcus donor / Alpha 3.2.0 / Alpha 3.2.1 full comparison",
        "",
        "This is a read-only, strictly sequential comparison. No winner is selected and no model is promoted. "
        "The `donor` column is the explicitly identified donor-derived, zero-update Alpha 3.2.1 initialization control; it is not mislabeled as a newly executed pure-donor run.",
        "",
        "The requested boundary was **%s input tokens**, but no exact Alpha 3.2.1 checkpoint survived. "
        "Alpha 3.2.1 is therefore identified by its actual **%s-token** exposure throughout this report."
        % (f"{boundary:,}", f"{alpha321_tokens:,}"),
        "",
        "## Evidence identity",
        "",
        "| Arm | Updates | Input tokens | Target tokens | Boundary relation | Checkpoint manifest |",
        "|---|---:|---:|---:|---|---|",
    ]
    for arm_id in ARM_IDS:
        cp = report["arms"][arm_id]["checkpoint"]
        lines.append("| %s | %s | %s | %s | %s | `%s` |" % (
            arm_id, cp["updates"], cp["input_tokens"], cp["target_tokens"],
            cp.get("boundary_relation", "not-applicable"), cp["checkpoint_manifest_sha256"]))
    lines += ["", "## Developmental evaluation", "",
              "The pure-donor column is compatible historical developmental evidence. The zero-update column is the donor-derived expanded initialization used as the benchmark control.", "",
              "| Measurement | Pure donor 1.711B | Donor-derived zero-update init | Alpha 3.2.0 | Alpha 3.2.1 |",
              "|---|---:|---:|---:|---:|"]
    metrics = [("NLL", lambda x: x["language"]["nll"]),
               ("Perplexity", lambda x: x["language"]["perplexity"])]
    categories = report["arms"]["donor"]["developmental"]["categories"]
    metrics += [(name + " passed", lambda x, n=name: "%s/%s" % (x["categories"][n]["passed"], x["categories"][n]["measured"]))
                for name in categories]
    for label, value in metrics:
        cells = [value(report["supplemental_pure_donor_developmental"])]
        cells += [value(report["arms"][arm]["developmental"]) for arm in ARM_IDS]
        lines.append("| %s | %s | %s | %s | %s |" % (label, *cells))
    lines += ["", "## Full donor-protocol benchmark", "",
              "All arms used the same complete task coverage; stderr values remain in `comparison.json`.", "",
              "| Task / metric | Donor | Alpha 3.2.0 | Alpha 3.2.1 |", "|---|---:|---:|---:|"]
    for task in report["benchmark_task_coverage"]:
        keys = sorted(set().union(*(_primary_metrics(report["arms"][arm]["benchmark"]["metrics"][task]) for arm in ARM_IDS)))
        for key in keys:
            values = [report["arms"][arm]["benchmark"]["metrics"][task].get(key, "") for arm in ARM_IDS]
            lines.append("| `%s` %s | %s | %s | %s |" % (task, key, *values))
    lines += ["", "## Limits", ""] + ["- " + item for item in report["limitations"]]
    return "\n".join(lines) + "\n"


def main(args):
    workspace = Path(__file__).resolve().parents[1]
    plan = read(args.config)
    validated = validate_plan(plan, workspace)
    root = Path(args.root).resolve()
    execution = read(root / "execution.json")
    report = build_report(plan, validated, execution)
    (root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (root / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"complete": True, "report": str(root / "comparison.json"), "automatic_promotion": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", required=True)
    main(parser.parse_args())
