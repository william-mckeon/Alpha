"""Pause Alpha 3.2.2 at each 1M input-token milestone for a local light check.

The active production policy and checkpoint state are never rewritten. Each
evaluation uses one stopped, hash-verified checkpoint; a new coordinator then
resumes the same optimizer, RNG and corpus cursor under the same sealed policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arcus3.expanded_checkpoint import verify as verify_checkpoint
from scripts.run_arcus3_model_comparison import _finish_developmental


WORKSPACE = Path(__file__).resolve().parents[1]
OWNED_RUNS = (WORKSPACE / "runs" / "arcus3").resolve()
CONTROL_ROOT = OWNED_RUNS / "alpha322-interim-light-001"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path: Path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".pending")
    try:
        with pending.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def owned_run(path: str | Path):
    result = Path(path).resolve()
    if result.parent != OWNED_RUNS:
        raise ValueError("Expected a direct owned runs/arcus3 child")
    return result


def validated_config(path: Path):
    cfg = read(path)
    if (cfg.get("schema") != "arcus3-alpha322-interim-light-v1"
            or cfg.get("lineage_id") != "alpha3.2.2-wsd-001"
            or cfg.get("milestone_input_tokens") != [million * 1_000_000 for million in range(1, 7)]
            or cfg.get("full_comparison_input_tokens") != 7_000_000
            or cfg.get("nll_regression_limit") != 0.2):
        raise ValueError("Unexpected interim evaluation plan")
    if [control.get("id") for control in cfg["controls"]] != ["donor", "alpha3.2.0", "alpha3.2.1"]:
        raise ValueError("Unexpected fixed comparison controls")
    for control in cfg["controls"]:
        output = (WORKSPACE / control["output"]).resolve()
        if output.parent != OWNED_RUNS:
            raise ValueError("Control output escaped owned runs")
        scores, transcripts = output / "scores.json", output / "transcripts.json"
        if digest(scores) != control["scores_sha256"] or digest(transcripts) != control["transcripts_sha256"]:
            raise ValueError("Pinned control result changed: " + control["id"])
        result = read(scores)
        if (not result.get("execution_complete") or result.get("suite_sha256") != cfg["suite_sha256"]
                or result.get("settings_sha256") != cfg["settings_sha256"]
                or result.get("tokenizer_revision") != "31b70e2e869a7173562077fd711b654946d38674"):
            raise ValueError("Incomplete or incompatible control result: " + control["id"])
    return cfg


def prompt_metrics(transcripts, ids):
    rows = {row["id"]: row for row in transcripts}
    if len(rows) != len(transcripts) or not set(ids).issubset(rows):
        raise ValueError("Light prompt IDs are not present in the fixed control")
    return [{"id": prompt, "task_success": rows[prompt]["metrics"].get("task_success"),
             "repeated_fourgram_fraction": rows[prompt]["metrics"].get("repeated_fourgram_fraction")}
            for prompt in ids]


def compare_light(cfg, current_scores, current_transcripts):
    if (current_scores.get("tier") != "light" or not current_scores.get("complete_generation")
            or not current_scores.get("execution_complete")
            or current_scores.get("suite_sha256") != cfg["suite_sha256"]
            or current_scores.get("settings_sha256") != cfg["settings_sha256"]
            or current_scores.get("tokenizer_revision") != "31b70e2e869a7173562077fd711b654946d38674"):
        raise ValueError("Alpha 3.2.2 light result is incomplete or incompatible")
    ids = [row["id"] for row in current_transcripts]
    if len(ids) != 4 or len(set(ids)) != 4:
        raise ValueError("The fixed light check must contain four distinct prompts")
    current_nll = float(current_scores["language"]["nll"])
    if not math.isfinite(current_nll):
        raise ValueError("Non-finite light language NLL")
    current_tokens = current_scores["language"]["target_tokens"]
    arms = [{"id": "alpha3.2.2", "input_tokens": None,
             "language": current_scores["language"],
             "light_prompts": prompt_metrics(current_transcripts, ids)}]
    for control in cfg["controls"]:
        output = WORKSPACE / control["output"]
        score = read(output / "scores.json")
        if score["language"]["target_tokens"] != current_tokens:
            raise ValueError("Language cohorts differ in target tokens")
        arms.append({"id": control["id"], "input_tokens": control["input_tokens"],
                     "language": score["language"],
                     "light_prompts": prompt_metrics(read(output / "transcripts.json"), ids),
                     "scores_sha256": control["scores_sha256"],
                     "transcripts_sha256": control["transcripts_sha256"]})
    donor_nll = float(next(arm for arm in arms if arm["id"] == "donor")["language"]["nll"])
    return {"schema": "arcus3-alpha322-interim-light-report-v1",
            "light_prompt_ids": ids, "suite_sha256": cfg["suite_sha256"],
            "settings_sha256": cfg["settings_sha256"], "arms": arms,
            "nll_regression_limit": cfg["nll_regression_limit"],
            "nll_review_required": current_nll > donor_nll + cfg["nll_regression_limit"],
            "limitations": ["Only Alpha 3.2.2 was rerun at this milestone.",
                            "The other models' pinned full transcripts were filtered to the same four light prompts.",
                            "The controls have different training exposures; this check is diagnostic, not a winner selection.",
                            "Human fluency and coherence ratings remain unscored."]}


def wait_for_pause(root: Path, threshold: int, *, timeout_seconds=8 * 3600):
    receipt_path = root / f"interim-{threshold}-pause.json"
    error_path = root / f"interim-{threshold}-error.json"
    end = time.monotonic() + timeout_seconds
    while time.monotonic() < end:
        if error_path.exists():
            raise RuntimeError("Milestone pause watcher failed: " + str(error_path))
        if receipt_path.exists():
            receipt = read(receipt_path)
            if receipt.get("status") == "verified-stopped" and receipt.get("verified_stopped"):
                return receipt
        if (root / "session-result.json").exists() and not receipt_path.exists():
            raise RuntimeError("Coordinator stopped before interim evaluation: " + str(root))
        time.sleep(10)
    raise TimeoutError("Interim milestone pause was not verified in time")


def verify_pause(root: Path, receipt, threshold: int):
    if receipt.get("requested_input_tokens") != threshold or not receipt.get("payload_hashes_verified"):
        raise ValueError("Unverified milestone pause receipt")
    saved = read(root / "controller-state.json")
    checkpoint = Path(saved["checkpoint"]).resolve()
    if checkpoint != Path(receipt["checkpoint"]).resolve():
        raise ValueError("Coordinator and pause receipt select different checkpoints")
    state = saved["state"]
    if (state["input_tokens"] != receipt["actual_input_tokens"]
            or not threshold <= state["input_tokens"] < threshold + 1_000_000
            or state["updates"] != receipt["actual_updates"]
            or state["config"]["lineage"]["id"] != "alpha3.2.2-wsd-001"
            or state.get("evaluation_pending") or state["scheduler"]["committed_input_tokens"] != state["input_tokens"]):
        raise ValueError("Milestone checkpoint state is inconsistent")
    manifest = verify_checkpoint(checkpoint, state["parent_sha256"], state["data_sha256"], state["config_sha256"])
    if digest(checkpoint / "manifest.json") != receipt["checkpoint_manifest_sha256"] or manifest["updates"] != state["updates"]:
        raise ValueError("Milestone checkpoint manifest changed")
    return checkpoint, state


def run_light(root: Path, checkpoint: Path, threshold: int, cfg):
    output = OWNED_RUNS / f"baseline-alpha322-interim-{threshold // 1_000_000}m-{uuid.uuid4().hex}"
    stop_at = (datetime.now(timezone.utc) + timedelta(minutes=29)).isoformat()
    command = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
               "scripts/start_arcus3.ps1", "-Mode", "baseline", "-Root", output.relative_to(WORKSPACE).as_posix(),
               "-StopAt", stop_at, "-RuntimeConfig", cfg["evaluation_runtime"],
               "-ConvertedPath", cfg["converted"], "-ExpandedPath", str(checkpoint),
               "-EvaluationTier", "light"]
    log = root / f"light-{threshold}-launcher.log"
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=WORKSPACE, stdout=stream, stderr=subprocess.STDOUT)
    scores = output / "scores.json"
    if result.returncode and scores.exists() and (output / "transcripts.json").exists():
        recovered = _finish_developmental(WORKSPACE, output, stop_at)
        if recovered != 0:
            raise RuntimeError("Light evaluation did not finish; inspect " + str(output))
    if not scores.exists():
        raise RuntimeError("Light evaluation did not produce scores; inspect " + str(output))
    value = read(scores)
    if value.get("expanded_manifest_sha256") != digest(checkpoint / "manifest.json"):
        raise ValueError("Light scores belong to a different checkpoint")
    if not value.get("execution_complete"):
        raise RuntimeError("Light evaluation is incomplete; inspect " + str(output))
    return output, value


def launch_continuation(old_root: Path, threshold: int, control_root: Path):
    new_root = OWNED_RUNS / f"alpha322-production-interim-{threshold // 1_000_000}m-{uuid.uuid4().hex}"
    command = [r"C:\Python314\python.exe", "scripts/run_arcus3_production.py",
               "--root", str(new_root), "--continue-from", str(old_root),
               "--runtime", "configs/arcus3/production_runtime_alpha322.json",
               "--qualification", "runs/arcus3/production-alpha322-qualification-001/receipt.json",
               "--policy", "configs/arcus3/production_alpha322.json",
               "--adaptation-config", "configs/arcus3/backbone_adaptation_alpha322.json"]
    environment = os.environ.copy()
    dependencies = WORKSPACE / ".python-deps"
    environment["PYTHONPATH"] = str(dependencies) + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    log = (control_root / f"continuation-{threshold}.log").open("w", encoding="utf-8")
    process = subprocess.Popen(command, cwd=WORKSPACE, env=environment, stdout=log, stderr=subprocess.STDOUT,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    log.close()
    end = time.monotonic() + 900
    while time.monotonic() < end:
        if process.poll() is not None:
            raise RuntimeError("Continuation coordinator exited before its worker started: " + str(new_root))
        session = new_root / "session.json"
        if session.exists():
            current = read(session)
            if current.get("mode") == "adaptation":
                return new_root, process.pid
        time.sleep(5)
    raise TimeoutError("Continuation coordinator did not start its worker")


def run(first_root: Path, config_path: Path, adopt_watching: bool = False):
    cfg = validated_config(config_path)
    first_root = owned_run(first_root)
    if adopt_watching:
        progress = read(CONTROL_ROOT / "state.json")
        if (progress.get("schema") != "arcus3-alpha322-interim-pipeline-v1"
                or progress.get("status") != "watching" or progress.get("milestones") != []
                or progress.get("current_root") != str(first_root)
                or progress.get("next_input_tokens") != 1_000_000):
            raise ValueError("Only an untouched, watching interim pipeline can be adopted")
    else:
        CONTROL_ROOT.mkdir(exist_ok=False)
        progress = {"schema": "arcus3-alpha322-interim-pipeline-v1", "status": "watching",
                    "current_root": str(first_root), "milestones": [], "next_input_tokens": 1_000_000,
                    "control_results_reused": True, "automatic_failure_retry": False}
        atomic_json(CONTROL_ROOT / "state.json", progress)
    root = first_root
    try:
        for threshold in cfg["milestone_input_tokens"]:
            if (CONTROL_ROOT / "pause-training").exists():
                progress["status"] = "user_paused"; break
            if threshold != 1_000_000:
                watcher = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                           "scripts/watch_arcus3_token_milestone.ps1", "-Root", root.relative_to(WORKSPACE).as_posix(),
                           "-CheckpointRoot", str(Path(read(WORKSPACE / "configs/arcus3/phase8_storage.json")["external_root"])
                                                  / "alpha3.2.2" / "checkpoints"), "-InputTokens", str(threshold)]
                with (CONTROL_ROOT / f"watcher-{threshold}.log").open("w", encoding="utf-8") as log:
                    code = subprocess.run(watcher, cwd=WORKSPACE, stdout=log, stderr=subprocess.STDOUT).returncode
                if code:
                    raise RuntimeError("Milestone watcher stopped without verified pause: " + str(root))
            receipt = wait_for_pause(root, threshold)
            checkpoint, state = verify_pause(root, receipt, threshold)
            progress.update(status="evaluating", current_root=str(root), next_input_tokens=threshold)
            atomic_json(CONTROL_ROOT / "state.json", progress)
            output, scores = run_light(CONTROL_ROOT, checkpoint, threshold, cfg)
            result = compare_light(cfg, scores, read(output / "transcripts.json"))
            result.update(milestone_input_tokens=threshold, actual_input_tokens=state["input_tokens"],
                          actual_target_tokens=state["target_tokens"], updates=state["updates"],
                          checkpoint=str(checkpoint), checkpoint_manifest_sha256=digest(checkpoint / "manifest.json"),
                          evaluation_output=str(output), scores_sha256=digest(output / "scores.json"),
                          transcripts_sha256=digest(output / "transcripts.json"))
            atomic_json(CONTROL_ROOT / f"million-{threshold // 1_000_000}-report.json", result)
            progress["milestones"].append({"threshold": threshold, "input_tokens": state["input_tokens"],
                                           "checkpoint_manifest_sha256": result["checkpoint_manifest_sha256"],
                                           "report": str(CONTROL_ROOT / f"million-{threshold // 1_000_000}-report.json"),
                                           "nll_review_required": result["nll_review_required"]})
            if result["nll_review_required"]:
                progress.update(status="nll_review_required", current_root=str(root))
                atomic_json(CONTROL_ROOT / "state.json", progress)
                return
            if (CONTROL_ROOT / "pause-training").exists():
                progress.update(status="user_paused", current_root=str(root))
                atomic_json(CONTROL_ROOT / "state.json", progress)
                return
            new_root, pid = launch_continuation(root, threshold, CONTROL_ROOT)
            progress.update(status="watching", current_root=str(new_root), coordinator_pid=pid,
                            next_input_tokens=threshold + 1_000_000)
            atomic_json(CONTROL_ROOT / "state.json", progress)
            root = new_root
        else:
            progress.update(status="interim_checks_complete", current_root=str(root),
                            next_input_tokens=7_000_000)
            atomic_json(CONTROL_ROOT / "state.json", progress)
    except BaseException as error:
        progress.update(status="failure_requires_review", error_type=type(error).__name__, error=str(error),
                        current_root=str(root))
        atomic_json(CONTROL_ROOT / "state.json", progress)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-root", required=True)
    parser.add_argument("--config", default="configs/arcus3/alpha322_interim_light.json")
    parser.add_argument("--adopt-watching", action="store_true")
    args = parser.parse_args()
    run(Path(args.first_root), WORKSPACE / args.config, args.adopt_watching)
