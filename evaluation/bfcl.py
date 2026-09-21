"""Catalog access for the pinned official BFCL runner (not a replacement verifier)."""
from __future__ import annotations

import json
import re
import hashlib
import copy
from pathlib import Path


def diagnostic_case(task_id: str) -> tuple[str, str]:
    """Explicit diagnostic-only raw catalog ID → official snippet-enabled variant."""
    if re.fullmatch(r"web_search_\d+", task_id):
        return "web_search_base", task_id.replace("web_search_", "web_search_base_", 1)
    category, _, index = task_id.rpartition("_")
    supported = {"simple_python", "simple_java", "simple_javascript", "multiple", "parallel", "parallel_multiple", "irrelevance", "live_simple", "live_multiple", "live_parallel", "live_parallel_multiple", "live_irrelevance", "live_relevance", "multi_turn_base", "multi_turn_long_context", "multi_turn_miss_param", "multi_turn_miss_func"}
    if category in supported and re.fullmatch(r"\d+(?:-\d+-\d+)?", index):
        return category, task_id
    raise ValueError("unsupported BFCL diagnostic task")


def category_group(category: str) -> str:
    if category.startswith("web_search_"):
        return "agentic"
    if category.startswith("multi_turn_"):
        return "multi_turn"
    if category.startswith("live_"):
        return "live"
    return "non_live"


def frozen_case(task_id: str, protocol: dict) -> tuple[str, str]:
    """Resolve only the explicitly approved search variant; never infer approval."""
    if re.fullmatch(r"web_search_\d+", task_id):
        if protocol.get("search", {}).get("bfcl_web_search_variant") != "web_search_base":
            raise ValueError("BFCL web-search qualification variant is not frozen")
    return diagnostic_case(task_id)


def load_answer_overrides(root: Path, protocol: dict) -> dict:
    policy = protocol.get("search", {}).get("answer_override")
    if policy is None:
        return {}
    if policy.get("path") != "evaluation/bfcl_answer_overrides.json":
        raise ValueError("unsupported answer correction authority")
    raw = (root / policy["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != policy.get("sha256"):
        raise ValueError("answer correction hash differs from frozen protocol")
    document = json.loads(raw)
    if document.get("harness_revision") != next(h["revision"] for h in protocol["harnesses"] if h["id"] == "function_calling"):
        raise ValueError("answer correction source revision mismatch")
    overrides = {}
    for row in document["overrides"]:
        if row["task_id"] in overrides or not row.get("ground_truth") or not row.get("source_url", "").startswith("https://"):
            raise ValueError("invalid answer correction")
        overrides[row["task_id"]] = row
    return overrides


def corrected_ground_truth(rows: list, overrides: dict) -> list:
    """Replace only frozen answer data; official grader and prompts stay unchanged."""
    result = copy.deepcopy(rows)
    for row in result:
        raw_id = row["id"].replace("web_search_base_", "web_search_", 1)
        if raw_id in overrides:
            correction = overrides[raw_id]
            if row["ground_truth"] != correction["original_ground_truth"]:
                raise ValueError("upstream answer no longer matches correction precondition")
            row["ground_truth"] = correction["ground_truth"]
    return result


def load_task_catalog(data_dir: Path) -> list[str]:
    tasks = []
    for path in sorted(data_dir.glob("BFCL_v4_*.json")):
        if path.name == "BFCL_v4_format_sensitivity.json":
            # This is a category-to-existing-ID index, not a task dataset.
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            task_id = entry.get("id")
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(f"BFCL catalog entry without ID: {path}")
            tasks.append(task_id)
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("BFCL catalog must be nonempty with unique IDs")
    return sorted(tasks)


def build_commands(python: Path, *, registry_alias: str, category: str, result_dir: Path, score_dir: Path, generation: dict) -> dict[str, list[str]]:
    """Use upstream CLI with an isolated one-ID run file and registered native FC handler."""
    if not registry_alias.endswith("-FC") or not category:
        raise ValueError("BFCL requires an explicit native-FC registry alias and category")
    return {
        "inference": [str(python), "-m", "bfcl_eval", "generate", "--model", registry_alias, "--run-ids", "--num-threads", "1", "--temperature", str(generation["temperature"]), "--include-input-log", "--result-dir", str(result_dir)],
        "verification": [str(python), "-m", "bfcl_eval", "evaluate", "--model", registry_alias, "--test-category", category, "--partial-eval", "--result-dir", str(result_dir), "--score-dir", str(score_dir)],
    }
