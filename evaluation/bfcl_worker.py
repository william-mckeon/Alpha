"""One frozen category task through pinned native BFCL and its official verifier.

This limited integration diagnostic is not a complete BFCL suite runner.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import statistics
from types import SimpleNamespace

from evaluation.worker_adapters import register_bfcl_route, singleton_stdev
from evaluation.search_service import RemoteSearchAPI
from evaluation.bfcl import frozen_case, load_answer_overrides, corrected_ground_truth


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--phase", choices=("generate", "evaluate"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to((ROOT / "evaluation/runs/bfcl").resolve()):
        raise ValueError("BFCL worker output must remain in its isolated runs directory")
    harness = json.loads((ROOT / "evaluation/harnesses/bfcl.json").read_text())
    vendor = ROOT / "evaluation/vendor/bfcl"
    git = ["git", "-c", "safe.directory=" + str(vendor).replace("\\", "/"), "-C", str(vendor)]
    revision = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    if revision != harness["revision"] or subprocess.run([*git, "diff", "--quiet", "HEAD"], check=False).returncode:
        raise ValueError("BFCL source does not match the clean pinned revision")
    suites = json.loads((ROOT / "evaluation/suites.json").read_text())["suites"]
    frozen = set(suites["smoke"]["function_calling"]["task_ids"]) | set(suites["qualification"]["function_calling"]["task_ids"])
    protocol = json.loads((ROOT / "evaluation/protocol.json").read_text())
    category, official_task = frozen_case(args.task, protocol)
    if args.task not in frozen:
        raise ValueError("BFCL worker requires a frozen task ID")
    os.environ["BFCL_PROJECT_ROOT"] = str(output)
    # Set isolated project root BEFORE importing upstream path constants. No .env here.
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
    from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
    from bfcl_eval.__main__ import cli
    if category == "web_search_base":
        from bfcl_eval.eval_checker.multi_turn_eval.func_source_code import web_search

        class BoundSearchAPI(RemoteSearchAPI):
            def __init__(self):
                super().__init__(os.environ["ARCUS_WORKER_BASE_URL"], os.environ["ARCUS_WORKER_TOKEN"], os.environ["ARCUS_WORKER_TRIAL_ID"])

        web_search.WebSearchAPI = BoundSearchAPI
    generation = json.loads((ROOT / "evaluation/protocol.json").read_text())["generation"]
    alias = register_bfcl_route(MODEL_CONFIG_MAPPING, ModelConfig, OpenAICompletionsHandler, candidate_id=args.candidate, gateway_url=os.environ["ARCUS_WORKER_BASE_URL"], proxy_token=os.environ["ARCUS_WORKER_TOKEN"], generation=generation)
    provider = next(item for item in json.loads((ROOT / "evaluation/providers.json").read_text())["providers"] if item["candidate_id"] == args.candidate)
    MODEL_CONFIG_MAPPING[alias].input_price = provider["prompt_per_million"]
    MODEL_CONFIG_MAPPING[alias].output_price = provider["completion_per_million"]
    from bfcl_eval.eval_checker import eval_runner_helper
    from bfcl_eval.eval_checker import eval_runner
    overrides = load_answer_overrides(ROOT, protocol)
    (output / ("bfcl-policy-" + args.phase + ".json")).write_text(json.dumps({"contract_revision": protocol["contract_revision"], "underscore_to_dot": True, "answer_override": protocol.get("search", {}).get("answer_override")}), encoding="utf-8")
    original_answers = eval_runner.load_ground_truth_entry
    eval_runner.load_ground_truth_entry = lambda *a, **k: corrected_ground_truth(original_answers(*a, **k), overrides)
    eval_runner_helper.statistics = SimpleNamespace(mean=statistics.mean, stdev=singleton_stdev)
    if args.phase == "generate":
        with (output / "test_case_ids_to_generate.json").open("x", encoding="utf-8") as stream:
            json.dump({category: [official_task]}, stream)
        arguments = ["generate", "--model", alias, "--run-ids", "--num-threads", "1", "--temperature", str(generation["temperature"]), "--include-input-log", "--result-dir", str(output / "result")]
    else:
        arguments = ["evaluate", "--model", alias, "--test-category", category, "--partial-eval", "--result-dir", str(output / "result"), "--score-dir", str(output / "score")]
    cli(args=arguments, standalone_mode=False)


if __name__ == "__main__":
    main()
