"""Pinned official SWE-bench entry point for a trusted Linux orchestration worker."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from evaluation.fixtures import sha256
from evaluation.openhands import SDK_REVISION, EXTENSIONS_REVISION, JSONL_SHA256, build_commands, guarded_workspace_command, disposable_conversation, run_bounded_conversation, gateway_iteration_stop
from evaluation.worker_adapters import validate_gateway


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--phase", choices=("infer", "verify"), required=True)
    args = parser.parse_args()
    work = Path("/work")
    if os.name != "posix" or not Path("/.dockerenv").exists() or not work.is_dir():
        raise ValueError("official OpenHands worker requires isolated Linux orchestration")
    if not args.trial_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.trial_id):
        raise ValueError("invalid trial identity")
    suites = json.loads((ROOT / "evaluation/suites.json").read_text())["suites"]
    frozen = {task for suite in suites.values() for task in suite["repository"]["task_ids"]}
    if args.task not in frozen or sha256(args.dataset) != JSONL_SHA256:
        raise ValueError("repository task or local dataset differs from frozen snapshot")
    vendor = ROOT / "evaluation/vendor/openhands"
    harness = json.loads((ROOT / "evaluation/harnesses/openhands.json").read_text())
    for source, expected in ((vendor, harness["revision"]), (vendor / "vendor/software-agent-sdk", SDK_REVISION)):
        git = ["git", "-c", "safe.directory=" + str(source), "-C", str(source)]
        if subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip() != expected or subprocess.run([*git, "diff", "--quiet", "HEAD"]).returncode:
            raise ValueError("OpenHands source differs from its clean pinned revision")
    validate_gateway(os.environ["ARCUS_WORKER_BASE_URL"], os.environ["ARCUS_WORKER_TOKEN"])
    protocol = json.loads((ROOT / "evaluation/protocol.json").read_text())
    generation = protocol["generation"]
    if os.environ.get("EXTENSIONS_REF") != EXTENSIONS_REVISION or harness.get("extensions_revision") != EXTENSIONS_REVISION:
        raise ValueError("OpenHands public extensions must use the retained immutable revision")
    selection, config = work / "selection.txt", work / "llm.json"
    if args.phase == "infer":
        if selection.exists() or (work / "inference").exists():
            raise ValueError("repository inference cannot resume or overwrite artifacts")
        selection.write_text(args.task + "\n")
        from openhands.sdk import LLM
        llm = {"model": "openai/" + args.candidate, "base_url": os.environ["ARCUS_WORKER_BASE_URL"], "api_key": os.environ["ARCUS_WORKER_TOKEN"], "num_retries": 0, "temperature": generation["temperature"], "top_p": generation["top_p"], "max_output_tokens": generation["max_output_tokens"]}
        LLM.model_validate(llm)
        config.write_text(json.dumps(llm))
    else:
        if selection.read_text().strip() != args.task:
            raise ValueError("inference task selection differs from verifier task")
    commands = build_commands(Path(sys.executable), dataset_path=args.dataset, selection_path=selection, llm_config_path=config, output_dir=work / "inference", prediction_path=work / "unused", run_id=args.trial_id, generation=generation)
    os.environ["BUILDKIT_RESET_ON_FAILURE"] = "0"
    # This pinned benchmark expects a phased Dockerfile-hash tag, but its legacy
    # source builder emits SDK-only tags. Use upstream's supported explicit
    # prefix override; clean source pins and resolved image IDs stay mandatory.
    os.environ["IMAGE_TAG_PREFIX"] = SDK_REVISION[:7]
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    # Upstream resolves git submodule metadata relative to cwd during import.
    os.chdir(vendor)
    if args.phase == "infer":
        # The trusted controller alone sees the Docker API. No socket or host
        # filesystem volume is passed to the model's terminal workspace.
        from openhands.workspace.docker import workspace as docker_workspace
        original = docker_workspace.execute_command

        def owned_command(command, *positional, **kwargs):
            if list(command[:2]) == ["docker", "run"]:
                command = list(command)
                index = command.index("--host") - 1
                image = command[index]
                image_id = subprocess.check_output(["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True).strip()
                command = guarded_workspace_command(command, trial_id=args.trial_id, image_id=image_id)
                with (work / "workspace-images.jsonl").open("a") as log:
                    log.write(json.dumps({"source_image": image, "image_id": image_id}) + "\n")
            return original(command, *positional, **kwargs)

        docker_workspace.execute_command = owned_command
        from benchmarks.utils import build_utils
        build_utils.maybe_prune_buildkit_cache = lambda *a, **k: False
        build_utils.maybe_reset_buildkit = lambda *a, **k: None
        from benchmarks.swebench import run_infer
        from openhands.sdk.conversation.exceptions import ConversationRunError
        original_run = run_infer.run_conversation_with_fake_user_response

        def bounded_run(conversation):
            return run_bounded_conversation(original_run, conversation, run_error=ConversationRunError, stop_path=work / "budget-stop.json", trial_id=args.trial_id, max_iterations=generation["max_agent_iterations"], stop_check=lambda: gateway_iteration_stop(os.environ["ARCUS_WORKER_BASE_URL"], os.environ["ARCUS_WORKER_TOKEN"], args.trial_id, generation["max_agent_iterations"]))

        run_infer.run_conversation_with_fake_user_response = bounded_run
        base = run_infer.DockerWorkspace

        class OwnedWorkspace(base):
            def _start_container(self, image, context):
                self.forward_env = []
                self.volumes = []
                self.host_port = docker_workspace.find_available_tcp_port()
                self.host = f"http://host.docker.internal:{self.host_port}"
                super()._start_container(image, context)

        run_infer.DockerWorkspace = OwnedWorkspace
        original_conversation = run_infer.Conversation

        def owned_conversation(*positional, **kwargs):
            return disposable_conversation(original_conversation, *positional, **kwargs)

        run_infer.Conversation = owned_conversation
        original_output_dir = run_infer.construct_eval_output_dir

        def safe_output_dir(*positional, **kwargs):
            # Upstream's error writer replaces '.jsonl' in the entire output path.
            # A local dataset filename in the directory name breaks that writer.
            kwargs["dataset_name"] = "arcus-frozen-swebench-verified-test"
            return original_output_dir(*positional, **kwargs)

        run_infer.construct_eval_output_dir = safe_output_dir
        command = commands["inference"]
        sys.argv = [command[2], *command[3:]]
        run_infer.main()
        extensions = Path.home() / ".openhands/cache/skills/public-skills"
        actual = subprocess.check_output(["git", "-C", str(extensions), "rev-parse", "HEAD"], text=True).strip()
        if actual != EXTENSIONS_REVISION:
            raise ValueError("loaded OpenHands public extensions differ from pin")
        from evaluation.execution import source_fingerprint
        (work / "controller-provenance.json").write_text(json.dumps({"harness_revision": harness["revision"], "sdk_revision": SDK_REVISION, "extensions_revision": actual, "dataset_sha256": JSONL_SHA256, "image_tag_prefix": SDK_REVISION[:7], "protocol_snapshot": protocol, "execution_fingerprint": source_fingerprint(ROOT)}), encoding="utf-8")
    else:
        predictions = list((work / "inference").rglob("output.jsonl"))
        if len(predictions) != 1:
            raise ValueError("official inference did not retain exactly one output file")
        rows = [json.loads(line) for line in predictions[0].read_text().splitlines() if line.strip()]
        if len(rows) != 1 or rows[0].get("instance_id") != args.task:
            raise ValueError("official repository output does not contain exactly the frozen task")
        command = commands["verification"]
        command[3] = str(predictions[0])
        sys.argv = [command[2], *command[3:]]
        from benchmarks.swebench import eval_infer
        os.chdir(work)
        eval_infer.main()


if __name__ == "__main__":
    main()
