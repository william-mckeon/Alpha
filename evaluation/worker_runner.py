"""Shared prerequisite and official-import checks for diagnostic integration sweeps."""
from pathlib import Path
import json
import hashlib

from evaluation.bfcl import frozen_case, category_group, load_answer_overrides
from evaluation.fixtures import sha256
from evaluation.openhands import JSONL_SHA256
from evaluation.control import validate_result, validate_artifact_hashes
from evaluation.execution import source_fingerprint


def registered_runners(root: Path, protocol: dict) -> set[str]:
    """Register workers only from retained current-contract normalized live proof."""
    registered = {"terminal"}
    for name, harness in (("bfcl", "function_calling"), ("mcpmark", "mcp"), ("openhands", "repository")):
        config = json.loads((root / "evaluation/harnesses" / (name + ".json")).read_text())
        proof = config.get("runner_evidence")
        if not isinstance(proof, dict):
            continue
        try:
            path = (root / proof["path"]).resolve()
            if not path.is_relative_to((root / "evaluation/runs").resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != proof["sha256"]:
                continue
            record = json.loads(path.read_text())
            if protocol["contract_revision"] >= 6 and record["run"].get("execution_fingerprint") != source_fingerprint(root):
                continue
            if protocol.get("execution_policy", {}).get("readiness_source_identity_required") and record["run"].get("launcher_sha256") != hashlib.sha256((root / "scripts/eval_foundations.py").read_bytes()).hexdigest():
                continue
            if (
                record["harness"]["id"] != harness
                or record["harness"]["revision"] != config["revision"]
                or record["run"]["contract_revision"] != protocol["contract_revision"]
                or record["run"].get("evidence_kind") != "diagnostic"
                or record["outcome"]["status"] not in {"passed", "failed"}
                or any(record["generation"].get(key) != value for key, value in protocol["generation"].items())
                or validate_result(record) or validate_artifact_hashes(record, root)
            ):
                continue
            registered.add(harness)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return registered


def control_mounts(root: Path, harness: str) -> list[str]:
    """Expose only nonsecret frozen controls to the owned orchestration worker."""
    if harness not in {"mcpmark", "openhands"}:
        raise ValueError("unsupported worker control authority")
    paths = ["protocol.json", "providers.json", "suites.json", "harnesses/" + harness + ".json"]
    args = []
    for label in paths:
        path = (root / "evaluation" / label).resolve()
        if not path.is_file():
            raise ValueError("missing worker control: " + label)
        args.extend(["--mount", f"type=bind,source={path},target=/app/evaluation/{label},readonly"])
    return args


def validate_worker_inputs(root: Path, plan: list[dict], protocol: dict) -> list[str]:
    """Audit every selected unit before any runner starts or spends money."""
    errors = []
    provider_file = root / "evaluation/providers.json"
    if provider_file.is_file():
        providers = json.loads(provider_file.read_text())["providers"]
        selected = {unit.get("candidate_id") for unit in plan}
        for provider in providers:
            maximum = provider.get("max_completion_tokens")
            if provider["candidate_id"] in selected and maximum is not None and protocol["generation"]["max_output_tokens"] > maximum:
                errors.append(provider["candidate_id"] + ": frozen output allowance exceeds pinned endpoint completion capacity")
    for harness, task in sorted({(unit["harness_id"], unit["task_id"]) for unit in plan}):
        try:
            if harness == "function_calling":
                frozen_case(task, protocol)
                load_answer_overrides(root, protocol)
                if not (root / "evaluation/runs/bfcl-venv/Scripts/python.exe").is_file():
                    raise ValueError("isolated BFCL runtime missing")
            elif harness == "mcp":
                parts = task.split("/")
                if len(parts) != 3 or parts[0] != "filesystem" or any(part in {"", ".", ".."} for part in parts):
                    raise ValueError("invalid isolated filesystem task")
                archive = root / "evaluation/runs/fixture-snapshots" / (parts[1] + ".zip")
                manifest = json.loads(archive.with_suffix(".manifest.json").read_text())
                if sha256(archive) != manifest["archive_sha256"]:
                    raise ValueError("fixture hash differs from retained snapshot")
                control_mounts(root, "mcpmark")
            elif harness == "repository":
                if sha256(root / "evaluation/runs/swebench-verified-c104f840.jsonl") != JSONL_SHA256:
                    raise ValueError("repository dataset hash differs from pin")
                control_mounts(root, "openhands")
            elif harness == "terminal":
                if not (root / "evaluation/vendor/harbor/.venv/Scripts/harbor.exe").is_file():
                    raise ValueError("pinned Harbor executable missing")
            else:
                raise ValueError("unsupported harness")
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"{harness}/{task}: {exc}")
    return errors


def worker_verifier_paths(directory: Path, harness: str, candidate: str, task: str, protocol: dict) -> list[Path]:
    if harness not in {"function_calling", "repository", "mcp"}:
        raise ValueError("unsupported official worker verifier")
    if harness == "function_calling":
        category, _ = frozen_case(task, protocol)
        alias = "arcus-" + candidate + "-FC"
        group = category_group(category)
        return [directory / "score" / alias / group / f"BFCL_v4_{category}_score.json", directory / "result" / alias / group / f"BFCL_v4_{category}_result.json"]
    pattern = "inference/**/*.report.json" if harness == "repository" else "results/**/meta.json"
    paths = list(directory.glob(pattern))
    if len(paths) != 1:
        raise ValueError("worker must retain exactly one official verifier result")
    return paths
