"""Release contracts for private, inference-only Alpha 3 packages.

The release spec selects an immutable training checkpoint.  It is deliberately
separate from a training configuration: publishing must never infer "latest" or
silently follow a moving checkpoint pointer.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


SCHEMA = "arcus3-private-release-v2"
AUTHORIZATION = "user-requested-alpha-3-family-private-publication"
MODEL_REPOS = {
    "alpha3.0": "Islanderintel/Alpha-3.0",
    "alpha3.2.0": "Islanderintel/Alpha-3.2.0",
    "alpha3.2.1": "Islanderintel/Alpha-3.2.1",
    "alpha3.2.2": "Islanderintel/Alpha-3.2.2",
}
MODEL_CARDS = {
    "alpha3.0": "docs/ALPHA_3_MODEL_CARD.md",
    "alpha3.2.0": "docs/ALPHA_3_2_0_MODEL_CARD.md",
    "alpha3.2.1": "docs/ALPHA_3_2_1_MODEL_CARD.md",
    "alpha3.2.2": "docs/ALPHA_3_2_2_MODEL_CARD.md",
}
INFERENCE_SUFFIXES = {".json", ".safetensors", ".py", ".md", ".txt"}
INFERENCE_BASENAMES = {"LICENSE", "NOTICE"}
FORBIDDEN_PARTS = {
    "optimizer", "state.pt", "training.jsonl", "metrics.jsonl", ".env",
    "credentials", "raw-data", "teacher-cache",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def digest(path: str | Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _sha(value: object, field: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ValueError(f"Invalid {field}")
    return value


def validate_spec(spec: dict, *, require_ready: bool = True) -> dict:
    """Validate and return a release spec without mutating it."""
    if spec.get("schema") != SCHEMA:
        raise ValueError("Unsupported release schema")
    label = spec.get("model_label")
    top_level = {"schema", "model_label", "repo_id", "private", "authorization", "status",
                 "release_kind", "package_kind", "unique_parameters", "context_tokens",
                 "model_card", "source"}
    if label == "alpha3.2.0":
        top_level.add("allow_legacy_missing_model_label")
    if set(spec) != top_level:
        raise ValueError("Release specification contains unapproved fields")
    if MODEL_REPOS.get(label) != spec.get("repo_id"):
        raise ValueError("Release repository does not match model label")
    if spec.get("private") is not True:
        raise ValueError("Alpha releases must remain private")
    if spec.get("authorization") != AUTHORIZATION:
        raise ValueError("Release authorization mismatch")
    status = spec.get("status")
    if status not in {"ready", "awaiting-evaluation", "awaiting-checkpoint"}:
        raise ValueError("Invalid release status")
    if require_ready and status != "ready":
        raise ValueError("Release is not ready")
    if spec.get("package_kind") != "inference-only":
        raise ValueError("Only inference-only releases are authorized")
    if spec.get("unique_parameters") not in {2013390848, 2013403142}:
        raise ValueError("Unexpected Alpha 3 parameter count")
    if spec.get("context_tokens") != 8192:
        raise ValueError("Release context must match the donor exactly")
    source = spec.get("source")
    if not isinstance(source, dict) or source.get("kind") not in {
        "conversion-initialization", "expanded-checkpoint"
    }:
        raise ValueError("Unsupported release source")
    source_fields = {"kind", "parent_manifest_sha256", "training_updates",
                     "input_tokens", "target_tokens"}
    if source["kind"] == "expanded-checkpoint":
        source_fields.add("checkpoint_manifest_sha256")
    if label == "alpha3.2.2":
        source_fields.update({"lineage_id", "adaptation_config_sha256",
                              "learning_rate_schedule_sha256"})
    if set(source) != source_fields:
        raise ValueError("Release source contains unapproved fields")
    if label == "alpha3.2.0" and spec.get("allow_legacy_missing_model_label") is not True:
        raise ValueError("Alpha 3.2.0 legacy checkpoint exception is required")
    _sha(source.get("parent_manifest_sha256"), "parent manifest SHA256")
    if source["kind"] == "conversion-initialization":
        if any(source.get(k) != 0 for k in ("training_updates", "input_tokens", "target_tokens")):
            raise ValueError("Initialization release has training exposure")
    else:
        if status == "ready" or source.get("checkpoint_manifest_sha256") is not None:
            _sha(source.get("checkpoint_manifest_sha256"), "checkpoint manifest SHA256")
        for field in ("training_updates", "input_tokens", "target_tokens"):
            value = source.get(field)
            if (status == "ready" and (type(value) is not int or value <= 0)) or (
                value is not None and (type(value) is not int or value < 0)
            ):
                raise ValueError(f"Invalid {field}")
    if label == "alpha3.2.2":
        if source.get("lineage_id") != "alpha3.2.2-wsd-001":
            raise ValueError("Alpha 3.2.2 release lineage mismatch")
        for field in ("adaptation_config_sha256", "learning_rate_schedule_sha256"):
            if status == "ready" or source.get(field) is not None:
                _sha(source.get(field), field)
    card = spec.get("model_card")
    if card != MODEL_CARDS[label]:
        raise ValueError("Model card does not match the approved release artifact")
    return spec


def read_spec(path: str | Path, *, require_ready: bool = True) -> dict:
    return validate_spec(json.loads(Path(path).read_text(encoding="utf-8")), require_ready=require_ready)


def verify_source(spec: dict, converted: str | Path, checkpoint: str | Path | None = None) -> dict:
    """Verify the immutable conversion/checkpoint identity selected by *spec*.

    Full tensor payload hashes are verified by the checkpoint implementation;
    this function also binds the counters in ``state.pt`` to release metadata.
    """
    from arcus3.checkpoint import digest as checkpoint_digest

    validate_spec(spec)
    converted = Path(converted)
    source = spec["source"]
    parent_sha = checkpoint_digest(converted / "manifest.json")
    if parent_sha != source["parent_manifest_sha256"]:
        raise ValueError("Release parent mismatch")
    if source["kind"] == "conversion-initialization":
        if checkpoint is not None:
            raise ValueError("Initialization release cannot include a trained checkpoint")
        return {"parent_manifest_sha256": parent_sha, "training_updates": 0,
                "input_tokens": 0, "target_tokens": 0}
    if checkpoint is None:
        raise ValueError("Trained release requires an explicit checkpoint")
    checkpoint = Path(checkpoint)
    manifest_sha = checkpoint_digest(checkpoint / "manifest.json")
    if manifest_sha != source["checkpoint_manifest_sha256"]:
        raise ValueError("Selected checkpoint manifest mismatch")
    from arcus3.expanded_checkpoint import verify as verify_checkpoint

    manifest = verify_checkpoint(checkpoint, parent_sha)
    if manifest.get("campaign") != "backbone-adaptation-v1":
        raise ValueError("Release checkpoint is not a backbone-adaptation checkpoint")
    if manifest.get("model_label") != spec["model_label"]:
        # Alpha 3.2.0 predates model_label persistence; its release spec records
        # the legacy exception explicitly rather than weakening all releases.
        if not (spec.get("allow_legacy_missing_model_label") is True and
                manifest.get("model_label") is None):
            raise ValueError("Checkpoint model label mismatch")
    if manifest.get("updates") != source["training_updates"]:
        raise ValueError("Checkpoint update count mismatch")
    import torch

    state = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=True, mmap=True)
    actual = {k: state.get(k) for k in ("updates", "input_tokens", "target_tokens")}
    expected = {"updates": source["training_updates"], "input_tokens": source["input_tokens"],
                "target_tokens": source["target_tokens"]}
    if actual != expected:
        raise ValueError("Checkpoint exposure counters mismatch")
    result = {"parent_manifest_sha256": parent_sha,
              "checkpoint_manifest_sha256": manifest_sha, **actual}
    if spec["model_label"] == "alpha3.2.2":
        from arcus3.learning_rate import identity, validate_state
        cfg = state.get("config", {})
        if (manifest.get("lineage_id") != source["lineage_id"]
                or cfg.get("lineage", {}).get("id") != source["lineage_id"]
                or manifest.get("config_sha256") != source["adaptation_config_sha256"]
                or state.get("config_sha256") != source["adaptation_config_sha256"]
                or identity(cfg) != source["adaptation_config_sha256"]
                or manifest.get("learning_rate_schedule_sha256") != source["learning_rate_schedule_sha256"]
                or identity(cfg.get("learning_rate_schedule")) != source["learning_rate_schedule_sha256"]):
            raise ValueError("Alpha 3.2.2 scheduler lineage mismatch")
        validate_state(state.get("scheduler"), cfg, state["input_tokens"])
        result.update(lineage_id=source["lineage_id"],
                      adaptation_config_sha256=source["adaptation_config_sha256"],
                      learning_rate_schedule_sha256=source["learning_rate_schedule_sha256"])
    return result


def inference_files(package: str | Path) -> list[Path]:
    """Return every permitted flat package file and reject private state."""
    package = Path(package)
    files = [p for p in package.iterdir() if p.is_file() and p.name != "manifest.json"]
    if any(p.is_dir() for p in package.iterdir()):
        raise ValueError("Release package must be flat")
    for path in files:
        lower = path.name.lower()
        if (path.name not in INFERENCE_BASENAMES and path.suffix.lower() not in INFERENCE_SUFFIXES):
            raise ValueError("Non-inference package file: " + path.name)
        if any(part in lower for part in FORBIDDEN_PARTS):
            raise ValueError("Private training material in package: " + path.name)
    if not any(path.suffix == ".safetensors" for path in files):
        raise ValueError("Inference weights are missing")
    return sorted(files, key=lambda p: p.name)


def write_manifest(package: str | Path, spec: dict) -> dict:
    package = Path(package)
    if (package / "manifest.json").exists():
        raise ValueError("Fresh manifest required")
    entries = {p.name: {"sha256": digest(p), "bytes": p.stat().st_size}
               for p in inference_files(package)}
    manifest = {"schema": "arcus3-inference-package-v2", "release": spec, "files": entries}
    (package / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    verify_package(package, expected_spec=spec)
    return manifest


def verify_package(package: str | Path, *, expected_spec: dict | None = None) -> dict:
    package = Path(package)
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "arcus3-inference-package-v2":
        raise ValueError("Unsupported package manifest")
    spec = validate_spec(manifest.get("release", {}))
    if expected_spec is not None and spec != expected_spec:
        raise ValueError("Package release identity mismatch")
    listed = manifest.get("files")
    if not isinstance(listed, dict) or not listed:
        raise ValueError("Empty package manifest")
    actual_names = {p.name for p in inference_files(package)}
    all_names = {p.name for p in package.iterdir() if p.is_file()}
    if all_names != actual_names | {"manifest.json"}:
        raise ValueError("Unexpected package files")
    if actual_names != set(listed):
        raise ValueError("Manifest does not cover the complete package")
    for name, entry in listed.items():
        if Path(name).name != name or set(entry) != {"sha256", "bytes"}:
            raise ValueError("Unsafe manifest entry")
        path = package / name
        if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise ValueError("Package tamper: " + name)
    verification = json.loads((package / "verification.json").read_text(encoding="utf-8"))
    if not verification.get("complete") or verification.get("release") != spec:
        raise ValueError("Package verification mismatch")
    if verification.get("unique_parameters") != spec["unique_parameters"]:
        raise ValueError("Verified parameter count mismatch")
    source = spec["source"]
    expected_source = {"parent_manifest_sha256": source["parent_manifest_sha256"],
                       "input_tokens": source["input_tokens"],
                       "target_tokens": source["target_tokens"]}
    if source["kind"] == "expanded-checkpoint":
        expected_source.update(checkpoint_manifest_sha256=source["checkpoint_manifest_sha256"],
                               updates=source["training_updates"])
        for field in ("lineage_id", "adaptation_config_sha256", "learning_rate_schedule_sha256"):
            if field in source:
                expected_source[field] = source[field]
    else:
        expected_source["training_updates"] = 0
    if verification.get("source") != expected_source:
        raise ValueError("Verified source identity mismatch")
    parity = verification.get("parity", {})
    if not parity or not all(item.get("exact") is True for item in parity.values()):
        raise ValueError("Package did not pass exact inference parity")
    return manifest
