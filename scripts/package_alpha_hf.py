"""Create verified, inference-only Hugging Face packages for the Alpha family."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from baby_arcus.shared_checkpoint import read_data
from safetensors.torch import load_model, save_model

from baby_arcus.shared_checkpoint import digest, load


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "huggingface"
THREE_WAY = ROOT / "runs" / "test2" / "three-way-8204" / "report.json"
FINAL = ROOT / "runs" / "test2" / "depth100-37000" / "report.json"

MODELS = {
    "Alpha-1.0.0": (ROOT / "runs/test2/depth100-seed-2101", "efa75913a35a499483975736e57f84f6"),
    "Alpha-0.0.0": (ROOT / "runs/arcus_shared_continuity025_v4", "b64d758b7b9f4157a24ffceef1471aa1"),
    "Alpha-0.0.0.25": (ROOT / "runs/test2/depth025-control-seed-2101", "8fbec755bce5440db53b69cee0b00654"),
    "Alpha-0.0.1": (ROOT / "runs/test2/depth100-seed-2101", "b49753d226ac4a5db6a0a2c06df26502"),
}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def metrics(name):
    if name == "Alpha-1.0.0":
        return json.loads(FINAL.read_text())["results"]
    report = json.loads(THREE_WAY.read_text())
    key = {"Alpha-0.0.0": "original", "Alpha-0.0.0.25": "fresh025", "Alpha-0.0.1": "fresh100"}[name]
    return report[key]["results"]


def summary_table(results):
    approach = results["approach"]
    moved = sum(bool(row["success"]) and row["steps"] > 0 for row in approach["records"])
    eligible = sum(row["steps"] > 0 for row in approach["records"])
    command = results["paired:commands"]["successes"] + results["unpaired:commands"]["successes"]
    color = results["paired:color_reference"]["successes"] + results["unpaired:color_reference"]["successes"]
    rest = results["paired:rest"]["successes"] + results["unpaired:rest"]["successes"]
    return f"""| Measurement | Result |
|---|---:|
| Standing | {results['standing']['successes']}/{results['standing']['episodes']} |
| Lying | {results['lying']['successes']}/{results['lying']['episodes']} |
| Sitting | {results['sitting']['successes']}/{results['sitting']['episodes']} |
| Approach requiring movement | {moved}/{eligible} |
| Command decisions | {command}/120 |
| Color decisions | {color}/120 |
| Rest decisions | {rest}/120 |
| Held-out language NLL (lower is better) | {results['language']['nll']:.4f} |
"""


def card(name, metadata, results):
    descriptions = {
        "Alpha-1.0.0": "Selected integrated Alpha release after 37,000 updates.",
        "Alpha-0.0.0": "Retained original Arcus research model assembled from previously trained components.",
        "Alpha-0.0.0.25": "Fresh integrated Alpha experiment at routing capacity 0.25 and 8,204 updates.",
        "Alpha-0.0.1": "Fresh integrated Alpha experiment at routing capacity 1.0 and 8,204 updates.",
    }
    return f"""---
license: apache-2.0
library_name: pytorch
tags:
- custom-code
- mixture-of-experts
- embodied-ai
- research
---

# {name}

{descriptions[name]} Alpha is the underlying model family for the Arcus embodied
application and character. This is a custom PyTorch architecture and is not a
drop-in Transformers `AutoModel` package.

## Architecture

- {metadata['parameters']:,} parameters including all experts
- 8 transformer layers, width 512, 4 experts per layer
- Expert-token routing capacity used by this checkpoint: {metadata['depth_capacity']}
- One shared core for body state, rendered RGB, simulated hearing/language,
  internal state, memory and task heads
- Tiktoken `o200k_base`, tokenizer package version 0.14.0

Routing capacity is separate from model version and parameter count. Adjustable
runtime depth is an architecture goal; autonomous per-task depth selection has
not been established by these weights.

## Measured validation results

{summary_table(results)}
These are small, single-seed simulator cohorts. Command scores compare predefined
decision labels, posture goals are selected externally, and the language result
uses 32 held-out next-token examples. They do not establish general reasoning,
conversational fluency, frontier capability or human-like development.

## Loading

Clone the repository, install the pinned requirements, and run:

```python
from load_alpha import load_alpha
model, metadata = load_alpha(".")
```

The package contains inference weights only. It excludes optimizer state, training
receipts, datasets, caregiver conversations and runtime logs. See `manifest.json`
for the source checkpoint identity and file hashes.

## Intended use

Research on a unified embodied learner and reproducible evaluation inside the
Arcus simulator. Further training and deployment require the Arcus source/runtime.
Do not interpret simulated body signals as real-world safety or robotics validation.
"""


LOADER = '''"""Load an Alpha shared checkpoint from this repository."""
import json
from pathlib import Path
from safetensors.torch import load_model
from arcus.model_config import ModelConfig
from baby_arcus.body_policy import BodyPolicy
from baby_arcus.language_model import LanguageAdapter
from baby_arcus.shared_continuity_model import ContinuityModel

def load_alpha(path, device="cpu"):
    path = Path(path)
    metadata = json.loads((path / "alpha_config.json").read_text())
    body = BodyPolicy(ModelConfig(**metadata["body_config"]), lying=True, sitting=True, approach=True)
    language = LanguageAdapter(body.cfg.dim, metadata["vocab_size"], metadata["text_dim"])
    model = ContinuityModel(body, language, metadata["schema_version"])
    load_model(model, path / "model.safetensors", strict=True, device="cpu")
    model.integrated_motor = metadata["integrated_motor"]
    if metadata.get("experiment_depth_capacity") is not None:
        model.experiment_depth_capacity = metadata["experiment_depth_capacity"]
    model.to(device).eval().requires_grad_(False)
    return model, metadata
'''


def package(name):
    root, generation = MODELS[name]
    source = root / f"{generation}.pt"
    data = read_data(source)
    manifest = {"generation": generation, "sha256": digest(source)}
    model, loaded = load(root, manifest, "cpu")
    if loaded["model"].keys() != model.state_dict().keys():
        raise RuntimeError("Loaded model differs from checkpoint state layout")
    destination = OUTPUT / name
    if destination.exists():
        raise ValueError("Refusing to replace an existing release directory")
    destination.mkdir(parents=True)
    weights = destination / "model.safetensors"
    parameter_count = sum(value.numel() for value in model.parameters())
    save_model(model, weights, metadata={"format": "pt", "family": "Alpha", "version": name.removeprefix("Alpha-")})
    del model
    metadata = {
        "family": "Alpha", "model_name": name, "schema": data["schema"],
        "schema_version": int(data["schema"].rsplit("v", 1)[1]),
        "body_config": data["body_config"], "vocab_size": data["vocab_size"],
        "text_dim": data["text_dim"], "integrated_motor": bool(data.get("integrated_motor", False)),
        "experiment_depth_capacity": data.get("experiment_depth_capacity"),
        "depth_capacity": data["body_config"]["capacity"],
        "parameters": parameter_count,
        "checkpoint_updates": data["progress"].get("updates"),
        "trained_tokens": data["progress"].get("trained_tokens"),
        "encoding": data["progress"].get("encoding", "o200k_base"),
    }
    (destination / "alpha_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (destination / "load_alpha.py").write_text(LOADER, encoding="utf-8")
    (destination / "requirements.txt").write_text("torch==2.11.0\nsafetensors>=0.4.0\ntiktoken==0.14.0\nPillow>=12.0\nnumpy>=2.0\n", encoding="utf-8")
    shutil.copy2(ROOT / "LICENSE", destination / "LICENSE")
    shutil.copytree(ROOT / "arcus", destination / "arcus", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(ROOT / "baby_arcus", destination / "baby_arcus", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "web"))
    result = metrics(name)
    (destination / "README.md").write_text(card(name, metadata, result), encoding="utf-8")
    files = {str(path.relative_to(destination)).replace("\\", "/"): sha256(path)
             for path in destination.rglob("*") if path.is_file() and path.name != "manifest.json"}
    package_manifest = {"model_name": name, "source_checkpoint": manifest,
                        "inference_only": True, "files": files}
    (destination / "manifest.json").write_text(json.dumps(package_manifest, indent=2), encoding="utf-8")
    del data
    # Reconstruct and compare every tensor after serialization.
    original = read_data(source)["model"]
    from arcus.model_config import ModelConfig
    from baby_arcus.body_policy import BodyPolicy
    from baby_arcus.language_model import LanguageAdapter
    from baby_arcus.shared_continuity_model import ContinuityModel
    body = BodyPolicy(ModelConfig(**metadata["body_config"]), lying=True, sitting=True, approach=True)
    verified = ContinuityModel(body, LanguageAdapter(body.cfg.dim, metadata["vocab_size"], metadata["text_dim"]), metadata["schema_version"])
    load_model(verified, weights, strict=True, device="cpu")
    reloaded = verified.state_dict()
    if reloaded.keys() != original.keys() or any(not torch.equal(reloaded[k], original[k]) for k in original):
        raise RuntimeError("Inference package changed model tensors")
    return {"name": name, "directory": str(destination), "weights_sha256": sha256(weights),
            "source_sha256": manifest["sha256"], "files": len(files), "private": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODELS), action="append")
    parser.add_argument("--spec", help="Explicit Alpha 2.0 release specification")
    args = parser.parse_args()
    if args.spec:
        from scripts.release_alpha_2 import package as package_v2
        print(package_v2(json.loads(Path(args.spec).read_text())))
        raise SystemExit(0)
    selected = args.model or list(MODELS)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reports = []
    for model_name in selected:
        print(json.dumps({"packaging": model_name}), flush=True)
        reports.append(package(model_name))
        print(json.dumps(reports[-1]), flush=True)
    index=OUTPUT / "packages.json"
    retained=json.loads(index.read_text()) if index.exists() else []
    index.write_text(json.dumps(retained+reports,indent=2),encoding="utf-8")
