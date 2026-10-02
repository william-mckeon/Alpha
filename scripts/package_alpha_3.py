"""Export a manifest-selected Alpha 3 checkpoint as inference-only weights."""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arcus3.config import deadline, check_live
from arcus3.donor import load, verify as verify_donor
from arcus3.release import read_spec, verify_source, write_manifest
from baby_arcus.gpu_job_control import gpu_job


def main(args):
    if os.environ.get("ARCUS3_CONTROLLED_DOCKER") != "1" or not Path("/.dockerenv").exists():
        raise RuntimeError("Docker CUDA required")
    root = Path(__file__).resolve().parents[1]
    spec_path = Path(args.release_spec)
    if not spec_path.is_absolute():
        spec_path = root / spec_path
    spec = read_spec(spec_path)
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    source = verify_source(spec, args.converted, checkpoint)
    verify_donor(args.donor)
    end = deadline(args.deadline)
    output = Path(args.output)
    package = output / "package"
    if package.exists():
        raise ValueError("Fresh package required")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from arcus3.hf_model import Alpha3Config
    from verify_arcus3_conversion import measure, compare

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    with gpu_job():
        torch.cuda.set_per_process_memory_fraction(.7)
        model, tokenizer = load(args.donor, converted=args.converted,
                                expanded=str(checkpoint) if checkpoint else None)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        before = measure(model, tokenizer)
        check_live(end, output)
        base_config = model.config.to_dict()
        base_config.pop("model_type", None)
        base_config.update(
            selected_layers=[3, 7, 11, 15, 19, 23],
            depth_enabled=spec["unique_parameters"] == 2013403142,
            depth_capacity=1.0,
            added_parameters_dtype="float32" if spec["unique_parameters"] == 2013403142 else "inherit",
            arcus_release={
                "model_label": spec["model_label"],
                "release_kind": spec["release_kind"],
                "training_updates": source.get("updates", source.get("training_updates", 0)),
                "input_tokens": source["input_tokens"],
                "target_tokens": source["target_tokens"],
                "source_manifest_sha256": source.get("checkpoint_manifest_sha256", source["parent_manifest_sha256"]),
                "lineage_id": source.get("lineage_id"),
                "adaptation_config_sha256": source.get("adaptation_config_sha256"),
                "learning_rate_schedule_sha256": source.get("learning_rate_schedule_sha256"),
            },
        )
        model.config = Alpha3Config(**base_config)
        model.save_pretrained(package, safe_serialization=True, max_shard_size="1GB")
        tokenizer.save_pretrained(package)
        config = json.loads((package / "config.json").read_text(encoding="utf-8"))
        config.update(
            _name_or_path="",
            architectures=["Alpha3ForCausalLM"],
            auto_map={"AutoConfig": "modeling_alpha3.Alpha3Config",
                      "AutoModelForCausalLM": "modeling_alpha3.Alpha3ForCausalLM"},
        )
        (package / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        shutil.copyfile(root / "arcus3/hf_model.py", package / "modeling_alpha3.py")
        shutil.copyfile(root / "arcus3/routing.py", package / "routing.py")
        shutil.copyfile(root / "arcus3/depth.py", package / "depth.py")
        shutil.copyfile(root / spec["model_card"], package / "README.md")
        shutil.copyfile(root / "NOTICE", package / "NOTICE")
        donor_license = Path(args.donor) / "files/LICENSE"
        shutil.copyfile(donor_license if donor_license.exists() else root / "LICENSE", package / "LICENSE")
        shutil.copyfile(Path(args.donor) / "files/README.md", package / "DONOR_MODEL_CARD.md")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        check_live(end, output)
        model, loading = AutoModelForCausalLM.from_pretrained(
            package, trust_remote_code=True, local_files_only=True,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            attn_implementation="sdpa", output_loading_info=True,
        )
        model = model.to("cuda").eval()
        tokenizer = AutoTokenizer.from_pretrained(package, local_files_only=True)
        if any(loading[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError("Export loading mismatch")
        parameter_count = sum(p.numel() for p in model.parameters())
        if parameter_count != spec["unique_parameters"]:
            raise ValueError("Export size mismatch")
        parity = compare(before, measure(model, tokenizer))
        if not all(item.get("exact") is True for item in parity.values()):
            raise ValueError("Export inference parity failed")
        report = {
            "complete": True, "release": spec, "source": source,
            "parity": parity, "loading": loading,
            "unique_parameters": parameter_count,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        }
        (package / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        shutil.rmtree(package / "__pycache__", ignore_errors=True)
        manifest = write_manifest(package, spec)
        (output / "package-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"complete": True, "model_label": spec["model_label"],
                          "files": len(manifest["files"]),
                          "bytes": sum(item["bytes"] for item in manifest["files"].values())}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--donor", default="/donor")
    parser.add_argument("--converted", required=True)
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--checkpoint")
    checkpoint_group.add_argument("--expanded", dest="checkpoint")
    parser.add_argument("--release-spec", default="configs/arcus3/alpha_3_release.json")
    parser.add_argument("--output", default="/output")
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    main(parser.parse_args())
