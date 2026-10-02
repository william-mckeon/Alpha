"""Read-only verification of an immutable Alpha 3 inference package."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arcus3.config import deadline, check_live
from arcus3.donor import load, verify as verify_donor
from arcus3.release import digest, read_spec, verify_package, verify_source
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
    package_manifest = verify_package(args.package, expected_spec=spec)
    verify_donor(args.donor)
    end = deadline(args.deadline)
    output = Path(args.output)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from verify_arcus3_conversion import measure, compare

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    with gpu_job():
        torch.cuda.set_per_process_memory_fraction(.7)
        model, tokenizer = load(args.donor, converted=args.converted,
                                expanded=str(checkpoint) if checkpoint else None)
        before = measure(model, tokenizer)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        check_live(end, output)
        model, loading = AutoModelForCausalLM.from_pretrained(
            args.package, trust_remote_code=True, local_files_only=True,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            attn_implementation="sdpa", output_loading_info=True,
        )
        model = model.to("cuda").eval()
        tokenizer = AutoTokenizer.from_pretrained(args.package, local_files_only=True)
        if any(loading[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError("Export key mismatch")
        count = sum(p.numel() for p in model.parameters())
        if count != spec["unique_parameters"]:
            raise ValueError("Export parameter mismatch")
        parity = compare(before, measure(model, tokenizer))
        if not all(item.get("exact") is True for item in parity.values()):
            raise ValueError("Export inference parity failed")
        report = {
            "complete": True, "release": spec, "source": source,
            "package_manifest_sha256": digest(Path(args.package) / "manifest.json"),
            "package_files": len(package_manifest["files"]), "parity": parity,
            "loading": loading, "unique_parameters": count,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        }
        check_live(end, output)
        (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"complete": True, "model_label": spec["model_label"],
                          "parameters": count}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--donor", default="/donor")
    parser.add_argument("--converted", required=True)
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--checkpoint")
    checkpoint_group.add_argument("--expanded", dest="checkpoint")
    parser.add_argument("--release-spec", default="configs/arcus3/alpha_3_release.json")
    parser.add_argument("--package", required=True)
    parser.add_argument("--output", default="/output")
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    main(parser.parse_args())
