"""Verify private Alpha repositories against local package manifests."""
import hashlib
import json
import os
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    root = Path("artifacts/huggingface")
    published = json.loads((root / "published.json").read_text())
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    results = []
    for item in published:
        local = root / item["model"]
        manifest = json.loads((local / "manifest.json").read_text())
        revision=item.get("revision",item.get("commit"))
        if not revision:raise ValueError("Immutable publication commit required")
        if "/commit/" in revision:revision=revision.rsplit("/",1)[-1]
        info = api.model_info(item["repo_id"], revision=revision, files_metadata=True)
        for name,expected_hash in {**manifest["files"],"manifest.json":sha256(local/"manifest.json")}.items():
            remote=hf_hub_download(item["repo_id"],name,revision=revision,token=os.environ.get("HF_TOKEN"))
            if sha256(remote)!=expected_hash:raise ValueError("Remote checksum mismatch: "+name)
        siblings = {entry.rfilename: entry for entry in info.siblings}
        expected = set(manifest["files"]) | {"manifest.json"}
        missing = expected - set(siblings)
        weights = siblings.get("model.safetensors")
        local_weights = local / "model.safetensors"
        result = {
            "model": item["model"], "repo_id": item["repo_id"],
            "private": info.private is True, "missing": sorted(missing),
            "remote_weight_bytes": weights.size if weights else None,
            "local_weight_bytes": local_weights.stat().st_size,
            "remote_weight_sha256": weights.lfs.sha256 if weights and weights.lfs else None,
            "local_weight_sha256": sha256(local_weights),
            "source_checkpoint_sha256": manifest["source_checkpoint"]["sha256"],
        }
        result["verified"] = (result["private"] and not missing
            and result["remote_weight_bytes"] == result["local_weight_bytes"]
            and result["remote_weight_sha256"] == result["local_weight_sha256"])
        if not result["verified"]:
            raise RuntimeError(json.dumps(result))
        results.append(result)
        print(json.dumps(result), flush=True)
    (root / "remote-verification.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
