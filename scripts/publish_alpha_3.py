"""Publish one verified Alpha 3 package and verify its private immutable revision."""
import argparse
import json
import logging
import os
from pathlib import Path
import sys

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.CRITICAL)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arcus3.release import digest, verify_package


def upload_metadata(api, repo, package, names):
    return api.upload_folder(
        repo_id=repo, folder_path=str(package), allow_patterns=names,
        commit_message="Complete verified private Alpha inference release",
    )


def staged_weights(api, repo, package, names, expected):
    """Commit and hash-verify one LFS shard at a time before final metadata."""
    info = api.model_info(repo, files_metadata=True)
    if not info.private:
        raise ValueError("Repository must remain private")
    remote = {item.rfilename: item for item in info.siblings}
    for name in names:
        if not name.endswith(".safetensors"):
            continue
        found = remote.get(name)
        if found is not None and found.lfs and found.lfs.sha256 == expected[name]["sha256"]:
            print(json.dumps({"stage": "shard_already_verified", "file": name}), flush=True)
            continue
        if not api.model_info(repo).private:
            raise ValueError("Repository must remain private")
        print(json.dumps({"stage": "uploading_shard", "file": name}), flush=True)
        commit = api.upload_file(
            repo_id=repo, path_or_fileobj=str(package / name), path_in_repo=name,
            commit_message="Stage private Alpha weight shard (release incomplete)",
        )
        current = api.model_info(repo, revision=commit.oid, files_metadata=True)
        match = next(item for item in current.siblings if item.rfilename == name)
        if not current.private or not match.lfs or match.lfs.sha256 != expected[name]["sha256"]:
            raise ValueError("Staged shard verification failed")
        print(json.dumps({"stage": "shard_verified", "file": name,
                          "revision": commit.oid}), flush=True)


def verify_remote(api, repo, package, manifest, revision, download):
    info = api.model_info(repo, revision=revision, files_metadata=True)
    if not info.private:
        raise ValueError("Privacy verification failed")
    siblings = {item.rfilename: item for item in info.siblings}
    names = sorted(manifest["files"]) + ["manifest.json"]
    unexpected = set(siblings) - set(names) - {".gitattributes"}
    if unexpected:
        raise ValueError("Remote repository contains files outside the inference manifest")
    verified = {}
    for name in names:
        if name not in siblings:
            raise ValueError("Remote file missing: " + name)
        expected = digest(package / name)
        item = siblings[name]
        actual = item.lfs.sha256 if item.lfs else digest(download(repo, name, revision=revision))
        if actual != expected:
            raise ValueError("Remote hash mismatch: " + name)
        verified[name] = actual
    return {"repo_id": repo, "revision": revision, "private": True,
            "model_label": manifest["release"]["model_label"],
            "manifest_sha256": verified["manifest.json"], "files": verified,
            "verification": "LFS server SHA256 for LFS objects; downloaded immutable bytes for Git objects"}


def main(package):
    package = Path(package)
    manifest = verify_package(package)
    spec = manifest["release"]
    repo = spec["repo_id"]
    from dotenv import load_dotenv
    load_dotenv(".env", override=True)
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import RepositoryNotFoundError

    api = HfApi()
    try:
        info = api.model_info(repo)
    except RepositoryNotFoundError:
        api.create_repo(repo, private=True, repo_type="model")
        info = api.model_info(repo)
    if not info.private:
        raise ValueError("Repository must be private before upload")
    names = sorted(manifest["files"]) + ["manifest.json"]
    unexpected = {item.rfilename for item in info.siblings} - set(names) - {".gitattributes"}
    if unexpected:
        raise ValueError("Remote repository contains files outside the inference manifest")
    staged_weights(api, repo, package, names, manifest["files"])
    if not api.model_info(repo).private:
        raise ValueError("Repository privacy changed during upload")
    metadata = [name for name in names if not name.endswith(".safetensors")]
    commit = upload_metadata(api, repo, package, metadata)
    receipt = verify_remote(api, repo, package, manifest, commit.oid, hf_hub_download)
    target = package.parent / ("publication-" + commit.oid + ".json")
    target.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verified": True, "repo_id": repo, "revision": commit.oid,
                      "private": True, "receipt": str(target)}), flush=True)
    return receipt


def sanitized_failure(package, error):
    """Record exception types/status only; never serialize signed request URLs."""
    chain = []
    seen = set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        chain.append({"type": type(current).__name__,
                      "http_status": getattr(response, "status_code", None)})
        current = current.__cause__ or current.__context__
    result = {"verified": False, "error_chain": chain}
    (Path(package).parent / "upload-error.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)
    return result


def cli(package):
    try:
        main(package)
        return 0
    except Exception as error:
        sanitized_failure(package, error)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    arguments = parser.parse_args()
    raise SystemExit(cli(arguments.package))
