"""Upload verified Alpha packages as private Hugging Face model repositories."""
import argparse
import json
import os
from pathlib import Path
from huggingface_hub import HfApi


def main(namespace):
    root = Path("artifacts/huggingface")
    packages = json.loads((root / "packages.json").read_text())
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set")
    api = HfApi(token=token)
    who = api.whoami()
    permitted = {who["name"], *(org["name"] for org in who.get("orgs", []))}
    if namespace not in permitted:
        raise ValueError(f"Authenticated identity cannot publish under {namespace}")
    results = []
    for package in packages:
        name = package["name"]
        directory = root / name
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["source_checkpoint"]["sha256"] != package["source_sha256"]:
            raise ValueError("Package manifest identity mismatch")
        repo_id = f"{namespace}/{name}"
        api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
        commit = api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=directory,
                                   commit_message=f"Publish verified private {name} checkpoint")
        info = api.model_info(repo_id=repo_id, files_metadata=True)
        if info.private is not True:
            raise RuntimeError(f"Repository is not private: {repo_id}")
        remote = {sibling.rfilename for sibling in info.siblings}
        missing = (set(manifest["files"]) | {"manifest.json"}) - remote
        if missing:
            raise RuntimeError(f"Remote repository is missing files: {sorted(missing)}")
        results.append({"model": name, "repo_id": repo_id, "url": f"https://huggingface.co/{repo_id}",
                        "private": True, "commit": str(commit)})
        print(json.dumps(results[-1]), flush=True)
    (root / "published.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", required=True)
    args = parser.parse_args()
    main(args.namespace)
