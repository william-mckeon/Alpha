"""Upload verified Alpha packages as private Hugging Face model repositories."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from huggingface_hub import HfApi


def main(namespace, selected):
    root = Path("artifacts/huggingface")
    packages = json.loads((root / "packages.json").read_text())
    packages=[p for p in packages if p["name"] in selected]
    if {p["name"] for p in packages}!=set(selected):raise ValueError("Unknown selected package")
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
        if api.model_info(repo_id).private is not True:
            raise ValueError("Refusing upload to public repository")
        from scripts.release_alpha_2 import verify_package
        verify_package(directory)
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
                        "private": True, "commit": commit.oid})
        print(json.dumps(results[-1]), flush=True)
    (root / "published.json").write_text(json.dumps((json.loads((root/"published.json").read_text()) if (root/"published.json").exists() else [])+results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace")
    parser.add_argument("--model",action="append")
    parser.add_argument("--spec")
    args = parser.parse_args()
    if args.spec:
        from scripts.release_alpha_2 import publish
        print(json.dumps(publish(json.loads(Path(args.spec).read_text()))))
    else:
        if not args.namespace or not args.model:parser.error("Select --namespace and --model, or --spec")
        main(args.namespace,args.model)
