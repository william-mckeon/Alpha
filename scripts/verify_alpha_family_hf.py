"""Independently verify a private Alpha package at an immutable HF revision."""
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
from arcus3.release import verify_package
from scripts.publish_alpha_3 import verify_remote


def main(package, revision):
    package = Path(package)
    manifest = verify_package(package)
    from dotenv import load_dotenv
    load_dotenv(".env", override=True)
    from huggingface_hub import HfApi, hf_hub_download
    receipt = verify_remote(HfApi(), manifest["release"]["repo_id"], package,
                            manifest, revision, hf_hub_download)
    target = package.parent / ("independent-publication-verification-" + revision + ".json")
    target.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verified": True, "private": True, "revision": revision,
                      "receipt": str(target)}), flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--revision", required=True)
    arguments = parser.parse_args()
    main(arguments.package, arguments.revision)
