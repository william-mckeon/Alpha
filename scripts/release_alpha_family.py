"""Validate and optionally publish one explicitly packaged private Alpha release."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arcus3.release import verify_package


def main(package, publish=False):
    manifest = verify_package(package)
    result = {"validated": True, "model_label": manifest["release"]["model_label"],
              "repo_id": manifest["release"]["repo_id"], "private": True,
              "files": len(manifest["files"]), "published": False}
    if publish:
        from scripts.publish_alpha_3 import main as publish_package, sanitized_failure
        try:
            receipt = publish_package(package)
        except Exception as error:
            sanitized_failure(package, error)
            raise SystemExit(1)
        result.update(published=True, revision=receipt["revision"])
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--publish", action="store_true")
    arguments = parser.parse_args()
    main(Path(arguments.package), arguments.publish)
