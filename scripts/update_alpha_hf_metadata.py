"""Republish a complete verified private package after model-card updates.

This intentionally uses the full publisher: unchanged LFS shards are reused only
after their hashes match, metadata is committed last, and the resulting revision
is fully reverified.  A metadata-only shortcut could leave an unverified release.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.publish_alpha_3 import cli


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    arguments = parser.parse_args()
    raise SystemExit(cli(Path(arguments.package)))
