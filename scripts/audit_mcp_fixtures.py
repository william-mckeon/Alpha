"""No-model-call Linux audit of retained MCP archive extraction metadata."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

from evaluation.fixtures import extract_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.output / "manifest.json").exists():
        raise ValueError("fixture audit already exists")
    outcomes = []
    for snapshot in sorted(args.snapshots.glob("*.manifest.json")):
        category = snapshot.name.removesuffix(".manifest.json")
        source = args.snapshots / (category + ".zip")
        expected = json.loads(snapshot.read_text())["archive_sha256"]
        destination = args.output / category
        result = extract_snapshot(source, destination, expected_sha256=expected)
        with zipfile.ZipFile(source) as archive:
            for entry in archive.infolist():
                path = destination.joinpath(*PurePosixPath(entry.filename).parts)
                timestamp = datetime(*entry.date_time, tzinfo=timezone.utc).timestamp()
                if abs(path.stat().st_mtime - timestamp) > 0.001:
                    raise ValueError("fixture timestamp fidelity failed")
                mode = entry.external_attr >> 16
                if mode and stat.S_IMODE(path.stat().st_mode) != (stat.S_IMODE(mode) & 0o777):
                    raise ValueError("fixture ordinary-permission fidelity failed")
        outcomes.append({"category": category, "status": "passed", **result})
    if not outcomes:
        raise ValueError("no fixture snapshots")
    with (args.output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump({"paid_model_calls": 0, "qualification_evidence": False, "archives": outcomes}, stream, indent=2)
    print(f"Audited {len(outcomes)} archives: hashes, safe extraction, ZIP timestamps and ordinary permissions passed.")


if __name__ == "__main__":
    main()
