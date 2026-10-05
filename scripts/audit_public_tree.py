"""Fail closed when the public source tree crosses its publication boundary."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = ROOT / "public-files.json"

FORBIDDEN_PATH_PARTS = {
    ".env",
    ".claude",
    "runs",
    "checkpoints",
    "dataset",
    "datasets",
    "teacher-cache",
    "teacher_cache",
    "training",
    "corpus",
    "prompts",
    "transcripts",
    "credentials",
}
FORBIDDEN_SUFFIXES = {
    ".arrow",
    ".bin",
    ".jsonl",
    ".parquet",
    ".pt",
    ".pth",
    ".safetensors",
}
CONTENT_RULES = {
    "absolute Windows user path": re.compile(r"[A-Za-z]:[\\/]Users[\\/]", re.I),
    "private dataset builder": re.compile(
        "dataset" + "forge|local_agent_" + "dataset", re.I
    ),
    "private data pipeline": re.compile(
        "build_alpha_" + "tool_correction_data", re.I
    ),
}
TEXT_SUFFIXES = {"", ".md", ".py", ".toml", ".txt", ".json", ".yml", ".yaml"}


def inventory() -> set[str]:
    return {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts
    }


def audit() -> list[str]:
    expected = set(json.loads(ALLOWLIST.read_text(encoding="utf-8"))["files"])
    actual = inventory()
    errors = []
    for path in sorted(actual - expected):
        errors.append(f"unlisted file: {path}")
    for path in sorted(expected - actual):
        errors.append(f"missing allowlisted file: {path}")

    for relative in sorted(actual):
        path = ROOT / relative
        lowered_parts = {part.lower() for part in Path(relative).parts}
        if lowered_parts & FORBIDDEN_PATH_PARTS:
            errors.append(f"forbidden path: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden payload: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in CONTENT_RULES.items():
            if pattern.search(text):
                errors.append(f"{label}: {relative}")
    return errors


def main() -> int:
    errors = audit()
    if errors:
        print("Public-tree audit failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"Public-tree audit passed ({len(inventory())} allowlisted files).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
