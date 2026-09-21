"""Retain immutable archive snapshots and safely extract fresh trial fixtures."""
from __future__ import annotations

import hashlib
import json
import shutil
import stat
import os
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import zipfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_archive(source: Path, target: Path, *, source_url: str, source_revision: str) -> dict:
    """Copy a downloaded archive once; retain its initial hash, never refetch at trial time."""
    if target.exists() or target.with_suffix(".manifest.json").exists():
        raise ValueError("fixture snapshot already exists")
    with zipfile.ZipFile(source) as archive:
        validate_archive(archive)
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as original, target.open("xb") as output:
        shutil.copyfileobj(original, output)
    evidence = {"archive_sha256": sha256(target), "source_url": source_url, "source_revision": source_revision, "size_bytes": target.stat().st_size}
    with target.with_suffix(".manifest.json").open("x", encoding="utf-8") as manifest:
        json.dump(evidence, manifest, indent=2)
    return evidence


def validate_archive(archive: zipfile.ZipFile, *, max_bytes: int = 2_147_483_648, max_entries: int = 100_000) -> list[tuple[zipfile.ZipInfo, PurePosixPath]]:
    entries = archive.infolist()
    if len(entries) > max_entries or sum(entry.file_size for entry in entries) > max_bytes:
        raise ValueError("fixture archive exceeds the extraction ceiling")
    validated, seen, files = [], set(), set()
    reserved = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
    for entry in entries:
        name = entry.filename
        path = PurePosixPath(name)
        mode = entry.external_attr >> 16
        if not name or path == PurePosixPath(".") or "\\" in name or path.is_absolute() or entry.flag_bits & 1 or any(part in {"..", ""} or ":" in part or part.endswith((".", " ")) or part.split(".")[0].casefold() in reserved for part in path.parts) or stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
            raise ValueError("fixture archive contains an unsafe path or special file")
        canonical = path.as_posix().casefold().rstrip("/")
        if canonical in seen:
            raise ValueError("fixture archive contains duplicate/colliding paths")
        seen.add(canonical)
        if not entry.is_dir():
            files.add(canonical)
        validated.append((entry, path))
    if any(parent.as_posix().casefold() in files for _, path in validated for parent in path.parents if parent != PurePosixPath(".")):
        raise ValueError("fixture archive contains a file/directory path collision")
    return validated


def extract_snapshot(archive_path: Path, destination: Path, *, expected_sha256: str, max_bytes: int = 2_147_483_648) -> dict:
    if destination.exists():
        raise ValueError("trial fixture destination already exists")
    if sha256(archive_path) != expected_sha256:
        raise ValueError("fixture archive hash differs from its frozen snapshot")
    with zipfile.ZipFile(archive_path) as archive:
        entries = validate_archive(archive, max_bytes=max_bytes)
        # Validate every path before creating any destination or writing any files.
        destination.mkdir(parents=True)
        total = 0
        for entry, relative in entries:
            target = destination.joinpath(*relative.parts)
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry) as source, target.open("xb") as output:
                while block := source.read(1_048_576):
                    total += len(block)
                    if total > max_bytes:
                        raise ValueError("fixture archive exceeded its declared size ceiling")
                    output.write(block)
        # Preserve the fixture timestamps used by filesystem-property tasks.
        for entry, relative in reversed(entries):
            target = destination.joinpath(*relative.parts)
            timestamp = datetime(*entry.date_time, tzinfo=timezone.utc).timestamp()
            os.utime(target, (timestamp, timestamp))
            mode = entry.external_attr >> 16
            if mode:
                # Preserve ordinary permission metadata, never setuid/setgid bits.
                os.chmod(target, stat.S_IMODE(mode) & 0o777)
    return {"archive_sha256": expected_sha256, "files": sum(not entry.is_dir() for entry, _ in entries), "uncompressed_bytes": total}
