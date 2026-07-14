"""
arcus/hf_utils.py

Shared HuggingFace helpers for the scripts — one place to fetch a checkpoint and ONE legible
failure for the private-repo / missing-or-unscoped-token case. Without this, `snapshot_download`
either prints a 30-line traceback on a 401 or — worse — silently returns a STALE cached snapshot
when the token can't authenticate, which once showed week-old samples as if they were fresh.
`resolve_checkpoint` gates on access first, so that failure is loud instead of silent-and-wrong.

huggingface_hub is imported lazily inside the functions, so `import arcus` stays dependency-light.
"""

from __future__ import annotations

import os


def hf_access_hint() -> str:
    """The one-line 'fix your token' hint, shared by every HF error path."""
    return ('set $env:HF_TOKEN to a token with READ access (run "huggingface-cli whoami" to check; '
            "a fine-grained token must list this repo in its scope)")


def _auth_error(repo: str, exc: Exception) -> SystemExit:
    return SystemExit(
        f"\n[HF] cannot access '{repo}' ({type(exc).__name__}: private, gated, or unauthorized).\n"
        f"     Fix: {hf_access_hint()}."
    )


def repo_last_modified(repo: str, token: str | None = None, required: bool = True):
    """Return the repo's `last_modified` timestamp. If it can't be read: raise a friendly error
    when `required` (the default — don't proceed against a repo you can't see), else return None."""
    token = token or os.environ.get("HF_TOKEN")
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    try:
        return HfApi(token=token).repo_info(repo, repo_type="model").last_modified
    except HfHubHTTPError as exc:
        if required:
            raise _auth_error(repo, exc)
        return None


def resolve_checkpoint(repo: str | None = None, ckpt: str | None = None,
                       token: str | None = None,
                       allow_patterns=("config.json", "*.safetensors")) -> str:
    """Return a local checkpoint directory from EITHER a local `ckpt` path or an HF `repo`.

    For a repo, ACCESS IS VERIFIED FIRST (a `repo_info` call): an auth failure stops here with a
    clear message instead of falling through to `snapshot_download`, which on a 401 can silently
    hand back a stale CACHED snapshot. Then download (only changed files re-download)."""
    if bool(repo) == bool(ckpt):
        raise SystemExit("resolve_checkpoint: pass exactly one of repo= or ckpt=")
    if ckpt:
        return ckpt

    token = token or os.environ.get("HF_TOKEN")
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError

    repo_last_modified(repo, token=token, required=True)          # gate: no silent stale-cache on 401
    try:
        return snapshot_download(repo, token=token, allow_patterns=list(allow_patterns))
    except HfHubHTTPError as exc:
        raise _auth_error(repo, exc)
