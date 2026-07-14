"""
scripts/hf_cleanup.py

Reclaim HuggingFace storage by SQUASHING a model repo's git history.

HF is git-based and keeps EVERY version you ever pushed, so a long run with frequent
`--save_every_steps` uploads accumulates hundreds of GB of old `model.safetensors`
checkpoints. `super_squash_history` collapses the whole history into a single commit,
keeping ONLY the latest weights — the accumulated old versions are reclaimed.

  # set your token, then squash the default Arcus repos:
  #   PowerShell:  $env:HF_TOKEN = "<your token>"
  python scripts/hf_cleanup.py

  # or name repos explicitly / add --yes to skip the prompt:
  python scripts/hf_cleanup.py Islanderintel/arcus-alpha-v0.1-1b --yes

WARNING: squashing is IRREVERSIBLE — the intermediate-checkpoint history is dropped
(the LATEST weights are kept). That's normally fine: the latest is what serving/backup need.
"""

import argparse
import os

DEFAULT_REPOS = [
    "Islanderintel/arcus-alpha-v0.1-1b",
    "Islanderintel/arcus-alpha-v0.5-0.5b",
]


def _human(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024 or unit == "TB":
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"


def main() -> None:
    ap = argparse.ArgumentParser(description="Squash HF model-repo history to reclaim storage")
    ap.add_argument("repos", nargs="*", help="repo ids (default: the Arcus 1B + 0.5B repos)")
    ap.add_argument("--token", default=None, help="HF token (else uses the HF_TOKEN env var)")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("no token: set HF_TOKEN in your shell, or pass --token <token>")
    api = HfApi(token=token)
    repos = args.repos or DEFAULT_REPOS

    print("Repos to squash (history -> one commit; only the LATEST weights are kept):\n")
    for repo in repos:
        try:
            info = api.repo_info(repo, repo_type="model", files_metadata=True)
            snap = sum((s.size or 0) for s in (info.siblings or []))
            used = getattr(info, "used_storage", None)
            if used:
                print(f"  - {repo}\n      total storage now ~{_human(used)}  |  latest snapshot ~{_human(snap)}  (the difference is reclaimed)")
            else:
                print(f"  - {repo}\n      latest snapshot ~{_human(snap)}  (accumulated version history is reclaimed)")
        except Exception as exc:
            print(f"  - {repo}   [could not read: {type(exc).__name__}] — "
                  f"is HF_TOKEN set with read access to this repo? (huggingface-cli whoami)")

    if not args.yes:
        ans = input("\nThis is IRREVERSIBLE (old checkpoint history is dropped). Proceed? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("aborted — nothing changed.")
            return

    print()
    for repo in repos:
        try:
            api.super_squash_history(
                repo_id=repo, repo_type="model",
                commit_message="Squash history to reclaim storage (keep latest weights)",
            )
            print(f"[ok]   squashed {repo}")
        except Exception as exc:
            print(f"[fail] {repo}: {type(exc).__name__}: {exc}")

    print("\nDone. HF can take a little while to reflect the reclaimed storage in your usage.")


if __name__ == "__main__":
    main()
