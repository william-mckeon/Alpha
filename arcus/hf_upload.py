"""
arcus/hf_upload.py

Per-epoch checkpoint save + HuggingFace Hub upload (backup + serving).

After each epoch the trainer calls `save_checkpoint` (local) and, if a repo id is
given, `upload_to_hf`. The repo is overwritten each epoch so it always holds the
LATEST weights — that's your backup if a run dies, and the artifact a server pulls.

Note: ArcusMoDE is a custom architecture; this saves a clean HF-style folder
(safetensors + config.json), but vLLM-style servers need a model plugin for
ArcusMoDE to load it. The upload is valid for backup and custom inference regardless.
"""

from __future__ import annotations

import json
import os


def save_checkpoint(model, model_cfg, encoding: str, out_dir: str) -> str:
    """Write model.safetensors + config.json to `out_dir` (created if needed)."""
    os.makedirs(out_dir, exist_ok=True)
    try:
        from safetensors.torch import save_model
        save_model(model, os.path.join(out_dir, "model.safetensors"))   # handles tied weights
    except Exception:
        import torch
        torch.save(model.state_dict(), os.path.join(out_dir, "pytorch_model.bin"))

    cfg = {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(model_cfg).items()}
    cfg.update({"architecture": "ArcusMoDE", "tokenizer": encoding,
                "model_type": "arcus_mode", "params": int(model.num_parameters())})
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return out_dir


def upload_to_hf(out_dir: str, repo_id: str, token: str | None = None,
                 commit_message: str = "epoch checkpoint") -> None:
    """Upload `out_dir` to `repo_id` (created private if missing). Token from arg or
    HF_TOKEN env. Raises on failure — the trainer catches so a bad upload can't kill
    a run."""
    from huggingface_hub import HfApi
    api = HfApi(token=token or os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id, exist_ok=True, private=True)
    api.upload_folder(folder_path=out_dir, repo_id=repo_id, commit_message=commit_message)
