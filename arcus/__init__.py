"""
arcus/__init__.py

Arcus — a from-scratch MoDE foundation model (Mixture-of-Depths + Experts) on a
modern transformer backbone, with a tiktoken tokenizer. boenet's validated mechanism,
modernized and built to scale (the Alpha ladder).

The Qwen-wrapper / upcycle path explored earlier is archived under `legacy/`.
"""

from arcus.mod_core import (
    MoDSelection,
    mod_select,
    ScalarRouter,
    straight_through_gate,
    capacity_penalty,
    pack_kept,
    unpack_kept,
)
from arcus.tokenizer import TiktokenTokenizer, get_tokenizer
from arcus.model_config import ModelConfig, get_config, PRESETS
from arcus.model import ArcusMoDE, MoDEBlock
from arcus.loss import chunked_cross_entropy
from arcus.generate import generate, load_model
from arcus.grow import grow_experts
from arcus.hf_utils import resolve_checkpoint, repo_last_modified

__version__ = "0.1.0"

__all__ = [
    "MoDSelection", "mod_select", "ScalarRouter", "straight_through_gate",
    "capacity_penalty", "pack_kept", "unpack_kept",
    "TiktokenTokenizer", "get_tokenizer",
    "ModelConfig", "get_config", "PRESETS",
    "ArcusMoDE", "MoDEBlock",
    "chunked_cross_entropy",
    "generate", "load_model",
    "grow_experts",
    "resolve_checkpoint", "repo_last_modified",
]
