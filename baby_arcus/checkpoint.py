"""Full training snapshots and transport-portable chunked binary checkpoints."""
from dataclasses import asdict
import base64
import hashlib
import os
from pathlib import Path
import random
import tempfile
import uuid
import numpy as np
import torch
from arcus.model_config import ModelConfig
from baby_arcus.model import BabyModel
from baby_arcus.learner import Learner,LearningConfig
from baby_arcus.vocabulary import VERSION,VOCABULARY_HASH
from baby_arcus.contracts import ContractError

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save(path, learner, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    payload = {"format":1,"vocabulary":VERSION,"vocabulary_hash":VOCABULARY_HASH,
               "model_config":asdict(learner.model.cfg),"learning_config":asdict(learner.config),
               "model":learner.model.state_dict(),"optimizer":learner.optimizer.state_dict(),
               "updates":learner.updates,"processed":sorted(learner.processed),
               "rng_python":random.getstate(),"rng_numpy":np.random.get_state(),
               "rng_torch":torch.get_rng_state(),
               "rng_cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
               "extra":extra or {}}
    descriptor,tmp = tempfile.mkstemp(dir=path.parent,prefix=".checkpoint-")
    os.close(descriptor)
    try:
        with open(tmp,"wb") as handle:
            torch.save(payload,handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

def load(path,device="cpu",training=True,restore_rng=True):
    # Only load our own hash-verified service artifacts. Never an arbitrary upload.
    payload = torch.load(path,map_location="cpu",weights_only=False)
    if payload.get("format") != 1 or payload.get("vocabulary_hash") != VOCABULARY_HASH:
        raise ContractError("Checkpoint/vocabulary incompatibility")
    model = BabyModel(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model"],strict=True)
    model.to(device)
    learner = None
    if training:
        learner = Learner(model,LearningConfig(**payload["learning_config"]))
        learner.optimizer.load_state_dict(payload["optimizer"])
        learner.updates = payload["updates"]
        learner.processed = set(payload["processed"])
    if restore_rng:
        random.setstate(payload["rng_python"])
        np.random.set_state(payload["rng_numpy"])
        torch.set_rng_state(payload["rng_torch"])
        if device.startswith("cuda") and payload["rng_cuda"] is not None:
            torch.cuda.set_rng_state_all(payload["rng_cuda"])
    return model,learner,payload["extra"]

from baby_arcus.binary_artifacts import Repository
