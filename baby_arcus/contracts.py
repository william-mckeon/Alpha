"""Strict Phase 1 wire records; no model or private-world data in action requests."""
from dataclasses import dataclass
import hashlib
import json
import re

SCHEMA_VERSION = 1
WORLD_VERSION = "baby-grid-v2"
MAX_BODY = 1_048_576

class ContractError(ValueError):
    pass

def canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                          allow_nan=False).encode("utf-8")
    except (ValueError, TypeError) as exc:
        raise ContractError("Value is not finite JSON") from exc

def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def decode(raw):
    try:
        value = json.loads(raw, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)),
                           object_pairs_hook=_unique)
    except (ValueError, UnicodeError) as exc:
        raise ContractError("Malformed, duplicate-key, or nonfinite JSON") from exc
    canonical(value)
    return value

def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate key")
        result[key] = value
    return result

def fields(value, required, optional=()):
    if not isinstance(value, dict) or set(value) - set(required) - set(optional) or set(required) - set(value):
        raise ContractError("Unexpected or missing fields")

def integer(value, low=0, high=2**31-1):
    if type(value) is not int or not low <= value <= high:
        raise ContractError("Integer outside allowed range")
    return value

def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value):
        raise ContractError("Invalid identifier")
    return value

def envelope(value, required, optional=()):
    fields(value, ("schema_version", "request_id", *required), optional)
    if type(value["schema_version"]) is not int or value["schema_version"] != SCHEMA_VERSION:
        raise ContractError("Unsupported schema version")
    identifier(value["request_id"])

@dataclass(frozen=True)
class EpisodeStart:
    request_id: str
    family: str
    seed: int
    max_steps: int = 64
    split: str = "train"
    layout_id: int | None = None
    difficulty: int = 0
    run_id: str | None = None
    checkpoint_id: str | None = None

    @classmethod
    def parse(cls, value):
        envelope(value, ("family", "seed"), ("max_steps","split","layout_id","difficulty","run_id","checkpoint_id"))
        if value["family"] not in ("switch_delivery", "clue_search"):
            raise ContractError("Unknown lesson")
        split = value.get("split","train")
        if split not in ("train","practice","evaluation","reserved"):
            raise ContractError("Invalid split")
        layout = value.get("layout_id")
        if layout is not None:
            integer(layout,0,9)
            allowed = {"train":(0,1,2,3),"practice":(4,5),"evaluation":(6,7),"reserved":(8,9)}
            if layout not in allowed[split]:
                raise ContractError("Layout does not belong to requested population")
        for key in ("run_id","checkpoint_id"):
            if value.get(key) is not None:
                identifier(value[key])
        return cls(value["request_id"], value["family"], integer(value["seed"]),
                   integer(value.get("max_steps", 64), 1, 256),split,layout,
                   integer(value.get("difficulty",0),0,2),value.get("run_id"),value.get("checkpoint_id"))
