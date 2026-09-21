"""Signals share one versioned meaning table; sender identity is explicit."""
import json
from pathlib import Path
from baby_arcus.contracts import ContractError

SIGNALS = ("none", "ready", "help", "follow_me", "good", "try_again",
           "door_open", "marker_red", "marker_blue")

def validate_signal(value):
    if value not in SIGNALS:
        raise ContractError("Unknown signal")
    return value

def validate_catalog(path):
    value = json.loads(Path(path).read_text())
    if value != {"schema_version": 1, "signals": list(SIGNALS)}:
        raise ContractError("Signal catalog does not match implemented vocabulary")

def deliver(actions, step):
    return [{"sender": agent, "signal": action.signal, "step": step}
            for agent, action in sorted(actions.items()) if action.signal != "none"]
