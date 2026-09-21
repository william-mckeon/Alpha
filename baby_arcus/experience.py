"""Eligible, single-policy PPO batches with separate agent/episode GAE streams."""
from collections import defaultdict
from copy import deepcopy
import math
from baby_arcus.contracts import ContractError
from baby_arcus.objectives import advantages
from baby_arcus.vocabulary import targets

def prepare(records, checkpoint_id, min_samples=1024):
    if len(records)<min_samples:
        raise ContractError("Insufficient samples; no update performed")
    groups = defaultdict(list)
    prepared = deepcopy(records)
    seen = set()
    for row in prepared:
        if row.get("checkpoint_id") != checkpoint_id or row.get("actor") != "agent" or row.get("split") != "train":
            raise ContractError("Stale, non-agent, or held-out experience rejected")
        identity = (row["episode_id"],row["agent_id"],row["step"])
        if row["agent_id"] not in ("a","b") or type(row["step"]) is not int or row["step"]<0:
            raise ContractError("Invalid trajectory identity")
        if len(row["mask"]) != 8 or any(type(v) is not bool for v in row["mask"]) or not any(row["mask"]):
            raise ContractError("Invalid action mask")
        if identity in seen:
            raise ContractError("Duplicate transition")
        seen.add(identity)
        for key in ("logp","value","next_value","reward"):
            if not math.isfinite(row[key]):
                raise ContractError("Nonfinite transition")
        if row["terminated"] and row["truncated"]:
            raise ContractError("Terminal/truncated flags conflict")
        if not row["context"] or not all(type(x) is int and 0<=x<512 for x in row["context"]):
            raise ContractError("Invalid context")
        if not 0<=row["action"]<8 or not 0<=row["signal"]<9:
            raise ContractError("Invalid sampled action")
        if not row["mask"][row["action"]]:
            raise ContractError("Sampled action was masked")
        if row["next_observation"]["agent_id"] != row["agent_id"] or row["next_observation"]["step"] != row["step"]+1:
            raise ContractError("Prediction target is not this agent's next observation")
        row["target"] = targets(row["next_observation"])
        groups[identity[:2]].append(row)
    for group in groups.values():
        group.sort(key=lambda r:r["step"])
        if group[0]["step"] != 0 or any(b["step"] != a["step"]+1 for a,b in zip(group,group[1:])):
            raise ContractError("Incomplete episode trajectory")
        if not (group[-1]["terminated"] or group[-1]["truncated"]):
            raise ContractError("Unfinished episode rejected")
        if any(r["terminated"] or r["truncated"] for r in group[:-1]):
            raise ContractError("Transitions after episode boundary")
        adv,returns = advantages([r["reward"] for r in group],[r["value"] for r in group],
                                 [r["next_value"] for r in group],[r["terminated"] for r in group],
                                 [r["terminated"] or r["truncated"] for r in group])
        for row,a,ret in zip(group,adv,returns):
            row["advantage"],row["return"] = a,ret
    values = [r["advantage"] for r in prepared]
    mean = sum(values)/len(values)
    variance = sum((v-mean)**2 for v in values)/len(values)
    for row in prepared:
        row["advantage"] = (row["advantage"]-mean)/max(variance**0.5,1e-8)
    return prepared
