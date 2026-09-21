"""Explicit, auditable budget scopes. Never migrate or reset historical spend."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math

from evaluation.provider import BudgetLedger, EvaluationBlocked


def canonical_path(path):
    """Normalize Windows extended-path spelling after resolving symlinks.

    During concurrent creation Windows can return the same path with or without
    the extended prefix. Containment must compare canonical spellings.
    """
    resolved = str(Path(path).resolve())
    if resolved.startswith("\\\\?\\UNC\\"):
        resolved = "\\\\" + resolved[8:]
    elif resolved.startswith("\\\\?\\"):
        resolved = resolved[4:]
    return Path(resolved)


@dataclass(frozen=True)
class BudgetScope:
    identity: str
    root: Path
    definition: dict

    @property
    def path(self):
        return canonical_path(self.root / self.definition["ledger"])

    def snapshot(self):
        return {"identity": self.identity, **self.definition}

    def ledger(self):
        if self.definition["status"] != "authorized":
            raise EvaluationBlocked(f"budget scope {self.identity} is pending approval; no paid request")
        return BudgetLedger(self.path, self.definition["caps"], self.definition["aggregate_cap"], self.definition.get("token_caps"))


def load_scope(root, identity="diagnostic", *, confirm=None, require_authorized=True):
    root = canonical_path(root)
    config = json.loads((root / "evaluation/budgets.json").read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or identity not in config.get("scopes", {}):
        raise EvaluationBlocked("unknown budget scope")
    paths = set()
    runs_root = canonical_path(root / "evaluation/runs")
    candidates = {item["candidate_id"] for item in json.loads((root / "evaluation/providers.json").read_text())["providers"]}
    for name, value in config["scopes"].items():
        if value.get("status") not in {"authorized", "pending_approval"}:
            raise EvaluationBlocked("invalid budget authorization status")
        path = canonical_path(root / value.get("ledger", ""))
        if not path.is_relative_to(runs_root) or path.suffix != ".json":
            raise EvaluationBlocked(f"budget ledger path is unsafe: {path}; expected below {runs_root}")
        if path in paths:
            raise EvaluationBlocked("budget ledger path is shared by multiple scopes")
        paths.add(path)
        if not value.get("caps") or not set(value["caps"]).issubset(candidates):
            raise EvaluationBlocked("budget scope must cover known frozen candidates")
        if "token_caps" in value:
            if set(value["token_caps"]) != set(value["caps"]):
                raise EvaluationBlocked("token limits must cover every candidate in the scope")
            for limits in value["token_caps"].values():
                if set(limits) != {"input", "output"} or any(type(n) is not int or n <= 0 for n in limits.values()):
                    raise EvaluationBlocked("token limits must be positive integers")
        for amount in [value.get("aggregate_cap"), *value["caps"].values()]:
            if type(amount) not in (float, int) or not math.isfinite(amount) or amount <= 0:
                raise EvaluationBlocked("budget caps must be finite and positive")
        if any(cap > value["aggregate_cap"] for cap in value["caps"].values()):
            raise EvaluationBlocked("model cap exceeds aggregate authorization")
        if path.exists():
            retained = json.loads(path.read_text())
            if retained.get("caps") != value["caps"] or retained.get("aggregate_cap") != value["aggregate_cap"]:
                raise EvaluationBlocked("retained budget caps differ from scope authorization; explicit reconciliation required")
    scope = BudgetScope(identity, root, config["scopes"][identity])
    if confirm is not None and (type(confirm) not in (float, int) or confirm != scope.definition["aggregate_cap"]):
        raise EvaluationBlocked("confirmed budget differs from selected scope authorization")
    if require_authorized and scope.definition["status"] != "authorized":
        raise EvaluationBlocked(f"budget scope {identity} is pending approval; no paid request")
    return scope


def retain_scope(scope, directory, trial_id):
    path = Path(directory) / "budget-scope.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump({"trial_id": trial_id, "scope": scope.snapshot()}, output, indent=2)
    return path


def attach_scope(record, proxy_log, root):
    path = Path(proxy_log).with_name("budget-scope.json")
    if not path.exists():
        return  # Historical fixtures remain readable; current launchers always retain it.
    document = json.loads(path.read_text())
    if document.get("trial_id") != record["run"]["id"]:
        raise ValueError("budget scope belongs to another trial")
    snapshot = document["scope"]
    scope = load_scope(root, snapshot["identity"])
    if snapshot != scope.snapshot():
        raise ValueError("retained budget scope differs from current authorization")
    record["run"]["budget_scope"] = snapshot["identity"]
    record["artifacts"]["budget_scope"] = str(path.resolve())
    record["artifacts"]["sha256"]["budget_scope"] = hashlib.sha256(path.read_bytes()).hexdigest()


def lifetime_accounting(root):
    """Read each unique ledger once; unknown reservations remain separate from actual."""
    config = json.loads((Path(root) / "evaluation/budgets.json").read_text())
    report = {"scopes": {}, "actual_usd": 0.0, "unknown_reserved_usd": 0.0}
    for name in config["scopes"]:
        scope = load_scope(root, name, require_authorized=False)
        ledger = BudgetLedger(scope.path, scope.definition["caps"], scope.definition["aggregate_cap"], scope.definition.get("token_caps"))
        actual = sum(ledger.spent.values())
        unknown = sum(row[1] for row in ledger._reserved.values())
        report["scopes"][name] = {"actual_usd": actual, "unknown_reserved_usd": unknown, "status": scope.definition["status"]}
        report["actual_usd"] += actual
        report["unknown_reserved_usd"] += unknown
    return report
