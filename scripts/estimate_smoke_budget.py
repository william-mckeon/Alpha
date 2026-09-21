"""Read-only historical token-volume scenario, not a spending authorization."""
import json
import argparse
from collections import defaultdict
from pathlib import Path


def estimate(root, current_results=(), contingency=0.25):
    directory = root / "evaluation/runs/suites/phase1-smoke-full-20260913-01"
    measured = defaultdict(lambda: {"attempts": 0, "prompt_tokens": 0, "completion_tokens": 0})
    for path in sorted(directory.glob("*-u*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        group = measured[record["harness"]["id"]]
        group["attempts"] += 1
        for line in Path(record["artifacts"]["raw_requests"]).read_text(encoding="utf-8").splitlines():
            usage = json.loads(line)["response"]["usage"]
            for key in ("prompt_tokens", "completion_tokens"):
                group[key] += usage[key]
    current = defaultdict(lambda: {"attempts": 0, "prompt_tokens": 0, "completion_tokens": 0})
    revision = json.loads((root / "evaluation/protocol.json").read_text(encoding="utf-8"))["contract_revision"]
    seen = set()
    for path in current_results:
        path = Path(path)
        if path.resolve() in seen:
            raise ValueError("duplicate current measurement")
        seen.add(path.resolve())
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["run"]["contract_revision"] != revision or record["outcome"]["status"] not in {"passed", "failed"}:
            raise ValueError("current measurements must be complete current-contract outcomes")
        group = current[record["harness"]["id"]]
        group["attempts"] += 1
        for line in Path(record["artifacts"]["raw_requests"]).read_text(encoding="utf-8").splitlines():
            usage = json.loads(line)["response"]["usage"]
            for key in ("prompt_tokens", "completion_tokens"):
                group[key] += usage[key]
    basis = {harness: "historical_step" for harness in measured}
    for harness, group in current.items():
        measured[harness] = group
        basis[harness] = "current_contract_diagnostic"
    suite = json.loads((root / "evaluation/suites.json").read_text(encoding="utf-8"))["suites"]["smoke"]
    attempts = json.loads((root / "evaluation/protocol.json").read_text(encoding="utf-8"))["generation"]["attempts"]
    projected = {key: sum(group[key] / group["attempts"] * suite[harness]["count"] * attempts
                          for harness, group in measured.items())
                 for key in ("prompt_tokens", "completion_tokens")}
    providers = json.loads((root / "evaluation/providers.json").read_text())
    scenarios = [{"candidate_id": item["candidate_id"], "nonterminal_usd":
                  (projected["prompt_tokens"] * item["prompt_per_million"] +
                   projected["completion_tokens"] * item["completion_per_million"]) / 1_000_000}
                 for item in providers["providers"]]
    for row, item in zip(scenarios, providers["providers"]):
        row["base_price_scenario_usd"] = row.pop("nonterminal_usd")
        high_prompt = max([item["prompt_per_million"], *[tier["prompt_per_million"] for tier in item.get("pricing_tiers", [])]])
        high_completion = max([item["completion_per_million"], *[tier["completion_per_million"] for tier in item.get("pricing_tiers", [])]])
        row["highest_price_tier_scenario_usd"] = (projected["prompt_tokens"] * high_prompt + projected["completion_tokens"] * high_completion) / 1_000_000
        row["with_contingency_usd"] = row["highest_price_tier_scenario_usd"] * (1 + contingency)
    return {"scenario_only": True, "price_snapshot_at": providers["price_snapshot_at"],
            "measured": dict(measured), "measurement_basis": basis, "projected_tokens_per_candidate": projected,
            "missing_harnesses": sorted({key for key, value in suite.items() if isinstance(value, dict) and "count" in value} - set(measured)),
            "models": scenarios, "aggregate_base_price_scenario_usd": sum(row["base_price_scenario_usd"] for row in scenarios),
            "aggregate_highest_tier_scenario_usd": sum(row["highest_price_tier_scenario_usd"] for row in scenarios),
            "contingency_fraction": contingency,
            "aggregate_with_contingency_usd": sum(row["with_contingency_usd"] for row in scenarios),
            "limitations": ["Same measured token volume assumed for every candidate",
                            "Historical output ceiling was 65536; current ceiling is 32768",
                            "Five repository attempts cover only two of three repository tasks",
                            "Current diagnostic samples may cover only one task per harness",
                            "Missing harnesses are unpriced, not zero; pricing-tier range is not a confidence interval",
                            "Contingency is a planning assumption, not spending authorization or a final-run guarantee"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-result", action="append", type=Path, default=[])
    parser.add_argument("--contingency", type=float, default=.25)
    args = parser.parse_args()
    if not 0 <= args.contingency <= 1:
        parser.error("contingency must be between zero and one")
    print(json.dumps(estimate(Path(__file__).resolve().parents[1], args.current_result, args.contingency), indent=2))
