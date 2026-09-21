"""No-paid-call fault probe of the real owned SDK budget-stop/grading path.

Synthetic gateway state is explicitly retained and can never be qualification
evidence: there are no completed model exchanges to pass the importer audit.
"""
import argparse
import json
import time
from pathlib import Path
from unittest.mock import patch, Mock

import eval_foundations as cli


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--financial", action="store_true", help="zero-funds synthetic ledger; verify must not launch")
    parser.add_argument("--provider", action="store_true", help="synthetic provider overload; no network completion")
    args = parser.parse_args()
    cli.load_credentials(cli.ROOT / ".env")
    protocol = cli.load_json(cli.ROOT / "evaluation/protocol.json")
    original = cli.ArcusProxyServer
    gateways = []

    def exhausted_gateway(*positional, **kwargs):
        server = original(*positional, **kwargs)
        gateways.append(server)
        if args.provider:
            from evaluation.provider import ProviderRequestError
            server.client = Mock()
            server.client.forward.side_effect = ProviderRequestError(429, "synthetic engine overload", {"fault_injection": True, "metadata": {"provider_error_code": "engine_overloaded"}})
        elif args.financial:
            from evaluation.provider import BudgetLedger
            server.ledger = BudgetLedger(cli.ROOT / "evaluation/runs/proxy" / args.run_id / "synthetic-budget.json", {"step-3.5-flash": 0}, 48)
        else:
            server.model_requests = protocol["generation"]["max_agent_iterations"]
            server.first_model_request_at = time.monotonic()
        return server

    run = argparse.Namespace(candidate="step-3.5-flash", task="django__django-14170", run_id=args.run_id, confirm_budget=48, port=8014)
    with patch.object(cli, "ArcusProxyServer", side_effect=exhausted_gateway):
        code = cli._run_repository_diagnostic(run)
    directory = cli.ROOT / "evaluation/runs/repository" / args.run_id
    proxy = cli.ROOT / "evaluation/runs/proxy" / args.run_id
    exchanges = proxy / "proxy-exchanges.jsonl"
    stop = directory / "budget-stop.json"
    if args.financial or args.provider:
        manifest = json.loads((directory / "validation.json").read_text())
        expected = "gateway_provider" if args.provider else "financial_budget"
        passed = code == 1 and manifest.get("stop_class") == expected and not (directory / "verify.log").exists() and not (exchanges.exists() and exchanges.stat().st_size)
        report = {"fault_injection": True, "qualification_eligible": False, "paid_model_calls": 0, "stop_class": manifest.get("stop_class"), "verifier_not_launched": not (directory / "verify.log").exists(), "passed": passed}
        if args.provider:
            report["forward_attempts"] = gateways[0].client.forward.call_count
            report["passed"] = passed and report["forward_attempts"] == 1
            passed = report["passed"]
        with (directory / "fault-probe.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
        print(json.dumps(report, indent=2))
        return int(not passed)
    report = {"fault_injection": True, "qualification_eligible": False, "paid_model_calls": 0, "synthetic_model_request_counter": protocol["generation"]["max_agent_iterations"], "worker_exit_code": code, "budget_stop_retained": stop.is_file(), "prediction_and_official_grading_completed": code == 0}
    if exchanges.exists() and exchanges.stat().st_size:
        raise RuntimeError("fault probe unexpectedly completed a model exchange")
    with (directory / "fault-probe.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))
    return int(code != 0 or not stop.is_file())


if __name__ == "__main__":
    raise SystemExit(main())
