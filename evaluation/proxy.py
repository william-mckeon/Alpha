"""Local OpenAI-compatible gateway that enforces Arcus provider and budget pins."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from evaluation.secrets import redact
from evaluation.mcp_client import MCPFailure
from evaluation.budget_scope import load_scope, retain_scope

from evaluation.provider import (
    BudgetLedger,
    EvaluationBlocked,
    FinancialBudgetExceeded,
    OpenRouterClient,
    ProviderRequestError,
    TrialControlStopped,
    load_local_api_key,
    load_provider_configs,
)


ROOT = Path(__file__).resolve().parents[1]


class ArcusProxyServer(ThreadingHTTPServer):
    """HTTP server carrying one immutable candidate configuration."""

    # Closing a trial must wait for accounting/logging of accepted requests.
    # Daemon handlers can otherwise disappear while a paid completion is pending.
    daemon_threads = False
    block_on_close = True

    def __init__(
        self,
        address: tuple[str, int],
        candidate_id: str,
        *,
        trial_id: str,
        artifact_dir: Path | None = None,
        proxy_token: str | None = None,
        search_service=None,
        budget_scope="diagnostic",
    ):
        providers, aggregate_cap = load_provider_configs(ROOT / "evaluation" / "providers.json")
        if candidate_id not in providers:
            raise ValueError(f"unknown candidate: {candidate_id}")
        self.config = providers[candidate_id]
        self.budget_scope = load_scope(ROOT, budget_scope)
        self.ledger = self.budget_scope.ledger()
        self.client = OpenRouterClient()
        self.proxy_token = proxy_token or os.environ.get("ARCUS_PROXY_TOKEN", "")
        if not self.proxy_token:
            raise EvaluationBlocked("ARCUS_PROXY_TOKEN is required for the evaluation gateway")
        self.generation = json.loads(
            (ROOT / "evaluation" / "protocol.json").read_text(encoding="utf-8")
        )["generation"]
        self.trial_id = trial_id
        self.search_service = search_service
        self.artifact_dir = artifact_dir
        if artifact_dir is not None:
            exchange_path = artifact_dir / "proxy-exchanges.jsonl"
            if exchange_path.exists() and exchange_path.stat().st_size:
                raise EvaluationBlocked(
                    f"refusing to append a new trial to existing artifact {exchange_path}"
                )
        self.tool_calls = 0
        self.model_requests = 0
        self.first_model_request_at = None
        self.terminal_error = None
        self.iteration_budget_exhausted = False
        self._state_lock = threading.Lock()
        super().__init__(address, ArcusProxyHandler)
        if artifact_dir is not None:
            try:
                retain_scope(self.budget_scope, artifact_dir, trial_id)
            except Exception:
                self.server_close()
                raise

    def record_exchange(self, exchange: dict[str, Any]) -> None:
        """Count tools and retain a credential-free raw exchange for this isolated trial."""
        exchange = redact(exchange)
        choices = exchange["response"].get("choices") or []
        calls = sum(
            len((choice.get("message") or {}).get("tool_calls") or []) for choice in choices
        )
        with self._state_lock:
            self.tool_calls += calls
            if self.artifact_dir is not None:
                self.artifact_dir.mkdir(parents=True, exist_ok=True)
                path = self.artifact_dir / "proxy-exchanges.jsonl"
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "trial_id": self.trial_id,
                                "tool_calls_total": self.tool_calls,
                                **exchange,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            if self.tool_calls > int(self.generation["tool_call_budget"]):
                raise EvaluationBlocked(
                    f"trial exceeded tool-call budget {self.generation['tool_call_budget']}"
                )

    def record_error(self, category: str, message: str, details=None) -> None:
        with self._state_lock:
            if category in {"financial_budget", "token_budget", "provider", "protocol", "timeout", "infrastructure", "context_capacity"} and self.terminal_error is None:
                self.terminal_error = {"category": category, "error": redact(message, extra_secrets=(self.proxy_token,))}
            if self.artifact_dir is not None:
                self.artifact_dir.mkdir(parents=True, exist_ok=True)
                with (self.artifact_dir / "gateway-errors.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"trial_id": self.trial_id, "category": category, "error": redact(message, extra_secrets=(self.proxy_token,)), **({"details": redact(details, extra_secrets=(self.proxy_token,))} if details is not None else {})}) + "\n")

    def raise_if_stopped(self):
        with self._state_lock:
            if self.terminal_error is not None:
                raise TrialControlStopped(self.terminal_error["category"], self.terminal_error["error"])


class ArcusProxyHandler(BaseHTTPRequestHandler):
    """Serve only the endpoints needed by OpenAI-compatible benchmark clients."""

    server: ArcusProxyServer

    def setup(self):
        super().setup()
        # Bound incomplete/slow client uploads while shutdown drains handlers.
        self.connection.settimeout(180)

    def _json(self, status: int, value: dict[str, Any]) -> None:
        body = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/arcus/control":
            if self.headers.get("Authorization") != f"Bearer {self.server.proxy_token}":
                self._json(401, {"error": {"message": "invalid local evaluation gateway credential"}})
                return
            with self.server._state_lock:
                state = {"trial_id": self.server.trial_id,
                         "model_requests": self.server.model_requests,
                         "max_agent_iterations": self.server.generation["max_agent_iterations"],
                         "iteration_budget_exhausted": self.server.iteration_budget_exhausted,
                         "terminal_error": self.server.terminal_error}
            self._json(200, state)
            return
        if self.path.rstrip("/") == "/health":
            self._json(200, {"status": "ok", "candidate": self.server.config.candidate_id})
            return
        if self.path.rstrip("/") in {"/v1/models", "/models"}:
            self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.server.config.candidate_id,
                            "object": "model",
                            "owned_by": "arcus-evaluation-proxy",
                        }
                    ],
                },
            )
            return
        self._json(404, {"error": {"message": "unsupported endpoint"}})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") not in {"/v1/chat/completions", "/chat/completions", "/arcus/tools"}:
            self._json(404, {"error": {"message": "unsupported endpoint"}})
            return
        if self.headers.get("Authorization") != f"Bearer {self.server.proxy_token}":
            self._json(401, {"error": {"message": "invalid local evaluation gateway credential"}})
            return
        if self.path.rstrip("/") != "/arcus/tools" and self.server.tool_calls >= int(self.server.generation["tool_call_budget"]):
            self._json(400, {"error": {"message": "trial tool-call budget exhausted"}})
            return
        try:
            self.server.raise_if_stopped()
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16_777_216 or self.headers.get("Transfer-Encoding"):
                raise ValueError("request body is missing or exceeds the gateway ceiling")
            self.connection.settimeout(10)
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("request must be a JSON object")
            if self.path.rstrip("/") == "/arcus/tools":
                if self.server.search_service is None:
                    raise ValueError("search service is not enabled for this trial")
                if set(payload) != {"trial_id", "method", "arguments"}:
                    raise ValueError("unsupported search RPC envelope")
                try:
                    result = self.server.search_service.call(payload["method"], payload["arguments"], trial_id=payload["trial_id"])
                except MCPFailure as exc:
                    if self.server.search_service.failed_calls or self.server.search_service.failure:
                        self.server.record_error("infrastructure", "fatal search service failure: " + str(exc), {"stage": "search_rpc"})
                    self._json(502, {"error": {"message": str(exc)}})
                    return
                self._json(200, {"result": result})
                return
            with self.server._state_lock:
                if self.server.terminal_error is not None:
                    raise TrialControlStopped(self.server.terminal_error["category"], self.server.terminal_error["error"])
                if self.server.model_requests >= self.server.generation["max_agent_iterations"]:
                    self.server.iteration_budget_exhausted = True
                    raise EvaluationBlocked("trial exceeded global agent-iteration budget")
                self.server.model_requests += 1
                if self.server.first_model_request_at is None:
                    self.server.first_model_request_at = time.monotonic()
            exchange = self.server.client.forward(
                self.server.config,
                self.server.ledger,
                payload,
                provider_retries=int(self.server.generation["retry_on_provider_error"]),
                generation_contract=self.server.generation,
            )
            self.server.record_exchange(exchange)
            self._json(200, exchange["response"])
        except (EvaluationBlocked, ValueError) as exc:
            category = "financial_budget" if isinstance(exc, FinancialBudgetExceeded) else "model_budget" if isinstance(exc, EvaluationBlocked) and str(exc) == "trial exceeded global agent-iteration budget" else "protocol"
            if getattr(exc, "stop_class", None) == "context_capacity":
                category = "context_capacity"
            if getattr(exc, "stop_class", None) == "token_budget":
                category = "token_budget"
            if isinstance(exc, TrialControlStopped):
                category = "control_stop"
            self.server.record_error(category, str(exc), getattr(exc, "details", None))
            self._json(400, {"error": {"message": str(exc)}})
        except ProviderRequestError as exc:
            self.server.record_error("provider", str(exc), exc.details)
            self._json(exc.status if 400 <= exc.status < 500 else 502, {"error": {"message": str(exc)}})
        except TimeoutError as exc:
            self.server.record_error("timeout", str(exc), {"stage": "provider_request_or_client_io", "classification": "excluded_infrastructure"})
            self._json(502, {"error": {"message": "gateway timeout; retained for infrastructure review"}})
        except Exception as exc:
            self.server.record_error("infrastructure", str(exc))
            self._json(502, {"error": {"message": "gateway infrastructure error; see retained sanitized evidence"}})

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--budget-scope", default="diagnostic")
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    load_local_api_key(ROOT / ".env")
    artifact_dir = args.artifact_dir
    if artifact_dir is not None:
        artifact_dir = artifact_dir if artifact_dir.is_absolute() else ROOT / artifact_dir
        runs = (ROOT / "evaluation" / "runs").resolve()
        if runs not in artifact_dir.resolve().parents:
            raise ValueError("proxy artifacts must be stored below evaluation/runs")
    server = ArcusProxyServer(
        (args.host, args.port),
        args.candidate,
        trial_id=args.trial_id,
        artifact_dir=artifact_dir,
        budget_scope=args.budget_scope,
    )
    print(f"Arcus evaluation proxy ready for {args.candidate} on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
