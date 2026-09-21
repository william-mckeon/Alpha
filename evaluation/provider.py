"""Pinned OpenRouter access with local hard-budget enforcement."""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import closing, contextmanager
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
import http.client
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from evaluation.secrets import load_credentials, redact


class EvaluationBlocked(RuntimeError):
    """Raised before a request when a prerequisite or budget gate is not satisfied."""


class TrialControlStopped(EvaluationBlocked):
    """Fatal trial outcome: no further paid requests may be accepted."""

    def __init__(self, category, message):
        super().__init__(f"trial stopped by {category}: {message}")
        self.stop_class = "financial_budget" if category == "financial_budget" else "gateway_" + category
        self.details = {"original_category": category}


class FinancialBudgetExceeded(EvaluationBlocked):
    """A local financial stop, never evidence of a model failure."""

    def __init__(self, candidate_id, requested, available):
        super().__init__("request would exceed shared remaining budget")
        self.details = {"candidate_id": candidate_id, "requested_upper_bound_usd": requested, "available_usd": available}


class TokenBudgetExceeded(EvaluationBlocked):
    """A candidate token allowance cannot accommodate the next full request."""
    stop_class = "token_budget"
    candidate_failure = True


class ProviderRequestError(RuntimeError):
    """A provider/API failure that is separate from model behavior."""

    def __init__(self, status: int, message: str, details=None):
        super().__init__(redact(f"OpenRouter HTTP {status}: {message}"))
        self.status = status
        self.details = redact(details) if details is not None else None


def is_retryable_http_status(status: int) -> bool:
    return status in {408, 409, 429, 500, 502, 503, 504}


def load_local_api_key(path: Path) -> bool:
    """Load only OPENROUTER_API_KEY from an ignored local env file, without overriding the host."""
    load_credentials(path)
    if os.environ.get("OPENROUTER_API_KEY", "").strip():
        return True
    if not path.exists():
        return False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == "OPENROUTER_API_KEY" and value.strip():
            os.environ["OPENROUTER_API_KEY"] = value.strip().strip('"').strip("'")
            return True
    return False


@dataclass(frozen=True)
class ProviderConfig:
    candidate_id: str
    model: str
    upstream: str
    provider_slug: str
    quantization: str
    prompt_per_million: float
    completion_per_million: float
    smoke_cap: float
    max_completion_tokens: int | None = None
    context_length: int | None = None
    prompt_cost_ceiling: float | None = None
    completion_cost_ceiling: float | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProviderConfig":
        return cls(
            candidate_id=value["candidate_id"],
            model=value["openrouter_model"],
            upstream=value["upstream"],
            provider_slug=value["provider_slug"],
            quantization=value["quantization"],
            prompt_per_million=float(value["prompt_per_million"]),
            completion_per_million=float(value["completion_per_million"]),
            smoke_cap=float(value["smoke_cap"]),
            max_completion_tokens=value.get("max_completion_tokens"),
            context_length=value.get("context_length"),
            prompt_cost_ceiling=max([float(value["prompt_per_million"]), *[float(tier["prompt_per_million"]) for tier in value.get("pricing_tiers", [])]]),
            completion_cost_ceiling=max([float(value["completion_per_million"]), *[float(tier["completion_per_million"]) for tier in value.get("pricing_tiers", [])]]),
        )


class BudgetLedger:
    """Persist actual API cost and refuse requests that could exceed a configured cap."""

    def __init__(self, path: Path, caps: dict[str, float], aggregate_cap: float, token_caps: dict | None = None):
        self.path = path
        self.caps = caps
        self.aggregate_cap = float(aggregate_cap)
        self.spent = {candidate_id: 0.0 for candidate_id in caps}
        self._reserved: dict[str, tuple[str, float]] = {}
        self.token_caps = token_caps or {}
        self.token_usage = {key: [0, 0] for key in self.token_caps}
        self.token_reservations = {}
        self._lock = threading.RLock()
        self._refresh()

    def _refresh(self):
        if self.path.exists():
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if value.get("token_caps", {}) != self.token_caps:
                raise EvaluationBlocked("retained token limits differ from authorization")
            self.token_usage = value.get("token_usage", {key: [0, 0] for key in self.token_caps})
            self.token_reservations = value.get("token_reservations", {})
            for counts in self.token_usage.values():
                if len(counts) != 2 or any(type(n) is not int or n < 0 for n in counts):
                    raise EvaluationBlocked("invalid retained token usage")
            self.spent = {key: 0.0 for key in self.caps}
            for candidate_id, cost in value.get("spent", {}).items():
                if type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
                    raise EvaluationBlocked("budget ledger contains invalid actual spend")
                self.spent[candidate_id] = float(cost)
            self._reserved = {}
            for identity, reservation in value.get("reservations", {}).items():
                candidate, amount = reservation
                if type(amount) not in (int, float) or not math.isfinite(amount) or amount < 0:
                    raise EvaluationBlocked("budget ledger contains an invalid reservation")
                self._reserved[identity] = (candidate, amount)

    @contextmanager
    def _transaction(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, closing(sqlite3.connect(self.path.with_suffix(".mutex.sqlite3"), timeout=30, isolation_level=None)) as database:
            database.execute("CREATE TABLE IF NOT EXISTS mutex(id INTEGER PRIMARY KEY)")
            database.execute("BEGIN IMMEDIATE")
            try:
                self._refresh()
                yield
                database.execute("COMMIT")
            except BaseException:
                database.execute("ROLLBACK")
                raise

    def _write(self):
        temporary = self.path.with_name(self.path.name + "." + uuid.uuid4().hex + ".tmp")
        with temporary.open("x", encoding="utf-8") as output:
            json.dump({"aggregate_cap": self.aggregate_cap, "caps": self.caps, "spent": self.spent, "reservations": self._reserved, **({"token_caps": self.token_caps, "token_usage": self.token_usage, "token_reservations": self.token_reservations} if self.token_caps else {})}, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        # Windows readers/sync scanners can briefly deny atomic replacement.
        # Retry only this already-written snapshot, under the transaction lock:
        # never repeat a reservation, settlement, or paid provider request.
        delays = (0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 1.0, 1.0, 1.0)
        for attempt in range(len(delays) + 1):
            try:
                os.replace(temporary, self.path)
                return
            except OSError as exc:
                if getattr(exc, "winerror", None) not in {5, 32, 33} or attempt == len(delays):
                    # Preserve the old durable ledger and failed snapshot for
                    # reconciliation. A failed settlement keeps its reservation.
                    raise
                time.sleep(delays[attempt])

    def _remaining(self, candidate_id):
        if candidate_id not in self.caps:
            raise EvaluationBlocked(f"no budget configured for {candidate_id}")
        candidate_reserved = sum(amount for candidate, amount in self._reserved.values() if candidate == candidate_id)
        aggregate_reserved = sum(amount for _, amount in self._reserved.values())
        return max(0.0, min(self.caps[candidate_id] - self.spent[candidate_id] - candidate_reserved, self.aggregate_cap - sum(self.spent.values()) - aggregate_reserved))

    def remaining(self, candidate_id: str) -> float:
        with self._transaction():
            return self._remaining(candidate_id)

    def authorize(self, candidate_id: str, worst_case_cost: float) -> None:
        if type(worst_case_cost) not in (int, float) or not math.isfinite(worst_case_cost) or worst_case_cost < 0:
            raise ValueError("request reservation must be finite and nonnegative")
        if worst_case_cost > self.remaining(candidate_id) + 1e-12:
            raise EvaluationBlocked(
                f"request could cost ${worst_case_cost:.6f}, but only "
                f"${self.remaining(candidate_id):.6f} remains for {candidate_id}"
            )

    def reserve(self, candidate_id: str, worst_case_cost: float, *, input_tokens: int | None = None, output_tokens: int | None = None) -> str:
        """Atomically reserve worst-case spend for one in-flight request."""
        if type(worst_case_cost) not in (int, float) or not math.isfinite(worst_case_cost) or worst_case_cost < 0:
            raise ValueError("request reservation must be finite and nonnegative")
        with self._transaction():
            if self.token_caps:
                if candidate_id not in self.token_caps or any(type(n) is not int or n < 0 for n in (input_tokens, output_tokens)):
                    raise EvaluationBlocked("token-limited requests require a candidate and bounded input/output")
                limits = self.token_caps[candidate_id]
                used = self.token_usage[candidate_id]
                pending = [sum(row[i + 1] for row in self.token_reservations.values() if row[0] == candidate_id) for i in range(2)]
                if any(used[i] + pending[i] + requested > limits[field] for i, (field, requested) in enumerate((("input", input_tokens), ("output", output_tokens)))):
                    raise TokenBudgetExceeded("candidate token allowance cannot fit the next full request; no request sent")
            if worst_case_cost > self._remaining(candidate_id) + 1e-12:
                raise FinancialBudgetExceeded(candidate_id, worst_case_cost, self._remaining(candidate_id))
            reservation_id = uuid.uuid4().hex
            self._reserved[reservation_id] = (candidate_id, worst_case_cost)
            if self.token_caps:
                self.token_reservations[reservation_id] = (candidate_id, input_tokens, output_tokens)
            self._write()
            return reservation_id

    def release(self, reservation_id: str) -> None:
        """Release an in-flight reservation after a request fails without a response."""
        with self._transaction():
            self._reserved.pop(reservation_id, None)
            self.token_reservations.pop(reservation_id, None)
            self._write()

    def settle(self, reservation_id: str, actual_cost: float, *, usage: dict | None = None) -> None:
        """Replace a reservation with provider-reported actual spend."""
        if type(actual_cost) not in (int, float) or not math.isfinite(actual_cost) or actual_cost < 0:
            raise ValueError("actual_cost must be finite and nonnegative")
        with self._transaction():
            if self.token_caps:
                counts = [(usage or {}).get(key) for key in ("prompt_tokens", "completion_tokens")]
                if any(type(n) is not int or n < 0 for n in counts):
                    raise EvaluationBlocked("provider omitted valid token usage; reservation retained")
                reserved_tokens = self.token_reservations.get(reservation_id)
                if reserved_tokens is None:
                    raise EvaluationBlocked("missing token reservation")
            reservation = self._reserved.pop(reservation_id, None)
            if reservation is None:
                raise EvaluationBlocked("unknown or already-settled budget reservation")
            candidate_id, authorized = reservation
            overrun = actual_cost > self._remaining(candidate_id) + 1e-9
            self.spent[candidate_id] += actual_cost
            token_overrun = False
            if self.token_caps:
                self.token_reservations.pop(reservation_id)
                self.token_usage[candidate_id] = [a + b for a, b in zip(self.token_usage[candidate_id], counts)]
                token_overrun = any(counts[i] > reserved_tokens[i + 1] for i in range(2))
            self._write()
            if token_overrun:
                raise EvaluationBlocked("provider exceeded token reservation; actual usage retained")
            if overrun or actual_cost > authorized + 1e-9:
                raise EvaluationBlocked(
                    "provider reported a cost above the request's reserved maximum; actual spend retained"
                )

    def record(self, candidate_id: str, actual_cost: float) -> None:
        with self._transaction():
            if type(actual_cost) not in (int, float) or not math.isfinite(actual_cost) or actual_cost < 0:
                raise ValueError("actual_cost must be finite and nonnegative")
            overrun = actual_cost > self._remaining(candidate_id) + 1e-9
            self.spent[candidate_id] += actual_cost
            self._write()
            if overrun:
                raise EvaluationBlocked(
                    "provider reported a cost above the remaining budget; actual spend retained"
                )


def estimate_worst_case_cost(
    config: ProviderConfig, payload: dict[str, Any], max_tokens: int
) -> float:
    """Conservatively bound request cost using at most one token per UTF-8 byte."""
    encoded = json.dumps(payload, ensure_ascii=False)
    prompt_tokens = max(1, len(encoded.encode("utf-8")) + 1024)
    return (
        prompt_tokens * (config.prompt_cost_ceiling if config.prompt_cost_ceiling is not None else config.prompt_per_million)
        + max_tokens * (config.completion_cost_ceiling if config.completion_cost_ceiling is not None else config.completion_per_million)
    ) / 1_000_000


class OpenRouterClient:
    """Minimal non-streaming client; upstream harnesses remain the benchmark runners."""

    def __init__(self, api_key: str | None = None, base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.base_url = base_url.rstrip("/")

    def require_key(self) -> None:
        if not self.api_key.strip():
            raise EvaluationBlocked(
                "OPENROUTER_API_KEY is not set; no paid request was attempted"
            )

    def chat(
        self,
        config: ProviderConfig,
        ledger: BudgetLedger,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.0,
        provider_retries: int = 1,
    ) -> dict[str, Any]:
        payload = {
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        return self.forward(config, ledger, payload, provider_retries=provider_retries)

    def forward(
        self,
        config: ProviderConfig,
        ledger: BudgetLedger,
        payload: dict[str, Any],
        *,
        provider_retries: int = 1,
        generation_contract: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Forward one non-streaming OpenAI-compatible request with immutable routing."""
        self.require_key()
        if generation_contract is not None:
            payload = enforce_generation_contract(payload, generation_contract)
        if payload.get("stream"):
            raise EvaluationBlocked("the Arcus evaluation proxy does not permit streaming")
        messages = payload.get("messages")
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")
        max_tokens = int(payload.get("max_tokens") or payload.get("max_completion_tokens") or 512)
        if config.max_completion_tokens is not None and max_tokens > config.max_completion_tokens:
            raise EvaluationBlocked("frozen output allowance exceeds pinned endpoint completion capacity")
        capacity_screen = None
        if config.context_length is not None:
            # This is a conservative screening bound, not an exact tokenizer count.
            # Preserve prompts and report uncertainty instead of false exclusions.
            prompt_upper_bound = len(json.dumps({"messages": payload["messages"], "tools": payload.get("tools", [])}, ensure_ascii=False).encode("utf-8")) + 1024
            capacity_screen = {"prompt_utf8_upper_bound": prompt_upper_bound, "output_allowance": max_tokens, "context_length": config.context_length, "exact_token_count": False, "fit_proven_by_screen": prompt_upper_bound + max_tokens <= config.context_length}
            # Exceeding an upper bound does NOT prove the actual count exceeds
            # capacity. Do not reject valid prompts using an approximate tokenizer.
        payload = dict(payload)
        payload["model"] = config.model
        payload["stream"] = False
        payload["provider"] = {
            "only": [config.provider_slug],
            "order": [config.provider_slug],
            "allow_fallbacks": False,
            "require_parameters": True,
        }
        estimate = estimate_worst_case_cost(config, payload, max_tokens)
        if ledger.token_caps:
            if not config.context_length:
                raise EvaluationBlocked("token-limited requests require a verified endpoint context limit")
            # Reserve the endpoint's whole context as an input upper bound.
            # This deliberately stops early near the limit rather than assuming
            # byte estimates are exact tokenizer counts or truncating prompts.
            reservation_id = ledger.reserve(config.candidate_id, estimate, input_tokens=config.context_length, output_tokens=max_tokens)
        else:
            reservation_id = ledger.reserve(config.candidate_id, estimate)
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-Title": "Arcus Foundation Evaluation",
            },
            method="POST",
        )
        result = None
        duration = (generation_contract or {}).get("wall_time_seconds", 180)
        deadline = time.monotonic() + duration
        last_rejection = None
        try:
            for attempt in range(provider_retries + 1):
                try:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        if last_rejection is not None:
                            raise ProviderRequestError(*last_rejection)
                        ledger.release(reservation_id)
                        raise TimeoutError("provider request exceeded its absolute execution deadline")
                    with urllib.request.urlopen(request, timeout=remaining) as response:
                        result = json.loads(read_deadlined_response(response, deadline).decode("utf-8"))
                    break
                except urllib.error.HTTPError as exc:
                    raw = read_deadlined_response(exc, deadline).decode("utf-8", errors="replace")
                    error_details = None
                    try:
                        error_details = json.loads(raw).get("error", {})
                        message = error_details.get(
                            "message", "provider request failed"
                        )
                    except json.JSONDecodeError:
                        message = "provider request failed"
                    last_rejection = (exc.code, message, error_details)
                    if attempt < provider_retries and is_retryable_http_status(exc.code):
                        delay = min(5.0, 2.0**attempt)
                        try:
                            delay = max(delay, min(60.0, float(exc.headers.get("Retry-After", delay))))
                        except (AttributeError, TypeError, ValueError):
                            pass
                        if deadline - time.monotonic() > delay:
                            time.sleep(delay)
                            continue
                    raise ProviderRequestError(exc.code, message, error_details) from exc
        except ProviderRequestError:
            # An explicit HTTP rejection did not return a billable completion.
            ledger.release(reservation_id)
            raise
        except Exception:
            # A timeout or malformed response can follow a charged completion.
            # Keep its durable reservation until spend can be reconciled.
            raise
        if result is None:
            ledger.release(reservation_id)
            raise ProviderRequestError(503, "provider retries exhausted")
        usage = result.get("usage") or {}
        if type(usage.get("cost")) not in (int, float) or not math.isfinite(usage["cost"]) or usage["cost"] < 0:
            raise RuntimeError("OpenRouter response omitted usage.cost; budget cannot be reconciled")
        if ledger.token_caps:
            ledger.settle(reservation_id, float(usage["cost"]), usage=usage)
        else:
            ledger.settle(reservation_id, float(usage["cost"]))
        returned_provider = str(result.get("provider", "")).strip()
        if returned_provider.casefold() != config.upstream.casefold():
            raise ProviderRequestError(
                502,
                f"expected upstream {config.upstream}, received {returned_provider or 'unreported'}",
            )
        return {"request": payload, "response": result, **({"capacity_screen": capacity_screen} if capacity_screen else {})}


def read_deadlined_response(response, deadline):
    """Bound real HTTP reads by remaining time, including whitespace keepalives.

    Do not cancel by abandoning accounting threads. An uncertain timeout leaves
    its durable reservation intact. Non-HTTP test transports retain normal read.
    """
    if isinstance(response, urllib.error.HTTPError):
        response = response.fp
    if not isinstance(response, http.client.HTTPResponse):
        return response.read()
    chunks = []
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("provider response exceeded request execution deadline")
        if response.fp is not None:
            response.fp.raw._sock.settimeout(remaining)
        chunk = response.read1(65536)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def enforce_generation_contract(
    payload: dict[str, Any], generation: dict[str, Any]
) -> dict[str, Any]:
    """Return a payload pinned to the frozen Arcus generation contract."""
    resolved = dict(payload)
    if resolved.get("stream") not in (None, False):
        raise EvaluationBlocked("streaming contradicts the frozen evaluation protocol")
    required = {
        "temperature": float(generation["temperature"]),
        "top_p": float(generation["top_p"]),
    }
    for key, expected in required.items():
        if key in resolved and float(resolved[key]) != expected:
            raise EvaluationBlocked(
                f"request {key}={resolved[key]!r} contradicts frozen value {expected!r}"
            )
        resolved[key] = expected
    expected_tokens = int(generation["max_output_tokens"])
    supplied = resolved.get("max_tokens", resolved.get("max_completion_tokens"))
    if supplied is not None and int(supplied) != expected_tokens:
        raise EvaluationBlocked(
            f"request output limit {supplied!r} contradicts frozen value {expected_tokens}"
        )
    resolved.pop("max_completion_tokens", None)
    resolved["max_tokens"] = expected_tokens
    resolved["stream"] = False
    return resolved


def load_provider_configs(path: Path) -> tuple[dict[str, ProviderConfig], float]:
    value = json.loads(path.read_text(encoding="utf-8"))
    configs = {
        item["candidate_id"]: ProviderConfig.from_dict(item)
        for item in value["providers"]
    }
    return configs, float(value["aggregate_smoke_cap"])
