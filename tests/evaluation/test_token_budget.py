import tempfile
from pathlib import Path
import pytest
from evaluation.provider import BudgetLedger, TokenBudgetExceeded, EvaluationBlocked


def test_token_limits_are_durable_and_independent():
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "ledger.json"
        caps = {"a": 10, "b": 10}
        tokens = {key: {"input": 100, "output": 50} for key in caps}
        ledger = BudgetLedger(path, caps, 20, tokens)
        rid = ledger.reserve("a", 1, input_tokens=80, output_tokens=40)
        with pytest.raises(TokenBudgetExceeded):
            BudgetLedger(path, caps, 20, tokens).reserve("a", 1, input_tokens=30, output_tokens=1)
        ledger.settle(rid, .5, usage={"prompt_tokens": 70, "completion_tokens": 30})
        reread = BudgetLedger(path, caps, 20, tokens)
        assert reread.token_usage["a"] == [70, 30]
        with pytest.raises(TokenBudgetExceeded):
            reread.reserve("a", 1, input_tokens=1, output_tokens=21)
        other = reread.reserve("b", 1, input_tokens=100, output_tokens=50)
        reread.release(other)
        assert not BudgetLedger(path, caps, 20, tokens).token_reservations


def test_unknown_token_usage_keeps_reservation():
    with tempfile.TemporaryDirectory() as folder:
        ledger = BudgetLedger(Path(folder) / "ledger.json", {"a": 10}, 10, {"a": {"input": 100, "output": 50}})
        rid = ledger.reserve("a", 1, input_tokens=100, output_tokens=50)
        with pytest.raises(EvaluationBlocked, match="omitted"):
            ledger.settle(rid, .5)
        with pytest.raises(TokenBudgetExceeded):
            ledger.reserve("a", 1, input_tokens=1, output_tokens=1)
