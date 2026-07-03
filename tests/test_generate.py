"""tests/test_generate.py — the sampler: it runs, greedy is deterministic, EOT stops it,
and a saved checkpoint round-trips back into a sampleable model with the head re-tied."""

import pytest

torch = pytest.importorskip("torch")

from arcus.model import ArcusMoDE
from arcus.model_config import get_config
from arcus.generate import generate, load_model


class _StubTok:
    """Minimal tokenizer for a fast CPU test — one id per char over a small vocab, so a
    decoded string's length equals its token count (lets us assert the EOT stop precisely)."""

    def __init__(self, vocab=512, eot=0):
        self.vocab_size = vocab
        self.eot_token = eot

    def encode(self, text, add_eot=False):
        ids = [((ord(c) % (self.vocab_size - 1)) + 1) for c in text] or [1]
        return ids + ([self.eot_token] if add_eot else [])

    def decode(self, ids):
        return "".join(chr(65 + (int(i) % 26)) for i in ids)


def _model(vocab=512):
    torch.manual_seed(0)
    cfg = get_config("tiny", vocab_size=vocab)
    return ArcusMoDE(cfg).eval(), cfg


def test_generate_runs_and_returns_text():
    m, cfg = _model()
    tok = _StubTok(vocab=cfg.vocab_size, eot=0)
    out = generate(m, tok, "hello", max_new_tokens=8, temperature=0.8, top_k=20, seed=0)
    assert isinstance(out, str) and len(out) > 0


def test_greedy_is_deterministic():
    m, cfg = _model()
    tok = _StubTok(vocab=cfg.vocab_size, eot=-1)   # eot=-1 -> never stops early
    a = generate(m, tok, "abc", max_new_tokens=10, temperature=0.0)
    b = generate(m, tok, "abc", max_new_tokens=10, temperature=0.0)
    assert a == b, "greedy decoding must be deterministic"


def test_stops_at_eot():
    m, cfg = _model()
    tok = _StubTok(vocab=cfg.vocab_size, eot=0)
    prompt = "stop"
    n_prompt = len(tok.encode(prompt))
    with torch.no_grad():                          # find the greedy first pick...
        ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
        first = int(m(ids)[0, -1, :].argmax())
    tok.eot_token = first                          # ...and make THAT the stop token
    text = generate(m, tok, prompt, max_new_tokens=50, temperature=0.0, stop_at_eot=True)
    assert len(text) == n_prompt + 1, "generation should stop right after emitting EOT"


def test_top_p_filter_runs():
    m, cfg = _model()
    tok = _StubTok(vocab=cfg.vocab_size, eot=0)
    out = generate(m, tok, "nucleus", max_new_tokens=6, temperature=0.9, top_p=0.9, seed=1)
    assert isinstance(out, str)


def test_checkpoint_round_trip(tmp_path):
    pytest.importorskip("safetensors")
    from arcus.hf_upload import save_checkpoint
    m, cfg = _model()
    out = save_checkpoint(m, cfg, "o200k_base", str(tmp_path), serve_dtype="bf16")
    m2, enc, cfg2 = load_model(out, device="cpu", dtype=torch.float32)
    assert enc == "o200k_base"
    assert (cfg2.dim, cfg2.n_layers, cfg2.n_experts) == (cfg.dim, cfg.n_layers, cfg.n_experts)
    assert m2.head.weight is m2.token_embed.weight, "tied head must be re-established after load"
    tok = _StubTok(vocab=cfg2.vocab_size, eot=0)
    assert isinstance(generate(m2, tok, "hi", max_new_tokens=4, temperature=0.0), str)
