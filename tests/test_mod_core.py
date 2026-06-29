"""
tests/test_mod_core.py

Phase 1 gate for the ported MoD core. Proves the selection matches a brute-force
reference, is causal, respects the fixed buffer, packs/unpacks losslessly, and the
straight-through gate passes the router's gradient. Pure tensor — runs on CPU, no
Qwen, no GPU needed.
"""

import math

import torch

from arcus.mod_core import (
    mod_select,
    ScalarRouter,
    straight_through_gate,
    pack_kept,
    unpack_kept,
)


def _reference_select(scores, capacity):
    """Brute-force pure-Python reference for mod_select (small tensors)."""
    B, T = scores.shape
    keep = torch.zeros(B, T, dtype=torch.bool)
    slot = torch.zeros(B, T, dtype=torch.long)
    kmax = max(1, int(capacity * T + 1.0 - 1e-9))
    for b in range(B):
        count = 0  # exclusive prefix count of keep_raw == boenet's `excl`
        for t in range(T):
            rank = sum(1 for j in range(t + 1) if scores[b, j].item() > scores[b, t].item())
            budget = max(1, math.ceil(capacity * (t + 1)))
            keep_raw = rank < budget
            excl = count
            slot[b, t] = min(excl, kmax - 1)
            keep[b, t] = keep_raw and (excl < kmax)
            if keep_raw:
                count += 1
    return keep, slot, kmax


def test_matches_reference():
    torch.manual_seed(0)
    for capacity in (0.25, 0.5, 0.75, 1.0):
        scores = torch.randn(3, 17)
        sel = mod_select(scores, capacity)
        ref_keep, ref_slot, ref_kmax = _reference_select(scores, capacity)
        assert sel.kmax == ref_kmax
        assert torch.equal(sel.keep, ref_keep), f"keep mismatch at capacity={capacity}"
        assert torch.equal(sel.slot, ref_slot), f"slot mismatch at capacity={capacity}"


def test_selection_is_causal():
    torch.manual_seed(1)
    scores = torch.randn(2, 12)
    sel = mod_select(scores, 0.5)
    perturbed = scores.clone()
    perturbed[:, -1] += 100.0           # change only the last position's score
    sel2 = mod_select(perturbed, 0.5)
    assert torch.equal(sel.keep[:, :-1], sel2.keep[:, :-1]), "future score leaked into the past"
    assert torch.equal(sel.slot[:, :-1], sel2.slot[:, :-1]), "future score changed an earlier slot"


def test_respects_fixed_buffer():
    torch.manual_seed(2)
    scores = torch.randn(4, 20)
    sel = mod_select(scores, 0.5)
    per_seq_kept = sel.keep.sum(dim=1)
    assert (per_seq_kept <= sel.kmax).all(), "kept more tokens than the buffer holds"
    assert sel.keep.float().mean().item() <= 0.5 + 0.15, "kept fraction far above capacity"


def test_capacity_one_keeps_everything():
    scores = torch.randn(2, 9)
    sel = mod_select(scores, 1.0)
    assert sel.kmax == 9
    assert sel.keep.all(), "capacity=1.0 must keep every token (the lossless reduction)"


def test_pack_unpack_roundtrip():
    torch.manual_seed(3)
    x = torch.randn(3, 14, 8)
    scores = torch.randn(3, 14)
    sel = mod_select(scores, 0.5)
    buf = pack_kept(x, sel)
    assert buf.shape == (3, sel.kmax, 8)
    back = unpack_kept(buf, sel, seq_len=14)
    keep = sel.keep.unsqueeze(-1).expand_as(x)
    assert torch.allclose(back[keep], x[keep], atol=1e-6), "kept tokens corrupted by pack/unpack"
    assert torch.all(back[~keep] == 0.0), "skipped tokens should be zero (residual passes them through)"


def test_straight_through_gate_gradient():
    p = torch.rand(2, 10, requires_grad=True)
    keep = (torch.arange(10) % 2 == 0).unsqueeze(0).expand(2, 10).bool()
    gate = straight_through_gate(p, keep)
    assert torch.equal(gate.squeeze(-1).detach(), keep.float()), "gate value must equal hard keep"
    gate.sum().backward()
    assert torch.allclose(p.grad, keep.float()), "gradient must flow to p_soft as d(gate)/d(p)=keep"


def test_router_outputs_probabilities():
    torch.manual_seed(4)
    router = ScalarRouter(dim=16)
    x = torch.randn(2, 7, 16)
    p = router(x)
    assert p.shape == (2, 7)
    assert (p > 0).all() and (p < 1).all(), "router must output probabilities in (0, 1)"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK  {name}")
    print("all mod_core tests passed")
