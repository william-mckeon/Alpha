"""
arcus/grow.py

The growth operator — grow a trained ArcusMoDE checkpoint into a bigger one by ADDING experts
(the MoE-width dial), reusing everything it has learned. The keystone of the ladder (specs/0010,
specs/0009 Stage 1); private (Arcus Code, specs/0012).

Recipe (near-lossless, NOT bit-identical — see the honest note below): each new expert is a COPY
of an existing expert (round-robin over the current experts) — a WARM start, not random — with
its router row copied and its router bias lowered by `dormant_margin`, so at init it is *dormant*
(its softmax logit sits `dormant_margin` below its source's, contributing ~exp(-margin) to the
gate) yet still TRACKS its source's routing, so training differentiates it.

Why near-lossless and not bit-identical: the gate is a softmax over ALL experts, so adding an
expert inflates the denominator and shrinks every probability — including the source's. There is
no way to add an *active* expert under top-1 + softmax and keep the output exactly. The dormant
margin makes the added contribution ~exp(-margin): at margin >= ~15 the perturbation is below
fp32 precision (effectively bit-identical); at a smaller margin the expert differentiates faster
in exchange for a small, quickly-recovered init bump. Under top-1 the copied expert also never
becomes the argmax at init (it sits a full margin below its source), so it steals no tokens.

Grows the MODEL weights + config only; continue training with a fresh optimizer
(`train_arcus.py --init_from`). Carrying the AdamW moments across a grow (to skip the brief
re-warmup) is a documented follow-up.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from arcus.model_config import ModelConfig

__all__ = ["grow_experts"]

# state-dict suffixes inside one MoE layer (arcus/moe.py: MoELayer.router + BatchedExperts)
_EXPERT_WEIGHTS = ("experts.gate_proj", "experts.up_proj", "experts.down_proj")  # [E, ...] along dim 0


def _source_indices(add: int, old_e: int, source) -> list[int]:
    """Which existing expert each new expert copies. `roundrobin` spreads copies over all experts
    (so no single expert is over-duplicated); an int copies that expert `add` times."""
    if isinstance(source, int):
        if not 0 <= source < old_e:
            raise ValueError(f"source expert {source} out of range [0, {old_e})")
        return [source] * add
    if source == "roundrobin":
        return [i % old_e for i in range(add)]
    raise ValueError(f"source must be 'roundrobin' or an int expert index; got {source!r}")


def grow_experts(state_dict: dict, cfg: ModelConfig, add: int = 1,
                 source="roundrobin", dormant_margin: float = 8.0):
    """Grow every MoE layer by `add` experts (near-lossless@grow). Returns (state_dict', cfg').

    A fresh `ArcusMoDE(cfg')` loads `state_dict'` with no shape error. `dormant_margin` trades
    init-losslessness (high margin ~ bit-identical) against differentiation speed (low margin).
    """
    if add < 1:
        raise ValueError(f"add must be >= 1; got {add}")
    old_e = cfg.n_experts
    new_cfg = replace(cfg, n_experts=old_e + add)
    src = _source_indices(add, old_e, source)

    sd = dict(state_dict)                                    # non-MoE keys pass through unchanged
    # every MoE layer, by the key prefix before ".moe."  (e.g. "blocks.0")
    prefixes = sorted({k.rsplit(".moe.", 1)[0] for k in sd if ".moe." in k})
    for pfx in prefixes:
        for suffix in _EXPERT_WEIGHTS:
            key = f"{pfx}.moe.{suffix}"
            w = sd[key]                                      # [E, ...]
            copies = torch.stack([w[s] for s in src], dim=0)          # [add, ...] warm copies
            sd[key] = torch.cat([w, copies], dim=0)                   # [E+add, ...]
        rw, rb = sd[f"{pfx}.moe.router.weight"], sd[f"{pfx}.moe.router.bias"]   # [E, dim], [E]
        rw_new = torch.stack([rw[s] for s in src], dim=0)             # copy source rows -> track it
        rb_new = torch.stack([rb[s] - dormant_margin for s in src], dim=0)      # dormant: sit below source
        sd[f"{pfx}.moe.router.weight"] = torch.cat([rw, rw_new], dim=0)
        sd[f"{pfx}.moe.router.bias"] = torch.cat([rb, rb_new], dim=0)

    return sd, new_cfg
