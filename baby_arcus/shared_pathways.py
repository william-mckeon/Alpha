"""Reversible expert-neuron probes for an isolated, single-threaded learner.

Neuron identity is (layer, expert, SwiGLU hidden channel). These interventions
are research instruments, not a new production routing policy.
"""
from contextlib import AbstractContextManager
import random
import torch
from torch.nn import functional as F


class NeuronProbe(AbstractContextManager):
    def __init__(self, model, masks=None, collect=False):
        self.modules = [block.moe.experts for block in model.core.blocks]
        self.masks = masks or {}
        self.collect = collect
        self.handles = []
        self.activation = {}
        self.salience = {}
        for layer, mask in self.masks.items():
            if type(layer) is not int or not 0 <= layer < len(self.modules):
                raise ValueError('Invalid layer')
            module = self.modules[layer]
            if mask.dtype != torch.bool or tuple(mask.shape) != (module.n_experts, module.hidden):
                raise ValueError('Neuron masks must be boolean [experts, hidden]')

    def __enter__(self):
        if self.handles:
            raise RuntimeError('Probe already entered')
        for layer, module in enumerate(self.modules):
            self.handles.append(module.register_forward_hook(self._hook(layer)))
        return self

    def _hook(self, layer):
        def hook(module, args, original):
            buf = args[0]
            batch, experts, capacity, dim = buf.shape
            packed = buf.transpose(0, 1).reshape(experts, batch * capacity, dim)
            hidden = F.silu(torch.bmm(packed, module.gate_proj)) * torch.bmm(packed, module.up_proj)
            if self.collect:
                # Non-gradient upstream parameters are fine: this probes the
                # contribution at this intermediate site, not weight gradients.
                hidden.requires_grad_(True)
                detached = hidden.detach()
                self.activation[layer] = self.activation.get(layer, 0) + detached.abs().sum(1).cpu()
                def gradient(grad):
                    value = (grad * detached).abs().sum(1).cpu()
                    self.salience[layer] = self.salience.get(layer, 0) + value
                hidden.register_hook(gradient)
            if layer in self.masks:
                hidden = hidden.masked_fill(self.masks[layer].to(hidden.device)[:, None, :], 0)
            output = torch.bmm(hidden, module.down_proj)
            return output.reshape(experts, batch, capacity, dim).transpose(0, 1)
        return hook

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def top_masks(scores, fraction):
    if not 0 < fraction <= 1:
        raise ValueError('Selection fraction must be in (0,1]')
    masks = {}
    for layer, score in scores.items():
        if score.ndim != 2 or not torch.isfinite(score).all() or (score < 0).any():
            raise ValueError('Invalid salience scores')
        mask = torch.zeros_like(score, dtype=torch.bool)
        for expert, values in enumerate(score):
            count = min(max(1, round(values.numel() * fraction)), int((values > 0).sum()))
            if count:
                order = torch.argsort(values, descending=True, stable=True)[:count]
                mask[expert, order] = True
        masks[layer] = mask
    return masks


def shared_masks(task_masks, minimum_tasks=2):
    if not 2 <= minimum_tasks <= len(task_masks):
        raise ValueError('Need at least two tasks')
    layers = set.intersection(*(set(m) for m in task_masks))
    return {layer: torch.stack([m[layer] for m in task_masks]).sum(0) >= minimum_tasks for layer in layers}


def matched_random(masks, seed):
    """Same count in every layer/expert; sampling may overlap selected neurons."""
    rng = random.Random(seed)
    result = {}
    for layer, mask in sorted(masks.items()):
        replacement = torch.zeros_like(mask)
        for expert, row in enumerate(mask):
            chosen = rng.sample(range(row.numel()), int(row.sum()))
            replacement[expert, chosen] = True
        result[layer] = replacement
    return result


def neuron_ids(masks):
    return [[layer, expert, channel] for layer, mask in sorted(masks.items())
            for expert, channel in mask.nonzero().tolist()]


class NeuronUpdate(AbstractContextManager):
    """Restore a bounded down-projection update even after an exception.

Only selected neurons' outgoing weights can change. Equal-norm updates allow
matched controls without conflating neuron count and update magnitude.
"""
    def __init__(self, model, masks, norm):
        if not 0 < norm < 1:
            raise ValueError('Update norm must be in (0,1)')
        self.model, self.masks, self.norm = model, masks, norm
        self.saved = []

    def __enter__(self):
        squared = 0.0
        for layer, mask in self.masks.items():
            param = self.model.core.blocks[layer].moe.experts.down_proj
            if param.grad is None:
                continue
            indices = mask.to(param.device)
            gradient = param.grad[indices].detach().clone()
            if not torch.isfinite(gradient).all():
                raise ValueError('Non-finite update')
            squared += float(gradient.double().square().sum())
            self.saved.append((param, indices, param[indices].detach().clone(), gradient))
        scale = self.norm / max(squared ** .5, 1e-30)
        with torch.no_grad():
            for param, indices, original, gradient in self.saved:
                param[indices] = original - gradient * scale
        return self

    def __exit__(self, *exc):
        with torch.no_grad():
            for param, indices, original, gradient in self.saved:
                param[indices] = original
        self.saved.clear()
