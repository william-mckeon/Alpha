"""Exact compact top-1 dispatch; optional measured alternative to padded BMM."""
import torch
import torch.nn.functional as F


def compact(experts, x, indices, keep):
    flat=x.reshape(-1,x.shape[-1])
    selected=indices.reshape(-1)
    valid=keep.reshape(-1)
    output=torch.zeros_like(flat)
    for expert in range(experts.n_experts):
        rows=torch.where(valid & (selected==expert))[0]
        values=flat.index_select(0,rows)
        hidden=F.silu(values @ experts.gate_proj[expert]) * (values @ experts.up_proj[expert])
        output=output.index_add(0,rows,hidden @ experts.down_proj[expert])
    return output.reshape_as(x)
