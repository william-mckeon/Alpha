"""Token-local top-1 routing. No capacity drops or cross-token statistics."""
import copy
import torch
from torch import nn


class SelectiveExperts(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.experts = nn.ModuleList([original, copy.deepcopy(original)])
        weight = original.gate_proj.weight
        self.router = nn.Linear(weight.shape[1], 2, bias=False,
                                device=weight.device, dtype=weight.dtype)
        nn.init.zeros_(self.router.weight)
        self.last_counts = None
        self.collect_aux = False
        self.last_aux = None
        self.collect_teaching = False
        self.last_teaching = None

    def forward(self, hidden):
        if hasattr(self,'depth_gate'): hidden=self.depth_gate(hidden)
        shape = hidden.shape
        flat = hidden.reshape(-1, shape[-1])
        probabilities = self.router(flat.to(self.router.weight.dtype)).float().softmax(-1)
        choices = probabilities.argmax(-1)
        output = torch.empty_like(flat)
        counts = []
        for index, expert in enumerate(self.experts):
            positions = torch.where(choices == index)[0]
            counts.append(positions.numel())
            if positions.numel():
                if index==1 and getattr(self,'expert_chunk_size',0):
                    from torch.utils.checkpoint import checkpoint
                    def run_expert(part,selected_expert=expert):
                        return selected_expert(part.to(selected_expert.gate_proj.weight.dtype)).to(flat.dtype)
                    values=torch.cat([checkpoint(run_expert,flat.index_select(0,p),use_reentrant=False) for p in positions.split(self.expert_chunk_size)])
                else:
                    values = expert(flat.index_select(0, positions).to(expert.gate_proj.weight.dtype) if hasattr(expert.gate_proj,'weight') else flat.index_select(0,positions)).to(flat.dtype)
                # Exactly one in the forward pass. Explicit surrogate gradient;
                # this is not the derivative of the hard argmax decision.
                p = probabilities[positions, index]
                scale = (1 + (p - p.detach())).to(values.dtype)
                output.index_copy_(0, positions, values * scale[:, None])
        self.last_counts = counts  # bounded diagnostic; no retained computation graph
        if self.collect_aux:
            fractions = torch.tensor(counts,device=flat.device,dtype=torch.float32) / flat.shape[0]
            self.last_aux = 2 * (fractions.detach() * probabilities.mean(0)).sum()
        if self.collect_teaching:
            if getattr(self,'teaching_chunk_size',0):
                from torch.utils.checkpoint import checkpoint
                def teach(part):
                    with torch.no_grad():
                        reference=self.experts[0](part.to(self.experts[0].gate_proj.weight.dtype)).float()
                        energy=reference.square().mean(-1).sqrt()
                        baseline=part.float().square().mean(-1).sqrt()
                        target=(energy/(energy+baseline+1e-6)).clamp(.01,.99)
                    student=self.experts[1](part.float()).float()
                    error=(student-reference).square().sum()
                    logits=torch.nn.functional.linear(part.float(),self.depth_gate.weight,self.depth_gate.bias).squeeze(-1)
                    gate=torch.nn.functional.binary_cross_entropy_with_logits(logits,target,reduction='sum')
                    return torch.stack([error,reference.square().sum(),gate])
                pieces=[checkpoint(teach,part,use_reentrant=False) for part in flat.detach().split(self.teaching_chunk_size)]
                error,denominator,gate=torch.stack(pieces).sum(0)
                self.last_teaching=(error/denominator.clamp_min(1e-6*flat.numel()),gate/flat.shape[0])
                return output.reshape(shape)
            # Local frozen-donor FFN teaching, not an assertion of global teacher parity.
            with torch.no_grad():
                reference = self.experts[0](flat.detach().to(self.experts[0].gate_proj.weight.dtype)).float()
                energy = reference.square().mean(-1).sqrt()
                baseline = flat.detach().float().square().mean(-1).sqrt()
                target = (energy / (energy + baseline + 1e-6)).clamp(.01,.99)
            student = self.experts[1](flat.detach().float()).float()
            expert_loss = (student-reference).square().mean() / reference.square().mean().clamp_min(1e-6)
            gate_logits = torch.nn.functional.linear(flat.detach().float(),self.depth_gate.weight,self.depth_gate.bias).squeeze(-1)
            gate_loss = torch.nn.functional.binary_cross_entropy_with_logits(gate_logits,target)
            self.last_teaching = (expert_loss,gate_loss)
        return output.reshape(shape)
