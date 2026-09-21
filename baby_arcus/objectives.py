"""PPO/GAE and visible-only prediction math, independent of collection."""
import torch
from torch.distributions import Categorical
from torch.nn import functional as F

def advantages(rewards, values, next_values, terminated, boundaries, gamma=0.99, lam=0.95):
    result = [0.0]*len(rewards)
    carry = 0.0
    for index in reversed(range(len(rewards))):
        bootstrap = 0.0 if terminated[index] else next_values[index]
        delta = rewards[index]+gamma*bootstrap-values[index]
        carry = delta + (0.0 if boundaries[index] else gamma*lam*carry)
        result[index] = carry
    return result,[a+v for a,v in zip(result,values)]

def loss(output, batch, prediction_coefficient=0.1, clip=0.2):
    device = output["value"].device
    def tensor(key,dtype=torch.float32):
        return torch.tensor([r[key] for r in batch],device=device,dtype=dtype)
    action = Categorical(logits=output["action"])
    signal = Categorical(logits=output["signal"])
    logp = action.log_prob(tensor("action",torch.long))+signal.log_prob(tensor("signal",torch.long))
    log_ratio = logp-tensor("logp")
    ratio = log_ratio.exp()
    advantage = tensor("advantage")
    policy = -torch.minimum(ratio*advantage,ratio.clamp(1-clip,1+clip)*advantage).mean()
    value = F.mse_loss(output["value"],tensor("return"))
    entropy = (action.entropy()+signal.entropy()).mean()
    targets = [r["target"] for r in batch]
    cells = torch.tensor([t["cells"] for t in targets],device=device)
    mask = torch.tensor([t["visible"] for t in targets],device=device).unsqueeze(-1)
    cell_loss = (F.binary_cross_entropy_with_logits(output["cells"],cells,reduction="none")*mask).sum() / (mask.sum()*cells.shape[-1]).clamp_min(1)
    inv = torch.tensor([t["inventory"] for t in targets],device=device,dtype=torch.long)
    res = torch.tensor([t["result"] for t in targets],device=device,dtype=torch.long)
    prediction = cell_loss+F.cross_entropy(output["inventory"],inv)+F.cross_entropy(output["result"],res)
    total = policy+0.5*value-0.01*entropy+prediction_coefficient*prediction+output["aux"]
    return total,{"policy":float(policy.detach()),"value":float(value.detach()),
                  "entropy":float(entropy.detach()),"prediction":float(prediction.detach()),
                  "aux":float(output["aux"].detach()),"overflow":output["overflow"],
                  "ratio_max":float(ratio.detach().max()),
                  "approx_kl":float((torch.expm1(log_ratio.detach())-log_ratio.detach()).mean()),
                  "clip_fraction":float(((ratio.detach()-1).abs()>clip).float().mean()),
                  **output.get("routing",{})}
