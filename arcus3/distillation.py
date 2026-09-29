"""Coarse-grained KL: top-k teacher probabilities and one residual bucket."""
def targets(logits, k=32):
    import torch
    logp = logits.detach().float().log_softmax(-1)
    values, indices = logp.topk(min(k, logits.shape[-1]-1), dim=-1)
    return {'indices':indices.cpu(), 'probabilities':values.exp().cpu()}


def loss(logits, target, mask):
    import torch
    ids=target['indices'].to(logits.device)
    probs=target['probabilities'].to(logits.device).float()
    if ids.shape != probs.shape or ids.shape[:-1] != logits.shape[:-1] or mask.shape != logits.shape[:-1]:
        raise ValueError('Teacher alignment mismatch')
    if not mask.any() or not torch.isfinite(probs).all() or (probs<0).any() or (probs.sum(-1)>1.00001).any():
        raise ValueError('Invalid teacher probabilities or mask')
    logp=logits.float().log_softmax(-1)
    chosen=logp.gather(-1,ids)
    tail=(1-probs.sum(-1,keepdim=True)).clamp_min(0)
    student_tail=(1-chosen.exp().sum(-1,keepdim=True)).clamp_min(1e-7).log()
    teacher=torch.cat([probs,tail],-1)
    student=torch.cat([chosen,student_tail],-1)
    kl=(teacher*(teacher.clamp_min(1e-7).log()-student)).sum(-1)
    return kl[mask].mean()
