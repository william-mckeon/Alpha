"""Single-update full added-expert adaptation; original donor tensors stay frozen."""
def update(model, optimizer, row, teacher, cfg):
    import torch
    from arcus3.routing import SelectiveExperts
    from arcus3.distillation import loss as teacher_loss
    blocks=[m for m in model.modules() if isinstance(m,SelectiveExperts)]
    model.train();model.config.use_cache=False
    if cfg.get('activation_checkpointing',False):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    optimizer.zero_grad(set_to_none=True)
    for m in blocks:m.collect_aux=True;m.collect_teaching=True;m.last_aux=None;m.last_teaching=None;m.teaching_chunk_size=cfg.get('teaching_chunk_size',0);m.expert_chunk_size=cfg.get('expert_chunk_size',0)
    try:
        x=torch.tensor([row['input_ids']],device='cuda');y=torch.tensor([row['labels']],device='cuda')
        mask=y[0,1:]!=-100
        if cfg.get('loss_chunk_size',0):
            from torch.utils.checkpoint import checkpoint
            hidden=model.model(input_ids=x,use_cache=False).last_hidden_state[0,:-1]
            count=mask.sum();pieces=[];chunk=cfg['loss_chunk_size']
            for start in range(0,len(hidden),chunk):
                labels=y[0,start+1:start+1+chunk];selected=labels!=-100
                if not selected.any():continue
                target={k:v[start:start+chunk] for k,v in teacher.items()}
                def part_loss(h,labels,selected,indices,probabilities):
                    logits=model.lm_head(h)
                    ce=torch.nn.functional.cross_entropy(logits.float(),labels,reduction='sum',ignore_index=-100)
                    kl=teacher_loss(logits,{'indices':indices,'probabilities':probabilities},selected)*selected.sum()
                    return torch.stack([ce,kl])
                pieces.append(checkpoint(part_loss,hidden[start:start+chunk],labels,selected,target['indices'],target['probabilities'],use_reentrant=False))
            task,distill=torch.stack(pieces).sum(0)/count
        else:
            out=model(input_ids=x,labels=y,use_cache=False)
            task=out.loss.float()
            distill=teacher_loss(out.logits[0,:-1],teacher,mask)
        local=torch.stack([m.last_teaching[0] for m in blocks]).mean()
        gate=torch.stack([m.last_teaching[1] for m in blocks]).mean()
        route=torch.stack([m.last_aux for m in blocks]).mean()
        total=task+cfg['teacher_coefficient']*distill+cfg['expert_coefficient']*local+cfg['gate_coefficient']*gate+cfg['router_coefficient']*route
        if not torch.isfinite(total):raise RuntimeError('Nonfinite loss')
        total.backward()
        parameters=[p for p in model.parameters() if p.requires_grad]
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        gradient_groups={kind:sum(float(p.grad.abs().sum()) for n,p in model.named_parameters() if key in n and p.grad is not None)
                         for kind,key in [('expert','.experts.1.'),('router','.router.'),('gate','.depth_gate.')]}
        optimizer.step()
        return {'task_nll':float(task.detach()),'teacher_kl':float(distill.detach()),'expert_mse':float(local.detach()),
                'gate_bce':float(gate.detach()),'router_aux':float(route.detach()),'gradient_norm':float(norm),
                'gradient_groups':gradient_groups,'routes':[m.last_counts for m in blocks],
                'input_tokens':len(row['input_ids']),'target_tokens':int(mask.sum())}
    finally:
        for m in blocks:m.collect_aux=False;m.collect_teaching=False;m.last_aux=None;m.last_teaching=None
