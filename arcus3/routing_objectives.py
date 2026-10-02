"""Versioned training-only routing signals; inference stays hard top-1."""

LEGACY = 'selected-probability-v1'
PAIRED = 'paired-output-v2'


def paired_correction(probabilities, reference, student):
    """Zero forward value; router derivative uses the difference of both outputs.

    This is a straight-through mixture surrogate, not an exact derivative of
    argmax or a claim that local output differences prove global superiority.
    Expert gradients still come from hard dispatch and local teaching.
    """
    difference = (student.float() - reference.float()).detach()
    value = probabilities[:, 1:2] * difference
    return value - value.detach()


def configure(model, cfg, initialize=False):
    import torch
    from arcus3.routing import SelectiveExperts
    version = cfg.get('routing_objective', LEGACY)
    if version not in (LEGACY, PAIRED):
        raise ValueError('Unknown routing objective')
    for index, block in enumerate(m for m in model.modules() if isinstance(m, SelectiveExperts)):
        block.routing_objective = version
        if initialize and cfg.get('router_init_std', 0):
            # Local generator: no mutation of the campaign RNG stream.
            generator = torch.Generator(device=block.router.weight.device)
            generator.manual_seed(cfg['seed'] + index)
            with torch.no_grad():
                vector = torch.randn(block.router.in_features, device=block.router.weight.device,
                                     generator=generator) * cfg['router_init_std']
                block.router.weight[0].copy_(-vector)
                block.router.weight[1].copy_(vector)


def reduce_balance(losses, cfg):
    import torch
    values = torch.stack(losses)
    weights = cfg.get('router_layer_weights')
    if weights is None:
        return values.mean()  # Preserve existing runs exactly.
    if len(weights) != len(losses):
        raise ValueError('Router weight/layer count mismatch')
    return (values * values.new_tensor(weights)).sum()
