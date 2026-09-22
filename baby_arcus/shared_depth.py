"""The caregiver's fixed shared-learner routing budget."""
DEPTH_CAPACITY = 0.25


def set_depth(model):
    """Explicit migration only; this changes behavior, not parameter count."""
    model.body.cfg.capacity = DEPTH_CAPACITY
    model.core.cfg.capacity = DEPTH_CAPACITY
    for block in model.core.blocks:
        block.capacity = DEPTH_CAPACITY
    verify_depth(model)


def verify_depth(model, config=None):
    expected = getattr(model, 'experiment_depth_capacity', DEPTH_CAPACITY)
    if expected not in (DEPTH_CAPACITY, 1.0):
        raise ValueError('Unsupported experiment depth capacity')
    if config is not None and config.get('depth_capacity', DEPTH_CAPACITY) != expected:
        raise ValueError(f'Shared depth capacity must be {expected}')
    if model.body.cfg.capacity != expected or model.core.cfg.capacity != expected or any(
        block.capacity != expected for block in model.core.blocks
    ):
        raise ValueError(f'Checkpoint is not qualified for {expected} depth capacity; migrate and requalify')
    def guard(block,args):
        if block.capacity!=expected:raise ValueError(f'An inference path attempted to override the {expected} depth budget')
    for block in model.core.blocks:
        if not hasattr(block,'_fixed_depth_guard'):
            block._fixed_depth_guard=block.register_forward_pre_hook(guard)
    return expected


def measurement(model):
    capacity = verify_depth(model)
    return {'capacity': capacity,
            'last_routed_fraction': float(model.core.last_compute_fraction),
            'scope': 'MoD expert-token routing; attention remains dense, with integer token rounding'}
