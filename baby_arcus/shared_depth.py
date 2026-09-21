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
    if config is not None and config.get('depth_capacity', DEPTH_CAPACITY) != DEPTH_CAPACITY:
        raise ValueError('Shared depth capacity must be 0.25')
    if model.body.cfg.capacity != DEPTH_CAPACITY or model.core.cfg.capacity != DEPTH_CAPACITY or any(
        block.capacity != DEPTH_CAPACITY for block in model.core.blocks
    ):
        raise ValueError('Checkpoint is not qualified for 0.25 depth capacity; migrate and requalify')
    def guard(block,args):
        if block.capacity!=DEPTH_CAPACITY:raise ValueError('An inference path attempted to override the 0.25 depth budget')
    for block in model.core.blocks:
        if not hasattr(block,'_fixed_depth_guard'):
            block._fixed_depth_guard=block.register_forward_pre_hook(guard)
    return DEPTH_CAPACITY


def measurement(model):
    verify_depth(model)
    return {'capacity': DEPTH_CAPACITY,
            'last_routed_fraction': float(model.core.last_compute_fraction),
            'scope': 'MoD expert-token routing; attention remains dense, with integer token rounding'}
