"""Read-only inference overrides, separate from checkpoint/training identity."""
import math


def configure(model, value):
    from arcus.model import MoDEBlock
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError('Invalid evaluation capacity')
    model.eval().requires_grad_(False)
    model.body.cfg.capacity = model.core.cfg.capacity = value
    blocks = [block for block in model.modules() if isinstance(block, MoDEBlock)]
    if not blocks:
        raise ValueError('No routing blocks')
    stats = {'requested_capacity': value, 'blocks': len(blocks), 'block_calls': 0,
             'fraction_sum': 0., 'min_fraction': 1., 'max_fraction': 0.}
    def guard(block, args):
        if block.capacity != value:
            raise ValueError('Evaluation path changed requested capacity')
    def measure(block, args, output):
        fraction = float(block.last_compute_fraction)
        stats['block_calls'] += 1
        stats['fraction_sum'] += fraction
        stats['min_fraction'] = min(stats['min_fraction'], fraction)
        stats['max_fraction'] = max(stats['max_fraction'], fraction)
    for block in blocks:
        if hasattr(block, '_fixed_depth_guard'):
            block._fixed_depth_guard.remove()
        block.capacity = value
        block._fixed_depth_guard = block.register_forward_pre_hook(guard)
        block.register_forward_hook(measure)
    return stats
