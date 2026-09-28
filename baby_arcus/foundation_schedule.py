"""SmolLM2 token budget adapted to one Arcus GPU; pure scheduling helpers."""
import math
from datetime import datetime, timezone


def accumulation(sequence_length=16384, micro_batch=1, world_size=1, tokens_per_update=1048576):
    values = (sequence_length, micro_batch, world_size, tokens_per_update)
    if any(type(v) is not int or v <= 0 for v in values):
        raise ValueError('Positive integer token geometry required')
    denominator = sequence_length * micro_batch * world_size
    if tokens_per_update % denominator:
        raise ValueError('Global token batch must be exactly divisible')
    return tokens_per_update // denominator


def learning_rate(update, cfg):
    """One-based optimizer update: warmup, stable, linear decay to final zero."""
    if not 1 <= update <= cfg['updates']:
        raise ValueError('Update outside schedule')
    peak = cfg['learning_rate']
    if update <= cfg['warmup_steps']:
        return peak * update / cfg['warmup_steps']
    if update <= cfg['decay_start']:
        return peak
    return max(cfg.get('minimum_lr', 0), peak * (cfg['updates']-update) / cfg['decay_steps'])


def stop_reason(root, deadline, now=None):
    from pathlib import Path
    if (Path(root) / 'pause-training').exists():
        return 'user_pause'
    if deadline:
        end = datetime.fromisoformat(deadline.replace('Z', '+00:00'))
        if end.tzinfo is None:
            raise ValueError('Deadline needs timezone')
        if (now or datetime.now(timezone.utc)) >= end:
            return 'deadline'
    return None


def estimate(seconds, tokens, target_tokens=2097152000000):
    if seconds <= 0 or tokens <= 0 or not math.isfinite(seconds):
        raise ValueError('Invalid measured exposure or duration')
    rate = tokens / seconds
    return {'tokens_per_second': rate, 'training_only_days': target_tokens / rate / 86400,
            'excludes': ['data preparation', 'evaluation', 'saving', 'downtime']}
