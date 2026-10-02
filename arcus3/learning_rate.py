"""Token-indexed learning-rate policy for fresh Arcus lineages.

Legacy adaptation configurations intentionally keep their original flat AdamW
parameter list, PyTorch defaults and constant learning rate.  The helpers in
this module only select the token clock when ``learning_rate_schedule`` exists.
"""
import hashlib
import json
import math
import re


TOKEN_WSD_SCHEMA = 'arcus3-token-wsd-v1'
SCHEDULER_STATE_SCHEMA = 'arcus3-token-wsd-state-v1'
LINEAGE_SCHEMA = 'arcus3-fresh-lineage-v1'
GROUP_PATTERNS = {
    'expert': '.experts.1.',
    'router': '.router.',
    'gate': '.depth_gate.',
}


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def uses_token_schedule(cfg):
    return 'learning_rate_schedule' in cfg


def validate_config(cfg):
    """Validate optional optimizer/schedule/lineage fields without changing cfg."""
    schedule = cfg.get('learning_rate_schedule')
    optimizer = cfg.get('optimizer')
    lineage = cfg.get('lineage')
    if schedule is None:
        if optimizer is not None or lineage is not None:
            raise ValueError('Optimizer/lineage extensions require a token schedule')
        return cfg
    if schedule.get('schema') != TOKEN_WSD_SCHEMA or schedule.get('clock') != 'cumulative-input-tokens-after-update':
        raise ValueError('Unsupported learning-rate schedule')
    peak = schedule.get('peak_learning_rate')
    warmup = schedule.get('warmup_input_tokens')
    if not isinstance(warmup, int) or warmup <= 0 or not isinstance(peak, (int, float)) or not math.isfinite(peak) or peak <= 0:
        raise ValueError('Invalid token warmup')
    if peak != cfg.get('learning_rate'):
        raise ValueError('Peak learning rate must match the adaptation learning rate')
    selection = schedule.get('selection', {})
    if selection.get('status') not in ('pending-calibration', 'qualified'):
        raise ValueError('Warmup selection status is required')
    if selection.get('selected_warmup_input_tokens') != warmup:
        raise ValueError('Selected warmup and active schedule differ')
    receipt = selection.get('receipt_sha256')
    if selection['status'] == 'qualified':
        if not isinstance(receipt, str) or not re.fullmatch(r'[0-9a-f]{64}', receipt):
            raise ValueError('Qualified warmup needs an immutable calibration receipt')
    elif receipt is not None or cfg.get('campaign_enabled'):
        raise ValueError('Pending warmup cannot launch a campaign')
    decay = schedule.get('decay', {})
    if decay.get('enabled'):
        start, length = decay.get('start_input_tokens'), decay.get('input_tokens')
        minimum = decay.get('minimum_learning_rate')
        if (decay.get('style') != 'linear' or not schedule.get('endpoint_authorized')
                or not isinstance(start, int) or start < warmup or not isinstance(length, int) or length <= 0
                or not isinstance(minimum, (int, float)) or not 0 <= minimum < peak
                or start + length > cfg['ceiling_input_tokens']):
            raise ValueError('Invalid or unauthorized decay endpoint')
    elif (schedule.get('endpoint_authorized') or any(decay.get(k) is not None for k in
                                                     ('start_input_tokens', 'input_tokens', 'minimum_learning_rate'))):
        raise ValueError('Disabled decay must not imply an endpoint')
    expected_optimizer = {'schema': 'arcus3-adamw-v1', 'betas': [0.9, 0.95], 'epsilon': 1e-8,
                          'weight_decay': 0.01, 'gradient_clip': 1.0,
                          'group_multipliers': {'expert': 1.0, 'router': 1.0, 'gate': 1.0}}
    if optimizer != expected_optimizer:
        raise ValueError('Alpha 3.2.2 optimizer policy mismatch')
    if (not isinstance(lineage, dict) or lineage.get('schema') != LINEAGE_SCHEMA
            or lineage.get('model_label') != cfg.get('model_label')
            or lineage.get('initialized_from') != 'verified-phase5-donor-derived-zero-update'
            or lineage.get('inherited_training_updates') != 0
            or lineage.get('inherited_optimizer_state') is not False):
        raise ValueError('Fresh lineage policy mismatch')
    if not isinstance(lineage.get('id'), str) or not lineage['id']:
        raise ValueError('Fresh lineage id required')
    return cfg


def learning_rate(schedule, input_tokens):
    """LR for an update ending at ``input_tokens``; the first update is nonzero."""
    if not isinstance(input_tokens, int) or input_tokens <= 0:
        raise ValueError('Positive committed token position required')
    peak = float(schedule['peak_learning_rate'])
    warmup = schedule['warmup_input_tokens']
    if input_tokens <= warmup:
        return peak * input_tokens / warmup, 'warmup'
    decay = schedule['decay']
    if not decay['enabled'] or input_tokens <= decay['start_input_tokens']:
        return peak, 'stable'
    progress = min(1.0, (input_tokens - decay['start_input_tokens']) / decay['input_tokens'])
    minimum = float(decay['minimum_learning_rate'])
    if progress >= 1:
        return minimum, 'minimum'
    return peak + (minimum - peak) * progress, 'decay'


def _parameter_groups(model, multipliers):
    groups = {name: [] for name in GROUP_PATTERNS}
    seen = set()
    for parameter_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        matches = [name for name, pattern in GROUP_PATTERNS.items() if pattern in parameter_name]
        if len(matches) != 1 or id(parameter) in seen:
            raise ValueError('Unexpected or duplicate trainable parameter: ' + parameter_name)
        seen.add(id(parameter));groups[matches[0]].append(parameter)
    if any(not values for values in groups.values()):
        raise ValueError('Expert, router and gate parameter groups are all required')
    return [{'params': groups[name], 'lr': 0.0, 'group_name': name,
             'lr_multiplier': multipliers[name]} for name in ('expert', 'router', 'gate')]


def build_optimizer(model, cfg):
    """Build the exact legacy optimizer or the explicit Alpha 3.2.2 optimizer."""
    import torch
    if not uses_token_schedule(cfg):
        # Do not spell out defaults here: old lineages must retain their exact constructor.
        return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=cfg['learning_rate'], foreach=False)
    validate_config(cfg)
    policy = cfg['optimizer'];peak = cfg['learning_rate']
    groups = _parameter_groups(model, policy['group_multipliers'])
    return torch.optim.AdamW(groups, lr=peak, betas=tuple(policy['betas']), eps=policy['epsilon'],
                             weight_decay=policy['weight_decay'], foreach=False)


def initial_state(cfg):
    if not uses_token_schedule(cfg):
        return 'constant-lr'
    validate_config(cfg);schedule = cfg['learning_rate_schedule']
    return {'schema': SCHEDULER_STATE_SCHEMA, 'schedule_sha256': identity(schedule),
            'committed_input_tokens': 0, 'last_applied_input_tokens': 0,
            'phase': 'warmup', 'base_learning_rate': 0.0,
            'group_learning_rates': {name: 0.0 for name in ('expert', 'router', 'gate')}}


def validate_state(state, cfg, input_tokens):
    if not uses_token_schedule(cfg):
        if state not in (None, 'constant-lr'):
            raise ValueError('Legacy constant-LR state changed')
        return True
    validate_config(cfg)
    if (not isinstance(state, dict) or state.get('schema') != SCHEDULER_STATE_SCHEMA
            or state.get('schedule_sha256') != identity(cfg['learning_rate_schedule'])
            or state.get('committed_input_tokens') != input_tokens
            or state.get('last_applied_input_tokens') != input_tokens):
        raise ValueError('Learning-rate scheduler state/config mismatch')
    if input_tokens == 0:
        base, phase = 0.0, 'warmup'
    else:
        base, phase = learning_rate(cfg['learning_rate_schedule'], input_tokens)
    rates = state.get('group_learning_rates')
    expected = {name: base * multiplier for name, multiplier in
                cfg['optimizer']['group_multipliers'].items()}
    if (state.get('phase') != phase or state.get('base_learning_rate') != base
            or not isinstance(rates, dict) or rates != expected):
        raise ValueError('Learning-rate scheduler state values changed')
    return True


def apply_for_update(optimizer, cfg, state, current_input_tokens, update_input_tokens):
    """Set LR before one update and return state to commit only after success."""
    if not uses_token_schedule(cfg):
        return None
    validate_state(state, cfg, current_input_tokens)
    if not isinstance(update_input_tokens, int) or update_input_tokens <= 0:
        raise ValueError('Positive update token count required')
    position = current_input_tokens + update_input_tokens
    base, phase = learning_rate(cfg['learning_rate_schedule'], position)
    expected = {'expert', 'router', 'gate'};names = [group.get('group_name') for group in optimizer.param_groups]
    if len(names) != len(expected) or set(names) != expected:
        raise ValueError('Optimizer parameter groups do not match schedule')
    rates = {}
    for group in optimizer.param_groups:
        rate = base * group['lr_multiplier'];group['lr'] = rate;rates[group['group_name']] = rate
    return {'schema': SCHEDULER_STATE_SCHEMA,
            'schedule_sha256': identity(cfg['learning_rate_schedule']),
            'committed_input_tokens': position, 'last_applied_input_tokens': position,
            'phase': phase, 'base_learning_rate': base, 'group_learning_rates': rates}


def checkpoint_metadata(state):
    scheduler = state.get('scheduler')
    if not isinstance(scheduler, dict):
        return {}
    return {'lineage_id': state['config']['lineage']['id'],
            'learning_rate_schedule_sha256': scheduler['schedule_sha256'],
            'learning_rate_phase': scheduler['phase'],
            'learning_rate_input_tokens': scheduler['committed_input_tokens']}
