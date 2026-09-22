"""Joint losses for a single shared learner, including sampled action outcomes."""
import copy
import torch
from baby_arcus.shared_experience import validate
from baby_arcus.shared_replay import partition
from baby_arcus.shared_depth import verify_depth


def loss(model, tokenizer, row, target):
    validate(row)
    if not row['eligibility']['training'] or partition(row) != 'training':
        raise ValueError('Held-out observations cannot train')
    verify_depth(model)
    supported = {'activity', 'posture_choice', 'gaze_choice', 'language_choice',
                 'rest', 'future_body', 'future_rgb', 'tokens', 'policy', 'perception',
                 'action_quality', 'curiosity', 'identity_match', 'visual_search', 'identity_risk'}
    if not target or set(target) - supported:
        raise ValueError('Unsupported learning target')
    heads = set(target) - {'tokens', 'policy'}
    if 'policy' in target:
        heads.add(target['policy']['head'])
    heads.add('hidden')
    policy_row = copy.deepcopy(row)
    policy_row.pop('executed_action', None)
    out = model([policy_row if 'policy' in target else row], tokenizer, requested=tuple(heads))
    if 'policy' in target and any(k in target for k in ('future_body', 'future_rgb')):
        forecast = model([row], tokenizer, requested=('future_body', 'future_rgb'))
        out.update(forecast)
    device = out['hidden'].device
    terms = {}
    for key in ('activity', 'posture_choice', 'gaze_choice', 'language_choice', 'action_quality'):
        if key in target:
            if type(target[key]) is not int or not 0 <= target[key] < out[key].shape[-1]:
                raise ValueError(f'Invalid {key} target {target[key]} for {out[key].shape[-1]} classes')
            terms[key] = torch.nn.functional.cross_entropy(out[key], torch.tensor([target[key]], device=device))
    for key in ('rest', 'future_body', 'future_rgb', 'curiosity'):
        if key in target:
            terms[key] = torch.nn.functional.mse_loss(out[key][0], torch.tensor(target[key], device=device, dtype=torch.float32).reshape_as(out[key][0]))
    for key in ('identity_match', 'visual_search', 'identity_risk'):
        if key in target:
            terms[key] = torch.nn.functional.binary_cross_entropy_with_logits(out[key][0], torch.tensor(target[key], device=device, dtype=torch.float32).reshape_as(out[key][0]))
    if 'perception' in target:
        labels = torch.tensor(target['perception'])
        if labels.min() < 0 or labels.max() >= 4:
            raise ValueError('Invalid perception class labels')
        terms['perception'] = torch.nn.functional.cross_entropy(
            out['perception'].permute(0, 2, 3, 1).reshape(-1, 4),
            torch.tensor(target['perception'], device=device).reshape(-1))
    if 'tokens' in target:
        ids = target['tokens']
        if not 2 <= len(ids) <= 65 or any(type(i) is not int or not 0 <= i < model.language.embedding.num_embeddings for i in ids):
            raise ValueError('Expected 2..65 valid language tokens')
        tokens = torch.tensor([ids], device=device)
        logits = model.language(model.core, tokens[:, :-1])
        # Every next-token position learns; context residual shares body/RGB/hearing.
        residual = model.text_context(out['hidden'])
        logits = logits + torch.nn.functional.linear(residual, model.language.embedding.weight)[:, None]
        terms['tokens'] = torch.nn.functional.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten())
    if 'policy' in target:
        policy = target['policy']
        if policy['head'] not in ('body', 'lying', 'sitting', 'approach', 'activity', 'gaze_choice'):
            raise ValueError('Unsupported policy head')
        reward = float(policy['advantage'])
        if not -10 <= reward <= 10:
            raise ValueError('Unbounded policy reward')
        dist = torch.distributions.Categorical(logits=out[policy['head']])
        choice = torch.tensor([policy['index']], device=device)
        terms['policy'] = (-dist.log_prob(choice) * reward - .001 * dist.entropy()).mean()
    total = sum(terms.values()) + model.core.last_aux_loss * .01
    if not bool(torch.isfinite(total)):
        raise ValueError('Nonfinite joint loss')
    return total, {key: float(value.detach()) for key, value in terms.items()}


def step(model, optimizer, tokenizer, row, target):
    from baby_arcus.shared_learning import configure_training
    configure_training()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total, metrics = loss(model, tokenizer, row, target)
    total.backward()
    groups = {'core': model.core, 'rgb': model.rgb, 'language': model.language,
              'body_sensation': model.body_sensation_input, 'internal': model.internal_input}
    gradients = {name: sum(float(p.grad.detach().square().sum()) for p in module.parameters() if p.grad is not None) ** .5
                 for name, module in groups.items()}
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    model.eval()
    return {'loss': float(total.detach()), 'losses': metrics, 'gradient_norms': gradients,
            'trained_tokens': max(0, len(target.get('tokens', [])) - 1)}
