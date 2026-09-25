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
                 'action_quality', 'curiosity', 'identity_match', 'visual_search', 'identity_risk', 'sft'}
    if not target or set(target) - supported:
        raise ValueError('Unsupported learning target')
    heads = set(target) - {'tokens', 'policy', 'sft'}
    if 'policy' in target:
        heads.add(target['policy']['head'])
    heads.add('hidden')
    # Preserve the historical motor-pass auxiliary loss for non-language tasks.
    # Language/SFT owns the final trunk pass and its auxiliary loss below.
    if not {'tokens','sft'}.intersection(target):heads.add('aux')
    policy_row = copy.deepcopy(row)
    policy_row.pop('executed_action', None)
    out = model([policy_row if 'policy' in target else row], tokenizer, requested=tuple(heads))
    if 'policy' in target and any(k in target for k in ('future_body', 'future_rgb')):
        forecast = model([row], tokenizer, requested=('future_body', 'future_rgb', 'aux'))
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
        if not 2 <= len(ids) <= model.body.cfg.max_seq_len+1 or any(type(i) is not int or not 0 <= i < model.language.embedding.num_embeddings for i in ids):
            raise ValueError('Expected 2..65 valid language tokens')
        tokens = torch.tensor([ids], device=device)
        # Every next-token position learns; context residual shares body/RGB/hearing.
        residual = model.text_context(out['hidden'])
        terms['tokens'] = model.language.loss(model.core,tokens[:,:-1],tokens[:,1:],residual)
    if 'sft' in target:
        # Conversation context travels through the same causal language trunk.
        # The sensory row must not contain the completion or future tool results.
        if row.get('hearing') or row.get('language_prefix_ids'):
            raise ValueError('SFT sensory context must not duplicate the target conversation')
        sequence = target['sft']
        from baby_arcus.sft_dataset import validate_window
        validate_window(sequence, model.language.embedding.num_embeddings, model.body.cfg.max_seq_len)
        ids = torch.tensor([sequence['ids']], device=device)
        labels = ids[:, 1:].clone()
        labels[~torch.tensor([sequence['mask'][1:]], device=device)] = -100
        terms['sft'] = model.language.loss(model.core,ids[:,:-1],labels,model.text_context(out['hidden']))
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
    values=torch.stack([value.detach() for value in terms.values()]).cpu().tolist() if terms else []
    return total, dict(zip(terms,values))


def step(model, optimizer, tokenizer, row, target):
    from baby_arcus.shared_learning import configure_training
    configure_training()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total, metrics = loss(model, tokenizer, row, target)
    total.backward()
    groups = {'core': model.core, 'rgb': model.rgb, 'language': model.language,
              'body_sensation': model.body_sensation_input, 'internal': model.internal_input}
    norms=[]
    for module in groups.values():
        sums=[p.grad.detach().float().square().sum() for p in module.parameters() if p.grad is not None]
        norms.append(torch.stack(sums).sum().sqrt() if sums else total.new_zeros(()))
    gradients = dict(zip(groups,torch.stack(norms).detach().cpu().tolist()))
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    model.eval()
    return {'loss': float(total.detach()), 'losses': metrics, 'gradient_norms': gradients,
            'trained_tokens': max(0, len(target.get('tokens', [])) - 1) + sum(target.get('sft', {}).get('mask', [])[1:])}
