"""Experimental continuation of the single shared model; not a second learner."""
import torch
from torch import nn
from baby_arcus.shared_model import SharedModel


def normalized_context(hidden):
    # Search expands one observation across many pairs. Normalize that observation
    # once; expansion preserves gradients and avoids repeated vector reductions.
    if hidden.ndim==2 and hidden.shape[0]>1 and hidden.stride(0)==0:
        return nn.functional.normalize(hidden[:1],dim=-1).expand_as(hidden)
    return nn.functional.normalize(hidden,dim=-1)


class ContinuityModel(SharedModel):
    def __init__(self, body, language, version=10):
        super().__init__(body, language, version=9)
        self.version = version
        self.object_association = nn.Sequential(nn.Linear(body.cfg.dim+33, 64), nn.GELU(), nn.Linear(64, 1))
        self.object_search = nn.Sequential(nn.Linear(body.cfg.dim+13, 64), nn.GELU(), nn.Linear(64, 1))
        if version >= 11:
            self.identity_uncertainty = nn.Sequential(nn.Linear(body.cfg.dim+7, 64), nn.GELU(), nn.Linear(64, 1))

    def uncertainty_logits(self, hidden, features):
        return self.identity_uncertainty(torch.cat((normalized_context(hidden), features), -1)).squeeze(-1)

    def association_logits(self, hidden, previous, observed):
        return self.object_association(torch.cat((normalized_context(hidden), previous,
                                                  observed, (previous-observed).abs()), -1)).squeeze(-1)

    def search_logits(self, hidden, remembered, gaze):
        return self.object_search(torch.cat((normalized_context(hidden), remembered, gaze), -1)).squeeze(-1)

    def forward(self, rows, tokenizer, requested=None):
        heads = {'identity_match', 'visual_search'} | ({'identity_risk'} if self.version >= 11 else set())
        if requested is not None and not heads.intersection(requested):
            return super().forward(rows, tokenizer, requested)
        base = None if requested is None else tuple((set(requested)-heads) | {'hidden'})
        result = super().forward(rows, tokenizer, base)
        hidden = result['hidden']
        device = hidden.device
        if requested is None or 'identity_match' in requested:
            pair = torch.tensor([row.get('identity_pair', [[0.0]*11, [0.0]*11]) for row in rows], device=device)
            result['identity_match'] = self.association_logits(hidden, pair[:, 0], pair[:, 1])[:, None]
        if requested is None or 'visual_search' in requested:
            search = torch.tensor([row.get('search_query', [0.0]*13) for row in rows], device=device)
            result['visual_search'] = self.search_logits(hidden, search[:, :11], search[:, 11:])[:, None]
        if self.version >= 11 and (requested is None or 'identity_risk' in requested):
            context = torch.tensor([row.get('identity_context', [0.0]*7) for row in rows], device=device)
            result['identity_risk'] = self.uncertainty_logits(hidden, context)[:, None]
        return result if requested is None else {key: value for key, value in result.items() if key in requested}


def load_candidate(root, manifest, device='cpu'):
    """Experimental loader, deliberately unavailable to the live v9 worker."""
    from pathlib import Path
    from baby_arcus.shared_checkpoint import digest
    from arcus.model_config import ModelConfig
    from baby_arcus.body_policy import BodyPolicy
    from baby_arcus.language_model import LanguageAdapter
    from baby_arcus.shared_depth import verify_depth
    generation = manifest['generation']
    if len(generation) != 32 or any(c not in '0123456789abcdef' for c in generation):
        raise ValueError('Invalid generation')
    path = Path(root)/(generation+'.pt')
    if digest(path) != manifest['sha256']:
        raise ValueError('Continuity checkpoint hash mismatch')
    from baby_arcus.shared_checkpoint import read_data
    data = read_data(path)
    if data['schema'] not in ('arcus-shared-v10', 'arcus-shared-v11') or manifest.get('depth_capacity') != .25:
        raise ValueError('Invalid continuity candidate')
    body = BodyPolicy(ModelConfig(**data['body_config']), lying=True, sitting=True, approach=True)
    model = ContinuityModel(body, LanguageAdapter(body.cfg.dim, data['vocab_size'], data['text_dim']), int(data['schema'].rsplit('v', 1)[1]))
    model.load_state_dict(data['model'])
    model.integrated_motor = data.get('integrated_motor', False)
    verify_depth(model)
    return model.to(device), data
