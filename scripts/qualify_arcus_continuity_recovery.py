"""Full-size deterministic recovery of all three new heads and their shared core."""
import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from baby_arcus.shared_continuity_model import load_candidate
from baby_arcus.shared_checkpoint import restore_optimizer, save
from baby_arcus.shared_learning import configure_training
from baby_arcus.shared_causal_curriculum import transition
from baby_arcus.language_stream import atomic_json


def main():
    from baby_arcus.shared_qualification import source_snapshot
    from baby_arcus.shared_continuity_qualification import bind
    initial_sources = source_snapshot()
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    root = Path(cfg['root'])
    manifest = json.loads((root/'candidate.json').read_text())
    configure_training()
    torch.set_num_threads(2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    from arcus.tokenizer import get_tokenizer
    tokenizer = get_tokenizer(cfg['encoding'])
    row, _, _ = transition(15, 'training')
    model, data = load_candidate(root, manifest, device)
    optimizer = restore_optimizer(model, data, cfg['learning_rate'])
    work = root/'continuity-recovery-work'
    before = save(work, model, optimizer, data['progress'])

    def step(current, opt):
        current.train().requires_grad_(True)
        opt.zero_grad(set_to_none=True)
        hidden = current([row], tokenizer, requested=('hidden',))['hidden']
        previous = torch.tensor([[.2, .4, .8, .1, .1, .2, .2, 0, 0, -1, 0]], device=device)
        observed = previous.clone()
        observed[:, 9] = 0
        loss = (current.association_logits(hidden, previous, observed)-1).square().mean()
        loss += current.search_logits(hidden, previous, torch.zeros(1, 2, device=device)).square().mean()
        loss += current.uncertainty_logits(hidden, torch.ones(1, 7, device=device)).square().mean()
        loss.backward()
        gradients = {name: module[-1].weight.grad is not None and bool(module[-1].weight.grad.abs().sum())
                     for name, module in (('association', current.object_association), ('search', current.object_search),
                                          ('uncertainty', current.identity_uncertainty))}
        gradients['core'] = current.core.token_embed.weight.grad is not None and bool(current.core.token_embed.weight.grad.abs().sum())
        torch.nn.utils.clip_grad_norm_(current.parameters(), 1, error_if_nonfinite=True)
        opt.step()
        return float(loss.detach()), gradients

    def hashes(current):
        return {key: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                for key, value in current.state_dict().items()}

    first, gradients = step(model, optimizer)
    expected = hashes(model)
    del model, optimizer, data
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    model, data = load_candidate(work, before, device)
    optimizer = restore_optimizer(model, data, cfg['learning_rate'])
    torch.set_rng_state(data['rng'])
    if device == 'cuda':
        torch.cuda.set_rng_state_all(data['cuda_rng'])
    second, resumed_gradients = step(model, optimizer)
    actual = hashes(model)
    mismatches = [key for key in expected if actual[key] != expected[key]]
    report = {'candidate': manifest, 'device': device, 'losses': [first, second],
              'mismatched_tensors': mismatches, 'gradients': gradients, 'resumed_gradients': resumed_gradients,
              'recovery': first == second and not mismatches and all(gradients.values()) and all(resumed_gradients.values()),
              'promoted': False}
    bind(report, initial_sources)
    atomic_json(root/('continuity-recovery-'+sys.platform+'.json'), report)
    print(json.dumps(report), flush=True)
    if not report['recovery']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
