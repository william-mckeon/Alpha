"""Continue the same candidate with a learned prior-survey uncertainty head."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from baby_arcus.shared_continuity_model import ContinuityModel, load_candidate
from baby_arcus.shared_continuity_curriculum import encode_scene
from baby_arcus.shared_identity_context import pair_features
from baby_arcus.shared_checkpoint import restore_optimizer, save
from baby_arcus.shared_learning import configure_training
from baby_arcus.language_stream import atomic_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--scenes', type=int, default=256)
    p.add_argument('--updates', type=int, default=3000)
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    configure_training()
    torch.set_num_threads(2)
    torch.manual_seed(11280919)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    source = json.loads((Path(cfg['root'])/'candidate.json').read_text())
    old, data = load_candidate(cfg['root'], source, device)
    model = ContinuityModel(old.body, old.language, 11).to(device)
    missing, extra = model.load_state_dict(old.state_dict(), strict=False)
    if extra or any(not k.startswith('identity_uncertainty.') for k in missing):
        raise ValueError('Unexpected uncertainty migration')
    model.eval().requires_grad_(False)
    from arcus.tokenizer import get_tokenizer
    tokenizer = get_tokenizer(cfg['encoding'])
    xs, ys = [], []
    for index in range(args.scenes):
        frames, ambiguous = encode_scene(model, tokenizer, index, 'training')
        inventory = [(d, label) for f in frames for d, label in zip(f['descriptors'], f['labels']) if label][:32]
        for frame in frames:
            for query in frame['descriptors']:
                for i, (first, a) in enumerate(inventory):
                    for second, b in inventory[i+1:]:
                        features = pair_features(first, second, query)
                        xs.append(torch.cat((torch.nn.functional.normalize(frame['hidden'], dim=-1), torch.tensor(features))))
                        ys.append(float(ambiguous and a != b))
        if (index+1) % 32 == 0:
            print(json.dumps({'scenes': index+1, 'examples': len(xs)}), flush=True)
    x, y = torch.stack(xs).to(device), torch.tensor(ys, device=device)
    positive, negative = torch.where(y > .5)[0], torch.where(y < .5)[0]
    if not len(positive) or not len(negative):
        raise ValueError('Missing uncertainty contrast')
    optimizer = restore_optimizer(model, data, cfg['learning_rate'])
    optimizer.add_param_group({'params': model.identity_uncertainty.parameters(), 'lr': .001})
    model.identity_uncertainty.requires_grad_(True)
    for step in range(args.updates):
        ids = torch.cat([group[torch.randint(len(group), (64,), device=device)] for group in (positive, negative)])
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model.identity_uncertainty(x[ids]).squeeze(-1), y[ids])
        loss.backward()
        optimizer.step()
        if (step+1) % 500 == 0:
            print(json.dumps({'updates': step+1, 'loss': float(loss.detach())}), flush=True)
    if any(not torch.equal(value, model.state_dict()[key]) for key, value in old.state_dict().items()):
        raise ValueError('Existing weights changed')
    progress = data['progress']
    progress['updates'] += args.updates
    progress['receipts'].append({'curriculum': 'prior-survey-identity-uncertainty', 'updates': args.updates, 'examples': len(ys), 'depth_capacity': .25})
    model.requires_grad_(True)
    result = save(root, model, optimizer, progress)
    atomic_json(root/'candidate.json', result)
    atomic_json(root/'config.json', dict(cfg, root=root.as_posix()))
    atomic_json(root/'uncertainty-training.json', {'candidate': result, 'source': source, 'examples': len(ys), 'promoted': False})
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
