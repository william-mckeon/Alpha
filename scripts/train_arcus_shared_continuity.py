"""Train new object heads within a full shared candidate; never promote it."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from baby_arcus.shared_checkpoint import load, save, restore_optimizer
from baby_arcus.shared_continuity_model import ContinuityModel
from baby_arcus.shared_continuity_curriculum import encode_scene
from baby_arcus.shared_depth import verify_depth
from baby_arcus.shared_learning import configure_training
from baby_arcus.language_stream import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--scenes', type=int, default=256)
    parser.add_argument('--updates', type=int, default=2000)
    args = parser.parse_args()
    if args.scenes < 32 or args.updates < 1:
        parser.error('At least 32 scenes and one update required')
    cfg = json.loads(Path(args.config).read_text())
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    configure_training()
    torch.set_num_threads(2)
    torch.manual_seed(10892518)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    source = json.loads((Path(cfg['root'])/'candidate.json').read_text())
    old, data = load(cfg['root'], source, device)
    verify_depth(old, cfg)
    model = ContinuityModel(old.body, old.language).to(device)
    missing, unexpected = model.load_state_dict(old.state_dict(), strict=False)
    if unexpected or any(not key.startswith(('object_association.', 'object_search.')) for key in missing):
        raise ValueError('Unexpected continuity migration')
    model.eval().requires_grad_(False)
    from arcus.tokenizer import get_tokenizer
    tokenizer = get_tokenizer(cfg['encoding'])
    association, search = [], []
    for index in range(args.scenes):
        encoded, ambiguous = encode_scene(model, tokenizer, index, 'training')
        for before in encoded:
            for descriptor, identity in zip(before['descriptors'], before['labels']):
                if not identity:
                    continue
                for after in encoded:
                    # Prediction must use a previously observed context, not the
                    # hidden representation of the view whose outcome is labelled.
                    search.append((before['hidden'], descriptor, after['gaze'], float(after['visible'][identity-1])))
                    for other, other_id in zip(after['descriptors'], after['labels']):
                        if other_id and not ambiguous:
                            association.append((after['hidden'], descriptor, other, float(identity == other_id)))
        if (index+1) % 16 == 0:
            print(json.dumps({'scenes': index+1, 'association_examples': len(association), 'search_examples': len(search)}), flush=True)
    if not association or not search:
        raise ValueError('No detected training examples')
    def tensors(items):
        return (torch.stack([x[0] for x in items]).to(device),
                *[torch.tensor([x[k] for x in items], device=device) for k in (1, 2, 3)])
    a, s = tensors(association), tensors(search)
    optimizer = restore_optimizer(model, data, cfg['learning_rate'])
    optimizer.add_param_group({'params': list(model.object_association.parameters())+list(model.object_search.parameters()), 'lr': .001})
    model.object_association.requires_grad_(True)
    model.object_search.requires_grad_(True)
    for step in range(args.updates):
        optimizer.zero_grad(set_to_none=True)
        loss = 0
        for examples, head in ((a, model.association_logits), (s, model.search_logits)):
            ids = torch.randint(len(examples[0]), (128,), device=device)
            loss = loss+torch.nn.functional.binary_cross_entropy_with_logits(head(*(x[ids] for x in examples[:3])), examples[3][ids])
        loss.backward()
        optimizer.step()
        if (step+1) % 250 == 0:
            print(json.dumps({'updates': step+1, 'loss': float(loss.detach())}), flush=True)
    # Prove every existing tensor was retained exactly, not just the motor heads.
    changed = [key for key, value in old.state_dict().items() if not torch.equal(value, model.state_dict()[key])]
    if changed:
        raise ValueError('Existing weights changed: '+str(changed))
    progress = data['progress']
    progress['updates'] += args.updates
    progress['receipts'].append({'curriculum': 'object-continuity-search', 'scenes': args.scenes,
        'updates': args.updates, 'depth_capacity': .25, 'existing_weights_unchanged': True})
    model.requires_grad_(True)
    manifest = save(root, model, optimizer, progress)
    atomic_json(root/'candidate.json', manifest)
    atomic_json(root/'config.json', dict(cfg, root=root.as_posix()))
    report = {'candidate': manifest, 'source': source, 'scenes': args.scenes, 'updates': args.updates,
              'association_examples': len(association), 'search_examples': len(search), 'existing_weights_unchanged': True,
              'qualified': False, 'promoted': False}
    atomic_json(root/'continuity-training.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
