"""Fresh shared learner construction; never imports a checkpoint loader."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch

from baby_arcus.body_policy import BodyPolicy
from baby_arcus.language_model import LanguageAdapter
from baby_arcus.presets import configuration
from baby_arcus.shared_continuity_model import ContinuityModel
from baby_arcus.shared_depth import verify_depth


def read_config(path):
    cfg = json.loads(Path(path).read_text(encoding='utf-8'))
    if cfg.get('schema') != 'arcus-test2-v1' or cfg.get('depth_capacity') not in (.25, 1.0):
        raise ValueError('Test 2 requires its own schema and capacity 0.25 or 1.0')
    if cfg.get('initialization') != 'random' or any(cfg.get(k) for k in ('body_checkpoint', 'language_root', 'parent_checkpoint')):
        raise ValueError('Fresh configuration cannot import trained parents')
    root = Path(cfg['root']).resolve()
    base = (Path(__file__).resolve().parents[1] / 'runs' / 'test2').resolve()
    # An environment-selected mount is allowed only inside a dedicated test2 directory.
    if base not in root.parents:
        raise ValueError('Experiment root must be under runs/test2')
    cfg['root'] = str(root)
    return cfg


def create(cfg, vocab_size, device='cpu'):
    if cfg['depth_capacity'] not in (.25, 1.0):
        raise ValueError('Capacity must be 0.25 or 1.0 before construction')
    torch.manual_seed(cfg['seed'])
    shape = configuration(cfg.get('preset', 'baby-125m-cap4'))
    shape.capacity = cfg['depth_capacity']
    shape.max_seq_len = max(512, shape.max_seq_len)
    body = BodyPolicy(shape, lying=True, sitting=True, approach=True)
    model = ContinuityModel(body, LanguageAdapter(shape.dim, vocab_size, cfg.get('text_dim', 128)), 11)
    # Migration bridges start at zero in the retained model. Fresh learning has no
    # behavior to retain, so expose gradients to every upstream modality immediately.
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) and not bool(module.weight.detach().any()):
            torch.nn.init.xavier_uniform_(module.weight)
    model.integrated_motor = True
    model.experiment_depth_capacity = cfg['depth_capacity']
    model.requires_grad_(True)
    verify_depth(model, cfg)
    return model.to(device)


def verify_run(cfg, data):
    """Mutable curriculum settings cannot silently change a checkpoint's identity."""
    original = json.loads((Path(cfg['root']) / 'experiment.json').read_text())
    for key in ('seed', 'preset', 'text_dim', 'depth_capacity', 'encoding', 'tiktoken_version'):
        if cfg[key] != original[key]:
            raise ValueError('Resume changed immutable experiment setting: ' + key)
    if data['progress'].get('initialization') != 'random' or data['progress'].get('sources'):
        raise ValueError('Not a fresh experiment lineage')
    if data['body_config']['capacity'] != cfg['depth_capacity']:
        raise ValueError('Checkpoint capacity differs from experiment')
    if importlib.metadata.version('tiktoken') != cfg['tiktoken_version']:
        raise ValueError('Tokenizer release changed')


def source_manifest():
    root = Path(__file__).resolve().parents[1]
    paths = sorted([*root.joinpath('baby_arcus').rglob('*.py'), *root.joinpath('arcus').rglob('*.py')])
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def initialize(config):
    from arcus.tokenizer import get_tokenizer
    from baby_arcus.shared_checkpoint import save
    from baby_arcus.language_stream import atomic_json
    cfg = read_config(config)
    root = Path(cfg['root'])
    if root.exists() and any(root.iterdir()):
        raise ValueError('Refusing to initialize a populated run root')
    tokenizer = get_tokenizer(cfg['encoding'])
    version = importlib.metadata.version('tiktoken')
    if version != cfg['tiktoken_version']:
        raise ValueError('Tokenizer version differs from experiment configuration')
    model = create(cfg, tokenizer.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'])
    identity = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    progress = {'updates': 0, 'receipts': [], 'sources': {}, 'initialization': 'random',
                'config_sha256': identity, 'seed': cfg['seed'], 'encoding': cfg['encoding'],
                'tiktoken_version': version, 'trained_tokens': 0, 'curriculum_index': 0,
                'parameters': sum(p.numel() for p in model.parameters())}
    progress['source_manifest'] = source_manifest()
    manifest = save(root, model, optimizer, progress)
    atomic_json(root / 'experiment.json', cfg)
    atomic_json(root / 'initial.json', manifest)
    atomic_json(root / 'candidate.json', manifest)
    return {**manifest, 'parameters': progress['parameters'], 'root': str(root)}
