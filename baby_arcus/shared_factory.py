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
    if type(cfg.get('gradient_checkpointing', False)) is not bool:
        raise ValueError('gradient_checkpointing must be Boolean')
    fraction = cfg.get('cuda_memory_fraction', 1.0)
    if type(fraction) not in (int, float) or not 0 < fraction <= 1:
        raise ValueError('Invalid CUDA memory fraction')
    if cfg.get('attention_backend', 'default') not in ('default', 'efficient'):
        raise ValueError('Unsupported training attention backend')
    if type(cfg.get('indexed_corpus',False)) is not bool:
        raise ValueError('indexed_corpus must be an explicit boolean')
    if type(cfg.get('training_cache_bytes',0)) is not int or not 0 <= cfg.get('training_cache_bytes',0) <= 2*1024**3:
        raise ValueError('training_cache_bytes must be 0–2 GiB; default 0 releases training state')
    from baby_arcus.context_contract import context_tokens
    context_tokens(cfg)
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
    if 'idle_learning' in cfg:
        idle = cfg['idle_learning']
        if type(idle.get('auto_resume')) is not bool or not 0 <= idle.get('idle_seconds', -1) <= 3600:
            raise ValueError('Invalid quiet-time resume policy')
        for name, maximum in (('chunk_updates', 64), ('checkpoint_every', 64), ('session_updates', 65536)):
            if type(idle.get(name)) is not int or not 1 <= idle[name] <= maximum:
                raise ValueError('Invalid quiet-time budget: '+name)
        if idle['checkpoint_every'] > idle['chunk_updates']:
            raise ValueError('Checkpoint interval exceeds quiet-time chunk')
    return cfg


def create(cfg, vocab_size, device='cpu'):
    from baby_arcus.runtime_contract import require_device
    require_device(device)
    if cfg['depth_capacity'] not in (.25, 1.0):
        raise ValueError('Capacity must be 0.25 or 1.0 before construction')
    shape = configuration(cfg.get('preset', 'baby-125m-cap4'))
    if cfg.get('preset', 'baby-125m-cap4') != 'tiny':
        from baby_arcus.runtime_contract import require_container
        require_container()
    torch.manual_seed(cfg['seed'])
    shape.capacity = cfg['depth_capacity']
    from baby_arcus.context_contract import context_tokens
    shape.max_seq_len = context_tokens(cfg)
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
    from baby_arcus.context_contract import context_tokens
    if context_tokens(cfg)!=context_tokens(original) or data['body_config']['max_seq_len']!=context_tokens(cfg):
        raise ValueError('Resume changed immutable context; prepare an explicit context extension')
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
