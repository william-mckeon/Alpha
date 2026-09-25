"""Count architecture parameters on the meta device; no weights or execution."""
import json
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from baby_arcus.presets import configuration
from baby_arcus.body_policy import BodyPolicy
from baby_arcus.language_model import LanguageAdapter
from baby_arcus.shared_continuity_model import ContinuityModel
from arcus.tokenizer import get_tokenizer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--preset', default='baby-256m-cap4', choices=['baby-256m-cap4', 'baby-128m-cap4'])
args = parser.parse_args()
with torch.device('meta'):
    shape = configuration(args.preset)
    model = ContinuityModel(BodyPolicy(shape, lying=True, sitting=True, approach=True),
                            LanguageAdapter(shape.dim, get_tokenizer('o200k_base').vocab_size, 128), 11)
print(json.dumps({'preset': args.preset, 'parameters': sum(p.numel() for p in model.parameters()),
                  'context_tokens': shape.max_seq_len, 'layers': shape.n_layers,
                  'experts_per_layer': shape.n_experts, 'allocated_weights': False}))
