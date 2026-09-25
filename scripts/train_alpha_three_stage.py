"""Explicit bounded three-stage training entry point. Never clears a pause flag."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.runtime_contract import require_container
if __name__ == '__main__':
    require_container()
import torch
from baby_arcus.three_stage_training import train


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True); parser.add_argument('--updates',type=int,required=True)
    args=parser.parse_args(); torch.set_num_threads(2)
    print(json.dumps(train(args.config,args.updates)))
