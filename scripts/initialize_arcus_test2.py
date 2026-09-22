"""Initialize an isolated random Arcus; never alter the existing learner."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from baby_arcus.shared_factory import initialize

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/baby_arcus/test2.json')
    args = p.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(initialize(args.config)), flush=True)
