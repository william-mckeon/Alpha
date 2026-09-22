import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from baby_arcus.shared_training_scheduler import train

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Bounded integrated fresh Arcus training')
    p.add_argument('--config', default='configs/baby_arcus/test2.json')
    p.add_argument('--updates', type=int)
    args = p.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(train(args.config, args.updates)), flush=True)
