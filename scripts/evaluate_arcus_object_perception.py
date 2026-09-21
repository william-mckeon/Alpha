import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.object_perception_learning import evaluate
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/baby_arcus/object_perception.json')
    if not evaluate(p.parse_args().config)['passed']:raise SystemExit(1)
