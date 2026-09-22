"""Isolated test-2 learner service: serialized inference and bounded training."""
import argparse
import json
import os
from pathlib import Path
import threading
import torch
from arcus.tokenizer import get_tokenizer
from baby_arcus.shared_factory import read_config, verify_run
from baby_arcus.shared_checkpoint import load
from baby_arcus.shared_training_scheduler import train
from baby_arcus.model_adapter import decide
from baby_arcus.embodiment_store import EmbodimentStore
from baby_arcus.transport import serve


class Learner:
    def __init__(self, config):
        self.config = config
        self.cfg = read_config(config)
        self.lock = threading.Lock()

    def __call__(self, method, path, body):
        root = Path(self.cfg['root'])
        if method == 'GET' and path in ('/health', '/ready'):
            ready = (root / 'candidate.json').exists()
            return (200 if ready else 503), {'ready': ready, 'mode': 'isolated-training', 'depth_capacity': self.cfg['depth_capacity']}
        if method != 'POST':
            raise KeyError(path)
        with self.lock:
            if path == '/train':
                if not isinstance(body.get('request_id'), str) or not 1 <= len(body['request_id']) <= 160:
                    raise ValueError('A stable training request_id is required')
                supplied = (body['row'], body['target']) if 'row' in body else None
                return 200, train(self.config, body.get('updates'), supplied, body['request_id'])
            if path == '/infer':
                lease = EmbodimentStore(root / 'learner-lease')
                try:
                    manifest = json.loads((root / 'candidate.json').read_text())
                    model, data = load(root, manifest, 'cuda' if torch.cuda.is_available() else 'cpu')
                    verify_run(self.cfg, data)
                    result = decide(model, get_tokenizer(self.cfg['encoding']), body['row'])
                    return 200, {**result, 'generation': manifest['generation']}
                finally:
                    lease.close()
            raise KeyError(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/baby_arcus/test2.json')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8901)
    args = p.parse_args()
    token = os.environ.get('ARCUS_TEST2_TOKEN')
    if not token:
        raise ValueError('ARCUS_TEST2_TOKEN required')
    torch.set_num_threads(2)
    server = serve(args.host, args.port, Learner(args.config), token)
    try:
        print(json.dumps({'ready': True, 'port': args.port}), flush=True)
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
