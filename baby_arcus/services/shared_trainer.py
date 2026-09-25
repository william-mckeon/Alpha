"""Isolated test-2 learner service: serialized inference and bounded training."""
import argparse
import json
import os
from pathlib import Path
import threading
import time
from baby_arcus.runtime_contract import require_container, identity
if __name__ == '__main__':
    require_container()
from baby_arcus.runtime_contract import model_device
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
        self.verified_checkpoint = None
        from baby_arcus.learner_session import LearnerSession
        self.session = LearnerSession()

    def __call__(self, method, path, body):
        root = Path(self.cfg['root'])
        if method == 'GET' and path == '/memory':
            # Do not race model device transfers, nor wait behind a training job.
            if not self.lock.acquire(blocking=False):
                return 200, {'busy':True,'memory':None}
            try:
                from baby_arcus.runtime_memory import snapshot
                return 200, {'busy':False,'memory':snapshot(self.session.model),
                             'generation':json.loads(self.session.key[1])['generation'] if self.session.key else None}
            finally:self.lock.release()
        if method == 'GET' and path in ('/health', '/ready'):
            try:
                if self.cfg.get('preset') != 'tiny':
                    from baby_arcus.runtime_contract import require_gpu
                    require_gpu()
                candidate = json.loads((root/'candidate.json').read_text())
                generation = candidate['generation']
                if len(generation) != 32 or any(c not in '0123456789abcdef' for c in generation):
                    raise ValueError('Invalid candidate generation')
                checkpoint = root/(generation+'.pt')
                stat = checkpoint.stat()
                key = (generation, candidate['sha256'], stat.st_size, stat.st_mtime_ns)
                if self.verified_checkpoint != key:
                    from baby_arcus.shared_checkpoint import digest
                    if digest(checkpoint) != candidate['sha256']: raise ValueError('Checkpoint hash mismatch')
                    self.verified_checkpoint = key
                return 200, {'ready':True,'runtime':identity(),'candidate':candidate,
                             'checkpoint_hash_verified':True,'model_loaded':self.session.model is not None,
                             'model_residency':'cpu' if self.session.model is not None else 'unloaded',
                             'inference_cache_generation':json.loads(self.session.key[1])['generation'] if self.session.key else None,
                             'mode':'isolated-training','depth_capacity':self.cfg['depth_capacity']}
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                return 503, {'ready':False,'runtime':identity(),'reason':str(exc),'model_loaded':False}
        if method != 'POST':
            raise KeyError(path)
        from baby_arcus.gpu_job_control import gpu_job
        with self.lock, gpu_job():
            if path == '/train':
                self.session.invalidate()
                if self.cfg.get('idle_learning'):
                    if 'row' in body or 'target' in body:
                        raise ValueError('Quiet-time mode preserves the sustained mixed curriculum')
                    from baby_arcus.shared_idle_training import train_idle
                    return 200, train_idle(self.config, body.get('updates'), body.get('request_id'))
                if not isinstance(body.get('request_id'), str) or not 1 <= len(body['request_id']) <= 160:
                    raise ValueError('A stable training request_id is required')
                supplied = (body['row'], body['target']) if 'row' in body else None
                return 200, train(self.config, body.get('updates'), supplied, body['request_id'])
            if path in ('/infer','/coding-infer'):
                from baby_arcus.training_session import SESSION
                SESSION.clear()
                lease = EmbodimentStore(root / 'learner-lease')
                try:
                    manifest = json.loads((root / 'candidate.json').read_text())
                    from baby_arcus.audit import TRACE
                    def stage(name):
                        print(json.dumps({'event':'learner.stage','stage':name,'time':time.time(),
                            'trace':TRACE.get(),'generation':manifest['generation']}),flush=True)
                    stage('checkpoint_load_started')
                    if not (root/'inference-artifacts'/(manifest['generation']+'.json')).exists():
                        from baby_arcus.shared_storage_budget import check
                        check(root,self.cfg['max_storage_bytes'],(root/(manifest['generation']+'.pt')).stat().st_size)
                    with self.session.use(root,manifest,model_device(self.cfg),lambda data:verify_run(self.cfg,data)) as model:
                        stage('checkpoint_loaded_inference_started')
                        if path == '/coding-infer':
                            from baby_arcus.coding_policy import decide as decide_coding
                            result = decide_coding(model, get_tokenizer(self.cfg['encoding']), body['row'],
                                                   body['messages'],body['definitions'],body.get('max_new_tokens',128),
                                                   cancelled=lambda:(root/'pause-practice').exists())
                        else:
                            result = decide(model, get_tokenizer(self.cfg['encoding']), body['row'])
                    stage('inference_complete')
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
