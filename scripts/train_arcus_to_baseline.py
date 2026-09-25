"""Resume the fresh .25 learner with continuous motor practice and corpus learning.

This writes candidates only. Reaching an update count is never a mastery claim.
"""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baby_arcus.runtime_contract import require_container
if __name__ == '__main__':
    require_container()
from baby_arcus.runtime_contract import model_device
import torch
from arcus.tokenizer import get_tokenizer
from baby_arcus.shared_checkpoint import load, save, restore_optimizer
from baby_arcus.shared_factory import read_config, verify_run, source_manifest
from baby_arcus.shared_depth import verify_depth
from baby_arcus.shared_objectives import step
from baby_arcus.shared_storage_budget import check
from baby_arcus.embodiment_store import EmbodimentStore
from baby_arcus.developmental_curriculum import lesson
from baby_arcus.shared_curriculum import example
from baby_arcus.sustained_curriculum import MotorStream, corpus_windows
from baby_arcus.language_stream import inventory, atomic_json

SCHEDULE = ('standing', 'lying', 'sitting', 'commands', 'color_reference', 'rest',
            'language', 'perception', 'causal', 'approach', 'continuity')


from baby_arcus.gpu_job_control import serialized

@serialized
def train(config, updates, checkpoint_every=1024):
    cfg = read_config(config)
    if cfg['depth_capacity'] not in (.25, 1.0) or not 1 <= updates <= 65536:
        raise ValueError('This run requires capacity .25 or 1.0 and 1..65536 updates')
    if not 1 <= checkpoint_every <= 4096:
        raise ValueError('Invalid checkpoint interval')
    root = Path(cfg['root'])
    lease = EmbodimentStore(root/'learner-lease')
    motor = None
    started = time.monotonic()
    try:
        manifest = json.loads((root/'candidate.json').read_text())
        model, data = load(root, manifest, model_device(cfg))
        verify_run(cfg, data)
        optimizer = restore_optimizer(model, data, cfg['learning_rate'])
        torch.set_rng_state(data['rng'])
        if torch.cuda.is_available() and data['cuda_rng']:
            torch.cuda.set_rng_state_all(data['cuda_rng'])
        progress = data['progress']
        del data
        sources = source_manifest()
        tokenizer = get_tokenizer(cfg['encoding'])
        state = progress.setdefault('sustained', {'index': 0, 'motor': {}, 'corpus_cursor': {}, 'seconds': 0.})
        motor = MotorStream(state['motor'])
        corpus_cfg = json.loads(Path(cfg['dataset_config']).read_text())
        corpus = inventory(corpus_cfg['dataset_root'], corpus_cfg['source_patterns'])
        if state.get('corpus_fingerprint', corpus['fingerprint']) != corpus['fingerprint']:
            raise ValueError('Training corpus changed across resume')
        state['corpus_fingerprint'] = corpus['fingerprint']
        language = corpus_windows(corpus, tokenizer, state['corpus_cursor'])
        reserve = sum(p.numel()*p.element_size() for p in model.parameters())*4
        check(root, cfg['max_storage_bytes'], reserve)
        last_commit = time.monotonic()
        corpus_exhausted = False
        def commit():
            nonlocal manifest, last_commit
            if source_manifest() != sources:
                raise RuntimeError('Runtime sources changed during training')
            verify_depth(model, cfg)
            check(root, cfg['max_storage_bytes'], reserve)
            state['seconds'] += time.monotonic()-last_commit
            progress['source_manifest'] = sources
            manifest = save(root, model, optimizer, progress)
            atomic_json(root/'candidate.json', manifest)
            atomic_json(root/f'sustained-{manifest["generation"]}.json', {
                'candidate': manifest, 'state': state, 'trained_tokens': progress['trained_tokens'],
                'parameters': sum(p.numel() for p in model.parameters()), 'mastery_established': False,
                'algorithm': 'continuous on-policy immediate-reward motor learning plus mixed supervised/reconstruction/corpus objectives'})
            last_commit = time.monotonic()
            print(json.dumps({'checkpoint': manifest}), flush=True)
        for offset in range(updates):
            if (root/'pause-training').exists():
                break
            index = state['index']
            family = SCHEDULE[index % len(SCHEDULE)]
            if family in ('standing', 'lying', 'sitting'):
                row = motor.observe(family)
                head = 'body' if family == 'standing' else family
                model.eval()
                with torch.no_grad():
                    choice = int(torch.distributions.Categorical(logits=model([row], tokenizer, requested=(head,))[head]).sample()[0])
                outcome = motor.advance(family, choice)
                # Actual sampled movement feedback; no scripted motor action labels.
                target = {'policy': {'head': head, 'index': choice, 'advantage': outcome['reward']}}
            elif family == 'language':
                try:
                    ids, cursor = next(language)
                except StopIteration:
                    corpus_exhausted = True
                    break
                row, _, _ = example(index, 'training', 'commands')
                row['hearing'] = []
                row['language_prefix_ids'] = ids[:1]
                target = {'tokens': ids}
                state['corpus_cursor'] = cursor
            else:
                row, target, actual = lesson(index//len(SCHEDULE), tokenizer, model, families=(family,))
            result = step(model, optimizer, tokenizer, row, target)
            progress['updates'] += 1
            progress['trained_tokens'] += result['trained_tokens']
            state['index'] += 1
            receipt = {'update': progress['updates'], 'experience_id': row['id'], 'family': family, **result}
            progress['receipts'].append(receipt)
            if (offset+1) % 64 == 0:
                atomic_json(root/'sustained-progress.json', {'committed': manifest, 'in_memory_updates': progress['updates'],
                    'trained_tokens': progress['trained_tokens'], 'motor': state['motor'],
                    'seconds_this_process': time.monotonic()-started, 'last': receipt, 'mastery_established': False})
                print(json.dumps({'updates': progress['updates'], 'family': family, 'loss': result['loss'],
                                  'trained_tokens': progress['trained_tokens']}), flush=True)
            if (offset+1) % checkpoint_every == 0 or offset+1 == updates:
                commit()
        if progress['updates'] != manifest['updates']:
            commit()
        return {'candidate': manifest, 'seconds': time.monotonic()-started,
                'corpus_exhausted': corpus_exhausted, 'mastery_established': False}
    finally:
        if motor:
            motor.close()
        lease.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/baby_arcus/test2.depth025-sustained.json')
    p.add_argument('--updates', type=int, default=4096)
    p.add_argument('--checkpoint-every', type=int, default=1024)
    a = p.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(train(a.config, a.updates, a.checkpoint_every)), flush=True)
