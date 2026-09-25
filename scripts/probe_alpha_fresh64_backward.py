"""Disposable CUDA backward qualification; never saves or promotes weights."""
import json
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lengths', type=int, nargs='+', default=[2048, 8192, 65536],
                        choices=[2048, 8192, 16384, 32768, 65536])
    parser.add_argument('--expandable-segments', action='store_true')
    args = parser.parse_args()
    if args.expandable_segments:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from arcus.tokenizer import get_tokenizer
    from baby_arcus.shared_checkpoint import load, digest
    from baby_arcus.shared_curriculum import example
    from baby_arcus.shared_objectives import step
    from baby_arcus.runtime_contract import require_gpu
    from baby_arcus.gpu_job_control import gpu_job

    require_gpu()
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.70)
    root = Path('/parent')
    manifest = json.loads((root / 'candidate.json').read_text())
    assert manifest['updates'] == 0 and manifest['depth_capacity'] == 1.0
    report = {'candidate': manifest, 'saved_updates': 0, 'lengths': [],
              'expandable_segments': args.expandable_segments,
              'full_64k_training_qualified': False}
    output = Path('/evidence/backward-qualification.json')
    try:
        with gpu_job():
            model, data = load(root, manifest, 'cuda')
            assert data['progress']['initialization'] == 'random'
            assert not data['progress']['sources']
            del data
            model.core.gradient_checkpointing = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
            tokenizer = get_tokenizer('o200k_base')
            row, _, _ = example(1, 'training', 'commands')
            row['hearing'] = []
            row.pop('language_prefix_ids', None)
            for length in args.lengths:
                torch.cuda.reset_peak_memory_stats()
                started = time.monotonic()
                entry = {'input_tokens': length, 'passed': False}
                report['lengths'].append(entry)
                output.write_text(json.dumps(report, indent=2))
                try:
                    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                        metrics = step(model, optimizer, tokenizer, row,
                                       {'tokens': [100 + i % 1000 for i in range(length + 1)]})
                    torch.cuda.synchronize()
                    entry.update(passed=True, metrics=metrics)
                except torch.OutOfMemoryError as exc:
                    entry['error'] = str(exc)
                    break
                finally:
                    entry.update(seconds=time.monotonic() - started,
                                 peak_allocated_bytes=torch.cuda.max_memory_allocated())
                    output.write_text(json.dumps(report, indent=2))
            report['full_64k_training_qualified'] = any(
                r['input_tokens'] == 65536 and r['passed'] for r in report['lengths'])
    finally:
        report['checkpoint_unchanged'] = digest(root / (manifest['generation'] + '.pt')) == manifest['sha256']
        output.write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
