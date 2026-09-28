"""Read-only developmental cohort on one hash-verified Docker CUDA checkpoint."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def evaluate(root, pointer, output, config='configs/baby_arcus/alpha_developmental_eval.json', mapping=False):
    import torch
    from arcus.tokenizer import get_tokenizer
    from baby_arcus.shared_checkpoint import load, digest
    from baby_arcus.runtime_contract import require_gpu
    from baby_arcus.gpu_job_control import gpu_job
    from baby_arcus.conversation_probe import respond
    from baby_arcus.developmental_evaluation import score, summarize, suite_identity
    from baby_arcus.model_inventory import inventory
    from baby_arcus.routing_trace import RoutingTrace, load_mapping_config
    from baby_arcus.language_stream import atomic_json
    from contextlib import nullcontext
    require_gpu()
    root, output = Path(root), Path(output)
    if output.exists():
        raise ValueError('Use a new output file; historical evaluations are immutable')
    cfg = json.loads(Path(config).read_text())
    map_cfg=load_mapping_config()
    if cfg['max_new_tokens'] > map_cfg['max_new_tokens']:
        raise ValueError('Generation exceeds mapping configuration')
    rows = [json.loads(line) for line in Path(cfg['prompts']).read_text().splitlines() if line]
    if len(rows) != 36 or len({r['id'] for r in rows}) != 36:
        raise ValueError('Expected the fixed 36-prompt cohort')
    identity = suite_identity([config, cfg['prompts'], cfg['rubric'], __file__,
        'baby_arcus/conversation_probe.py', 'baby_arcus/developmental_evaluation.py',
        'baby_arcus/routing_trace.py', 'baby_arcus/route_usage.py','baby_arcus/model_inventory.py',
        'configs/baby_arcus/alpha_mapping.json'])
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.70)
    with gpu_job():
        model, data = load(root, pointer, 'cuda')
        if data['progress']['updates'] != pointer['updates']:
            raise ValueError('Checkpoint update metadata mismatch')
        del data
        model.eval().requires_grad_(False)
        tokenizer = get_tokenizer('o200k_base')
        records, traces = [], []
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        for item in rows:
            # Exact streaming use counts for every prompt; detailed tensors are sampled.
            with (RoutingTrace(model,max_events=map_cfg['max_events'],max_positions=map_cfg['max_positions'],count_usage=True) if mapping else nullcontext()) as trace:
                response = respond(model, tokenizer, item['prompt'], cfg['max_new_tokens'], item.get('tool_schema'))
            records.append({'id': item['id'], 'category': item['category'], **response, 'scores': score(item, response)})
            if trace is not None:
                traces.append({'prompt_id': item['id'], **trace.report()})
            print(json.dumps({'completed': item['id'], 'tokens': response['generated_tokens']}), flush=True)
        unchanged = digest(root/(pointer['generation']+'.pt')) == pointer['sha256']
        if not unchanged:
            raise ValueError('Checkpoint changed during evaluation')
        report = {'schema': 'alpha-development-v1', 'candidate': pointer, 'identity': identity,
                  'complete': True, 'checkpoint_unchanged': unchanged, 'records': records,
                  'summary': summarize(records), 'seconds': time.monotonic()-started,
                  'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'limitations': 'Small repeated single-seed cohort. Human judgments are pending. No mastery or 16k competence claim.'}
        atomic_json(output, report)
        if mapping:
            from baby_arcus.route_usage import aggregate
            mapped = {'candidate': pointer, 'checkpoint_unchanged': unchanged, 'inventory': inventory(model),
                      'traces': traces, 'mapping_config':{**map_cfg,'all_prompts_counted':True}, 'schema': 'alpha-map-v2', 'complete': True,
                      'routing_usage':aggregate([t['usage'] for t in traces],sum(r['generated_tokens'] for r in records))}
            mapped_path = output.with_name('mapping-'+str(pointer['updates'])+'.json')
            atomic_json(mapped_path, mapped)
            from scripts.map_alpha_routes import write_html
            write_html(mapped_path.with_suffix('.html'), mapped)
        return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True); p.add_argument('--pointer', default='candidate.json')
    p.add_argument('--output', required=True); p.add_argument('--config', default='configs/baby_arcus/alpha_developmental_eval.json')
    p.add_argument('--mapping', action='store_true')
    a = p.parse_args()
    pointer = json.loads((Path(a.root)/a.pointer).read_text())
    evaluate(a.root, pointer, a.output, a.config, a.mapping)
