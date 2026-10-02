"""Validate disposable warmup arms and seal an explicit selection; never train."""
import argparse
import json
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arcus3.checkpoint import digest
from arcus3.config import read
from baby_arcus.language_stream import atomic_json


def validate_protocol(config):
    if config.get('schema') != 'arcus3-alpha322-schedule-calibration-v1':
        raise ValueError('Unknown Alpha 3.2.2 calibration protocol')
    arms=config.get('arms',[]);values=[arm.get('warmup_input_tokens') for arm in arms]
    required_flags=('fresh_initialization_required','same_records_and_order_required',
                    'scheduler_replay_qualification_required','frozen_backbone_required',
                    'all_gradient_groups_required')
    if (config.get('model_label')!='alpha3.2.2' or config.get('lineage_id')!='alpha3.2.2-wsd-001'
            or values != [1_000_000,1_350_000,2_000_000] or len({arm.get('id') for arm in arms})!=3
            or config.get('provisional_selection_input_tokens') not in values
            or config.get('tokens_per_arm')!=250_000
            or config.get('peak_learning_rate')!=1e-5
            or config.get('maximum_nll_regression')!=0.02
            or config.get('selection_mode')!='explicit-human-or-user-authorized-review'
            or any(config.get(flag) is not True for flag in required_flags)
            or config.get('upstream_global_batch_reference',{}).get('warmup_input_tokens')!=4_194_304_000
            or config['upstream_global_batch_reference'].get('warmup_updates')!=2_000
            or config['upstream_global_batch_reference'].get('tokens_per_update')!=2_097_152
            or config['upstream_global_batch_reference'].get('use_as_local_selection') is not False):
        raise ValueError('Changed Alpha 3.2.2 calibration grid/provenance')
    return config


def assess(config, paths, selected):
    validate_protocol(config)
    arms = {arm['warmup_input_tokens']: arm['id'] for arm in config['arms']}
    if selected not in arms:
        raise ValueError('Selection is outside the frozen calibration grid')
    rows = []
    for path in map(Path, paths):
        item = read(path)
        if (item.get('schema') != 'arcus3-alpha322-warmup-arm-v1' or not item.get('disposable')
                or item.get('model_label') != config['model_label']
                or item.get('lineage_id') != config['lineage_id']
                or item.get('warmup_input_tokens') not in arms
                or item.get('arm_id') != arms.get(item.get('warmup_input_tokens'))
                or not config['tokens_per_arm']-8192 < item.get('input_tokens',0) <= config['tokens_per_arm']):
            raise ValueError('Calibration arm identity/budget mismatch')
        rows.append((path, item))
    if len(rows) != len(arms) or {item['warmup_input_tokens'] for _, item in rows} != set(arms):
        raise ValueError('Every frozen calibration arm is required exactly once')
    identity_fields = ('initialization_manifest_sha256', 'initialization_recipe_sha256',
                       'data_sha256', 'record_order_sha256')
    if (any(len({item[field] for _, item in rows}) != 1 for field in identity_fields)
            or len({item['input_tokens'] for _,item in rows})!=1):
        raise ValueError('Calibration arms did not share initialization and data order')
    eligible = []
    for path, item in rows:
        gradients = item.get('gradient_groups', {})
        delta = item.get('nll_after', math.inf) - item.get('nll_before', -math.inf)
        passed = (item.get('complete') is True and item.get('qualification_exact_replay') is True
                  and item.get('frozen_unchanged') is True and math.isfinite(delta)
                  and delta <= config['maximum_nll_regression']
                  and all(gradients.get(name, 0) > 0 for name in ('expert', 'router', 'gate')))
        if passed:
            eligible.append(item['warmup_input_tokens'])
    if selected not in eligible:
        raise ValueError('Selected warmup did not pass the frozen safety gates')
    if len({item['scheduler_qualification_sha256'] for _,item in rows})!=1:
        raise ValueError('Calibration arms used different replay qualifications')
    return {'schema': 'arcus3-alpha322-schedule-selection-v1', 'model_label': config['model_label'],
            'lineage_id': config['lineage_id'], 'selected_warmup_input_tokens': selected,
            'eligible_warmup_input_tokens': sorted(eligible),
            'selection_mode': config['selection_mode'],
            'calibration_config_sha256': digest(config['_path']),
            'arm_reports': {str(path.resolve()): digest(path) for path, _ in rows},
            'campaign_updates': 0,
            'note': 'All arm updates are disposable. This receipt selects configuration only and is not a training checkpoint.'}


def main(args):
    config = read(args.config);config['_path'] = args.config
    result = assess(config, args.results, args.select_warmup_input_tokens)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/arcus3/alpha322_schedule_calibration.json')
    parser.add_argument('--results', nargs='+', required=True)
    parser.add_argument('--select-warmup-input-tokens', type=int, required=True)
    parser.add_argument('--output', required=True)
    main(parser.parse_args())
