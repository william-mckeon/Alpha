"""Deterministic versioned schedule; no implicit default mixture or approvals."""
from baby_arcus.contracts import digest
import json
import math
from pathlib import Path


def validate(config):
    language_only = config.get('schema') == 'alpha-phase2b-v1'
    if config.get('schema') not in ('alpha-three-stage-v1', 'alpha-phase2b-v1'):
        raise ValueError('Invalid training schema')
    if config.get('exhaustion_policy','stop') not in ('stop', 'repeat-sft'):
        raise ValueError('Unsupported exhaustion policy')
    if config.get('exhaustion_policy') == 'repeat-sft' and (not language_only or config.get('baseline_policy') not in ('random-initialization','tool-correction')):
        raise ValueError('SFT repetition requires an explicit fresh language-only plan')
    if not config.get('fixture') and config.get('exhaustion_policy') not in ('stop', 'repeat-sft'):
        raise ValueError('Explicit exhaustion policy required for real training')
    if 'target_total_updates' in config and (type(config['target_total_updates']) is not int or config['target_total_updates'] < 1):
        raise ValueError('Invalid total update budget')
    sequence = config.get('mixture')
    if not isinstance(sequence,list) or not 3 <= len(sequence) <= 128:
        raise ValueError('An explicit mixed schedule is required')
    expected = {'language','coding_corpus','sft'} if language_only else {'embodied','coding_corpus','sft'}
    if set(sequence) != expected:
        raise ValueError('Mixture must retain all three learning streams')
    if type(config.get('token_budget')) is not int or config['token_budget'] < 1:
        raise ValueError('Explicit additional target-token budget required')
    if config.get('training_enabled') is not True:
        raise ValueError('Training remains paused')
    if config.get('baseline_policy') == 'tool-correction':
        if ((config.get('source_updates'), config.get('target_total_updates')) not in ((40000,41000),(41000,60000))
                or config.get('evaluation_every') != 1000 or config.get('balanced_sft') is not True):
            raise ValueError('Correction requires an authorized frozen parent, bounded target and 1000-update evaluations')
    if language_only:
        from baby_arcus.context_contract import context_tokens
        context_tokens(config)
        window=config.get('language_window_tokens',64)
        if type(window) is not int or not 1 <= window <= config['context_tokens']:
            raise ValueError('Language window exceeds configured context')
    else:
        from baby_arcus.context_contract import context_tokens
        context_tokens(config)
        window=config.get('language_window_tokens',64)
        if type(window) is not int or not 1<=window<=config['context_tokens']:
            raise ValueError('Language window exceeds configured context')
    if config.get('preserve_embodied_schedule') is not (not language_only) or config.get('automatic_promotion') is not False:
        raise ValueError('Embodied retention and no automatic promotion are required')
    if config.get('resume_after_seconds') != 60:
        raise ValueError('Caregiver inactivity interval must remain 60 seconds')
    if not config.get('fixture'):
        try:
            gates=json.loads(Path(config['gates_file']).read_text(encoding='utf-8'))
            if (gates.get('training_authorized') is not True or digest(gates)!=config.get('gates_sha256')
                    or not valid_thresholds(gates.get('capability_thresholds'))
                    or (config.get('baseline_policy')=='tool-correction' and not valid_correction_gates(gates))):
                raise ValueError('Agreed, content-pinned acceptance gates required')
        except (OSError,KeyError,TypeError) as exc:
            raise ValueError('Real training acceptance gates unavailable') from exc
    return identity(config)


def family(config,index):
    validate(config)
    if type(index) is not int or index < 0:
        raise ValueError('Invalid durable mixture cursor')
    return config['mixture'][index % len(config['mixture'])]


def identity(config):
    # Locations and enablement are deployment state, not the learning algorithm.
    semantic = {key:value for key,value in config.items()
                if key not in ('staging_store','source_checkpoint','training_enabled')}
    return digest({'contract':'alpha-three-stage-context-v2','plan':semantic})


def migrate_state(state, plan, new_identity, updates):
    if state['plan_hash']==new_identity:return
    migration=plan.get('migration',{})
    if state['plan_hash']!=migration.get('previous_plan_hash') or updates!=migration.get('at_updates'):
        raise ValueError('Mixture, approvals or budget changed; prepare a new continuation')
    state.setdefault('plan_migrations',[]).append({'from':state['plan_hash'],'to':new_identity,'updates':updates})
    state['plan_hash']=new_identity


def valid_thresholds(values):
    keys={'max_retention_rate_drop','max_language_nll_increase','min_coding_solved','min_coding_gain'}
    return (isinstance(values,dict) and set(values)==keys
            and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in values.values())
            and values['max_retention_rate_drop']<=1
            and all(type(values[k]) is int for k in ('min_coding_solved','min_coding_gain')))


def valid_correction_gates(gates):
    values=gates.get('tool_correction',{})
    return (set(values)=={'min_valid_call_rate','min_discovery_solved','max_external_transcript_rate','max_nll_increase'}
            and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in values.values())
            and values['min_valid_call_rate']<=1 and values['max_external_transcript_rate']<=1
            and type(values['min_discovery_solved']) is int)
