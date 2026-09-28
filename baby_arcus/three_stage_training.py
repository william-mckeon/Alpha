"""Opt-in mixed continuation: original embodied learning, coding corpus, reviewed SFT.

The original trainer remains unchanged. One model, optimizer and checkpoint contain
all streams and their cursors. This module never approves, promotes or resumes data.
"""
import json
import time
from contextlib import nullcontext
from pathlib import Path
from baby_arcus.runtime_contract import model_device
import torch
from arcus.tokenizer import get_tokenizer
from baby_arcus import training_mixture
from baby_arcus.training_progress import emit, progress_line
from baby_arcus.data_staging import StagingStore
from baby_arcus.data_manifest import validate_manifest
from baby_arcus.sft_dataset import windows
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
from scripts.train_arcus_to_baseline import SCHEDULE


from baby_arcus.gpu_job_control import serialized

@serialized
def train(config, updates, checkpoint_every=1):
    cfg = read_config(config)
    root = Path(cfg['root'])
    def stage(name, **values):
        atomic_json(root/'training-stage.json', {'stage':name,'updated_at':time.time(),**values})
        if name != 'optimizer_step':
            emit(f"Stage: {name} | " + ' '.join(f'{k}={v}' for k,v in values.items()))
    if (root/'pause-training').exists():
        return {'updates_this_call':0,'stop_reason':'paused','mastery_established':False}
    from baby_arcus.routing_trace import load_mapping_config
    trace_config=load_mapping_config() if cfg.get('routing_telemetry',False) else None
    plan = json.loads(Path(cfg['three_stage_config']).read_text())
    identity = training_mixture.validate(plan)
    if plan['context_tokens'] != cfg.get('context_tokens',512):
        raise ValueError('Training plan and checkpoint context configuration differ')
    limit = cfg.get('max_job_updates', 64) if plan.get('baseline_policy') == 'random-initialization' else 64
    if type(limit) is not int or not 1 <= limit <= 256:
        raise ValueError('Invalid fresh-job update bound')
    if type(updates) is not int or not 1 <= updates <= limit or type(checkpoint_every) is not int or not 1 <= checkpoint_every <= limit:
        raise ValueError('Training request exceeds configured job bound')
    root = Path(cfg['root'])
    if not (root/'three-stage-continuation.json').exists():
        raise ValueError('An isolated, explicitly prepared continuation is required')
    fixture = plan.get('fixture') is True
    marker = json.loads((root/'three-stage-continuation.json').read_text())
    if marker.get('fixture') is not fixture:
        raise ValueError('Continuation fixture identity mismatch')
    if not fixture and plan.get('source_sha256') != marker.get('sha256'):
        raise ValueError('Continuation parent differs from frozen plan')
    if plan.get('baseline_policy') == 'random-initialization':
        initial = json.loads((root/'initial.json').read_text())
        if initial['updates'] != 0 or initial['sha256'] != marker['sha256']:
            raise ValueError('Fresh training requires the frozen zero-update checkpoint')
    stage('validating_dataset')
    with_store = StagingStore(plan['staging_store'],fixture=fixture,readonly=True)
    try:
        approved = with_store.approved_stream(plan['approved_batches'],allow_fixture=fixture)
        if len(plan['approved_source_manifests']) != 1:
            raise ValueError('One explicitly reviewed coding-source manifest required')
        coding = with_store.approved_sources(plan['approved_source_manifests'][0],allow_fixture=fixture)
        try:
            validate_manifest(coding,verify_files=True,cancelled=lambda:(root/'pause-training').exists())
        except InterruptedError:
            return {'updates_this_call':0,'stop_reason':'paused','mastery_established':False}
    finally:
        with_store.close()
    from baby_arcus.dataset_paths import resolve_root
    coding = dict(coding, root=str(resolve_root(coding)))
    if coding['schema'] in ('alpha-coding-corpus-v2','alpha-coding-corpus-v3','alpha-coding-corpus-v4'):
        coding['files'] = [dict(entry, mtime_ns=(Path(coding['root'])/entry['path']).stat().st_mtime_ns) for entry in coding['files']]
    if (root/'pause-training').exists():
        return {'updates_this_call':0,'stop_reason':'paused','mastery_established':False}
    tokenizer = get_tokenizer(cfg['encoding'])
    from baby_arcus.packed_training_store import PackedTrainingStore
    stage('preparing_sft_cache', device='cpu')
    packed_store=PackedTrainingStore(root/'packed-sft',approved,tokenizer,plan['context_tokens'],
                                    {'encoding':cfg['encoding'],'version':cfg['tiktoken_version']},
                                    interleave=plan.get('interleave_sft_sources',False),
                                    consumed_prefix=plan.get('sft_preserved_prefix',0),
                                    balance=plan.get('balanced_sft',False),
                                    max_bytes=min(cfg.get('packed_sft_max_bytes',1024**3),cfg['max_storage_bytes']//4))
    packing=packed_store.report
    atomic_json(root/'sft-packing-report.json',packing)
    if packing['quarantined']:
        packed_store.close()
        raise ValueError('Approved SFT does not fit the context; review sft-packing-report.json before training')
    lease = EmbodimentStore(root/'learner-lease')
    motor = None
    from baby_arcus.training_session import SESSION
    successful=False
    try:
        manifest = json.loads((root/'candidate.json').read_text())
        from scripts.prepare_alpha_fresh_training import preflight
        if plan.get('baseline_policy')=='random-initialization' and not fixture:
            preflight(config)
        stage('loading_model', device=model_device(cfg), updates=manifest['updates'])
        model,optimizer,progress = SESSION.acquire(root,manifest,cfg,model_device(cfg),verify_run)
        sources = source_manifest()
        state = progress.setdefault('three_stage',{'version':1,'plan_hash':identity,'index':0,
            'sft_cursor':0,'coding_cursor':{},'additional_target_tokens':0,'seconds':0.})
        if (plan.get('baseline_policy')=='tool-correction' and state['plan_hash']!=identity
                and progress['updates']==plan['source_updates'] and manifest['sha256']==plan['source_sha256']
                and not plan.get('migration')):
            progress.setdefault('previous_three_stage',[]).append(state)
            state={'version':1,'plan_hash':identity,'index':0,'sft_cursor':0,'coding_cursor':{},
                   'additional_target_tokens':0,'seconds':0.}
            progress['three_stage']=state
        if state['plan_hash'] != identity:
            training_mixture.migrate_state(state,plan,identity,progress['updates'])
        import itertools
        supervised = packed_store.resume(state['sft_cursor'])
        sustained = progress.setdefault('sustained',{'index':0,'motor':{},'corpus_cursor':{},'seconds':0.})
        if plan['schema'] != 'alpha-phase2b-v1':
            motor = MotorStream(sustained['motor'])
        original_cfg = json.loads(Path(cfg['dataset_config']).read_text())
        original = inventory(original_cfg['dataset_root'],original_cfg['source_patterns'])
        if sustained.get('corpus_fingerprint',original['fingerprint']) != original['fingerprint']:
            raise ValueError('Original training corpus changed')
        sustained['corpus_fingerprint'] = original['fingerprint']
        window=plan.get('language_window_tokens',64)
        reserve = sum(p.numel()*p.element_size() for p in model.parameters())*4
        cache_options={'cache_root':root/'packed-corpus','tokenizer_identity':{'encoding':cfg['encoding'],'version':cfg['tiktoken_version']},
                       'cancelled':lambda:(root/'pause-training').exists(),
                       'cache_max_bytes':min(1024**3,cfg['max_storage_bytes']//4),
                       'cache_reserve':lambda size:check(root,cfg['max_storage_bytes'],reserve+size)}
        # Small shards benchmark faster with the existing streaming reader.
        # Indexing is explicit until a workload demonstrates an advantage.
        if not cfg.get('indexed_corpus',False):cache_options={}
        original_stream = corpus_windows(original,tokenizer,sustained['corpus_cursor'],window,**cache_options)
        coding_stream = corpus_windows(coding,tokenizer,state['coding_cursor'],window,**cache_options)
        if plan.get('interleave_corpus_files',False):
            from baby_arcus.interleaved_corpus import windows as interleaved
            original_stream=interleaved(original,tokenizer,sustained['corpus_cursor'],window)
            coding_stream=interleaved(coding,tokenizer,state['coding_cursor'],window)
        check(root,cfg['max_storage_bytes'],reserve)
        started = time.monotonic(); completed = 0; reason = 'update_budget'
        chunk_started = started
        last_display = started
        def commit():
            nonlocal manifest,started
            if source_manifest() != sources:
                raise RuntimeError('Runtime sources changed during training')
            verify_depth(model,cfg); check(root,cfg['max_storage_bytes'],reserve)
            state['seconds'] += time.monotonic()-started
            progress['source_manifest'] = sources
            manifest = save(root,model,optimizer,progress)
            atomic_json(root/'candidate.json',manifest)
            atomic_json(root/'three-stage-progress.json',{'candidate':manifest,'state':state,'mastery_established':False})
            emit(f"Checkpoint saved: step {manifest['updates']:,} | sha256={manifest['sha256']}")
            started = time.monotonic()
        for offset in range(updates):
            if progress['updates'] >= plan.get('target_total_updates', 2**63):
                reason = 'total_update_budget'; break
            if (root/'pause-training').exists():
                reason = 'paused'; break
            if state['additional_target_tokens'] >= plan['token_budget']:
                reason = 'token_budget'; break
            stream = training_mixture.family(plan,state['index'])
            index = sustained['index']
            family = SCHEDULE[index % len(SCHEDULE)] if stream == 'embodied' else stream
            cursor = None
            if family in ('standing','lying','sitting'):
                row = motor.observe(family)
                head = 'body' if family == 'standing' else family
                model.eval()
                with torch.no_grad():
                    choice = int(torch.distributions.Categorical(logits=model([row],tokenizer,requested=(head,))[head]).sample()[0])
                outcome = motor.advance(family,choice)
                target = {'policy':{'head':head,'index':choice,'advantage':outcome['reward']}}
            elif family in ('language','coding_corpus','sft'):
                row,_,_ = example(index,'training','commands')
                row['hearing'] = []; row.pop('language_prefix_ids',None)
                if family == 'sft':
                    try:
                        target = {'sft':next(supervised)}
                    except StopIteration:
                        if plan.get('exhaustion_policy') != 'repeat-sft' or packed_store.count == 0:
                            reason = 'sft_exhausted'; break
                        # Commit epoch and cursor only after a successful optimizer step.
                        supervised = packed_store.resume(0)
                        target = {'sft':next(supervised)}
                    count = sum(target['sft']['mask'][1:])
                else:
                    try:
                        ids,cursor = next(original_stream if family == 'language' else coding_stream)
                    except InterruptedError:
                        reason = 'paused'; break
                    except StopIteration:
                        reason = family+'_exhausted'; break
                    target = {'tokens':ids}; count = len(ids)-1
                # Never silently overshoot or discard part of an approved example.
                if state['additional_target_tokens']+count > plan['token_budget']:
                    reason = 'next_window_exceeds_token_budget'; break
            else:
                row,target,_ = lesson(index//len(SCHEDULE),tokenizer,model,families=(family,))
            from torch.nn.attention import sdpa_kernel, SDPBackend
            stage('optimizer_step', device=str(next(model.parameters()).device), updates=progress['updates'], stream=stream)
            attention = sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION) if cfg.get('attention_backend') == 'efficient' else nullcontext()
            from baby_arcus.routing_trace import RoutingTrace
            sampled = trace_config is not None and (progress['updates']+1) % trace_config['training_sample_every'] == 0
            with (RoutingTrace(model, max_events=trace_config['training_max_events'], max_positions=trace_config['training_max_positions']) if sampled else nullcontext()) as trace:
                with attention:
                    result = step(model,optimizer,tokenizer,row,target)
            if trace is not None:
                atomic_json(root/('routing-update-'+str(progress['updates']+1)+'.json'), trace.report())
            progress['updates'] += 1; progress['trained_tokens'] += result['trained_tokens']
            state['additional_target_tokens'] += result['trained_tokens']; state['index'] += 1
            if stream == 'embodied': sustained['index'] += 1
            if family == 'language': sustained['corpus_cursor'] = cursor
            if family == 'coding_corpus': state['coding_cursor'] = cursor
            if family == 'sft':
                if state['sft_cursor'] == packed_store.count:
                    state['sft_epoch'] = state.get('sft_epoch', 0) + 1
                    state['sft_cursor'] = 0
                provenance=packed_store.metadata_at(state['sft_cursor'])
                exposure=state.setdefault('target_exposures',{}).setdefault(
                    provenance['source']+'/'+provenance['target_kind'],{'updates':0,'target_tokens':0,'loss_sum':0.})
                exposure['updates']+=1;exposure['target_tokens']+=result['trained_tokens'];exposure['loss_sum']+=result['loss']
                state['sft_cursor'] += 1
            progress['receipts'].append({'update':progress['updates'],'family':family,'plan_hash':identity,**result})
            metrics = state.setdefault('streams', {}).setdefault(stream, {'updates':0,'target_tokens':0})
            metrics['updates'] += 1
            metrics['target_tokens'] += result['trained_tokens']
            completed += 1
            now = time.monotonic()
            if completed == 1 or completed == updates or now-last_display >= 10:
                emit(progress_line(progress['updates'],plan.get('target_total_updates',progress['updates']),
                                   manifest['updates'],completed,now-chunk_started,result['loss'],stream))
                last_display = now
            if completed % checkpoint_every == 0: commit()
        if progress['updates'] != manifest['updates']: commit()
        SESSION.release(root,manifest,cfg)
        successful=True
        stage('chunk_complete', updates=manifest['updates'], stop_reason=reason)
        return {'candidate':manifest,'updates_this_call':completed,'stop_reason':reason,
                'additional_target_tokens':state['additional_target_tokens'],'mastery_established':False,
                'training_method':'alpha-three-stage-v1','exhaustion_policy':plan.get('exhaustion_policy','stop')}
    finally:
        if not successful:SESSION.clear()
        packed_store.close()
        if motor: motor.close()
        lease.close()
