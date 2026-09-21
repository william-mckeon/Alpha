"""Bind qualification to measured, immutable candidate-specific artifacts."""
import json,hashlib,math
from pathlib import Path
REQUIRED={'integration','posture','retention','transfer','pixels','live','recovery','curiosity'}
LIVE_CHECKS={'learned_standing','expression','listening','pause','paused_cursor_stable','resume','shutdown',
    'replay_recovery','language_replay','stale_response_rejected','human_override','delayed_replay','sequence_replay','memory_recovery','depth_capacity'}
GATES=Path(__file__).resolve().parents[1]/'configs/baby_arcus/shared_gates.json'
SOURCE_ROOT=Path(__file__).resolve().parents[1]
SOURCES=('baby_arcus/shared_model.py','baby_arcus/shared_runtime.py','baby_arcus/services/shared_worker.py',
    'baby_arcus/services/playroom.py','baby_arcus/shared_experience.py','baby_arcus/shared_replay.py',
    'baby_arcus/body_policy.py','baby_arcus/body_dynamics.py','baby_arcus/visual_model.py','baby_arcus/shared_temporal.py',
    'baby_arcus/shared_learning.py','baby_arcus/shared_checkpoint.py','baby_arcus/shared_pooling.py','baby_arcus/shared_depth.py',
    'baby_arcus/shared_memory.py','baby_arcus/shared_causal.py','baby_arcus/shared_causal_curriculum.py',
    'scripts/evaluate_arcus_shared_curiosity.py','scripts/qualify_arcus_shared_recovery.py',
    'baby_arcus/shared_continuity_model.py','baby_arcus/shared_continuity_session.py',
    'baby_arcus/shared_identity_context.py','baby_arcus/shared_object_memory.py','baby_arcus/shared_object_planning.py',
    'baby_arcus/services/shared_continuity_worker.py','baby_arcus/shared_continuity_qualification.py',
    'configs/baby_arcus/continuity_gates.json','baby_arcus/shared_continuity_curriculum.py',
    'baby_arcus/object_observation.py','baby_arcus/gaze.py',
    'scripts/evaluate_arcus_shared_object_continuity.py','scripts/evaluate_arcus_shared_object_tracks.py',
    'scripts/evaluate_arcus_shared_planning.py','scripts/qualify_arcus_continuity_service.py',
    'scripts/qualify_arcus_continuity_recovery.py')
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def source_snapshot():
    # Bind all learner and host modules, including backbone/expert routing and
    # sensory delivery. New runtime modules must not silently evade binding.
    paths=set(SOURCES)
    for directory in ('arcus','baby_arcus'):
        paths.update(path.relative_to(SOURCE_ROOT).as_posix() for path in (SOURCE_ROOT/directory).rglob('*.py'))
    paths.update(path.relative_to(SOURCE_ROOT).as_posix() for path in (SOURCE_ROOT/'baby_arcus/web').glob('*.png'))
    return {path:digest(SOURCE_ROOT/path) for path in sorted(paths)}
def number(value,default=-1e9):return value if type(value) in (int,float) and math.isfinite(value) else default

def compile_report(root,manifest,paths):
    if set(paths) not in (REQUIRED,REQUIRED|{'continuity'}):raise ValueError('All eight qualification artifacts are required')
    base=Path(root).resolve() if root else None;g=json.loads(GATES.read_text())
    reports={};evidence={}
    for name,path in paths.items():
        path=Path(path).resolve();value=json.loads(path.read_text(encoding='utf-8'))
        candidate=value.get('candidate',value.get('settings',{}).get('candidate',{}))
        if any(candidate.get(key)!=manifest[key] for key in ('generation','sha256')):
            raise ValueError(f'{name} used a different candidate')
        if value.get('runtime_sources')!=source_snapshot():raise ValueError(f'{name} used different runtime sources')
        reports[name]=value;evidence[name]={'path':path.relative_to(base).as_posix() if base and path.is_relative_to(base) else str(path),'sha256':digest(path)}
    i,p,r,t,v,l,q=(reports[key] for key in ('integration','posture','retention','transfer','pixels','live','recovery'))
    diagnostics=i.get('diagnostics',{});gradients=diagnostics.get('gradients',{})
    integration=(i.get('integration') is True and all(diagnostics.get(key) is True for key in ('authenticated_http','one_core','original_body_unchanged'))
        and all(gradients.get(key) is True for key in ('rgb','body_sensations','text','internal','pixels','gaze','future_body','future_rgb','action_quality','memory')))
    episodes=number(p.get('settings',{}).get('episodes'),0);results=p.get('results',{})
    posture=(episodes>=g['minimum_behavior_episodes'] and all(
        number(results.get(goal,{}).get('shared'))>=g['minimum_posture_success']*episodes
        and number(results.get(goal,{}).get('shared'))>=number(results.get(goal,{}).get('parent'),1e9)-g['maximum_parent_success_drop']*episodes
        for goal in ('standing','lying','sitting')))
    episodes=number(r.get('episodes'),0);approach=r.get('approach',{});nll=r.get('language_nll',{})
    retention=(posture and episodes>=g['minimum_behavior_episodes']
        and number(approach.get('shared'))>=g['minimum_approach_success']*episodes
        and number(approach.get('shared'))>=number(approach.get('parent'),1e9)-g['maximum_parent_success_drop']*episodes
        and number(r.get('language_examples'),0)>=200
        and number(nll.get('shared'),1e9)<=number(nll.get('parent'))+g['maximum_language_nll_increase'])
    scores=t.get('accuracy',{});commands=t.get('per_command',{})
    full=number(scores.get('color_reference:full'));negative=max(number(scores.get('color_reference:'+key),1e9) for key in ('no_rgb','no_hearing','shuffled_rgb'))
    cross_modal=(t.get('split')=='confirmation' and number(t.get('examples_per_condition'),0)>=200
        and number(scores.get('commands:full'))>=g['minimum_command_accuracy'] and set(commands)==set(g['required_commands'])
        and min(number(value) for value in commands.values())>=g['minimum_per_command_accuracy']
        and number(scores.get('rest:full'))>=g['minimum_rest_accuracy'] and full>=g['minimum_transfer_accuracy'] and full-negative>=g['minimum_ablation_gap']
        and v.get('split')=='confirmation' and number(v.get('examples'),0)>=1000
        and number(v.get('count_accuracy'))>=g['minimum_pixel_count_accuracy'] and number(v.get('ball_iou'))>=g['minimum_ball_iou']
        and number(v.get('surface_color_accuracy'))>=g['minimum_surface_color_accuracy']
        and number(v.get('count_accuracy'))-number(v.get('blank_count_accuracy'),1e9)>=.2)
    live=(l.get('live') is True and l.get('recovery') is True and l.get('actions_passed') is True and l.get('hearing_passed') is True
        and set(l.get('checks',{}))==LIVE_CHECKS and all(l['checks'][key] is True for key in LIVE_CHECKS)
        and q.get('recovery') is True and q.get('device')=='cuda'
        and q.get('mismatched_tensors')==[] and len(q.get('losses',[]))==2 and q['losses'][0]==q['losses'][1])
    depth=manifest.get('depth_capacity')==g['required_depth_capacity']
    c=reports['curiosity'];prediction=c.get('prediction',{});exploration=c.get('exploration',{})
    causal=(c.get('split')=='confirmation' and number(c.get('examples'),0)>=g['minimum_causal_examples']
        and number(c.get('scenes'),0)>=g['minimum_exploration_scenes'] and c.get('depth_capacity')==g['required_depth_capacity']
        and 0<=number(prediction.get('body'))<=g['maximum_body_persistence_ratio']*number(prediction.get('body_persistence'))
        and number(prediction.get('body_shuffled'))>=g['minimum_shuffled_action_ratio']*number(prediction.get('body'))
        and 0<=number(prediction.get('rgb'))<=g['maximum_rgb_persistence_ratio']*number(prediction.get('rgb_persistence'))
        and (number(exploration.get('learned',{}).get('discoveries'))-number(exploration.get('random',{}).get('discoveries'),1e9))/5>=g['minimum_discovery_gain']
        and number(exploration.get('learned',{}).get('discoveries'))>number(exploration.get('no_action',{}).get('discoveries'),1e9))
    continuity=True
    if 'continuity' in reports:
        from baby_arcus.shared_continuity_qualification import assess
        continuity=assess(base,manifest,reports['continuity'])
    return {'candidate':manifest,'sha256':manifest['sha256'],'integration':integration,'retention':retention,'depth':depth,
        'continuity':continuity if 'continuity' in reports else None,
        'cross_modal':cross_modal,'live':live,'curiosity':causal,'evidence':evidence,'gate_thresholds':g,'gate_source_sha256':digest(GATES),
        'measurements':{'commands':scores.get('commands:full'),'color_reference':scores.get('color_reference:full'),
            'depth_capacity':manifest.get('depth_capacity'),'causal_prediction':prediction,'exploration':exploration,
            'rest':scores.get('rest:full'),'pixel_count':v.get('count_accuracy'),'ball_iou':v.get('ball_iou'),
            'postures':results,'approach':approach,'language_nll':nll},
        'runtime_sources':source_snapshot(),
        'qualified':all((integration,retention,cross_modal,live,depth,causal,continuity))}

def verify_evidence(report,root=None):
    evidence=report.get('evidence',{})
    if set(evidence) not in (REQUIRED,REQUIRED|{'continuity'}):raise ValueError('Missing measured qualification evidence')
    paths={}
    for key,item in evidence.items():
        path=Path(item['path'])
        if not path.is_absolute():
            if root is None:raise ValueError('Qualification evidence root required')
            path=Path(root)/path
        if digest(path)!=item['sha256']:raise ValueError('Qualification evidence changed')
        paths[key]=path
    if report.get('gate_source_sha256')!=digest(GATES):raise ValueError('Qualification thresholds changed')
    if report.get('runtime_sources')!=source_snapshot():
        raise ValueError('Qualified runtime source changed')
    rebuilt=compile_report(root,report['candidate'],paths)
    if not rebuilt['qualified']:raise ValueError('Measured qualification gates not passed')
