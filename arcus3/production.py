"""Production identities, token thresholds and explicit checkpoint transitions."""
import copy
import hashlib
import json
from pathlib import Path
from arcus3.checkpoint import digest
from arcus3.config import read

def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def thresholds(tokens, schedule):
    return {tier:(tokens//interval+1)*interval for tier,interval in schedule.items()}

def token_due(previous, current, schedule, final=False):
    if current<previous:raise ValueError('Exposure moved backwards')
    crossed=[tier for tier,interval in schedule.items() if current//interval>previous//interval]
    if final:crossed.append('full')
    return next(([tier] for tier in ('full','developmental','light') if tier in crossed),[])

def validate_policy(policy):
    if policy['schema']!='arcus3-production-v1' or policy['review_input_tokens']!=100_000_000:
        raise ValueError('Unapproved production review budget')
    expected_ceiling=4_000_000_000_000 if policy.get('model_label')=='alpha3.2.2' else 12_000_000_000_000
    if policy['ceiling_input_tokens']!=expected_ceiling or not 2<=policy['batch_input_tokens']<=10_000_000:
        raise ValueError('Unapproved production ceiling/batch')
    if policy['cache_limit_bytes']>50*1024**3 or policy['mixture']!={'general':.4,'code':.2,'math':.1,'instruction_tools':.2,'local':.1}:
        raise ValueError('Unapproved cache/mixture')
    if policy['evaluation']!={'light':1_000_000,'developmental':10_000_000,'full':100_000_000} or policy['retention']!={'recovery':2,'milestones':2}:
        raise ValueError('Unapproved evaluation/retention policy')
    if policy['nll_regression_limit']!=.2 or not policy['local_reuse'] or not policy['inbox_auto_admit']:
        raise ValueError('Unapproved production behavior')
    if policy.get('model_label')=='alpha3.2.2':
        if policy.get('lineage_id')!='alpha3.2.2-wsd-001':
            raise ValueError('Alpha 3.2.2 production lineage mismatch')
        selection=policy.get('schedule_selection',{})
        if selection.get('warmup_input_tokens') not in (1_000_000,1_350_000,2_000_000):
            raise ValueError('Alpha 3.2.2 warmup selection mismatch')
        receipt=selection.get('calibration_receipt_sha256')
        if policy.get('launch_ready') and (not isinstance(receipt,str) or len(receipt)!=64 or any(c not in '0123456789abcdef' for c in receipt)):
            raise ValueError('Launch-ready Alpha 3.2.2 needs a calibration receipt')
    return policy

def accept_donor_receipt(result, checkpoint_sha, tier, policy):
    if not result.get('complete') or result.get('checkpoint_sha256')!=checkpoint_sha or result.get('tier')!=tier:
        raise ValueError('Incomplete or unrelated donor evaluation')
    if result.get('protocol',{}).get('revision')!=policy['donor_evaluation_revision'] or result.get('lighteval_revision')!=policy['lighteval_revision']:
        raise ValueError('Donor evaluation protocol mismatch')
    if not result.get('donor_context') or result.get('model_context')!=result.get('donor_context') or not result.get('results'):
        raise ValueError('Donor context/results missing')
    return {'checkpoint_sha256':checkpoint_sha,'tier':tier,'protocol':result['protocol']['revision']}

def transition(state, receipt, manifest_sha, new_data, new_teacher, policy, same_data=False):
    """Only metadata changes; callers first verify and restore old tensors/RNG/optimizer."""
    if receipt['checkpoint_sha256']!=manifest_sha or receipt['old_data_sha256']!=state['data_sha256']:
        raise ValueError('Transition checkpoint mismatch')
    if receipt['new_data_sha256']!=new_data or receipt['new_teacher_sha256']!=new_teacher:
        raise ValueError('Transition data/teacher mismatch')
    if receipt['policy_sha256']!=identity(policy):raise ValueError('Transition policy mismatch')
    if state.get('production') and state['production']['policy_sha256']!=identity(policy):
        raise ValueError('Production policy changed')
    if state['accumulation_position']!=0 or state['evaluation_pending']:
        raise ValueError('Transition needs update boundary and accepted evaluation')
    result=copy.deepcopy(state)
    if not state.get('production'):
        result['preproduction_exposure']={'input_tokens':state['input_tokens'],'target_tokens':state.get('target_tokens'),
                                         'updates':state['updates'],'data_sha256':state['data_sha256']}
    if not same_data and not state.get('batch_complete'):raise ValueError('Previous batch is not complete')
    result['data_sha256']=new_data;result['teacher_sha256']=new_teacher
    result['production']={'campaign_id':policy['campaign_id'],'policy_sha256':identity(policy),
        'batch_id':receipt['batch_id'],'review_input_tokens':policy['review_input_tokens'],
        'next_evaluation':thresholds(state['input_tokens'],policy['evaluation'])}
    result.setdefault('transitions',[]).append(receipt)
    result.pop('stage_end_pending',None);result['batch_complete']=False
    if not same_data:result['stream']=None
    return result

def batch_receipt(checkpoint, data, teacher, policy, batch_id):
    from arcus3.expanded_checkpoint import verify
    old=read(Path(checkpoint)/'manifest.json');verify(checkpoint,old['parent_sha256'])
    t=read(Path(teacher)/'manifest.json');data_sha=digest(Path(data)/'manifest.json')
    if t['data_sha256']!=data_sha:raise ValueError('Teacher does not cover batch')
    return {'schema':'arcus3-production-transition-v1','checkpoint_sha256':digest(Path(checkpoint)/'manifest.json'),
        'old_data_sha256':old['data_sha256'],'new_data_sha256':data_sha,
        'new_teacher_sha256':digest(Path(teacher)/'manifest.json'),'policy_sha256':identity(policy),'batch_id':batch_id}

def disk_budget(root, limit, reserve=12*1024**3):
    import shutil
    root=Path(root)
    used=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    if used>limit or shutil.disk_usage(root).free<reserve:raise RuntimeError('Production cache/storage budget exhausted')
    return used
