"""Delayed action outcomes; labels never enter the observation used for inference."""
from io import BytesIO
from PIL import Image
from baby_arcus.shared_experience import validate
from baby_arcus.body_dynamics import JOINTS

def body_vector(row):
    s=row['senses']
    return [s['joint_positions'][key] for key in JOINTS]+[s['height'],s['tilt']['roll'],s['tilt']['pitch']]+[
        float(s['contacts'][key]) for key in ('front_left','front_right','rear_left','rear_right')]+[float(s['supported'])]

def targets(before,outcome,later):
    validate(before);raw=validate(later);immediate=outcome['after'];validate(immediate)
    if outcome.get('experience_id')!=before['id'] or not outcome.get('action'):raise ValueError('Missing attempted action')
    if type(outcome.get('executed')) is not bool:raise ValueError('Missing verified execution result')
    own_change=outcome['action'].get('kind') in ('gaze','head','eyelids','move','sleep_when_ready','wake_voluntarily')
    if immediate['tick']<before['tick'] or (immediate['epoch']!=before['epoch'] and not (own_change and immediate['epoch']==before['epoch']+1)):
        raise ValueError('Unexplained immediate outcome boundary')
    if any(before[key]!=later[key] or before[key]!=immediate[key] for key in ('session','entity_id','scope_id')):
        raise ValueError('Delayed outcome crossed sensory scope')
    if later['epoch']!=immediate['epoch'] or not 1<=later['tick']-immediate['tick']<=30:
        raise ValueError('Delayed outcome requires 1–30 later ticks in the same visual epoch')
    if not 1<=later['tick']-before['tick']<=90:
        raise ValueError('Delayed prediction exceeds the 90-tick model horizon')
    labels={'future_body':body_vector(later),'action_quality':int(outcome['executed'])}
    if raw:
        with Image.open(BytesIO(raw)) as image:
            labels['future_rgb']=[value/255 for value in image.convert('RGB').resize((4,4)).tobytes()]
    return labels


def sequence_targets(transitions):
    """Join verified adjacent actions; reject unobserved boundaries or long gaps."""
    from copy import deepcopy
    if not 1<=len(transitions)<=8:raise ValueError('Action sequence requires 1–8 transitions')
    first=transitions[0][0];last=None;actions=[];accepted=True
    for before,outcome,later in transitions:
        targets(before,outcome,later)
        if last is not None and any(before[key]!=last[key] for key in ('id','session','entity_id','scope_id','epoch','tick')):
            raise ValueError('Noncontiguous action sequence')
        actions.append(deepcopy(outcome['action']));accepted=accepted and outcome['executed'];last=later
    horizon=last['tick']-first['tick']
    if not 1<=horizon<=90:raise ValueError('Sequence exceeds 90-tick horizon')
    row=deepcopy(first);row['executed_action']={'kind':'sequence','actions':actions};row['prediction_horizon']=horizon
    label={'future_body':body_vector(last),'action_quality':int(accepted)}
    raw=validate(last)
    if raw:
        with Image.open(BytesIO(raw)) as image:label['future_rgb']=[v/255 for v in image.convert('RGB').resize((4,4)).tobytes()]
    if any(not before['eligibility']['training'] or not later['eligibility']['training'] for before,_,later in transitions):row['eligibility']['training']=False
    return row,label
