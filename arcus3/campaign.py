"""Pure campaign policy; no automatic stage advancement or background scheduling."""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

def session_deadline(stop_at,settings,now=None):
    """Explicit deadline overrides the bounded default for a chat start/resume."""
    from datetime import timedelta
    now=now or datetime.now(timezone.utc)
    end=datetime.fromisoformat(stop_at.replace('Z','+00:00')) if stop_at else now+timedelta(seconds=settings.get('default_session_seconds',7200))
    if end.tzinfo is None or not 0<(end-now).total_seconds()<=86400:
        raise ValueError('Session deadline must be timezone-aware, future and within 24 hours')
    return end

def accept_evaluation(state,result,checkpoint_sha,suite_sha,settings_sha,limit=.2):
    import copy,math
    if not state['evaluation_pending']:raise ValueError('No pending evaluation')
    if result.get('expanded_manifest_sha256')!=checkpoint_sha or not result.get('complete_generation') or not result.get('execution_complete'):
        raise ValueError('Incomplete or unrelated evaluation')
    if result.get('suite_sha256')!=suite_sha or result.get('settings_sha256')!=settings_sha:raise ValueError('Evaluation protocol mismatch')
    tier='full' if state['evaluation_pending']==['baseline-full'] else state['evaluation_pending'][0]
    if result.get('tier','full')!=tier:raise ValueError('Evaluation tier mismatch')
    nll=result['language']['nll']
    if not math.isfinite(nll) or nll>state.get('baseline_nll',nll)+limit:raise ValueError('NLL regression requires review')
    updated=copy.deepcopy(state);updated.setdefault('baseline_nll',nll)
    updated['evaluation_completed'].append({'update':state['updates'],'tier':tier,'nll':nll,'checkpoint_sha256':checkpoint_sha})
    updated['evaluation_pending']=[]
    return updated

def due(updates, schedule, final=False):
    tracks=[]
    if updates==0 or final or updates%schedule['full_every']==0:tracks.append('full')
    if updates>0 and updates%schedule['developmental_every']==0:tracks.append('developmental')
    if updates in schedule['early'] or (updates>0 and updates%schedule['light_every']==0):tracks.append('light')
    return ['full'] if 'full' in tracks else ['developmental'] if 'developmental' in tracks else tracks

def in_window(settings, now=None):
    if not settings['enabled']:return False
    if settings.get('mode')=='chat-deadline':
        start=datetime.fromisoformat(settings['start_at'].replace('Z','+00:00'))
        end=datetime.fromisoformat(settings['stop_at'].replace('Z','+00:00'))
        if start.tzinfo is None or end.tzinfo is None or not 0<(end-start).total_seconds()<=86400:
            raise ValueError('Explicit bounded timezone-aware session required')
        return start <= (now or datetime.now(timezone.utc)) < end
    local=(now or datetime.now(timezone.utc)).astimezone(ZoneInfo(settings['timezone']))
    minute=local.hour*60+local.minute
    for window in settings['windows']:
        start,end=[int(t.split(':')[0])*60+int(t.split(':')[1]) for t in (window['start'],window['end'])]
        if not 0<=start<end<=1440:raise ValueError('Use same-day windows; split overnight windows')
        if local.weekday() in window['weekdays'] and start<=minute<end:return True
    return False

def window_end(settings, now=None):
    from datetime import timedelta
    now=now or datetime.now(timezone.utc)
    if not in_window(settings,now):raise ValueError('Outside enabled window')
    if settings.get('mode')=='chat-deadline':return datetime.fromisoformat(settings['stop_at'].replace('Z','+00:00')).astimezone(timezone.utc)
    local=now.astimezone(ZoneInfo(settings['timezone']));minute=local.hour*60+local.minute
    ends=[]
    for w in settings['windows']:
        start,end=[int(t.split(':')[0])*60+int(t.split(':')[1]) for t in (w['start'],w['end'])]
        if local.weekday() in w['weekdays'] and start<=minute<end:
            ends.append((local.replace(hour=0,minute=0,second=0,microsecond=0)+timedelta(minutes=end)).astimezone(timezone.utc))
    return min(ends)

def validate(cfg):
    if cfg['schema']!='arcus3-backbone-adaptation-v1' or cfg['trainability']!='added-expert1-full-fp32':raise ValueError('Unsupported adaptation')
    if cfg['ceiling_input_tokens']!=12_000_000_000_000 or not 1<=cfg['stage_input_tokens']<=10_000_000:raise ValueError('Unapproved token stage')
    if cfg['depth_capacity']!=1 or cfg['layers']!=[3,7,11,15,19,23]:raise ValueError('Full-depth architecture required')
    if not 2<=cfg['max_length']<=512 or not 0<cfg['learning_rate']<=1e-4:raise ValueError('Qualification precision/budget')
    if cfg['save_every']<1 or cfg['accumulation']!=1:raise ValueError('Update-boundary accumulation-one required')
    for key in ('teacher_coefficient','expert_coefficient','gate_coefficient','router_coefficient'):
        if not 0<=cfg[key]<=1:raise ValueError('Invalid objective')
    return cfg
