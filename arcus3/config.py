"""Phase 1 scope, paths and bounded inference controls."""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

REPO = 'HuggingFaceTB/SmolLM2-1.7B-Instruct'
REVISION = '31b70e2e869a7173562077fd711b654946d38674'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def safe_child(root, name):
    root = Path(root).resolve()
    child = (root / name).resolve()
    if not child.is_relative_to(root) or child == root:
        raise ValueError('Path must be a child of the selected root')
    return child

def authorize(project, operation):
    if project['donor']['repo_id'] != REPO or project['donor']['revision'] != REVISION:
        raise ValueError('Unapproved donor revision')
    if operation not in ('download', 'inference','training','conversion','expanded_preflight') or project['authorization'].get(operation) is not True:
        raise ValueError('Operation not authorized')
    if any(project['authorization'].get(k) for k in ('cloud', 'publication')):
        raise ValueError('Cloud/publication not authorized by this runtime')
    if project['authorization'].get('training') and project.get('training_scope')!='dense-control-v1':
        raise ValueError('Explicit bounded dense-control scope required')
    if operation == 'conversion' and (project['authorization'].get('training') or project.get('conversion_scope') != 'selective-experts-parity-v1'):
        raise ValueError('Construction-only conversion scope required')
    if operation=='expanded_preflight' and (project['authorization'].get('training') or project.get('expanded_scope')!='qualification-v1'):
        raise ValueError('Expanded qualification-only scope required')


def validate_expanded(cfg):
    if cfg.get('schema')!='arcus3-expanded-preflight-v1' or cfg.get('freeze')!='base-weights':raise ValueError('Expanded scope')
    for key,cap in {'rank':8,'alpha':16,'max_length':512,'accumulation':2,'max_updates':8,'max_target_tokens':4096,'max_train_seconds':300,'save_every':2}.items():
        if type(cfg.get(key)) is not int or not 1<=cfg[key]<=cap:raise ValueError('Expanded budget '+key)
    if not 0<cfg['learning_rate']<=1e-4 or cfg['router_aux_coefficient']!=0.01:raise ValueError('Expanded objective')
    return cfg


def validate_conversion(cfg):
    expected = {'schema':'arcus3-selective-experts-v1','layers':[3,7,11,15,19,23],
                'experts':2,'top_k':1,'capacity_limit':None,'depth_enabled':False,
                'router_bias':False,'router_initialization':'zeros-first-index-tie',
                'output_scale':'unit','router_gradient':'selected-softmax-straight-through',
                'context':8192,'expected_parameters':2013390848}
    if cfg != expected: raise ValueError('Unapproved conversion architecture')
    return cfg

def validate_control(cfg):
    limits={'rank':(1,16),'alpha':(1,32),'max_length':(64,1024),'accumulation':(1,4),
            'max_updates':(1,64),'max_target_tokens':(1,32768),'max_train_seconds':(1,900),'save_every':(1,8)}
    for name,(low,high) in limits.items():
        if type(cfg.get(name)) is not int or not low<=cfg[name]<=high:raise ValueError('Control budget: '+name)
    if cfg['targets']!=['gate_proj','up_proj','down_proj'] or not 0<cfg['learning_rate']<=1e-4:
        raise ValueError('Unapproved adapter configuration')
    return cfg

def deadline(value):
    value = re.sub(r'\.(\d+)(?=Z|[+-]\d{2}:\d{2}$)',
                   lambda match: '.'+(match.group(1)+'000000')[:6], value)
    end = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if end.tzinfo is None:
        raise ValueError('Timezone required')
    remaining = (end - datetime.now(timezone.utc)).total_seconds()
    if not 0 < remaining <= 1800:
        raise ValueError('Deadline must be within the next 30 minutes')
    return end

def check_live(end, output):
    if datetime.now(timezone.utc) >= end or (Path(output) / 'pause-inference').exists():
        raise RuntimeError('Inference paused or deadline reached')


def validate_application(settings):
    limits={'max_input_tokens':(128,2048),'max_new_tokens':(1,128),'max_model_calls':(1,4),
            'max_tool_calls':(0,4),'max_turns':(1,8),'max_history_chars':(100,24000)}
    for key,(low,high) in limits.items():
        if type(settings.get(key)) is not int or not low<=settings[key]<=high:
            raise ValueError('Invalid application budget: '+key)
    if settings.get('tools')!=['echo','calculate'] or settings.get('persist_memory') is not False:
        raise ValueError('Default application must use approved tools and ephemeral memory')
    return settings
