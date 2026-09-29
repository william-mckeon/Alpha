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
    if operation not in ('download', 'inference') or project['authorization'].get(operation) is not True:
        raise ValueError('Operation not authorized')
    if any(project['authorization'].get(k) for k in ('training', 'cloud', 'publication')):
        raise ValueError('Phase 1 cannot authorize training, cloud or publication')

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
