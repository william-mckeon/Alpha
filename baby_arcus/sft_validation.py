"""Versioned, bounded conversation records. Imported text is never an instruction."""
from baby_arcus.contracts import fields, canonical
import re


def validate(record):
    fields(record, ('version', 'source', 'group', 'split', 'messages'), ('provenance',))
    if 'provenance' in record:
        p=record['provenance']
        fields(p, ('adapter','input_sha256','target_kinds','terminal_result_observed'), ('lesson','dataset'))
        if (p['adapter'] not in ('local-agent-log-v1','hf-phase2b-v1','opencode-v1','opencode-literal-lesson-v1','alpha-synthetic-tool-lesson-v1') or not isinstance(p['input_sha256'],str)
                or not re.fullmatch('[a-f0-9]{64}',p['input_sha256'])
                or not isinstance(p['target_kinds'],list) or not p['target_kinds']
                or any(k not in ('text','action_prediction') for k in p['target_kinds'])
                or type(p['terminal_result_observed']) is not bool):
            raise ValueError('Invalid adapter provenance')
        if p['adapter'] == 'hf-phase2b-v1':
            from baby_arcus.hf_training_sources import SOURCES
            data = p.get('dataset', {})
            fields(data, ('repo','revision','license','outcome','identity_changes'))
            if (data['repo'] not in SOURCES or not re.fullmatch('[a-f0-9]{40}',data['revision'])
                    or not data['license'] or data['outcome'] != 'source_reports_success'
                    or not isinstance(data['identity_changes'],list)):
                raise ValueError('Invalid HF provenance')
        elif 'dataset' in p:
            raise ValueError('Dataset metadata requires HF adapter')
        if p['adapter'] in ('opencode-literal-lesson-v1','alpha-synthetic-tool-lesson-v1'):
            lesson = p.get('lesson')
            fields(lesson, ('source_file_sha256','source_session_sha256','source_path',
                            'teacher','original_task_solved','code_executed','verified_content_sha256'))
            if (lesson['teacher'] != 'deterministic' or lesson['original_task_solved'] is not False
                    or lesson['code_executed'] is not False or p['terminal_result_observed'] is not True
                    or not isinstance(lesson['source_path'], str) or not lesson['source_path']
                    or any(not isinstance(lesson[k], str) or not re.fullmatch('[a-f0-9]{64}', lesson[k])
                           for k in ('source_file_sha256','source_session_sha256','verified_content_sha256'))):
                raise ValueError('Invalid source lesson provenance')
        elif 'lesson' in p:
            raise ValueError('Lesson provenance requires its explicit adapter')
    if type(record['version']) is not int or record['version'] != 1:
        raise ValueError('Unsupported SFT format')
    if record['split'] not in ('training', 'validation', 'test'):
        raise ValueError('Invalid split')
    for key in ('source', 'group'):
        if not isinstance(record[key], str) or not 1 <= len(record[key]) <= 300:
            raise ValueError('Source and leakage group required')
    messages = record['messages']
    if not isinstance(messages, list) or not 2 <= len(messages) <= 64:
        raise ValueError('Expected 2..64 messages')
    assistant = False
    previous = None
    for item in messages:
        fields(item, ('role', 'content'), ('train',))
        if 'train' in item and (item['role'] != 'assistant' or type(item['train']) is not bool):
            raise ValueError('Only assistant messages may have a Boolean train flag')
        if item['role'] not in ('system', 'user', 'assistant', 'tool'):
            raise ValueError('Unknown role')
        if not isinstance(item['content'], str) or not item['content']:
            raise ValueError('Nonempty textual content required; serialize tool calls explicitly')
        if item['role'] == 'tool' and previous not in ('assistant','tool'):
            raise ValueError('Tool observation must follow an assistant action')
        if re.search(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\bhf_[A-Za-z0-9]{20,}|\bsk-[A-Za-z0-9_-]{20,}',item['content']):
            raise ValueError('Possible credential requires redaction before staging')
        assistant |= item['role'] == 'assistant' and item.get('train', True)
        previous = item['role']
    if not assistant or messages[0]['role'] not in ('system', 'user'):
        raise ValueError('Missing prompt or assistant target')
    if len(canonical(record)) > 262144:
        raise ValueError('SFT record exceeds limit')
    return record
