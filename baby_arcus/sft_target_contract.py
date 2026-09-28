"""Classify supervised content, never execute or repair imported instructions."""
import html
import json
import re
from baby_arcus.coding_contracts import validate_call

VERSION = 'alpha-target-v1'
EXTERNAL = re.compile(r'\[/?external_agent_tool_(?:call|result)\b|"(?:external_tool|source_only_tool_call)"\s*:|<\|(?:tool|function)_', re.I)


def source_family(source):
    """A numbered lesson is not a new independently weighted data source."""
    return ':'.join(source.split(':')[:2])


def classify(text):
    if EXTERNAL.search(html.unescape(text)):
        return 'external_transcript'
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return 'text'
    if isinstance(value, dict) and {'name', 'version', 'arguments'} & set(value):
        try:
            validate_call(value)
            return 'action_prediction'
        except ValueError:
            return 'malformed_action'
    return 'text'


def validate_target(item):
    kind = classify(item['content'])
    declared = item.get('target_kind')
    if kind in ('external_transcript', 'malformed_action'):
        raise ValueError('Quarantine assistant target: ' + kind)
    if declared is not None and declared != kind:
        raise ValueError('Declared target kind differs from content')
    return kind


def annotate(record):
    import copy
    result = copy.deepcopy(record)
    for item in result['messages']:
        if item['role'] == 'assistant' and item.get('train', True):
            item['target_kind'] = validate_target(item)
    return result
