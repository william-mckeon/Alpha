"""Versioned, bounded model-facing coding records."""
import json
import re
from baby_arcus.contracts import canonical, digest


def validate_args(schema, value):
    from baby_arcus.tool_schema import validate_schema, validate_value
    validate_schema(schema)
    return validate_value(schema, value)


def validate_call(call):
    if not isinstance(call, dict) or set(call) != {'name', 'version', 'arguments'}:
        raise ValueError('Expected name, version and arguments')
    if not isinstance(call['name'], str) or not re.fullmatch(r'[a-z_]{1,48}', call['name']):
        raise ValueError('Invalid tool name')
    if type(call['version']) is not int or call['version'] != 1 or not isinstance(call['arguments'],dict) or len(canonical(call))>16000:
        raise ValueError('Invalid tool version or oversized call')
    return call


def message(role, content):
    return {'role': role, 'content': content if isinstance(content, str) else json.dumps(content, sort_keys=True)}
