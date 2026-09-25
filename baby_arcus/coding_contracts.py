"""Versioned, bounded model-facing coding records."""
import json
import re
from baby_arcus.contracts import canonical, digest


def validate_args(schema, value):
    if not isinstance(value, dict) or set(value)-set(schema['properties']) or set(schema.get('required', []))-set(value):
        raise ValueError('Unexpected or missing tool arguments')
    for key, item in value.items():
        spec = schema['properties'][key]
        kind = spec['type']
        if kind == 'string' and (not isinstance(item, str) or len(item)>spec.get('maxLength', 12000)):
            raise ValueError('Invalid string argument: '+key)
        if kind == 'integer' and (type(item) is not int or not spec.get('minimum', 0)<=item<=spec.get('maximum', 100)):
            raise ValueError('Invalid integer argument: '+key)
    return value


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
