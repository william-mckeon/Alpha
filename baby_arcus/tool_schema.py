"""Bounded JSON argument schemas for dynamically registered tools."""
import math


def validate_schema(schema, depth=0):
    if depth > 6 or not isinstance(schema, dict): raise ValueError('Schema nesting limit')
    kind = schema.get('type')
    allowed = {'type', 'description'}
    if kind == 'object':
        allowed |= {'properties','required','additionalProperties'}
        props, required = schema.get('properties'), schema.get('required')
        if not isinstance(props,dict) or len(props)>64 or not isinstance(required,list) or any(not isinstance(k,str) for k in required):
            raise ValueError('Invalid object schema')
        if schema.get('additionalProperties') is not False or set(required)-set(props): raise ValueError('Closed schema required')
        for key, value in props.items():
            if not isinstance(key,str) or not 1<=len(key)<=128: raise ValueError('Invalid argument name')
            validate_schema(value,depth+1)
    elif kind == 'array':
        allowed |= {'items','maxItems'}
        if type(schema.get('maxItems')) is not int or not 0<=schema['maxItems']<=256: raise ValueError('Bounded array required')
        validate_schema(schema.get('items'),depth+1)
    elif kind == 'string':
        allowed.add('maxLength')
        if type(schema.get('maxLength')) is not int or not 1<=schema['maxLength']<=12000: raise ValueError('Bounded string required')
    elif kind in ('integer','number'):
        allowed |= {'minimum','maximum'}
        if any(type(schema.get(k)) not in (int,float) or not math.isfinite(schema[k]) for k in ('minimum','maximum')) or schema['minimum']>schema['maximum']:
            raise ValueError('Bounded number required')
    elif kind not in ('boolean','null'): raise ValueError('Unsupported schema type')
    if set(schema)-allowed: raise ValueError('Unsupported schema keyword')
    return schema


def validate_value(schema,value,depth=0):
    if depth>6: raise ValueError('Argument nesting limit')
    kind=schema['type']
    if kind=='object':
        if not isinstance(value,dict) or set(value)-set(schema['properties']) or set(schema['required'])-set(value): raise ValueError('Unexpected or missing tool arguments')
        for key,item in value.items(): validate_value(schema['properties'][key],item,depth+1)
    elif kind=='array':
        if not isinstance(value,list) or len(value)>schema['maxItems']: raise ValueError('Invalid array argument')
        for item in value: validate_value(schema['items'],item,depth+1)
    elif kind=='string':
        if not isinstance(value,str) or len(value)>schema['maxLength']: raise ValueError('Invalid string argument')
    elif kind in ('integer','number'):
        types=(int,) if kind=='integer' else (int,float)
        if type(value) not in types or not math.isfinite(value) or not schema['minimum']<=value<=schema['maximum']: raise ValueError('Invalid numeric argument')
    elif kind=='boolean':
        if type(value) is not bool: raise ValueError('Invalid Boolean argument')
    elif kind=='null':
        if value is not None: raise ValueError('Invalid null argument')
    else: raise ValueError('Unsupported schema type')
    return value
