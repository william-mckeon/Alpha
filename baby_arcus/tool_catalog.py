"""Registered tool definitions; search never creates a capability."""
import copy
from baby_arcus.contracts import digest
from baby_arcus.coding_contracts import validate_args, validate_call


class ToolCatalog:
    def __init__(self, definitions, allowed=None):
        self.tools = {}
        seen = set()
        for tool in definitions:
            if set(tool) != {'name', 'version', 'description', 'input_schema'}:
                raise ValueError('Invalid catalog definition')
            validate_call({'name': tool['name'], 'version': tool['version'], 'arguments': {}})
            if tool['name'] in seen or tool['input_schema'].get('type') != 'object':
                raise ValueError('Duplicate tool or invalid schema')
            seen.add(tool['name'])
            schema = tool['input_schema']
            if not isinstance(tool['description'],str) or not 1 <= len(tool['description']) <= 1000:
                raise ValueError('Bounded tool description required')
            if set(schema) != {'type','properties','required','additionalProperties'} or schema['additionalProperties'] is not False:
                raise ValueError('Closed object schema required')
            if not isinstance(schema['properties'],dict) or not isinstance(schema['required'],list) or set(schema['required']) - set(schema['properties']):
                raise ValueError('Invalid required properties')
            for spec in schema['properties'].values():
                if spec.get('type') not in ('string','integer'):
                    raise ValueError('Unsupported schema type')
                if spec['type'] == 'string' and (type(spec.get('maxLength')) is not int or not 1 <= spec['maxLength'] <= 12000):
                    raise ValueError('Bounded string schema required')
                if spec['type'] == 'integer' and (type(spec.get('minimum')) is not int or type(spec.get('maximum')) is not int or spec['minimum'] > spec['maximum']):
                    raise ValueError('Bounded integer schema required')
            if allowed is None or tool['name'] in allowed:
                self.tools[tool['name']] = copy.deepcopy(tool)
        self.version = digest(self.tools)

    def resolve(self, call):
        validate_call(call)
        tool = self.tools.get(call['name'])
        if not tool or tool['version'] != call['version']:
            raise ValueError('Tool unavailable or schema version changed')
        validate_args(tool['input_schema'], call['arguments'])
        return copy.deepcopy(tool)
