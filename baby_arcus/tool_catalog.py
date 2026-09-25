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
            from baby_arcus.tool_schema import validate_schema
            validate_schema(schema)
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
