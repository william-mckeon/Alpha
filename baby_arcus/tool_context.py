"""Task-local discovery receipts; invocation still revalidates permissions."""
from baby_arcus.contracts import canonical


class ToolContext:
    def __init__(self, catalog, max_bytes=16000):
        if 'tool_search' not in catalog.tools:
            raise ValueError('Bootstrap tool_search required')
        self.catalog, self.max_bytes = catalog, max_bytes
        self.version, self.names = catalog.version, {'tool_search'}
        self.recent = []

    def accept(self, result):
        if result['catalog_version'] != self.catalog.version:
            raise ValueError('Stale tool catalog')
        names = self.names | {t['name'] for t in result['tools']}
        for tool in result['tools']:
            if self.catalog.tools.get(tool['name']) != tool:
                raise ValueError('Unregistered search result')
        if len(canonical([self.catalog.tools[n] for n in sorted(names)]))>self.max_bytes:
            raise ValueError('Discovered tool context budget exhausted')
        self.names = names
        self.recent = [tool['name'] for tool in result['tools']]

    def definitions(self):
        order=['tool_search',*self.recent,*sorted(self.names-set(self.recent)-{'tool_search'})]
        return [self.catalog.tools[name] for name in order]

    def resolve(self, call):
        if not isinstance(call,dict) or self.version != self.catalog.version or call.get('name') not in self.names:
            raise ValueError('Search for a current tool definition before using it')
        return self.catalog.resolve(call)
