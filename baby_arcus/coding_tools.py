"""Small executable capability catalog for initial coding practice."""
from baby_arcus.tool_catalog import ToolCatalog
from baby_arcus.tool_search import search
from baby_arcus.tool_context import ToolContext


def definition(name, description, properties, required):
    return {'name': name, 'version': 1, 'description': description,
            'input_schema': {'type':'object','properties':properties,'required':required,'additionalProperties':False}}

TEXT = {'type':'string','maxLength':12000}
DEFINITIONS = [
    definition('tool_search','Find available capabilities by describing the task.',
               {'query':{'type':'string','maxLength':300},'limit':{'type':'integer','minimum':1,'maximum':5}},['query']),
    definition('read_file','Read inspect open Python source code in a file.',{'path':TEXT},['path']),
    definition('write_file','Write edit replace fix Python source code in a file.',{'path':TEXT,'content':TEXT},['path','content']),
    definition('run_tests','Run tests execute verify Python solution correctness.',{},[]),
    definition('read_docs','Read Python documentation functions sum set comprehension examples.',{'topic':TEXT},['topic']),
    definition('list_files','List files available in this Python task workspace.',{},[]),
    definition('search_code','Search a literal symbol or text in the task source code.',{'query':{'type':'string','maxLength':300}},['query']),
    definition('patch_file','Patch one exact occurrence in a Python source file.',{'path':TEXT,'old':TEXT,'new':TEXT},['path','old','new']),
]
DOCS = {'sum':'sum(iterable) adds its numbers. A comprehension can filter values: [x for x in values if condition].',
        'set':'set(iterable) keeps distinct values. len(collection) returns its number of elements.'}


class CodingTools:
    def __init__(self, environment):
        self.environment = environment
        self.catalog = ToolCatalog(DEFINITIONS)
        self.context = ToolContext(self.catalog)

    def execute(self, call):
        self.context.resolve(call)
        args, name = call['arguments'], call['name']
        if name=='tool_search':
            result=search(self.catalog, **args)
            self.context.accept(result)
            return result
        if name=='read_file':return self.environment.read(**args)
        if name=='write_file':return self.environment.write(**args)
        if name=='run_tests':return self.environment.tests()
        if name=='read_docs':return {'text': DOCS.get(args['topic'],'No documentation found; available topics: sum, set')}
        if name=='list_files':return {'files':['solution.py']}
        if name=='search_code':
            text=self.environment.read('solution.py')['content']
            return {'matches':[{'line':number,'text':line[:500]} for number,line in enumerate(text.splitlines(),1) if args['query'] in line][:20]}
        if name=='patch_file':
            text=self.environment.read(args['path'])['content']
            if not args['old'] or text.count(args['old']) != 1:
                raise ValueError('Patch must match exactly once')
            return self.environment.write(args['path'],text.replace(args['old'],args['new'],1))
        raise ValueError('No registered provider')
