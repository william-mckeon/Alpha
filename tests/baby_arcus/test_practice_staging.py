import json
import unittest
from baby_arcus.practice_staging import examples
from baby_arcus.coding_policy import prepare_prompt
from baby_arcus.coding_tools import DEFINITIONS
from baby_arcus.tool_catalog import ToolCatalog
from baby_arcus.tool_context import ToolContext
from baby_arcus.tool_search import search


class PracticeStagingTests(unittest.TestCase):
    def test_exact_prompt_and_only_current_action_are_targets(self):
        context=[{'role':'system','content':'Actual selected schemas'},
                 {'role':'user','content':'Task'},{'role':'assistant','content':'Old action'},
                 {'role':'tool','content':'Old result'}]
        call={'name':'read_file','version':1,'arguments':{'path':'solution.py'}}
        intent={'kind':'intent','decision':{'input_messages':context,'call':call}}
        rows=examples('episode',[intent,{'kind':'outcome','result':{'content':'source'}}])
        self.assertEqual(rows[0]['messages'][0],context[0])
        self.assertFalse(rows[0]['messages'][2]['train'])
        self.assertEqual(json.loads(rows[0]['messages'][-1]['content']),call)
        self.assertEqual(examples('episode',[intent,{'kind':'outcome','result':{'status':'tool_error'}}]),[])

    def test_real_tokenizer_schema_selection_fits_existing_context(self):
        from arcus.tokenizer import get_tokenizer
        catalog=ToolCatalog(DEFINITIONS); context=ToolContext(catalog)
        result=search(catalog,'read inspect file',limit=1); context.accept(result)
        messages=[{'role':'user','content':'Inspect solution.py'},
                  {'role':'assistant','content':json.dumps({'name':'tool_search','version':1,'arguments':{'query':'read inspect file','limit':1}})},
                  {'role':'tool','content':json.dumps(result)}]
        packed,ids,selected=prepare_prompt(messages,context.definitions(),get_tokenizer('o200k_base'),384)
        self.assertLessEqual(len(ids),384)
        self.assertIn('read_file',[definition['name'] for definition in selected])
        self.assertEqual(packed[-1]['role'],'tool')
