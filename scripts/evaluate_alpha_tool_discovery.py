"""Interface retrieval diagnostics, explicitly separate from learned Alpha policy."""
import copy
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.coding_tools import DEFINITIONS
from baby_arcus.tool_catalog import ToolCatalog
from baby_arcus.tool_search import search


def evaluate():
    cases=[('inspect source','read_file'),('execute tests correctness','run_tests'),('documentation sum','read_docs'),('list files','list_files')]
    definitions=copy.deepcopy(DEFINITIONS)
    for tool in definitions:
        if tool['name']=='run_tests': tool['name']='verify_solution'
    variants=[('registered',ToolCatalog(DEFINITIONS)),('renamed',ToolCatalog(definitions))]
    rows=[]
    for label,catalog in variants:
        for query,expected in cases:
            if label=='renamed' and expected=='run_tests': expected='verify_solution'
            found=search(catalog,query,limit=1)['tools']
            rows.append({'condition':label,'query':query,'expected':expected,'matched':bool(found and found[0]['name']==expected)})
    return {'cases':rows,'matched':sum(r['matched'] for r in rows),'total':len(rows),
            'evaluates':'deterministic retriever only, not Alpha task reasoning'}


if __name__=='__main__': print(json.dumps(evaluate()))
