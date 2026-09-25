"""Create only a fresh tiny learner and synthetic data for service qualification."""
import importlib.metadata
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.runtime_contract import require_container
require_container()
from baby_arcus.shared_factory import initialize
from baby_arcus.language_stream import atomic_json
from baby_arcus.data_manifest import build
import zstandard


if __name__=='__main__':
    root=Path(sys.argv[1]).resolve()
    if Path('/app/runs/test2') not in root.parents or (root/'config.json').exists():
        raise ValueError('Fresh fixture root required')
    root.mkdir(parents=True,exist_ok=True)
    corpus=root/'corpus'; corpus.mkdir()
    text='\n'.join(json.dumps({'text':'def add(a, b): return a + b\nUse a test to verify the result.'}) for _ in range(12))
    (corpus/'python.jsonl.zst').write_bytes(zstandard.ZstdCompressor().compress(text.encode()))
    atomic_json(root/'dataset.json',{'dataset_root':str(corpus),'source_patterns':['*.zst']})
    plan=json.loads(Path('configs/baby_arcus/alpha_three_stage.json').read_text())
    plan.update(fixture=True,staging_store='/review/staging.sqlite',mixture=['embodied','coding_corpus','sft'],token_budget=1024)
    atomic_json(root/'plan.json',plan)
    cfg={'schema':'arcus-test2-v1','initialization':'random','root':str(root/'learner'),
         'seed':2101,'preset':'tiny','text_dim':16,'depth_capacity':1.,'encoding':'o200k_base',
         'tiktoken_version':importlib.metadata.version('tiktoken'),'learning_rate':.00001,
         'dataset_config':str(root/'dataset.json'),'three_stage_config':str(root/'plan.json'),
         'max_storage_bytes':1024**3,'max_graph_records':100,
         'idle_learning':{'auto_resume':True,'idle_seconds':60,'chunk_updates':3,'checkpoint_every':3,'session_updates':3}}
    atomic_json(root/'config.json',cfg); initial=initialize(str(root/'config.json'))
    atomic_json(root/'learner'/'three-stage-continuation.json',{'fixture':True,'sha256':initial['sha256']})
    (root/'learner'/'pause-training').touch()
    atomic_json(root/'corpus-manifest.json',build(corpus,['*.zst'],['Python'],fixture=True))
    # Exercise the same reviewed source-lesson path using explicitly synthetic data.
    from scripts.prepare_alpha_source_lessons import prepare
    from arcus.tokenizer import get_tokenizer
    source = root/'source-fixture.jsonl'
    source.write_text(json.dumps({'meta':{'session_id':'fixture-source-lesson'},'messages':[],
        'completion':{'tool_calls':[{'type':'function','function':{'name':'edit_file',
            'arguments':{'path':'synthetic.py','old_string':'return 0','new_string':'return 1'}}}]}}),encoding='utf-8')
    lesson_report = prepare(source,root/'source-lessons',get_tokenizer('o200k_base'),fixture=True)
    if lesson_report['accepted_lessons'] != 1: raise ValueError('Source lesson fixture failed')
    evidence=json.loads((root/'source-lessons/lesson-0.json').read_text())
    atomic_json(root/'source-lesson-records.json',evidence['records'])
