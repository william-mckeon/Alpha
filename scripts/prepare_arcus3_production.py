"""Pin production source inventories, admit local inbox files, and prepare batches."""
import argparse,json,os,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from arcus3.config import read
from arcus3.checkpoint import digest
from arcus3.production import validate_policy
from arcus3.production_data import prepare,normalize
from baby_arcus.language_stream import atomic_json

SOURCES=[
 ('general','HuggingFaceFW/fineweb-edu','87f09149ef4734204d70ed1d046ddc9ca3f2b8f9',['data/'], 'ODC-By-1.0; source web terms retained'),
 ('general','mlfoundations/dclm-baseline-1.0','a3b142c183aebe5af344955ae20836eb34dcf69b',['global-shard_'], 'ODC-BY; source web terms retained; public baseline differs from donor filtering'),
 ('code','bigcode/the-stack-dedup','17cad72c886a2858e08d4c349a00d6466f54df63',['data/'], 'Per-record explicit permissive licenses only; attribution retained'),
 ('math','HuggingFaceTB/finemath','e92b25a616738fe95dc186b64dfb19f9c8525594',['finemath-4plus/','infiwebmath-4plus/'], 'ODC-By-1.0; source web terms retained'),
 ('instruction_tools','HuggingFaceTB/smoltalk','5feaf2fd3ffca7c237fc38d1861bc30365d48ffa',
  ['data/smol-magpie-ultra/train-','data/smol-constraints/train-','data/smol-rewrite/train-','data/smol-summarize/train-','data/apigen-80k/train-','data/everyday-conversations/train-'],
  'New smol components Apache-2.0; APIGen retains original Synth-APIGen/xLAM and Llama-generated-data terms; not relicensed'),
]

def catalog(root,inbox):
    from dotenv import load_dotenv
    from huggingface_hub import HfApi
    load_dotenv('.env',override=True);api=HfApi(token=os.getenv('HF_TOKEN'))
    root=Path(root);root.mkdir(parents=True,exist_ok=True);sources=[]
    for category,repo,revision,prefixes,terms in SOURCES:
        paths=api.list_repo_files(repo,repo_type='dataset',revision=revision)
        files=sorted(p for p in paths if p.endswith(('.parquet','.jsonl.zst')) and any(p.startswith(v) for v in prefixes))
        if not files:raise ValueError('No reviewed shards found: '+repo)
        # Deterministic interleaving avoids exhausting the first component/language first.
        import random
        random.Random(2101).shuffle(files)
        sources.append({'category':category,'repo_id':repo,'revision':revision,'terms':terms,'files':[{'path':p} for p in files]})
    original=Path('runs/test2/tool-correction-data-v4/records.jsonl').resolve()
    local=[{'path':str(original),'sha256':digest(original)}];admitted=root/'admitted';admitted.mkdir(exist_ok=True)
    inbox=Path(inbox);inbox.mkdir(parents=True,exist_ok=True);events=[]
    for path in sorted(inbox.glob('*.jsonl')):
        if path.is_symlink():raise ValueError('Inbox symlink refused')
        source_sha=digest(path);dest=admitted/(source_sha+'.jsonl');rejected={};accepted=0
        if not dest.exists():
            temporary=dest.with_suffix('.tmp')
            with path.open(encoding='utf-8') as f,temporary.open('w',encoding='utf-8') as out:
                for line in f:
                    try:
                        row=json.loads(line);normalize(row,'local')
                        out.write(json.dumps(row)+'\n');accepted+=1
                    except (ValueError,KeyError,TypeError):rejected['invalid_or_sensitive']=rejected.get('invalid_or_sensitive',0)+1
            if digest(path)!=source_sha:raise ValueError('Inbox file changed during admission')
            temporary.replace(dest)
            atomic_json(dest.with_suffix('.receipt.json'),{'source_sha256':source_sha,'accepted':accepted,'rejected':rejected,
                        'heldout_and_token_checks':'enforced during batch encoding; no content echoed'})
        if dest.stat().st_size:local.append({'path':str(dest.resolve()),'sha256':digest(dest)})
        events.append({'source_sha256':source_sha,'admitted_file':dest.name})
    order_path=root/'local-order.json'
    ordered=read(order_path) if order_path.exists() else []
    for item in local:
        if item not in ordered:ordered.append(item)
    atomic_json(order_path,ordered)
    sources.append({'category':'local','local':True,'files':ordered,'terms':'User-authorized train split only; private, never uploaded'})
    value={'schema':'arcus3-source-catalog-v1','reviewed':True,'sources':sources,'inbox':events,
           'scope':'Donor-published source families plus local; public filtered-source differences documented'}
    from arcus3.production import identity
    path=root/('catalog-'+identity(value)+'.json')
    if not path.exists():atomic_json(path,value)
    return path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache-root',required=True);p.add_argument('--inbox',required=True)
    p.add_argument('--donor',required=True);p.add_argument('--output');p.add_argument('--exclusions',default='runs/arcus3/phase8-stage-final-001/exclusions.json')
    p.add_argument('--policy',default='configs/arcus3/production.json');a=p.parse_args()
    cfg=validate_policy(read(a.policy));path=catalog(Path(a.cache_root)/'sources',a.inbox)
    if a.output:result=prepare(a.output,a.donor,path,a.exclusions,cfg,a.cache_root)
    else:result={'catalog':str(path)}
    print(json.dumps({'catalog':str(path),'records':result.get('records'),'input_tokens':result.get('input_tokens_per_pass')}))
