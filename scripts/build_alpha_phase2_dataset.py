"""Build a CPU-only, versioned Phase 2 dataset; never enable training or approve SFT."""
import argparse
import itertools
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baby_arcus.contracts import digest
from baby_arcus.data_manifest import file_digest, build, validate_manifest
from baby_arcus.language_stream import atomic_json, documents, inventory
from baby_arcus.sft_source_lessons import demonstrate
from baby_arcus.sft_dataset import packing_report
from baby_arcus.data_staging import StagingStore


def round_robin(groups):
    for batch in itertools.zip_longest(*groups):
        for value in batch:
            if value is not None:
                yield value


def write_corpus(path, training, validation):
    """Preserve the existing reader's every-tenth-document held-out contract."""
    import zstandard
    held = iter(validation)
    count = padding = 0
    with path.open('wb') as raw, zstandard.ZstdCompressor(level=3).stream_writer(raw) as stream:
        for row in training:
            if count % 10 == 0:
                val = next(held, None)
                if val is None:
                    val = {'text': '', 'split': 'padding', 'reason': 'reserved validation slot'}
                    padding += 1
                stream.write((json.dumps(val, ensure_ascii=False)+'\n').encode())
                count += 1
            stream.write((json.dumps(row, ensure_ascii=False)+'\n').encode())
            count += 1
    return {'physical_records': count, 'empty_validation_slots': padding}


def select_coding(root, manifest, tokenizer, per_language):
    selected = {}; seen = set(); skipped = {'duplicate': 0, 'oversize': 0}
    for language in ('Python', 'JavaScript', 'Go', 'Rust'):
        rows = {'training': [], 'validation': []}; tokens = dict.fromkeys(rows, 0)
        targets = {'training': per_language, 'validation': math.ceil(per_language / 9)}
        paths = [item for item in manifest['files'] if '/'+language.lower()+'/' in '/'+item['path'].replace('_data_', '/')]
        if not paths: raise ValueError('Missing language source: '+language)
        for entry in paths:
            for number, text in documents(Path(root)/entry['path']):
                split = 'validation' if number % 10 == 0 else 'training'
                if tokens[split] >= targets[split]:
                    if all(tokens[k] >= targets[k] for k in rows): break
                    continue
                if len(text.encode()) > 65536:
                    skipped['oversize'] += 1; continue
                identity = digest(text)
                if identity in seen:
                    skipped['duplicate'] += 1; continue
                ids = tokenizer.encode(text)
                if len(ids) < 2: continue
                seen.add(identity)
                rows[split].append({'text':text, 'split':split, 'language':language,
                    'source_path':entry['path'], 'source_document':number,
                    'source_sha256':entry['sha256'], 'text_sha256':identity,
                    'target_tokens':len(ids)-1})
                tokens[split] += len(ids)-1
            if all(tokens[k] >= targets[k] for k in rows): break
        if any(tokens[k] < targets[k] for k in rows): raise ValueError('Insufficient source data: '+language)
        selected[language] = {'rows':rows, 'target_tokens':tokens}
        print(json.dumps({'event':'coding_selected','language':language,'target_tokens':tokens}),flush=True)
    return selected, skipped


def generated_spec(index):
    # Explicit synthetic literal mechanics. These are not solved repository tasks.
    old = f'def offset_{index}(value):\n    return value + {index}\n'
    new = f'def offset_{index}(value):\n    return value - {index+1}\n'
    return {'kind':'patch_file' if index % 2 == 0 else 'write_file',
            'old':old, 'new':new, 'source_path':f'synthetic/offset_{index}.py'}


def prepare(args):
    from arcus.tokenizer import get_tokenizer
    from scripts.train_arcus_to_baseline import SCHEDULE
    tokenizer = get_tokenizer('o200k_base')
    output = Path(args.output)
    if output.exists(): raise ValueError('Use a new version directory')
    output.mkdir(parents=True)
    report = {'schema':'alpha-phase2-dataset-v1','complete':False,'training_enabled':False,
              'human_approved':False,'model_loaded':False,'encoding':'o200k_base',
              'context_tokens':512,'original_schedule':list(SCHEDULE)}
    atomic_json(output/'report.json', report)
    source = json.loads(Path(args.coding_manifest).read_text())
    validate_manifest(source, verify_files=True)
    atomic_json(output/'full-coding-source-manifest.json',source)
    original_cfg = json.loads(Path(args.original_config).read_text())
    original = inventory(original_cfg['dataset_root'],original_cfg['source_patterns'])
    # Preserve the original corpus and checkpoint cursor exactly, not a sampled replacement.
    atomic_json(output/'original-corpus-inventory.json', original)
    atomic_json(output/'original-curriculum.json',json.loads(Path('configs/baby_arcus/test2_curriculum.json').read_text()))
    selected, skipped = select_coding(args.dataset_root,source,tokenizer,args.coding_tokens_per_language)
    train = list(round_robin([item['rows']['training'] for item in selected.values()]))
    validation = list(round_robin([item['rows']['validation'] for item in selected.values()]))
    corpus = output/'coding'; corpus.mkdir()
    layout = write_corpus(corpus/'balanced.jsonl.zst',train,validation)
    with (output/'coding-validation.jsonl').open('w',encoding='utf-8') as stream:
        for row in validation: stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    corpus_manifest = build(corpus, ['*.jsonl.zst'], list(selected), dataset_id='phase2-coding-v1')
    atomic_json(output/'coding-manifest.json',corpus_manifest)
    report['coding'] = {'scope':'balanced pilot; complete selected upstream shards separately indexed',
        'languages':{k:{'documents':{s:len(v) for s,v in item['rows'].items()},
                        'target_tokens':item['target_tokens']} for k,item in selected.items()},
        'deduplication':'exact document text across all four languages and both splits',
        'source_holdout':'original source document index modulo 10 equals zero',
        'skipped':skipped, **layout}
    # Verify emitted data against the exact held-out convention used by the trainer.
    actual = {'training':set(), 'validation':set()}
    for number,text in documents(corpus/'balanced.jsonl.zst'):
        actual['validation' if number%10==0 else 'training'].add(digest(text))
    if actual['training'] != {r['text_sha256'] for r in train} or actual['training'] & {r['text_sha256'] for r in validation}:
        raise ValueError('Coding split verification failed')
    records = {'training':[], 'validation':[]}; seen = set(); batches = []
    specs = [generated_spec(index) for index in range(args.lessons)]
    spec_path = output/'synthetic-lesson-specs.jsonl'
    spec_path.write_text(''.join(json.dumps(spec)+'\n' for spec in specs),encoding='utf-8')
    spec_hash = file_digest(spec_path)
    store = StagingStore(output/'staging.sqlite')
    try:
        corpus_batch = store.stage_sources(corpus_manifest)
        trajectories = output/'trajectories'; trajectories.mkdir()
        # Different numeric variants are held out; templates overlap deliberately and are disclosed.
        for index in range(args.lessons):
            split = 'validation' if index % 10 == 0 else 'training'
            spec = specs[index]
            rows, evidence = demonstrate(spec,output/'workspaces'/str(index),tokenizer,
                'alpha:synthetic-tool-mechanics-v1:'+str(index),'synthetic-mechanics:'+str(index),split,
                {'input_sha256':digest(spec),'lesson':{'source_file_sha256':spec_hash,
                  'source_session_sha256':digest(['synthetic-tool-mechanics-v1',index])}},
                adapter='alpha-synthetic-tool-lesson-v1')
            for row in rows:
                key = digest(row['messages'])
                if key in seen: continue
                seen.add(key); records[split].append(row)
            atomic_json(trajectories/(str(index)+'.json'),{'spec':spec,**evidence})
            if index % 50 == 0: print(json.dumps({'event':'trajectory_verified','lesson':index}),flush=True)
        for split, rows in records.items():
            packed = packing_report(rows,tokenizer)
            if packed['quarantined']: raise ValueError('Tool sequence packing failed')
            with (output/('react-'+split+'.jsonl')).open('w',encoding='utf-8') as stream:
                for row in rows: stream.write(json.dumps(row,ensure_ascii=False)+'\n')
            for start in range(0,len(rows),100):
                batches.append({'split':split,'id':store.stage(rows[start:start+100])})
            report.setdefault('react',{})[split] = {'examples':len(rows),'target_tokens':packed['target_tokens'],
                'windows':sum(r['windows'] for r in packed['accepted'])}
        report['react'].update(lessons=args.lessons,teacher='deterministic; actual tools executed and readback verified',
            scope='discovery, inspection, literal patch/write, rediscovery, verification; not general reasoning or coding mastery',
            validation_limit='unseen values using shared templates; not unseen task families',
            existing_sft='separate optional review-v3 package; no personal interactions auto-admitted')
        report['batches'] = batches; report['coding_batch'] = corpus_batch
    finally: store.close()
    # Original schedule remains one shared optimizer stream, alongside coding and action supervision.
    schedule = ['embodied']*len(SCHEDULE)+['coding_corpus','sft']
    atomic_json(output/'run-proposal.json',{'training_enabled':False,'additional_updates':1300,
        'evaluation_after_additional_updates':[130,650,1300], 'mixture':schedule,
        'sft_stream_meaning':'ReAct action-supervision records; additional reviewed conversations optional',
        'repetition':'none; stop on exhaustion','token_ceiling':1000000,
        'original_dataset_config':args.original_config,'coding_dataset_id':'phase2-coding-v1',
        'approved_batches':[], 'proposed_batches':[b['id'] for b in batches if b['split']=='training'],
        'held_out_batches':[b['id'] for b in batches if b['split']=='validation'],
        'proposed_coding_manifest':corpus_batch,'source_updates':37000})
    report['original'] = {'files':len(original['files']),'fingerprint':original['fingerprint'],
        'verification':'original names/sizes/mtimes retained; full original shards not copied or rehashed'}
    report['complete'] = True
    atomic_json(output/'report.json',report)
    files = {str(path.relative_to(output)):file_digest(path) for path in output.rglob('*')
             if path.is_file() and 'workspaces' not in path.parts and path.suffix != '.sqlite'}
    atomic_json(output/'checksums.json',files)
    return report


if __name__ == '__main__':
    from baby_arcus.runtime_contract import require_container
    require_container()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--coding-manifest',required=True)
    parser.add_argument('--dataset-root',default='/dataset')
    parser.add_argument('--original-config',default='configs/baby_arcus/alpha_dataset.container.json')
    parser.add_argument('--coding-tokens-per-language',type=int,default=250000)
    parser.add_argument('--lessons',type=int,default=360)
    args = parser.parse_args()
    if not 100 <= args.coding_tokens_per_language <= 10000000 or not 10 <= args.lessons <= 2000:
        parser.error('Invalid bounded build size')
    print(json.dumps(prepare(args)))
