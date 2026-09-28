"""Prepare a review candidate with prompt parity and a simulated pilot schedule.

Never approves records, changes weights or enables training.
"""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def review(source, teacher, output, config):
    from arcus.tokenizer import get_tokenizer
    from baby_arcus.contracts import digest
    from baby_arcus.sft_validation import validate
    from baby_arcus.sft_target_contract import classify,source_family
    from baby_arcus.conversation_format import content
    from baby_arcus.coding_policy import prepare_prompt
    from baby_arcus.sft_dataset import windows
    from baby_arcus.data_staging import StagingStore
    source,teacher,output,config=map(Path,(source,teacher,output,config))
    output.mkdir(parents=True,exist_ok=False);config.mkdir(parents=True,exist_ok=False)
    tokenizer=get_tokenizer('o200k_base');store=StagingStore(output/'review.sqlite')
    counts=Counter();targets=Counter();batches=[];batch=[];excluded=[];strata={};buckets={};seen={};groups={}
    parity_checks=0;prompt_migrations=[]
    teacher_report=json.loads((teacher/'report.json').read_text())
    for episode in teacher_report['episodes']:
        outcomes=[event['result'].get('passed') for event in episode['events'] if event['call']['name']=='run_tests']
        if not episode['verified'] or outcomes!=[False,True]:raise ValueError('Unverified teacher trajectory')
    def records():
        with (source/'records.jsonl').open(encoding='utf-8') as stream:
            for line in stream:yield json.loads(line)
        for episode in teacher_report['episodes']:yield from episode['records']
    try:
        with (output/'records.jsonl').open('x',encoding='utf-8') as stream:
            for record in records():
                if record['source']=='alpha:tool-discovery-v1':
                    excluded.append({'sha256':digest(record),'reason':'replaced_legacy_discovery_prompt','split':record['split']})
                    continue
                validate(record)
                trainable=[i for i,m in enumerate(record['messages']) if m['role']=='assistant' and m.get('train',True)]
                actions=[i for i in trainable if classify(record['messages'][i]['content'])=='action_prediction']
                if actions:
                    if len(trainable)!=1 or actions[0]!=len(record['messages'])-1:
                        raise ValueError('Action migration requires a single terminal target')
                    original_hash=digest(record)
                    marker='Available definitions: '
                    if marker not in record['messages'][0]['content']:raise ValueError('No trusted runtime tool definitions')
                    definitions=json.loads(record['messages'][0]['content'].split(marker,1)[1])
                    packed,_,_=prepare_prompt(record['messages'][:-1],definitions,tokenizer,16384-128)
                    messages=[dict(m,train=False) if m['role']=='assistant' else dict(m) for m in packed]+[record['messages'][-1]]
                    record=dict(record,messages=messages)
                    if digest(record)!=original_hash:
                        prompt_migrations.append({'source_sha256':original_hash,'derived_sha256':digest(record),
                                                  'reason':'runtime_prompt_canonicalization'})
                key=digest(record['messages']);split=record['split']
                if key in seen:
                    if seen[key]!=split:raise ValueError('Cross-split duplicate')
                    continue
                if record['group'] in groups and groups[record['group']]!=split:raise ValueError('Cross-split group')
                groups[record['group']]=split;seen[key]=split
                history=[];kinds=[]
                for item in record['messages']:
                    if item['role']=='assistant' and item.get('train',True):
                        kind=classify(item['content']);kinds.append(kind)
                        if kind=='action_prediction':
                            if len(tokenizer.encode(content(item)+'\n</assistant>\n'))>128:raise ValueError('Action exceeds decoding budget')
                            marker='Available definitions: '
                            if marker not in history[0]['content']:raise ValueError('Action missing runtime definitions')
                            definitions=json.loads(history[0]['content'].split(marker,1)[1])
                            _,ids,_=prepare_prompt(history,definitions,tokenizer,16384-128)
                            one=dict(record,messages=[dict(m,train=False) if m['role']=='assistant' else m for m in history]+[item])
                            window=next(windows(one,tokenizer,16384))
                            if ids!=[i for i,m in zip(window['ids'],window['mask']) if not m]:raise ValueError('Action prompt parity failed')
                            parity_checks+=1
                        if split=='training':
                            category=(kind,source_family(record['source']))
                            targets['/'.join(category)]+=1
                            buckets.setdefault(category,[]).append(digest(item))
                    history.append(item)
                stream.write(json.dumps(record,ensure_ascii=False)+'\n');counts[split]+=1
                if split=='training':
                    batch.append(record)
                    if len(batch)==20:batches.append(store.stage(batch));batch=[]
                elif split=='validation':
                    category=(source_family(record['source']),kinds[0])
                    strata.setdefault(category,[]).append((key,record))
            if batch:batches.append(store.stage(batch))
        prior_manifest=json.loads((source/'manifest.json').read_text())
        prior=StagingStore(source/'review.sqlite',readonly=True)
        try:source_id=store.stage_sources(prior.get(prior_manifest['pending_source_manifest'])['manifest'])
        finally:prior.close()
    finally:store.close()
    evaluation=[e[1] for row in itertools.zip_longest(*(sorted(strata[k],key=lambda x:x[0]) for k in sorted(strata))) for e in row if e is not None][:24]
    with (output/'evaluation.jsonl').open('x',encoding='utf-8') as stream:
        for record in evaluation:stream.write(json.dumps(record,ensure_ascii=False)+'\n')
    shutil.copy2(teacher/'report.json',output/'teacher-receipts.json')
    (output/'excluded.json').write_text(json.dumps(excluded,indent=2))
    (output/'prompt-migrations.json').write_text(json.dumps(prompt_migrations,indent=2))
    # Dry-run source exposure for 500 SFT steps in the proposed four-step mixture.
    keys={kind:sorted(k for k in buckets if k[0]==kind) for kind in sorted({k[0] for k in buckets})}
    schedule=Counter()
    for index in range(250):
        for kind in sorted(keys):
            category=keys[kind][index%len(keys[kind])];schedule['/'.join(category)]+=1
    def sha(path):
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
        return h.hexdigest()
    report={'training_authorized':False,'counts':counts,'targets_by_family':targets,'action_prompt_parity_checks':parity_checks,
            'excluded_legacy_records':len(excluded),'pending_batches':batches,'pending_source_manifest':source_id,
            'canonicalized_action_prompts':len(prompt_migrations),
            'records_sha256':sha(output/'records.jsonl'),'evaluation_sha256':sha(output/'evaluation.jsonl'),
            'teacher_sha256':sha(output/'teacher-receipts.json'),'proposed_500_sft_updates':schedule,
            'limitations':'Sampling counts are projected updates, not token equality. Repeated small validation cohort, not an untouched benchmark.'}
    (output/'review.json').write_text(json.dumps(report,indent=2))
    plan=json.loads(Path('configs/baby_arcus/alpha_tool_correction_training.json').read_text())
    gates=json.loads(Path('configs/baby_arcus/alpha_tool_correction_gates.json').read_text())
    plan.update(approved_batches=batches,approved_source_manifests=[source_id],gates_sha256=digest(gates))
    (config/'training.json').write_text(json.dumps(plan,indent=2));(config/'gates.json').write_text(json.dumps(gates,indent=2))
    shutil.copy2('configs/baby_arcus/alpha_tool_correction.json',config/'learner.json')
    shutil.copy2('runs/test2/alpha-tool-correction-run-config-v1/language.json',config/'language.json')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('source','teacher','output','config'):p.add_argument('--'+name,required=True)
    a=p.parse_args();r=review(a.source,a.teacher,a.output,a.config)
    print(json.dumps({k:v for k,v in r.items() if k!='pending_batches'}))
