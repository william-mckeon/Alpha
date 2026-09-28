"""Assemble pending review materials and a disabled runnable configuration."""
import argparse
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def stage(data, config, old_config, old_review, with_coding=False):
    from baby_arcus.data_staging import StagingStore
    from baby_arcus.contracts import digest
    from scripts.audit_alpha_fresh_holdouts import audit_splits
    data=Path(data);config=Path(config);config.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((data/'manifest.json').read_text())
    store=StagingStore(data/'review.sqlite');old=StagingStore(old_review,readonly=True)
    try:
        old_plan=json.loads((Path(old_config)/'training.json').read_text())
        source=old.approved_sources(old_plan['approved_source_manifests'][0])
        source_id=store.stage_sources(source)
        batches=list(manifest['batches_pending_review'])
        if with_coding:
            from arcus.tokenizer import get_tokenizer
            from baby_arcus.coding_curriculum import correction_lessons
            episodes=list(correction_lessons(data/'teacher-workspaces',get_tokenizer('o200k_base')))
            with (data/'coding-teacher-receipts.json').open('x') as stream:json.dump(episodes,stream,indent=2)
            with (data/'records.jsonl').open('a',encoding='utf-8') as stream:
                for episode in episodes:
                    batches.append(store.stage(episode['records']))
                    for record in episode['records']:stream.write(json.dumps(record)+'\n')
        plan=json.loads(Path('configs/baby_arcus/alpha_tool_correction_training.json').read_text())
        gates=json.loads(Path('configs/baby_arcus/alpha_tool_correction_gates.json').read_text())
        plan.update(approved_batches=batches,approved_source_manifests=[source_id],gates_sha256=digest(gates))
        (config/'training.json').write_text(json.dumps(plan,indent=2))
        (config/'gates.json').write_text(json.dumps(gates,indent=2))
        shutil.copy2('configs/baby_arcus/alpha_tool_correction.json',config/'learner.json')
        shutil.copy2(Path(old_config)/'language.json',config/'language.json')
        import hashlib
        # Round-robin strata so the fixed evaluation budget cannot select only
        # the alphabetically first source. Keep each record's first target kind.
        from baby_arcus.sft_target_contract import validate_target
        import itertools
        strata={}
        with (data/'records.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                record=json.loads(line)
                if record['split']!='validation':continue
                item=next(m for m in record['messages'] if m['role']=='assistant' and m.get('train',True))
                key=(record['source'],validate_target(item))
                strata.setdefault(key,[]).append((digest(record),record))
        selected=[entry[1] for row in itertools.zip_longest(*(sorted(strata[k],key=lambda x:x[0]) for k in sorted(strata)))
                  for entry in row if entry is not None][:24]
        if len(selected)!=24:raise ValueError('Need 24 reviewable validation records')
        with (data/'evaluation.jsonl').open('w',encoding='utf-8') as stream:
            for record in selected:stream.write(json.dumps(record,ensure_ascii=False)+'\n')
        h=hashlib.sha256()
        with (data/'records.jsonl').open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
        manifest.update(records_sha256=h.hexdigest(),batches_pending_review=batches,
                        pending_source_manifest=source_id,coding_teacher_executed=with_coding,
                        evaluation_records=len(selected),evaluation_sha256=hashlib.sha256((data/'evaluation.jsonl').read_bytes()).hexdigest())
        (data/'manifest.json').write_text(json.dumps(manifest,indent=2))
    finally:store.close();old.close()
    return {'training_authorized':False,'pending_batches':len(batches),'pending_source':source_id,
            'split_audit':audit_splits(data/'records.jsonl')}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--config',required=True)
    p.add_argument('--old-config',required=True);p.add_argument('--old-review',required=True);p.add_argument('--with-coding',action='store_true')
    a=p.parse_args();result=stage(a.data,a.config,a.old_config,a.old_review,a.with_coding)
    Path(a.data,'review-summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
