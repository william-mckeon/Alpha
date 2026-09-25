"""Build bounded, unapproved Phase 2B review packs without loading a model."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.contracts import digest
from baby_arcus.codebase_corpus import collect
from baby_arcus.hf_training_sources import rows
from baby_arcus.trajectory_adapters import adapt
from baby_arcus.dataset_split_audit import audit
from baby_arcus.data_manifest import file_digest, validate_manifest
from baby_arcus.data_staging import StagingStore
from baby_arcus.sft_dataset import packing_report


def build(config, output, tokenizer):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    accepted=[]; rejected=[]; corpus=[]
    if config.get('include_local_logs') is not False:
        raise ValueError('Local logs must be explicitly excluded')
    for source in config.get('hf',[]):
        if not source.get('enabled'): continue
        path=Path(source['path'])
        receipt=json.loads(path.with_name(path.name+'.receipt.json').read_text())
        if (receipt['repo']!=source['repo'] or receipt['revision']!=source['revision']
                or receipt['sha256']!=file_digest(path)):
            raise ValueError('HF source receipt mismatch')
        for index,row in enumerate(rows(path,source.get('limit',100))):
            try:
                group=row.get('repo') or source.get('group')
                if not group: raise ValueError('Repository group unavailable')
                row={**row,'_split':source.get('split')}
                record=adapt(row,source['repo'],source['revision'],group,source['license'])
                result=packing_report([record],tokenizer)
                if result['quarantined']: raise ValueError(result['quarantined'][0]['reason'])
                accepted.append(record)
            except (ValueError,KeyError,TypeError) as exc:
                rejected.append({'source':source['repo'],'row':index,'reason':str(exc)})
    for source in config.get('codebases',[]):
        corpus.extend(collect(source['root'],source['paths'],source['group']))
    # Deduplicate globally before assigning approval batches; cross-split collisions fail.
    evidence=audit(accepted+corpus,config.get('heldout_groups',[]))
    seen=set(); unique=[]
    for record in accepted:
        key=digest(record['messages'])
        if key not in seen: unique.append(record); seen.add(key)
    accepted=unique
    store=StagingStore(output/'staging.sqlite')
    batches=[]; files=[]
    try:
        for start in range(0,len(accepted),50): batches.append(store.stage(accepted[start:start+50]))
        for split in ('training','validation','test'):
            selected=[r for r in corpus if r['split']==split]
            if not selected: continue
            path=output/(split+'.jsonl')
            path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected),encoding='utf-8')
            files.append({'path':path.name,'size':path.stat().st_size,'sha256':file_digest(path),'split':split})
        manifest={'schema':'alpha-coding-corpus-v3','fixture':False,'dataset_id':'phase2b-code',
                  'languages':['Python','JavaScript','Go','Rust'],'files':files,'fingerprint':digest(files)}
        corpus_batch=None
        if files:
            validate_manifest(manifest)
            corpus_batch=store.stage_sources(manifest)
            (output/'corpus-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    finally: store.close()
    report={'schema':'alpha-phase2b-review-pack-v1','training_enabled':False,'approved':False,
            'source_config_sha256':digest(config),'accepted_sft':len(accepted),'code_documents':len(corpus),
            'rejected':rejected,'split_audit':evidence,'batches':batches,'corpus_batch':corpus_batch,
            'packing':packing_report(accepted,tokenizer)}
    (output/'source-config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True); parser.add_argument('--output',required=True)
    args=parser.parse_args()
    from arcus.tokenizer import get_tokenizer
    print(json.dumps(build(json.loads(Path(args.config).read_text()),args.output,get_tokenizer('o200k_base')),indent=2))
