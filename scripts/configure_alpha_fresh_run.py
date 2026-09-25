"""Record the user's run authorization against exact prepared batch identities."""
import argparse
import json
import secrets
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.contracts import digest
from baby_arcus.data_staging import StagingStore


def configure(dataset, output, migration=None):
    dataset, output=Path(dataset),Path(output);output.mkdir(exist_ok=False)
    report=json.loads((dataset/'manifest.json').read_text())
    if not report.get('complete') or report['context_tokens']!=16384: raise ValueError('Incomplete 16K release')
    credential=secrets.token_urlsafe(32)
    store=StagingStore(dataset/'review.sqlite',review_secret=credential)
    try:
        for identity in report['sft_batches']+[report['coding_batch']]:
            store.review(identity,'approved','User-authorized fresh run, recorded by assistant',credential)
    finally:store.close()
    gates={'schema':'alpha-three-stage-gates-v1','training_authorized':True,
           'capability_thresholds':{'max_retention_rate_drop':0,'max_language_nll_increase':0,'min_coding_solved':0,'min_coding_gain':0},
           'automatic_promotion':False,'note':'Diagnostic thresholds only; zero-init has no retained mastery. No hardware clearance asserted.'}
    plan={'schema':'alpha-phase2b-v1','training_enabled':True,'baseline_policy':'random-initialization',
          'source_sha256':'135590bb84b9b7031cab157188289acc632ff982690875eb12218b17df5ea658',
          'staging_store':'/review/review.sqlite','approved_batches':report['sft_batches'],
          'approved_source_manifests':[report['coding_batch']],
          'mixture':['language','coding_corpus','sft'],'token_budget':40000*16384,
          'context_tokens':16384,'language_window_tokens':16384,'resume_after_seconds':60,
          'preserve_embodied_schedule':False,'automatic_promotion':False,'exhaustion_policy':'stop',
          'interleave_corpus_files':True,'interleave_sft_sources':True,'sft_preserved_prefix':0,
          'target_total_updates':40000,'gates_file':'/run-config/gates.json','gates_sha256':digest(gates)}
    if migration is not None:
        plan['migration']=migration
        plan['sft_preserved_prefix']=1
    config=json.loads(Path('configs/baby_arcus/alpha_fresh128m_16k.json').read_text())
    config['three_stage_config']='/run-config/training.json'
    config['dataset_config']='/run-config/language.json'
    language=json.loads(Path('configs/baby_arcus/language.json').read_text())
    language['dataset_root']='/dataset'
    for name,value in [('training.json',plan),('gates.json',gates),('learner.json',config),('language.json',language),
                       ('authorization.json',{'user_authorized_total_updates':40000,'extension_to_64000_authorized':False,
                        'dataset_manifest_sha256':digest(report),'parameters':128353994,'context_tokens':16384})]:
        (output/name).write_text(json.dumps(value,indent=2))
    print(json.dumps({'configured':True,'run_config':str(output),'training_launched':False}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
    p.add_argument('--migration-file', help='Explicit reviewed migration JSON; omit for zero-update runs')
    a=p.parse_args();configure(a.dataset,a.output,json.loads(Path(a.migration_file).read_text()) if a.migration_file else None)
