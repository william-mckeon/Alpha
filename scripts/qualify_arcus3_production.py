"""Seal evidence from completed live checks; never substitutes for executing tests."""
import argparse,json,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from arcus3.config import read
from arcus3.campaign import validate as validate_adaptation
from arcus3.checkpoint import digest
from arcus3.production import identity,validate_policy
from baby_arcus.language_stream import atomic_json

def adaptation_for_policy(policy,path):
    if policy.get('model_label')=='alpha3.2.2' and not path:
        raise ValueError('Alpha 3.2.2 qualification requires its exact adaptation configuration')
    if not path:return None
    cfg=validate_adaptation(read(path))
    if policy.get('model_label') and (cfg.get('model_label')!=policy['model_label']
            or cfg.get('lineage',{}).get('id')!=policy.get('lineage_id')):
        raise ValueError('Qualification policy/adaptation lineage mismatch')
    return cfg

def qualify(a):
    report=read(Path(a.context_run)/'report.json');runtime=read(a.runtime)
    actual=read(Path(a.context_run)/'runtime.json');exit_state=read(Path(a.context_run)/'container-state.json')
    if not report.get('qualified') or not report.get('exact_replay') or not report.get('frozen_unchanged') or report['max_input_tokens_tested']!=report['tokenizer_contract']['context_tokens']:
        raise ValueError('Full-context live replay failed or incomplete')
    if actual['image_id']!=runtime['image_id'] or exit_state['Running'] or exit_state['ExitCode']!=0:raise ValueError('Image/exit mismatch')
    tests=read(a.test_receipt)
    if tests['image_id']!=runtime['image_id'] or tests['exit_code']!=0 or tests['tests_run']<3 or tests.get('skipped',0):raise ValueError('CUDA integration tests incomplete')
    if tests.get('checkpoint_recovery_suite') is not True:raise ValueError('Checkpoint retention/recovery integration required')
    policy=validate_policy(read(a.policy))
    adaptation=getattr(a,'adaptation_config',None);cfg=adaptation_for_policy(policy,adaptation)
    if cfg:
        expected=hashlib.sha256(json.dumps({**cfg,'campaign_enabled':False},sort_keys=True).encode()).hexdigest()
        if report['config_sha256']!=expected:raise ValueError('Qualification objective mismatch')
        if cfg.get('routing_objective')=='paired-output-v2' and tests.get('routing_repair_suite') is not True:
            raise ValueError('Routing repair CUDA tests required')
    files=['scripts/run_arcus3_production.py','scripts/start_arcus3.ps1','scripts/train_arcus3_backbone_adaptation.py','arcus3/campaign.py','arcus3/production.py','arcus3/production_data.py','arcus3/production_cache.py','scripts/prepare_arcus3_production.py','arcus3/learning_rate.py']
    files+=['arcus3/checkpoint_retention.py','arcus3/checkpoint_recovery.py','arcus3/expanded_checkpoint.py','scripts/resume_arcus3_training.py']
    deferred=policy.get('model_label')=='alpha3.2.2' and policy.get('defer_startup_evaluation') is True
    result={'qualified':True,'image_id':runtime['image_id'],'policy_sha256':identity(policy),'context_report_sha256':digest(Path(a.context_run)/'report.json'),
            'test_receipt_sha256':digest(a.test_receipt),'host_files':{f:digest(f) for f in files},'context':report['tokenizer_contract'],
            'peak_cuda_bytes':report['peak_cuda_bytes'],'campaign_updates':0,
            'joint_evaluation_input_tokens':policy['joint_evaluation_input_tokens'] if deferred else None,
            'note':('Disposable CUDA qualification only; capability evaluation is deferred to the joint 7M-token checkpoint.' if deferred
                    else 'Disposable CUDA qualification only; startup donor evaluations still gate production updates.')}
    if adaptation:result['adaptation_config_file_sha256']=digest(adaptation)
    atomic_json(a.output,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('context-run','runtime','test-receipt','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--policy',default='configs/arcus3/production.json');p.add_argument('--adaptation-config');a=p.parse_args();print(json.dumps(qualify(a)))
