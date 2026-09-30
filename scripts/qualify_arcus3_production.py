"""Seal evidence from completed live checks; never substitutes for executing tests."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from arcus3.config import read
from arcus3.checkpoint import digest
from arcus3.production import identity,validate_policy
from baby_arcus.language_stream import atomic_json

def qualify(a):
    report=read(Path(a.context_run)/'report.json');runtime=read(a.runtime)
    actual=read(Path(a.context_run)/'runtime.json');exit_state=read(Path(a.context_run)/'container-state.json')
    if not report.get('qualified') or not report.get('exact_replay') or not report.get('frozen_unchanged') or report['max_input_tokens_tested']!=report['tokenizer_contract']['context_tokens']:
        raise ValueError('Full-context live replay failed or incomplete')
    if actual['image_id']!=runtime['image_id'] or exit_state['Running'] or exit_state['ExitCode']!=0:raise ValueError('Image/exit mismatch')
    tests=read(a.test_receipt)
    if tests['image_id']!=runtime['image_id'] or tests['exit_code']!=0 or tests['tests_run']!=3:raise ValueError('CUDA integration tests incomplete')
    policy=validate_policy(read(a.policy))
    files=['scripts/run_arcus3_production.py','scripts/start_arcus3.ps1','arcus3/production.py','arcus3/production_data.py','arcus3/production_cache.py','scripts/prepare_arcus3_production.py']
    result={'qualified':True,'image_id':runtime['image_id'],'policy_sha256':identity(policy),'context_report_sha256':digest(Path(a.context_run)/'report.json'),
            'test_receipt_sha256':digest(a.test_receipt),'host_files':{f:digest(f) for f in files},'context':report['tokenizer_contract'],
            'peak_cuda_bytes':report['peak_cuda_bytes'],'campaign_updates':0,'note':'Disposable CUDA qualification only; startup donor evaluations still gate production updates.'}
    atomic_json(a.output,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('context-run','runtime','test-receipt','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--policy',default='configs/arcus3/production.json');a=p.parse_args();print(json.dumps(qualify(a)))
