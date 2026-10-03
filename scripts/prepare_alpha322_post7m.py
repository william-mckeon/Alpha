"""Prepare a hash-bound 7M evaluation handoff; never launch a GPU job."""
import argparse
import copy
import json
import os
import shutil
import sys
import uuid
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from arcus3.config import read
from arcus3.checkpoint import digest
from arcus3.campaign import accept_evaluation
from arcus3.production import accept_donor_receipt,identity,validate_policy,_accept_alpha322_schedule_transition

WORKSPACE=Path(__file__).resolve().parents[1]

def atomic_json(path,value):
    path=Path(path);temporary=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.pending')
    try:
        with temporary.open('w',encoding='utf-8') as stream:
            json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)

def verify_checkpoint(path,manifest):
    if (manifest.get('schema')!='arcus3-expanded-delta-v1'
            or set(manifest.get('files',{}))!={'delta.safetensors','state.pt'}):
        raise ValueError('Incomplete expanded checkpoint')
    for name,expected in manifest['files'].items():
        if digest(path/name)!=expected:raise ValueError('Checkpoint payload hash mismatch: '+name)

def prepare(source,developmental,benchmark,output,policy_path='configs/arcus3/production_alpha322_post7m.json'):
    source=Path(source).resolve();developmental=Path(developmental).resolve()
    benchmark=Path(benchmark).resolve();output=Path(output).resolve()
    owned=(WORKSPACE/'runs'/'arcus3').resolve()
    if any(p.parent!=owned for p in (source,developmental,benchmark,output)):
        raise ValueError('All inputs and output must be direct owned runs/arcus3 children')
    if output.exists():raise FileExistsError(output)
    policy=validate_policy(read(policy_path));old_path='configs/arcus3/production_alpha322.json'
    old=validate_policy(read(old_path));saved=read(source/'controller-state.json')
    state=saved['state'];cp=Path(saved['checkpoint']).resolve()
    manifest=read(cp/'manifest.json');verify_checkpoint(cp,manifest)
    checkpoint_sha=digest(cp/'manifest.json')
    if (policy.get('prior_policy_sha256')!=identity(old)
            or saved['policy_sha256']!=identity(old)
            or state['production']['policy_sha256']!=identity(old)
            or checkpoint_sha!=policy['continuation_checkpoint_sha256']
            or state['updates']!=manifest['updates']
            or state['production']!=manifest['production']
            or state['evaluation_pending']!=['full']
            or not 7_000_000<=state['input_tokens']<10_000_000):
        raise ValueError('Not the verified 7M handoff checkpoint')
    terminal=read(source/'session-result.json')
    if (terminal.get('reason')!='joint_7m_evaluation_required'
            or terminal.get('updates')!=state['updates']
            or terminal.get('input_tokens')!=state['input_tokens']):
        raise ValueError('Source has not stopped at the 7M evaluation boundary')
    scores=read(developmental/'scores.json');donor=read(benchmark/'donor-scores.json')
    if scores.get('tier')!='full' or donor.get('tier')!='full':
        raise ValueError('Full 7M results required')
    from arcus3.evaluation import sha
    accepted=accept_evaluation(state,scores,checkpoint_sha,
        sha('evaluation/arcus3/baseline-v1.json'),sha('configs/arcus3/evaluation.json'),.2)
    accept_donor_receipt(donor,checkpoint_sha,'full',policy)
    manifest_path=Path(read('configs/arcus3/phase8_storage.json')['external_root'])/'production-cache-v1/benchmarks/manifest.json'
    if donor['benchmark_manifest_sha256']!=digest(manifest_path):
        raise ValueError('Benchmark manifest changed')
    receipt={'schema':'arcus3-alpha322-evaluation-cadence-v1','prior_policy_path':old_path,
        'prior_policy_sha256':identity(old),'checkpoint_sha256':checkpoint_sha,
        'old_data_sha256':state['data_sha256'],'new_data_sha256':state['data_sha256'],
        'new_teacher_sha256':state['teacher_sha256'],'policy_sha256':identity(policy),
        'batch_id':state['production']['batch_id']}
    if not _accept_alpha322_schedule_transition(accepted,receipt,checkpoint_sha,policy):
        raise ValueError('Policy change is not the authorized schedule-only handoff')
    output.mkdir(parents=True)
    evaluation=output/'evaluation';evaluation.mkdir()
    shutil.copyfile(developmental/'scores.json',evaluation/'scores.json')
    shutil.copyfile(benchmark/'donor-scores.json',evaluation/'donor-scores.json')
    atomic_json(output/'transition.json',receipt)
    result=copy.deepcopy(saved)
    result.update(policy_sha256=identity(policy),transition=str(output/'transition.json'),
                  evaluation=str(evaluation))
    result['recovery_evidence']={str(cp/'manifest.json'):checkpoint_sha,
        str(source/'controller-state.json'):digest(source/'controller-state.json'),
        str(source/'session-result.json'):digest(source/'session-result.json'),
        str(developmental/'scores.json'):digest(developmental/'scores.json'),
        str(benchmark/'donor-scores.json'):digest(benchmark/'donor-scores.json'),
        str(evaluation/'scores.json'):digest(evaluation/'scores.json'),
        str(evaluation/'donor-scores.json'):digest(evaluation/'donor-scores.json'),
        str(output/'transition.json'):digest(output/'transition.json')}
    atomic_json(output/'controller-state.json',result)
    proof={'schema':'arcus3-alpha322-post7m-preparation-v1','checkpoint':str(cp),
        'checkpoint_manifest_sha256':checkpoint_sha,'updates':state['updates'],
        'input_tokens':state['input_tokens'],'target_tokens':state['target_tokens'],
        'old_policy_sha256':identity(old),'new_policy_sha256':identity(policy),
        'developmental_sha256':digest(developmental/'scores.json'),
        'benchmark_sha256':digest(benchmark/'donor-scores.json'),
        'controller_sha256':digest(output/'controller-state.json'),
        'checkpoint_payload_hashes_verified':True,'launch_started':False}
    atomic_json(output/'preparation.json',proof)
    return proof

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('source','developmental','benchmark','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--policy',default='configs/arcus3/production_alpha322_post7m.json')
    a=p.parse_args()
    print(prepare(a.source,a.developmental,a.benchmark,a.output,a.policy))
