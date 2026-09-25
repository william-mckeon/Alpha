"""Live qualification in marked fixtures; never authorizes or trains real data."""
import argparse
import json
import secrets
import sys
import threading
import uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.runtime_contract import require_container
if __name__ == '__main__':
    require_container()
import torch
import zstandard
from baby_arcus.language_stream import atomic_json
from baby_arcus.coding_environment import CodingEnvironment
from baby_arcus.coding_tools import CodingTools
from baby_arcus.coding_practice import run_episode
from baby_arcus.trajectory_store import TrajectoryStore
from baby_arcus.data_manifest import build
from baby_arcus.data_staging import StagingStore
from baby_arcus.services.data_review import Application
from baby_arcus.transport import serve, Client, RemoteError
from baby_arcus.shared_idle_training import train_idle
from baby_arcus.shared_checkpoint import load, digest
from baby_arcus.shared_curriculum import example
from baby_arcus.coding_policy import decide
from arcus.tokenizer import get_tokenizer
from scripts.prepare_alpha_three_stage import prepare


def qualify(root, source_pointer, learner_config):
    root=Path(root).resolve()
    base=(Path(__file__).resolve().parents[1]/'runs'/'test2').resolve()
    if base not in root.parents or root.exists():
        raise ValueError('Use a new qualification directory inside runs/test2')
    root.mkdir(parents=True)
    torch.set_num_threads(2)
    report={'fixture':True,'mastery_established':False,'real_data_approved':False}
    source=Path(source_pointer)
    retained=json.loads(source.read_text())
    report['retained_candidate']=retained
    env=CodingEnvironment(root/'workspace','positive_sum')
    before=env.tests()
    tools=CodingTools(env); store=TrajectoryStore(root/'trajectories.sqlite')
    calls=iter([
        {'name':'tool_search','version':1,'arguments':{'query':'write edit file','limit':1}},
        {'name':'write_file','version':1,'arguments':{'path':'solution.py','content':'def solve(values):\n    return sum(x for x in values if x > 0)\n'}},
        {'name':'tool_search','version':1,'arguments':{'query':'run tests','limit':1}},
        {'name':'run_tests','version':1,'arguments':{}},
    ])
    # This is deliberately an expert fixture, not a claimed Alpha-generated solution.
    episode=run_episode('fixture-expert',tools,lambda state:{'status':'call','call':next(calls)},store,4)
    store.close()
    report['sandbox']={'initial_test_passed':before['passed'],'fixture_expert_solved':episode['solved'],'steps':episode['steps']}
    if before['passed'] or not episode['solved']: raise AssertionError('Live sandbox failure')
    review_secret=secrets.token_urlsafe(32); ingest_secret=secrets.token_urlsafe(32)
    staging=StagingStore(root/'staging.sqlite',review_secret,fixture=True)
    server=serve('127.0.0.1',0,Application(staging,ingest_secret))
    thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    client=Client('http://127.0.0.1:'+str(server.server_port),timeout=30)
    try:
        record={'version':1,'source':'fixture:expert-tool-discovery','group':'fixture-positive-sum',
                'split':'training','messages':episode['messages']}
        batch=client.request('POST','/stage',{'credential':ingest_secret,'records':[record]})['batch_id']
        try: staging.approved([batch],allow_fixture=True)
        except ValueError: report['unapproved_training_refused']=True
        else: raise AssertionError('Unapproved data admitted')
        try: client.request('POST','/review',{'credential':ingest_secret,'batch_id':batch,'decision':'approved','reviewer':'test fixture'})
        except RemoteError as exc:
            if exc.status != 403: raise
            report['machine_approval_refused']=True
        else: raise AssertionError('Machine approved data')
        client.request('POST','/review',{'credential':review_secret,'batch_id':batch,'decision':'approved','reviewer':'AUTOMATED FIXTURE ONLY'})
        corpus=root/'corpus'; corpus.mkdir()
        text='Python functions return values. Test edge cases and inspect failures. '*100
        raw='\n'.join(json.dumps({'text':text}) for _ in range(3))+'\n'
        (corpus/'python.jsonl.zst').write_bytes(zstandard.ZstdCompressor().compress(raw.encode()))
        manifest=build(corpus,['*.zst'],['Python'],fixture=True)
        source_batch=staging.stage_sources(manifest)
        staging.review(source_batch,'approved','AUTOMATED FIXTURE ONLY',review_secret)
    finally:
        server.shutdown(); server.server_close(); thread.join(); staging.close()
    plan=json.loads(Path('configs/baby_arcus/alpha_three_stage.json').read_text())
    plan.update(training_enabled=True,fixture=True,source_checkpoint=str(source),
                staging_store=str(root/'staging.sqlite'),approved_batches=[batch],
                approved_source_manifests=[source_batch],mixture=['embodied']*11+['coding_corpus','sft'],
                token_budget=4096)
    atomic_json(root/'plan.json',plan)
    cfg=json.loads(Path(learner_config).read_text())
    cfg.update(root=str(root/'learner'),three_stage_config=str(root/'plan.json'))
    cfg['idle_learning'].update(chunk_updates=13,checkpoint_every=13,session_updates=13)
    atomic_json(root/'config.json',cfg)
    prepare(root/'config.json')
    learner=root/'learner'; (learner/'pause-training').unlink()
    try:
        result=train_idle(root/'config.json',13,'fixture-mixed-cycle')
        retry=train_idle(root/'config.json',13,'fixture-mixed-cycle')
    finally: (learner/'pause-training').touch()
    report['training']=result
    report['retry_did_not_train']=retry.get('already_trained') is True
    model,data=load(learner,result['candidate'],'cuda' if torch.cuda.is_available() else 'cpu')
    report['parameters']=sum(p.numel() for p in model.parameters())
    report['new_receipts']=data['progress']['receipts'][-13:]
    report['durable_state']=data['progress']['three_stage']
    if result['candidate']['updates'] != retained['updates']+13: raise AssertionError('Wrong update count')
    if not all(f in [r['family'] for r in report['new_receipts']] for f in ('standing','lying','sitting','coding_corpus','sft')):
        raise AssertionError('Incomplete mixed cycle')
    row,_,_=example(0,'training','commands'); row['hearing']=[]
    report['alpha_decoding']=decide(model,get_tokenizer(cfg['encoding']),row,
        [{'role':'user','content':'Find a tool to inspect a Python file.'}],
        [tools.catalog.tools['tool_search']],max_new_tokens=16)
    report['source_pointer_unchanged']=json.loads(source.read_text())==retained
    report['source_hash_verified']=digest(source.parent/(retained['generation']+'.pt'))==retained['sha256']
    report['real_idle_still_paused']=json.loads((source.parent/'idle-state.json').read_text())['enabled'] is False
    if not all(report[k] for k in ('retry_did_not_train','source_pointer_unchanged','source_hash_verified','real_idle_still_paused')):
        raise AssertionError('Preservation or idempotence failed')
    atomic_json(root/'report.json',report)
    return {'report':str(root/'report.json'),'updates':result['candidate']['updates'],'alpha_decoding_status':report['alpha_decoding']['status']}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='runs/test2/three-stage-qualification-'+uuid.uuid4().hex[:8])
    parser.add_argument('--source-pointer',required=True)
    parser.add_argument('--learner-config',required=True)
    args=parser.parse_args()
    print(json.dumps(qualify(args.root,args.source_pointer,args.learner_config)),flush=True)
