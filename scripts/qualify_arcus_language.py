"""Exercise real GPU learning, message receipts and transactional restart in an isolated pilot."""
import argparse
import json
from pathlib import Path
import random
import shutil
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.language_stream import atomic_json
from baby_arcus.services.language_worker import LanguageWorker
from baby_arcus.large_body_learning import file_hash


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(Path('configs/baby_arcus/language.json').read_text(encoding='utf-8'));source=Path(cfg['language_root'])
    for name in ('bootstrap.pt','qualification.json','dataset-manifest.json'):
        shutil.copyfile(source/name,root/name)
    cfg['language_root']=str(root.resolve());config=root/'config.json';atomic_json(config,cfg)
    worker=LanguageWorker(config)
    message={'request_id':'language-qualification-hello','sender':'you','text':'Hello Arcus. You are in your playpen. I am here with you.'}
    before=worker.snapshot();worker.request({'op':'control','action':'pause'})
    paused=worker.snapshot()['listening'] is False
    worker.request({'op':'control','action':'resume'})
    snapshots=[]
    for index in range(18):
        state=worker.request({'op':'step','messages':[message]})
        snapshots.append(state)
        print(json.dumps({'decision':state['decisions'],'choice':state['last_choice'],
                          'tokens':state['training_tokens'],'validation_loss':state['validation_loss']}),flush=True)
    trained=state['message_receipts'][message['request_id']]['trained_tokens']
    fingerprint=file_hash(cfg['body_checkpoint']);saved=worker.snapshot();worker.close()
    # Force non-ASCII into the atomic pointer (Windows' default cp1252 must not be used).
    active=json.loads((root/'active.json').read_text(encoding='utf-8'))
    active['qualification_unicode_probe']='“Arcus” 🐉 你好'
    atomic_json(root/'active.json',active)
    # Simulate a cursor sidecar write whose associated weight transaction never committed.
    pending=json.loads((root/'listening.json').read_text(encoding='utf-8'))
    pending['token']+=7;atomic_json(root/'listening.json',pending)
    restored=LanguageWorker(config)
    after=restored.snapshot()
    resumed=all(saved[k]==after[k] for k in ('training_tokens','decisions','cursor','message_receipts','expressions'))
    count=after['message_receipts'][message['request_id']]['trained_tokens']
    # Retried delivery is not counted or trained twice.
    retry=restored.request({'op':'step','messages':[message]})
    deduplicated=retry['message_receipts'][message['request_id']]['trained_tokens']==count
    restored.close()
    report={'gate_passed':bool(paused and resumed and deduplicated and trained>0 and fingerprint==saved['parent_sha256']),
        'pause_control':paused,'restart_preserves_state':resumed,'message_retry_deduplicated':deduplicated,
        'unicode_pointer_loaded':True,'uncommitted_cursor_discarded':saved['cursor']==after['cursor'],
        'human_message_trained_targets':trained,'motor_parent_unchanged':fingerprint==saved['parent_sha256'],
        'initial':before,'final':retry,'scope':'Pipeline, genuine weight updates, policy choices and persistence; not comprehension'}
    atomic_json(root/'live-report.json',report);print(json.dumps({'gate_passed':report['gate_passed']}),flush=True)
    if not report['gate_passed']:raise SystemExit(1)


if __name__=='__main__':main()
