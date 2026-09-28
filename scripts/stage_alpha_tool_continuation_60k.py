"""Freeze the user-authorized 41k-to-60k continuation without loading a model."""
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.contracts import digest
import hashlib


def file_digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def stage():
    old=Path('runs/test2/alpha-tool-correction-run-config-v3')
    parent=Path('runs/test2/alpha-tool-correction-001')
    out=Path('runs/test2/alpha-tool-correction-60k-config-v1')
    if out.exists():raise ValueError('Use a fresh configuration directory')
    pointer=json.loads((parent/'candidate.json').read_text())
    expected='7ca24f7edc4b814fcd3472ea7b3163785157b3ad2fb3111bc407c270f3147a91'
    if pointer['updates']!=41000 or pointer['sha256']!=expected:raise ValueError('Wrong 41k parent')
    if file_digest(parent/(pointer['generation']+'.pt'))!=expected:raise ValueError('Parent hash mismatch')
    plan=json.loads((old/'training.json').read_text())
    progress=json.loads((parent/'three-stage-progress.json').read_text())
    cfg=json.loads((old/'learner.json').read_text())
    gates=json.loads((old/'gates.json').read_text())
    cfg.update(root='runs/test2/alpha-tool-correction-60k-001',evaluation_actions=32)
    plan.update(source_checkpoint=str(parent/'candidate.json').replace('\\','/'),source_updates=41000,
                source_sha256=expected,target_total_updates=60000,evaluation_every=1000,
                # Upper bound for all 20,000 updates including the preserved pilot exposure.
                token_budget=20000*16384,
                migration={'previous_plan_hash':progress['state']['plan_hash'],'at_updates':41000})
    gates['review_note']='User approved continuation to exactly 60000, evaluating every 1000. Retention regression pauses for review; no automatic promotion.'
    plan['gates_sha256']=digest(gates)
    out.mkdir()
    for name,value in [('learner.json',cfg),('training.json',plan),('gates.json',gates)]:
        (out/name).write_text(json.dumps(value,indent=2)+'\n')
    shutil.copy2(old/'language.json',out/'language.json')
    data=Path('runs/test2/tool-correction-data-v4')
    manifest={name:file_digest(data/name) for name in ('records.jsonl','evaluation.jsonl','review.sqlite')}
    (out/'authorization.json').write_text(json.dumps({'source_updates':41000,'target_updates':60000,
        'evaluation_every':1000,'evaluation_actions':32,'checkpoint_max_updates':64,
        'user_instruction':'Continue to 60k and reevaluate every 1k; implement and live test the approved file plan.',
        'source_sha256':expected,'dataset_files':manifest,'plan_sha256':digest(plan),
        'gates_sha256':digest(gates),'automatic_promotion':False},indent=2)+'\n')
    for name,value in [('alpha_tool_continuation_60k.json',cfg),
                       ('alpha_tool_continuation_60k_training.json',plan),
                       ('alpha_tool_continuation_60k_gates.json',gates)]:
        template=dict(value)
        if 'training_enabled' in template:
            template.update(training_enabled=False,approved_batches=[],approved_source_manifests=[])
        if 'training_authorized' in template:template['training_authorized']=False
        (Path('configs/baby_arcus')/name).write_text(json.dumps(template,indent=2)+'\n')
    print(json.dumps({'config':str(out),'parent':pointer,'target':60000,'actions':32}))


if __name__=='__main__':stage()
