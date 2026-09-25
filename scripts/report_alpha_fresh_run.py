"""Assemble a verified final report without loading a model."""
import sys
import hashlib
import json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def build(root, initial, final):
    root=Path(root)
    candidate=json.loads((root/'candidate.json').read_text())
    if candidate['updates']!=40000 or final.get('candidate')!=candidate:
        raise ValueError('Final report requires exactly 40000 saved updates')
    for report in (initial,final):
        pointer=report['candidate']
        if not report.get('complete') or not report.get('checkpoint_unchanged') or not report.get('coding_execution_evaluated'):
            raise ValueError('Incomplete evaluation')
        if file_hash(root/(pointer['generation']+'.pt'))!=pointer['sha256']:
            raise ValueError('Checkpoint hash mismatch')
    if initial['candidate']['updates']!=0 or initial.get('evaluation_identity')!=final.get('evaluation_identity'):
        raise ValueError('Initial and final cohorts differ')
    progress=json.loads((root/'three-stage-progress.json').read_text())
    if progress['candidate']!=candidate: raise ValueError('Stale token exposure evidence')
    samples={'count':0,'min_host_free_bytes':None,'max_gpu_used_mib':0}
    for path in root.glob('alpha-fresh128m-*/host-samples.jsonl'):
        with path.open() as stream:
            for line in stream:
                try: row=json.loads(line)
                except ValueError: continue
                if 'host_free_bytes' not in row or 'gpu_used_mib' not in row: continue
                samples['count']+=1
                samples['min_host_free_bytes']=min(samples['min_host_free_bytes'] if samples['min_host_free_bytes'] is not None else row['host_free_bytes'],row['host_free_bytes'])
                samples['max_gpu_used_mib']=max(samples['max_gpu_used_mib'],row['gpu_used_mib'])
    return {'resource_samples_all_attempts':samples,'initial':initial,'final':final,'target_total_updates':40000,
            'exposure':progress['state'],'single_seed':True,'automatic_promotion':False,
            'extension_to_64000_authorized':False,
            'limitations':'Small fixed cohorts; repeated validation is not an untouched final test. Source eligibility does not mean every source token was consumed.'}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    from baby_arcus.language_stream import atomic_json
    out=root/'run-40000'
    atomic_json(out/'report.json',build(root,json.loads((out/'initial/evaluation.json').read_text()),json.loads((out/'evaluation-40000/evaluation.json').read_text())))
