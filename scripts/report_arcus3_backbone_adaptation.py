"""CPU-only verified campaign summary; no completion inference from step counts."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from arcus3.config import read
from arcus3.checkpoint import digest
from arcus3.expanded_checkpoint import verify

def report(root,replay=None):
    root=Path(root);r=read(root/'report.json')
    if replay:
        proof=read(replay);pointer=read(root/'checkpoints/latest.json');cp=root/'checkpoints'/pointer['generation']
        from arcus3.config import safe_child
        cp=safe_child(root/'checkpoints',pointer['generation']);m=read(cp/'manifest.json')
        verify(cp,m['parent_sha256'])
        if digest(cp/'manifest.json')!=pointer['manifest_sha256'] or proof['checkpoint_manifest_sha256']!=pointer['manifest_sha256'] or proof.get('qualified') is not True:
            raise ValueError('Replay proof does not match durable checkpoint')
        return {'schema':'arcus3-phase8-qualified-summary-v1','qualified':True,'campaign_complete':False,
                'checkpoint_manifest_sha256':pointer['manifest_sha256'],'parent_sha256':m['parent_sha256'],
                'config_sha256':m['config_sha256'],'data_sha256':m['data_sha256'],'updates':proof['updates'],
                'input_tokens':proof['input_tokens'],'target_tokens':sum(x['target_tokens'] for x in r['records']),
                'trainable_parameters':r['trainable_parameters'],'trainability':'added-expert1-full-fp32',
                'max_input_tokens_tested':max(x['input_tokens'] for x in r['records']),
                'peak_cuda_bytes':proof['peak_cuda_bytes'],'exact_replay':True,'frozen_unchanged':proof['frozen_unchanged'],
                'qualification_report_sha256':digest(root/'report.json'),'replay_report_sha256':digest(replay),
                'note':'Original post-save fingerprint error repaired; separate read-only checkpoint replay passed. Two-update qualification only.'}
    if 'checkpoint' in r:
        cp=root/'checkpoints'/Path(r['checkpoint']).name
        verify(cp,r['state']['parent_sha256'],r['state']['data_sha256'],r['state']['config_sha256'])
        if digest(cp/'manifest.json')!=r['checkpoint_manifest_sha256']:raise ValueError('Report checkpoint mismatch')
    return {'qualified':r.get('qualified',False),'campaign_complete':r.get('complete',False),'reason':r.get('reason'),
            'updates':r.get('state',{}).get('updates'),'input_tokens':r.get('state',{}).get('input_tokens'),
            'target_tokens':r.get('state',{}).get('target_tokens'),'trainable_parameters':r.get('trainable_parameters'),
            'exact_replay':r.get('exact_replay'),'frozen_unchanged':r.get('frozen_unchanged'),
            'peak_cuda_bytes':r.get('peak_cuda_bytes'),'seconds':r.get('seconds')}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--replay');p.add_argument('--output');a=p.parse_args();result=report(a.root,a.replay)
    if a.output:
        from baby_arcus.language_stream import atomic_json
        atomic_json(a.output,result)
    print(json.dumps(result,indent=2))
