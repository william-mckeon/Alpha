"""Small matched-comparison summary; never promotes a candidate automatically."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def summarize(report):
    if not report.get('complete') or not report.get('bounded_check_passed'):
        raise ValueError('Comparison is incomplete or requires review')
    if set(report['arms'])!={'control','balance-only','paired-balance'}:
        raise ValueError('Missing comparison arm')
    baseline=report['arms']['control'];result={}
    for name,arm in report['arms'].items():
        if not arm['frozen_unchanged'] or arm['updates']!=len(report['training_row_hashes']):
            raise ValueError('Unmatched updates or changed backbone')
        if arm['input_tokens']!=baseline['input_tokens'] or arm['before']!=baseline['before']:
            raise ValueError('Unmatched exposure or starting behavior')
        layers=[]
        for index in range(len(arm['metrics'][0]['routing_layers'])):
            samples=[m['routing_layers'][index] for m in arm['metrics']]
            positions=sum(s['positions'] for s in samples)
            layers.append({'new_expert_fraction':sum(s['counts'][1] for s in samples)/positions,
                'new_expert_mean_probability':sum(s['probability_mean'][1]*s['positions'] for s in samples)/positions})
        result[name]={'before':arm['before'],'after':arm['after'],
            'nll_delta':arm['after']['nll']-arm['before']['nll'],
            'updates':arm['updates'],'input_tokens':arm['input_tokens'],
            'layers':layers,'seconds':arm['seconds'],'peak_cuda_bytes':arm['peak_cuda_bytes']}
    return {'arms':result,'campaign_updates':0,'automatic_promotion':False,'limitations':report['limitations']}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    root=Path(a.root);result=summarize(json.loads((root/'report.json').read_text()))
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
