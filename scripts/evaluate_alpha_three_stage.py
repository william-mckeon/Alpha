"""Matched diagnostic retention and coding evaluation, with no promotion side effects."""
import argparse
import json
import math
import sys
import uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.runtime_contract import require_container
if __name__ == '__main__':
    require_container()
import torch
from scripts.evaluate_arcus_baseline_parity import evaluate as retained_skills
from scripts.evaluate_alpha_coding import evaluate as coding
from baby_arcus.language_stream import atomic_json
from baby_arcus.shared_checkpoint import digest
from baby_arcus.shared_factory import read_config


def retention(config, directory):
    cfg=read_config(config)
    candidate=json.loads((Path(cfg['root'])/'candidate.json').read_text())
    path=directory/f'baseline-validation-{candidate["generation"]}.json'
    if path.exists():
        report=json.loads(path.read_text())
        if (report.get('candidate') != candidate or not report.get('complete') or
                not report.get('checkpoint_unchanged') or report.get('split') != 'validation' or
                report.get('evaluator_sha256') != digest(Path('scripts/evaluate_arcus_baseline_parity.py')) or
                report.get('cohort') != {'split':'validation','episodes':8,'cases':60,'seed_offset':40000000} or
                any(report['results'][family].get('episodes') != 8 for family in ('standing','lying','sitting')) or
                digest(Path(cfg['root'])/(candidate['generation']+'.pt')) != candidate['sha256']):
            raise ValueError('Existing retention evidence cannot be reused; preserve it and choose a new output directory')
        return {'report':str(path),'reused_verified_report':True,
                'results':{k:{n:v for n,v in value.items() if n != 'records'} for k,value in report['results'].items()}}
    return retained_skills(config,'validation',8,60,report_directory=directory)


def assess(before, after, thresholds):
    """Fail closed on incomplete evidence; thresholds never default to a pass."""
    try:
        for bundle in (before,after):
            retention,coding=bundle['retention'],bundle['coding']
            if (retention.get('complete') is not True or retention.get('checkpoint_unchanged') is not True
                    or coding.get('complete') is not True or coding.get('checkpoint_unchanged') is not True
                    or retention['candidate'] != coding['candidate']):
                raise ValueError('Incomplete or mismatched checkpoint evidence')
        a,b=before['retention'],after['retention']
        if (a.get('split')!='validation' or b.get('split')!='validation' or not a.get('sources')
                or a['sources']!=b.get('sources') or set(a['results'])!=set(b['results'])):
            raise ValueError('Unmatched retention cohorts or evaluator sources')
        required={'standing','lying','sitting','approach','language'} | {prefix+family for prefix in ('paired:','unpaired:') for family in ('commands','color_reference','rest')}
        if not required.issubset(a['results']): raise ValueError('Missing retention capability results')
        if (not a.get('evaluator_sha256') or a['evaluator_sha256'] != b.get('evaluator_sha256')
                or a.get('cohort') != b.get('cohort') or not a.get('cohort')):
            raise ValueError('Unmatched evaluator or cohort provenance')
        if before['coding'].get('cohort') != after['coding'].get('cohort') or not before['coding'].get('cohort'):
            raise ValueError('Unmatched coding cohorts')
        for family,metric in a['results'].items():
            other=b['results'][family]
            denominator='episodes' if 'episodes' in metric else 'examples'
            if metric.get(denominator,0)<=0 or metric[denominator]!=other.get(denominator):
                raise ValueError('Different or empty cohort: '+family)
        if thresholds is None:
            return {'status':'inconclusive','passed':False,'reason':'Acceptance thresholds not agreed'}
        expected={'max_retention_rate_drop','max_language_nll_increase','min_coding_solved','min_coding_gain'}
        if set(thresholds)!=expected or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in thresholds.values()):
            raise ValueError('Invalid acceptance thresholds')
        if thresholds['max_retention_rate_drop']>1 or any(type(thresholds[k]) is not int for k in ('min_coding_solved','min_coding_gain')):
            raise ValueError('Invalid acceptance threshold types or bounds')
        checks={}
        for family,metric in a['results'].items():
            other=b['results'][family]
            if family=='language':
                if not all(math.isfinite(float(m['nll'])) for m in (metric,other)): raise ValueError('Nonfinite language loss')
                checks[family]=other['nll']<=metric['nll']+thresholds['max_language_nll_increase']
            else:
                n=metric.get('episodes',metric.get('examples'))
                if any(type(m.get('successes')) is not int or not 0<=m['successes']<=n for m in (metric,other)):
                    raise ValueError('Invalid success count')
                checks[family]=other['successes']/n >= metric['successes']/n-thresholds['max_retention_rate_drop']
        ca,cb=before['coding'],after['coding']
        if ca['total']!=cb['total'] or ca['total']<=0 or any(not 0<=c['solved']<=c['total'] for c in (ca,cb)):
            raise ValueError('Invalid coding counts')
        checks['coding_minimum']=cb['solved']>=thresholds['min_coding_solved']
        checks['coding_gain']=cb['solved']-ca['solved']>=thresholds['min_coding_gain']
        return {'status':'passed' if all(checks.values()) else 'failed','passed':all(checks.values()),'checks':checks}
    except (KeyError,ValueError,TypeError,ZeroDivisionError) as exc:
        return {'status':'inconclusive','passed':False,'reason':str(exc)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--baseline-config',default='configs/baby_arcus/alpha_idle.container.json')
    parser.add_argument('--output',required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--gates',default='configs/baby_arcus/alpha_three_stage_gates.json')
    args=parser.parse_args(); root=Path(args.output)
    if root.exists() and any(root.iterdir()) and not args.resume: raise ValueError('Use --resume to preserve and verify existing components')
    root.mkdir(parents=True,exist_ok=True); torch.set_num_threads(2)
    report={'mastery_established':False,'comparison':{},'limitations':'Small reused single-seed retention cohorts and two held-out toy coding templates. Fixture training is not a real data experiment.'}
    for label,config in (('before',args.baseline_config),('after',args.config)):
        report['comparison'][label]={'retention':retention(config,root/(label+'-retention'))}
        atomic_json(root/'report.json',report)
        # An interrupted coding episode is never replayed over its old workspace.
        report['comparison'][label]['coding']=coding(config,root/(label+'-coding-'+uuid.uuid4().hex[:8]))
        atomic_json(root/'report.json',report)
    raw={}
    for label in ('before','after'):
        raw[label]={'retention':json.loads(Path(report['comparison'][label]['retention']['report']).read_text()),
                    'coding':report['comparison'][label]['coding']}
    gates=json.loads(Path(args.gates).read_text())
    report['acceptance']=assess(raw['before'],raw['after'],gates.get('capability_thresholds'))
    report['complete']=True
    atomic_json(root/'report.json',report)
    print(json.dumps({'report':str(root/'report.json')}))


if __name__ == '__main__': main()
