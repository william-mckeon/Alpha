"""Compare only matched developmental suites; preserve every source report."""
import argparse
import json
from pathlib import Path


def compare(reports):
    if not reports or any(not r.get('complete') or not r.get('checkpoint_unchanged') for r in reports):
        raise ValueError('Incomplete developmental evidence')
    if any(r['identity'] != reports[0]['identity'] for r in reports):
        raise ValueError('Developmental suite identity differs')
    if any(r.get('scoring_identity') != reports[0].get('scoring_identity') for r in reports):
        raise ValueError('Scoring cohorts differ')
    if any(r.get('execution_identity') != reports[0].get('execution_identity') for r in reports):
        raise ValueError('Execution cohorts differ')
    return {'schema':'alpha-development-comparison-v1', 'identity':reports[0]['identity'],
            'checkpoints':[{'candidate':r['candidate'],'summary':r['summary']} for r in reports],
            'limitations':'Descriptive small-cohort comparison. Null scores are unmeasured, not failures. No combined intelligence score.'}


def summary_html(summary):
    import html
    rows=[]
    for category,values in summary.items():
        repetition=values.get('repetition',{}).get('mean_repeated_4gram_fraction')
        rep='unmeasured' if repetition is None else f'{repetition:.2%}'
        generation=values.get('generation',{})
        qualitative='; '.join(name+': '+('pending' if values.get(name,{}).get('mean') is None else str(values[name]['mean'])) for name in ('fluency','coherence','relevance'))
        success=values['task_success']
        result=f"{success['passed']}/{success['measured']}" if success['measured'] else 'not automatically scored'
        cells=[category,result,rep,str(generation.get('truncated','unknown'))+'/'+str(generation.get('truncation_measured',0)),qualitative]
        rows.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in cells)+'</tr>')
    return '<table><tr><th>Category</th><th>Task success</th><th>Mean repeated 4-grams</th><th>Truncated/measured</th><th>Qualitative review</th></tr>'+''.join(rows)+'</table><p>Repetition and truncation are descriptive measurements, not fluency scores. Pending means unmeasured, not failed.</p>'


def write_comparison_html(comparison,path):
    sections=[]
    present={entry['candidate']['updates'] for entry in comparison['checkpoints']}
    missing=[step for step in (40000,45000,53192,60000) if step not in present]
    for entry in comparison['checkpoints']:
        sections.append('<h2>'+str(entry['candidate']['updates'])+' updates</h2>'+summary_html(entry['summary']))
    Path(path).write_text('<!doctype html><meta charset="utf-8"><title>Alpha developmental comparison</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto}td,th{padding:10px;border-bottom:1px solid #aaa;text-align:left}</style><h1>Matched developmental comparison</h1><p>Separate measurements of emerging language behavior, correctness and task success. Missing requested checkpoints: '+(', '.join(map(str,missing)) or 'none')+'.</p>'+''.join(sections),encoding='utf-8')


def write_review_html(report, path):
    import html
    cards=[]
    for row in report['records']:
        prompt=next(m['content'] for m in row['messages'] if m['role']=='user')
        cards.append('<section><h2>'+html.escape(row['id'])+'</h2><p>'+html.escape(prompt)+'</p><pre>'+html.escape(row['response'])+'</pre><p>Truncated: '+str(row['truncated'])+'</p><details><summary>Measurements and pending review</summary><pre>'+html.escape(json.dumps(row['scores'],indent=2))+'</pre></details></section>')
    Path(path).write_text('<!doctype html><meta charset="utf-8"><title>Alpha developmental responses</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;color:#dde6ef;background:#101721}td,th{padding:8px;text-align:left}section{border-top:1px solid #445;padding:16px 0}pre{white-space:pre-wrap;background:#1b2634;padding:18px}summary{cursor:pointer}</style><h1>Alpha at '+str(report['candidate']['updates'])+' updates</h1><p>Actual greedy responses. Subjective judgments are pending human review. Exact checks and restricted code tests are separate measurements.</p>'+summary_html(report['summary'])+''.join(cards),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('reports',nargs='+');p.add_argument('--output',required=True);a=p.parse_args()
    output=Path(a.output)
    if output.exists():raise ValueError('Refusing to overwrite comparison')
    output.write_text(json.dumps(compare([json.loads(Path(x).read_text()) for x in a.reports]),indent=2))
