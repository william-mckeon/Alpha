"""Bounded route map and a self-contained, escaped HTML viewer."""
import argparse
import html
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def write_html(path, report):
    # Text nodes, never executable model-generated HTML.
    rows=[]
    for trace in report['traces']:
        events=[e for e in trace['events'] if e['kind']=='expert']
        totals=(trace.get('usage') or {}).get('totals',{})
        rows.append('<tr><td>'+html.escape(trace['prompt_id'])+'</td><td>'+str(totals.get('expert_invocations',len(events)))+'</td><td>'+str(totals.get('expert_token_visits',sum(e['logical_accepted_tokens'] for e in events)))+'</td><td>'+str(totals.get('physical_expert_slots',sum(e['physical_slots'] for e in events)))+'</td><td>'+str(totals.get('expert_overflow_token_visits',sum(e['overflow_tokens'] for e in events)))+'</td></tr>')
    table='<table><tr><th>Prompt</th><th>Expert invocations</th><th>Accepted token assignments</th><th>Physical slots</th><th>Overflow</th></tr>'+''.join(rows)+'</table>'
    sections = ''.join('<details><summary>'+html.escape(t['prompt_id'])+'</summary><pre>'+
                       html.escape(json.dumps(t, indent=2))+'</pre></details>' for t in report['traces'])
    usage=report.get('routing_usage',{})
    usage_view='<h2>Repeated computational uses</h2><p>Every visit counts again, across tokens, forward passes and separate calls. Counts cover all evaluated prompts; detailed event arrays are sampled. Expert weight-token uses exclude non-expert weights. Module parameter-invocation uses are a different unit and must not be added to them.</p><pre>'+html.escape(json.dumps({k:v for k,v in usage.items() if k not in ('node_visits','route_traversals','module_invocations','direct_parameter_invocation_uses')},indent=2))+'</pre>'
    for field in ('node_visits','route_traversals','module_invocations','direct_parameter_invocation_uses'):
        usage_view+='<details><summary>'+html.escape(field)+'</summary><pre>'+html.escape(json.dumps(usage.get(field,{}),indent=2))+'</pre></details>'
    Path(path).write_text('<!doctype html><meta charset="utf-8"><title>Alpha route map</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;background:#101721;color:#e0e8ef}pre{white-space:pre-wrap}summary{cursor:pointer;padding:12px}</style>'
        '<h1>Alpha repeated route traversals and computational uses</h1>'+usage_view+
        '<p>Stored unique parameters (separate inventory): '+str(report['inventory']['unique_parameters'])+
        '; unique tensor weight bytes: '+str(report['inventory'].get('unique_weight_bytes','not measured'))+
        '. Observed routes are not an exhaustive neural connectome or a dense-equivalent size.</p>'+table+sections+
        '<details><summary>Parameter inventory and shared aliases</summary><pre>'+html.escape(json.dumps(report['inventory'],indent=2))+'</pre></details>', encoding='utf-8')


if __name__ == '__main__':
    from scripts.evaluate_alpha_development import evaluate
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--pointer',default='candidate.json');a=p.parse_args()
    evaluate(a.root,json.loads((Path(a.root)/a.pointer).read_text()),a.output,mapping=True)
