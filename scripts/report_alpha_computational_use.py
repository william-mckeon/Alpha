"""Full evaluation report with repeated computational uses as the mapping headline."""
import html
import json
from pathlib import Path
from baby_arcus.route_usage import aggregate
from scripts.report_alpha_tool_correction import scores


def build(directory):
    out=Path(directory)
    agent=json.loads((out/'agent/evaluation.json').read_text())
    dev=json.loads((out/'developmental/evaluation-53192-scored.json').read_text())
    mapping=json.loads((out/'developmental/mapping-53192.json').read_text())
    assert all(r['complete'] and r['checkpoint_unchanged'] for r in (agent,dev,mapping))
    assert agent['candidate']==dev['candidate']==mapping['candidate']
    coding=agent['coding']
    assert coding['complete'] and coding['checkpoint_unchanged'] and coding['candidate']==agent['candidate']
    decisions=[d for t in coding['tasks'] for d in t['decisions']]
    discovery=[d for t in agent['unseen_tools'] for d in t['decisions']]
    language=[r['routing_usage'] for r in agent['sft']]
    development=[r['usage'] for r in mapping['traces']]
    assert all(d.get('routing_usage') for d in decisions+discovery)
    cohorts={'coding':aggregate([d['routing_usage'] for d in decisions],sum(d['generated_tokens'] or 0 for d in decisions)),
             'discovery':aggregate([d['routing_usage'] for d in discovery],sum(d['generated_tokens'] or 0 for d in discovery)),
             'language_validation':aggregate(language), 'developmental':mapping['routing_usage']}
    growth=[]
    for budget in (1,8,16,32,64,96,128):
        chosen=[d for t in coding['tasks'] for d in t['decisions'][:budget]]
        usage=aggregate([d['routing_usage'] for d in chosen],sum(d['generated_tokens'] or 0 for d in chosen))
        growth.append({'actions_per_task':budget,'observations':len(chosen),
                       'distinct_routes':usage['distinct_routes_retained'], **usage['totals']})
    all_usage=aggregate(language+development+[d['routing_usage'] for d in decisions+discovery])
    metrics=scores(agent)
    report={'schema':'alpha-full-computational-evaluation-v1','candidate':agent['candidate'],
            'complete':True,'checkpoint_unchanged':True,'scores':metrics,'developmental':dev['summary'],
            'cohorts':cohorts,'overall':all_usage,'coding_coverage_growth':growth,
            'definitions':decisions[0]['routing_usage']['definitions'],
            'resources':{'agent_seconds':agent['seconds'],'coding_seconds':coding['seconds'],
                         'agent_peak_cuda_allocated_bytes':agent['peak_cuda_allocated_bytes'],
                         'developmental_seconds':dev['seconds'],
                         'developmental_peak_cuda_allocated_bytes':dev['peak_cuda_allocated_bytes']},
            'limitations':['Component/expert nodes, not an enumeration of scalar neuron paths.',
                           'Expert weight-token uses cover routed expert matrices only; repeated uses are not new independent parameters.',
                           'Registered module parameter-invocation counts exclude functional access and custom methods; not a full-network weight-use total.',
                           'Counts include prompt prefill and sensory processing. Cached versus uncached execution must be matched for comparisons.',
                           'Detailed developmental event arrays are sampled; streaming aggregate totals are not cut off by that sampling.',
                           'Distinct route identities are capped per observation; dropped identities are disclosed. Distinct counts may be lower bounds.',
                           'Small fixed cohorts, one checkpoint/seed. Subjective review remains pending. No training, RL, promotion or release.']}
    (out/'full-report.json').write_text(json.dumps(report,indent=2))
    rows=['| Cohort | Observations | Token route traversals | Expert token visits | Expert weight-token uses | Distinct routes |',
          '|---|---:|---:|---:|---:|---:|']
    for name,u in cohorts.items():
        t=u['totals']
        rows.append(f"| {name} | {u['observations']:,} | {t.get('token_route_traversals',0):,} | {t.get('expert_token_visits',0):,} | {t.get('expert_weight_token_uses',0):,} | {u['distinct_routes_retained']:,} |")
    lines=['# Alpha full evaluation and repeated computational-use mapping',
           f"Checkpoint: **{agent['candidate']['updates']:,} updates**. All evaluated checkpoint hashes verified unchanged.",
           '## Mapping: repeated visits count again',*rows,
           'A token route traversal is one token position through the routed block stack in one forward pass. Repeated calls and token processing count again. Distinct routes count different phase-labelled expert/skip/overflow sequences.',
           'Expert weight-token uses multiply each accepted expert token visit by the number of weights in that expert. This is a reuse count for expert matrices, not the total stored parameters or a full-network parameter-use count.',
           '## Additional routing evidence from longer coding episodes',
           '| Actions per task | Recorded decisions | Route traversals | Distinct routes |', '|---|---:|---:|---:|']
    for row in growth:
        lines.append(f"| {row['actions_per_task']} | {row['observations']} | {row.get('token_route_traversals',0):,} | {row['distinct_routes']:,} |")
    lines.extend(['The progression above measures additional observations and routing coverage, independently of whether coding tasks succeed.',
          '## Evaluation scores',
          f"- Weighted NLL **{metrics['language']['nll']:.8f}**, perplexity **{metrics['language']['perplexity']:.4f}**, {metrics['language']['target_tokens']:,} target tokens.",
          f"- Coding: **{coding['solved']}/{coding['total']}** solved; {len(decisions)} attempts with a maximum of 128 per task.",
          f"- Parseable calls: {sum(t['parseable_calls'] for t in coding['tasks'])}; executed calls: {sum(t['executed_calls'] for t in coding['tasks'])}; tool errors: {metrics['tool_errors']}.",
          f"- Discovery: **{metrics['discovery_solved']}/{len(agent['unseen_tools'])}** solved (separate existing three-action protocol).",
          '- Developmental results by category:'])
    for name,item in dev['summary'].items():
        measure=item['task_success']
        lines.append(f"  - {name}: {measure['passed']}/{measure['measured']} automated successes; subjective review pending for {item['review_pending']} responses.")
        repetition=item.get('repetition',{}).get('mean_repeated_4gram_fraction')
        generation=item.get('generation',{})
        if repetition is not None:
            lines.append(f"    Mean repeated four-grams: {repetition:.2%}; truncated: {generation.get('truncated',0)}/{generation.get('truncation_measured',0)}; mean generated tokens: {generation.get('mean_generated_tokens')}. Fluency, coherence and relevance are separate review dimensions; see the transcript report for measurement coverage.")
    lines.extend(['## Resources and integrity','```json',json.dumps(report['resources'],indent=2),'```',
                  'Checkpoint SHA256: `'+agent['candidate']['sha256']+'`.',
                  '## Scope and limitations',*['- '+x for x in report['limitations']],
                  '## Detailed evidence',
                  '- [All cohort totals, node visits and route counts](full-report.json)',
                  '- [All coding prompts, decisions and per-call mapping counts](agent/coding/report.json)',
                  '- [Developmental prompts, responses and scores](developmental/evaluation-53192-scored.html)',
                  '- [Developmental mapping viewer](developmental/mapping-53192.html)'])
    text='\n\n'.join(lines).replace('|\n\n|','|\n|')
    (out/'RESULTS.md').write_text(text)
    # Standalone viewer: compact tables plus expandable exact aggregate evidence.
    body='<h1>Alpha full evaluation: repeated computational uses</h1><pre>'+html.escape(text)+'</pre>'
    from scripts.report_alpha_development import summary_html
    body+='<h2>Developmental dimensions</h2>'+summary_html(dev['summary'])
    for name,u in cohorts.items():
        body+='<details><summary>'+html.escape(name)+' — node visits and route counts</summary><pre>'+html.escape(json.dumps(u,indent=2))+'</pre></details>'
    body+='<p><a href="developmental/evaluation-53192-scored.html">Prompts and responses</a> · <a href="developmental/mapping-53192.html">Developmental mapping</a> · <a href="full-report.json">Full JSON evidence</a></p>'
    (out/'full-report.html').write_text('<!doctype html><meta charset="utf-8"><title>Alpha full evaluation</title><style>body{font:16px system-ui;max-width:1200px;margin:36px auto;background:#101721;color:#e0e8ef}pre{white-space:pre-wrap;overflow-wrap:anywhere}summary{cursor:pointer;padding:12px}a{color:#86c9ff}</style>'+body)
    return report
