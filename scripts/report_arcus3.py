"""Finalize generated Python in isolated Docker executors and render all transcripts."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from arcus3.config import deadline, check_live
from arcus3.evaluation import summarize, aggregate, sha, load_suite,compatible
from baby_arcus.services.coding_executor import execute


def executor_budget(end, root):
    check_live(end, root)
    # Existing broker permits 45s execution plus 15s removal and 5s process wait.
    if (end-datetime.now(timezone.utc)).total_seconds() < 70:
        raise RuntimeError('Insufficient deadline margin for restricted executor cleanup')


def finish(root, stop_at):
    root=Path(root); end=deadline(stop_at)
    report=json.loads((root/'scores.json').read_text())
    rows=json.loads((root/'transcripts.json').read_text())
    load_suite(Path(__file__).resolve().parents[1])
    if report['suite_sha256'] != sha(Path(__file__).resolve().parents[1]/'evaluation/arcus3/baseline-v1.json'):
        raise ValueError('Generated suite differs from executor suite')
    for row in rows:
        if row.get('check')=='python':
            executor_budget(end,root)
            receipt=execute(row['executor_task'],row['metrics']['source'])
            row['metrics'].update(execution='restricted-docker',execution_receipt=receipt,task_success=receipt['passed'])
    report.update(execution_complete=True,categories=summarize(rows))
    render(root,report,rows)
    return report


def render(root, report, rows):
    root=Path(root)
    report['language_by_domain']={domain:aggregate([r for r in report['language_records'] if r['domain']==domain])
                                for domain in sorted({r['domain'] for r in report['language_records']})}
    tools=[r for r in rows if r.get('check')=='tool']
    report['tool_metrics']={key:sum(int(r['metrics'][key]) for r in tools)
                           for key in ('parseable','schema_valid','semantic_correct','executed','task_success')}
    report['tool_metrics']['total']=len(tools)
    (root/'transcripts.json').write_text(json.dumps(rows,indent=2))
    (root/'scores.json').write_text(json.dumps(report,indent=2))
    variant = 'expanded adapter checkpoint' if report.get('expanded_manifest_sha256') else 'converted experts' if report.get('conversion_manifest_sha256') else 'adapted dense control' if report.get('adapter_manifest_sha256') else 'unchanged donor'
    text=['# Arcus 3 matched evaluation — '+variant,'',report['limitations'],'',
          'Read-only evaluation. Conversation fluency, relevance and coherence await human review.','',
          '```json',json.dumps({k:v for k,v in report.items() if k!='language_records'},indent=2),'```']
    for row in rows:
        text += ['', '## '+row['id'],'', '**Prompt:** '+row['prompt'],'','**Response:**','',
                 '````text',row['response'],'````','','**Measurements:**',
                 '```json',json.dumps(row['metrics'],indent=2),'```']
    (root/'report.md').write_text('\n'.join(text),encoding='utf-8')

def compare_reports(baseline, candidate):
    if not compatible(baseline,candidate):raise ValueError('Incompatible evaluation identities')
    if not baseline['execution_complete'] or not candidate['execution_complete']:raise ValueError('Incomplete evaluation')
    return {'nll_delta':candidate['language']['nll']-baseline['language']['nll'],
            'perplexity_before':baseline['language']['perplexity'],'perplexity_after':candidate['language']['perplexity'],
            'categories':{key:{'before':value,'after':candidate['categories'][key]} for key,value in baseline['categories'].items()},
            'adapter_manifest_sha256':candidate.get('adapter_manifest_sha256'),'conversion_manifest_sha256':candidate.get('conversion_manifest_sha256')}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--deadline',required=True)
    args=p.parse_args();finish(args.root,args.deadline)
