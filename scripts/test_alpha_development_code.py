"""Score saved generated Python through the restricted Docker executor; no model runs."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.developmental_evaluation import extract_code, summarize, suite_identity, score


def evaluate_saved(report_path, output, execute=None):
    from baby_arcus.services.coding_executor import execute as restricted
    execute=execute or restricted
    report=json.loads(Path(report_path).read_text())
    if Path(output).exists():raise ValueError('Use a new scored report')
    # Controls distinguish a broken executor from a failing model answer.
    good=execute('development-1','def solve(x):\n return x+1\n')
    bad=execute('development-1','def solve(x):\n return 0\n')
    if not good.get('passed') or bad.get('passed') or good.get('returncode')!=0:
        raise RuntimeError('Developmental executor control failed')
    report['executor_controls']={'correct_passed':True,'incorrect_rejected':True}
    config=json.loads(Path('configs/baby_arcus/alpha_developmental_eval.json').read_text())
    prompts={r['id']:r for r in map(json.loads,Path(config['prompts']).read_text().splitlines())}
    report['scoring_identity']=suite_identity(['baby_arcus/developmental_evaluation.py',config['rubric']])
    report['execution_identity']=suite_identity(['baby_arcus/developmental_tasks.py','baby_arcus/services/coding_executor.py'])
    for row in report['records']:
        row['scores']=score(prompts[row['id']],row)
        if row['category']!='python':continue
        task='development-'+str(int(row['id'].rsplit('-',1)[1])-1)
        result=execute(task,extract_code(row['response']))
        row['scores'].update(correctness=result['passed'],task_success=result['passed'],
                             execution='restricted_docker',execution_result=result)
    report['summary']=summarize(report['records'])
    Path(output).write_text(json.dumps(report,indent=2))
    from scripts.report_alpha_development import write_review_html
    write_review_html(report,Path(output).with_suffix('.html'))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('report');p.add_argument('--output',required=True);a=p.parse_args()
    evaluate_saved(a.report,a.output)
