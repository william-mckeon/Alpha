"""Bounded corrective continuation with matched milestone evaluations."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def next_count(current, start=40000, target=41000, interval=1000):
    if (any(type(v) is not int for v in (current,start,target,interval))
            or interval<1 or start>=target or not start<=current<=target):
        raise ValueError('Outside continuation budget')
    if current==target:return 0
    milestone=min(target,start+((current-start)//interval+1)*interval)
    return min(64,milestone-current)


def worker_progress(current, now, count, paused):
    if paused and current <= now <= current + count:
        return 'paused'
    if not current < now <= current + count:
        raise RuntimeError('Worker failed to make bounded progress')
    return 'training'


def supervise(config):
    from baby_arcus.shared_factory import read_config
    from baby_arcus.shared_checkpoint import digest
    from baby_arcus.language_stream import atomic_json
    from baby_arcus.process_lock import ProcessLock
    from baby_arcus.training_mixture import validate
    from scripts.run_alpha_fresh_40000 import run_logged
    from scripts.evaluate_alpha_fresh import evaluation_identity,reusable
    from scripts.report_alpha_tool_correction import build
    cfg=read_config(config);root=Path(cfg['root']);plan=json.loads(Path(cfg['three_stage_config']).read_text())
    validate(plan)
    start,target,interval=plan['source_updates'],plan['target_total_updates'],plan['evaluation_every']
    out=root/'pilot';out.mkdir(exist_ok=True)
    def pointer():
        value=json.loads((root/'candidate.json').read_text())
        if digest(root/(value['generation']+'.pt'))!=value['sha256']:raise ValueError('Checkpoint checksum mismatch')
        next_count(value['updates'],start,target,interval)
        return value
    def evaluation(baseline=False):
        p=json.loads((root/'baseline.json').read_text()) if baseline else pointer()
        if baseline and (p['updates']!=start or p['sha256']!=plan['source_sha256']):
            raise ValueError('Wrong frozen baseline pointer')
        identity=evaluation_identity(config)
        revision=cfg.get('evaluation_revision','') if baseline else ''
        folder=out/('evaluation-'+str(p['updates'])+('-'+revision if revision else ''))
        report=folder/'evaluation.json'
        if report.exists() and reusable(json.loads(report.read_text()),p,identity):return json.loads(report.read_text())
        if folder.exists():folder.rename(folder.with_name(folder.name+'-incomplete-'+str(time.time_ns())))
        with (out/('eval-'+str(time.time_ns())+'.log')).open('x') as log:
            result=run_logged([sys.executable,'-u','scripts/evaluate_alpha_fresh.py','--config',config,'--output',str(report),'--coding']+(['--checkpoint-pointer','baseline.json'] if baseline else []),log)
        if result.returncode:raise RuntimeError('Pilot evaluation failed')
        value=json.loads(report.read_text())
        if not reusable(value,p,identity):raise ValueError('Incomplete pilot evaluation')
        return value
    lock=ProcessLock(root/'correction-supervisor.lock')
    try:
        if (root/'pause-training').exists():
            state={'state':'paused','saved':pointer()['updates'],'target':target}
            atomic_json(out/'status.json',state);return state
        if pointer()['updates']==start:evaluation()
        revision=cfg.get('evaluation_revision','')
        baseline=evaluation(baseline=True) if revision else json.loads((out/f'evaluation-{start}/evaluation.json').read_text())
        if (baseline['candidate']['sha256']!=plan['source_sha256'] or not
                reusable(baseline,baseline['candidate'],evaluation_identity(config))):
            raise ValueError('Frozen baseline evaluation mismatch')
        gates=json.loads(Path(plan['gates_file']).read_text())
        while True:
            p=pointer();current=p['updates']
            if (root/'pause-training').exists():
                state={'state':'paused','candidate':p,'saved':current,'target':target}
                atomic_json(out/'status.json',state);return state
            if current>start and ((current-start)%interval==0 or current==target):
                report=build(baseline,evaluation(),gates)
                if cfg.get('developmental_evaluation'):
                    diagnostic=root/'developmental'/('evaluation-'+str(current)+'.json')
                    if not diagnostic.exists():
                        with (out/('development-'+str(current)+'.log')).open('a') as log:
                            result=run_logged([sys.executable,'-u','scripts/evaluate_alpha_development.py','--root',str(root),'--output',str(diagnostic),'--config',cfg['developmental_evaluation'],'--mapping'],log)
                        if result.returncode:raise RuntimeError('Developmental evaluation failed')
                    from scripts.test_alpha_development_code import evaluate_saved
                    scored=diagnostic.with_name(diagnostic.stem+'-scored.json')
                    if not scored.exists():
                        import os
                        from baby_arcus.transport import Client
                        client=Client(os.environ['ALPHA_EXECUTOR_URL'],os.environ['ALPHA_EXECUTOR_TOKEN'],timeout=65,attempts=1)
                        evaluate_saved(diagnostic,scored,execute=lambda task,source:client.request('POST','/execute',{'task':task,'source':source}))
                    from scripts.report_alpha_tool_correction import attach_developmental
                    report=attach_developmental(report,json.loads(scored.read_text()))
                    report['developmental_report']=str(scored)
                report['evaluation_identity']=baseline['evaluation_identity']
                report['target_updates']=target
                report['evaluation_every']=interval
                report['previous_milestone']=None
                previous=out/f'comparison-{current-interval}.json'
                if previous.exists():
                    previous_report=json.loads(previous.read_text())
                    if previous_report.get('evaluation_identity')==report['evaluation_identity']:
                        report['previous_milestone']=previous_report['after']
                    else:report['previous_comparison_note']='Previous milestone used a different evaluator revision; not treated as matched.'
                atomic_json(out/('comparison-'+str(current)+'.json'),report)
                # Early capability failures are expected; stop early on retention regression.
                if current==target or not report['gates']['language_retention']:
                    (root/'pause-training').touch();atomic_json(out/'report.json',report)
                    state='complete' if current==target else 'review_required'
                    atomic_json(out/'status.json',{'state':state,'saved':current,'target':target,'accepted':report['accepted']})
                    return {'state':state,'report':report}
            count=next_count(current,start,target,interval)
            atomic_json(out/'status.json',{'state':'training','saved':current,'target':target})
            with (out/('worker-'+str(time.time_ns())+'.log')).open('x') as log:
                result=run_logged([sys.executable,'-u',__file__,'--config',config,'--worker','--updates',str(count)],log)
            if result.returncode:raise RuntimeError('Pilot worker failed')
            now=pointer()['updates']
            if worker_progress(current,now,count,(root/'pause-training').exists())=='paused':
                state={'state':'paused','saved':now,'target':target}
                atomic_json(out/'status.json',state);return state
    except Exception as exc:
        (root/'pause-training').touch();atomic_json(out/'status.json',{'state':'failed','error':str(exc)});raise
    finally:lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--worker',action='store_true');p.add_argument('--updates',type=int);a=p.parse_args()
    from baby_arcus.runtime_contract import require_gpu
    require_gpu()
    if a.worker:
        import torch
        torch.set_num_threads(2)
        from baby_arcus.three_stage_training import train
        print(json.dumps(train(a.config,a.updates,checkpoint_every=a.updates)))
    else:print(json.dumps(supervise(a.config)))
