"""Checkpointed .25-only continuation. A budget boundary is never success.

The supervisor runs frozen validation between blocks, and the original-sized
confirmation suite only when basic validation is competent. It never trains 1.0.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baby_arcus.language_stream import atomic_json
from baby_arcus.shared_checkpoint import digest
from baby_arcus.shared_factory import read_config


def validation_ready(report):
    r = report['results']
    return (report.get('complete') and all(r[f]['successes'] == r[f]['episodes'] for f in ('standing','lying','sitting','approach'))
        and all(r[k]['successes']/r[k]['examples'] >= (.8 if k == 'unpaired:rest' else .95)
                for k in r if ':' in k) and r['language']['nll'] <= 8.6869118285)


def primary_checks(report, targets):
    r = report['results']
    result = {f: r[f]['episodes'] == 200 and r[f]['successes'] == 200 for f in ('standing','lying','sitting','approach')}
    result.update({k: r[k]['examples'] == 300 and r[k]['successes'] >= v for k,v in targets['decision_successes'].items()})
    result['language'] = r['language']['examples'] == 200 and r['language']['nll'] <= targets['maximum_language_nll']+1e-8
    return result


def run(config, maximum_updates, wait_pid=None):
    cfg = read_config(config)
    if cfg['depth_capacity'] != .25:
        raise ValueError('Supervisor is .25-only')
    root = Path(cfg['root'])
    targets = json.loads(Path('configs/baby_arcus/test2_baseline_targets.json').read_text())
    baseline = Path(targets['baseline_root'])/(targets['baseline_generation']+'.pt')
    full_root = Path('runs/test2/depth100-seed-2101')
    full_pointer = (full_root/'candidate.json').read_bytes()
    full = json.loads(full_pointer)
    if digest(baseline) != targets['baseline_sha256'] or digest(full_root/(full['generation']+'.pt')) != full['sha256']:
        raise ValueError('Protected checkpoint identity mismatch')
    status = {'state': 'running', 'config': config, 'pid': os.getpid(), 'mastery_established': False,
              'full_depth_training_authorized': False, 'maximum_updates_this_invocation': maximum_updates}
    status_path = root/'baseline-supervisor.json'

    def publish(**extra):
        status.update(extra)
        status['updated_at'] = time.time()
        atomic_json(status_path, status)

    def command(script, arguments, name):
        publish(stage=name)
        with (root/(name+'.log')).open('a', encoding='utf-8') as log:
            completed = subprocess.run([sys.executable, 'scripts/'+script, *arguments], stdout=log, stderr=subprocess.STDOUT)
        return completed.returncode

    publish(stage='waiting_for_initial_training' if wait_pid else 'starting')
    if wait_pid:
        # psutil is available in the project environment; PID wait does not train concurrently.
        import psutil
        try:
            process = psutil.Process(wait_pid)
            process.wait()
        except psutil.NoSuchProcess:
            pass
    try:
        while True:
            manifest = json.loads((root/'candidate.json').read_text())
            publish(candidate=manifest)
            if (root/'pause-training').exists():
                publish(state='paused', stage='requested_pause')
                return
            validation = root/f'baseline-validation-{manifest["generation"]}.json'
            if not validation.exists():
                code = command('evaluate_arcus_baseline_parity.py', ['--config', config], 'validation-'+manifest['generation'])
                if code:
                    raise RuntimeError('Validation process failed; inspect stage log')
            report = json.loads(validation.read_text())
            publish(validation=str(validation))
            if validation_ready(report):
                confirm = root/f'baseline-confirmation-{manifest["generation"]}.json'
                if not confirm.exists():
                    code = command('evaluate_arcus_baseline_parity.py', ['--config',config,'--split','confirmation','--episodes','200','--cases','300'], 'confirmation-'+manifest['generation'])
                    if code:
                        raise RuntimeError('Confirmation process failed')
                checks = primary_checks(json.loads(confirm.read_text()), targets)
                if all(checks.values()):
                    evidence = root/('parity-'+manifest['generation'])
                    evidence.mkdir(exist_ok=True)
                    linked = evidence/(manifest['generation']+'.pt')
                    if not linked.exists():
                        os.link(root/linked.name, linked)
                    atomic_json(evidence/'candidate.json', manifest)
                    evalcfg = json.loads(Path('configs/baby_arcus/shared.json').read_text())
                    evalcfg.update(root=str(evidence), depth_capacity=.25)
                    evalpath = evidence/'config.json'
                    atomic_json(evalpath, evalcfg)
                    stages = [('evaluate_arcus_shared_pixels.py',['--confirmation']),
                              ('evaluate_arcus_shared_curiosity.py',[]),
                              ('qualify_arcus_continuity_candidate.py',['--split','confirmation','--scenes','256'])]
                    for script, args in stages:
                        code = command(script, ['--config',str(evalpath),*args], script[:-3]+'-'+manifest['generation'])
                        checks[script] = code == 0
                    names = ['confirmation-pixels.json','confirmation-curiosity.json','confirmation-object-continuity.json',
                             'confirmation-moving-object-continuity.json','confirmation-object-tracks.json']
                    if all((evidence/n).exists() for n in names):
                        pixels,curiosity,stationary,moving,tracks = [json.loads((evidence/n).read_text()) for n in names]
                        checks['pixel_count'] = pixels['count_accuracy'] >= targets['minimum_pixel_count_accuracy']
                        checks['ball_iou'] = pixels['ball_iou'] >= targets['minimum_ball_iou']-1e-8
                        checks['curiosity_discoveries'] = curiosity['exploration']['learned']['discoveries'] >= targets['minimum_curiosity_discoveries']
                        for label, values in [('stationary',stationary),('moving',moving)]:
                            for metric in ('precision','recall'):
                                checks[label+'_'+metric] = values['association'][metric] >= targets['minimum_'+label+'_'+metric]-1e-6
                        checks['memory_recovery'] = tracks['exact_memory_recoveries'] == 256
                        diagnostics = evidence/'confirmation-continuity-diagnostics.json'
                        checks['continuity_diagnostics'] = diagnostics.exists() and json.loads(diagnostics.read_text()).get('diagnostics_passed') is True
                    else:
                        checks['complete_evidence'] = False
                publish(checks=checks)
                if all(checks.values()):
                    if digest(baseline) != targets['baseline_sha256'] or (full_root/'candidate.json').read_bytes() != full_pointer or digest(full_root/(full['generation']+'.pt')) != full['sha256']:
                        raise ValueError('Protected baseline or full-depth experiment changed')
                    publish(state='parity_reached', stage='report_required', mastery_established=True)
                    return
            if manifest['updates'] >= maximum_updates:
                publish(state='review_required', stage='training_budget_boundary', reason='Parity not established; assess learning curves and adjust before more training.')
                return
            block = min(4096, maximum_updates-manifest['updates'])
            code = command('train_arcus_to_baseline.py', ['--config', config, '--updates',str(block)], 'training-after-'+str(manifest['updates']))
            if code:
                raise RuntimeError('Training process failed; inspect stage log')
    except Exception as exc:
        publish(state='error', error=str(exc))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/baby_arcus/test2.depth025-sustained.json')
    p.add_argument('--maximum-updates', type=int, default=16396)
    p.add_argument('--wait-pid', type=int)
    args = p.parse_args()
    if not 1 <= args.maximum_updates <= 1000000:
        p.error('Invalid update budget')
    run(args.config, args.maximum_updates, args.wait_pid)
