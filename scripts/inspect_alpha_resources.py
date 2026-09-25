"""Host-side Docker metadata inspection only; emits no credentials or model data."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import re
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baby_arcus.runtime_resources import assess


def memory_bytes(text):
    match = re.fullmatch(r'\s*([0-9.]+)\s*(B|KiB|MiB|GiB|TiB|kB|MB|GB)\s*', text)
    if not match: raise ValueError('Unrecognized Docker memory measurement')
    factors = {'B':1,'KiB':1024,'MiB':1024**2,'GiB':1024**3,'TiB':1024**4,
               'kB':1000,'MB':1000**2,'GB':1000**3}
    return int(float(match[1]) * factors[match[2]])


def docker(*args):
    process = subprocess.run(['docker', *args], capture_output=True, text=True, timeout=30)
    if process.returncode:
        raise RuntimeError('Docker metadata command failed; inspect Docker status separately')
    return process.stdout


def inspect(mode='stack', service='learner', env_file='.env'):
    compose = json.loads(docker('compose', '--env-file', env_file, '--profile', '*', '-f',
                        'docker/baby-arcus/compose.alpha-three-stage.yaml', 'config', '--format', 'json'))
    engine = json.loads(docker('info', '--format', '{{json .}}'))
    ids = docker('ps', '-q').split()
    running = json.loads(docker('inspect', *ids)) if ids else []
    if ids:
        stats = [json.loads(line) for line in docker('stats','--no-stream','--format','{{json .}}',*ids).splitlines()]
        for container in running:
            sample = next((row for row in stats if container['Id'].startswith(row['ID'])), None)
            if sample:
                container['MeasuredMemoryBytes'] = memory_bytes(sample['MemUsage'].split('/')[0])
    return assess(compose, engine, running, mode, service)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['stack','job'], default='stack')
    parser.add_argument('--service', default='learner')
    parser.add_argument('--env-file', default='.env')
    parser.add_argument('--output')
    args = parser.parse_args()
    result = inspect(args.mode, args.service, args.env_file)
    if args.output:
        output = Path(args.output)
        if output.exists(): raise ValueError('Use a fresh report path')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))
    if not result['complete']: raise SystemExit('Alpha runtime resource preflight failed')
