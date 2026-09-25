"""Model execution contract. This module deliberately does not import torch."""
import os
import platform
import json
import re
import time
from pathlib import Path


def identity():
    return {'system': platform.system(), 'python': platform.python_version(),
            'container': platform.system() == 'Linux' and Path('/.dockerenv').is_file(),
            'profile': os.environ.get('ALPHA_RUNTIME_PROFILE', '')}


def require_container():
    runtime = identity()
    if not runtime['container'] or runtime['profile'] != 'alpha-container-v1':
        raise RuntimeError('Alpha model execution requires the configured Linux Docker runtime. '
                           'Use scripts/run_alpha_job.ps1; native execution is disabled.')
    return runtime


def scope():
    values = {key: os.environ.get(env, '') for key, env in (
        ('image_id', 'ALPHA_RUNTIME_IMAGE'), ('host_fingerprint', 'ALPHA_HOST_FINGERPRINT'))}
    if not re.fullmatch(r'sha256:[a-f0-9]{64}', values['image_id']) or not re.fullmatch(r'[a-f0-9]{64}', values['host_fingerprint']):
        raise RuntimeError('Exact runtime image and host fingerprint required')
    return values


def require_evidence(variable, schema):
    require_container()
    try:
        path = Path(os.environ[variable])
        if path.stat().st_size > 65536:
            raise ValueError('Oversized runtime evidence')
        report = json.loads(path.read_text(encoding='utf-8'))
        age = time.time() - report['created_at']
        if (report.get('schema') != schema or report.get('complete') is not True
                or report.get('scope') != scope() or not 0 <= age <= 86400
                or not isinstance(report.get('checks'), dict) or not report['checks']
                or not all(v is True for v in report['checks'].values())):
            raise ValueError('Failed, stale or mismatched runtime evidence')
        return report
    except (KeyError, OSError, ValueError, TypeError) as exc:
        raise RuntimeError('Valid ' + variable + ' evidence required before GPU execution') from exc


def require_gpu():
    if os.environ.get('ALPHA_GPU_MODE') == 'controlled-docker':
        return require_controlled_docker()
    require_evidence('ALPHA_HOST_QUALIFICATION', 'alpha-host-qualification-v1')
    return require_evidence('ALPHA_GPU_QUALIFICATION', 'alpha-gpu-qualification-v1')


def require_controlled_docker():
    """Explicit operating mode, not a hardware-health or training approval receipt."""
    require_container()
    runtime_scope = scope()
    try:
        root = Path('/sys/fs/cgroup')
        memory = int((root / 'memory.max').read_text().strip())
        pids = int((root / 'pids.max').read_text().strip())
        quota, period = map(int, (root / 'cpu.max').read_text().split())
        if not (0 < memory <= 10 * 1024**3 and 0 < pids <= 256
                and period > 0 and 0 < quota <= 2 * period):
            raise ValueError('Unbounded or excessive container limits')
    except (OSError, ValueError) as exc:
        raise RuntimeError('Controlled Docker mode requires cgroup v2 memory, CPU and PID limits') from exc
    return {'mode': 'controlled-docker', 'scope': runtime_scope,
            'host_stability_established': False, 'training_authorized': False,
            'limits': {'memory_bytes': memory, 'pids': pids, 'cpus': quota / period}}


def require_device(device):
    if isinstance(device,int) or str(getattr(device,'device',device)).split(':')[0] == 'cuda':
        require_gpu()


def require_checkpoint(path, device='cpu'):
    # Tiny CPU unit-test snapshots remain usable without loading a production model.
    if Path(path).stat().st_size > 64 * 1024 * 1024:
        require_container()
    require_device(device)


def model_device(config):
    if config.get('preset') == 'tiny':
        return 'cpu'
    require_gpu()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Production Alpha requires its configured GPU; refusing silent CPU fallback')
    return 'cuda'
