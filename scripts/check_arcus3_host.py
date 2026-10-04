"""Fail closed on host dependencies and pinned dataset access before production starts."""
import argparse
import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REQUIRED_MODULES = (
    'huggingface_hub', 'dotenv', 'pyarrow.parquet', 'zstandard',
    'transformers', 'safetensors', 'torch',
)


def preflight(check_remote=True):
    failures = []
    for name in REQUIRED_MODULES:
        try:
            importlib.import_module(name)
        except Exception:
            failures.append(name)
    if failures:
        raise RuntimeError('Production host Python lacks usable modules: ' + ', '.join(failures))

    from dotenv import load_dotenv
    from huggingface_hub import HfApi
    from scripts.prepare_arcus3_production import SOURCES

    load_dotenv('.env', override=True)
    if check_remote:
        token = os.getenv('HF_TOKEN')
        if not token:
            raise RuntimeError('HF_TOKEN is missing from the production host environment')
        api = HfApi(token=token)
        for _category, repo, revision, _prefixes, _terms in SOURCES:
            try:
                paths = api.list_repo_files(repo, repo_type='dataset', revision=revision)
            except Exception:
                raise RuntimeError('Pinned dataset access check failed: ' + repo) from None
            if not paths:
                raise RuntimeError('Pinned dataset contains no listed files: ' + repo)
    return {'python': sys.executable, 'modules_ok': True,
            'pinned_datasets_checked': len(SOURCES) if check_remote else 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--offline', action='store_true', help='Check imports without accessing datasets')
    args = parser.parse_args()
    result = preflight(check_remote=not args.offline)
    print('Production host ready: ' + result['python'])
