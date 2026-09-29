"""Explicit pinned acquisition or offline verification; no model loading."""
import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
from arcus3.config import read, safe_child, REVISION
from arcus3.donor import acquire, verify

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['download', 'verify'])
    args = parser.parse_args()
    project = read('configs/arcus3/project.json')
    destination = safe_child('artifacts/arcus3', 'donor/'+REVISION)
    result = acquire(project, destination) if args.operation == 'download' else verify(destination)
    print(json.dumps({'revision':result['revision'], 'verified_files':len(result['files'])}))
