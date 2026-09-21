"""Time unchanged qualified worker startup without starting the desktop loop."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    started = time.monotonic()
    print(json.dumps({'stage': 'import_started'}), flush=True)
    from baby_arcus.services.shared_worker import Worker
    if json.loads(Path(args.config).read_text()).get('worker_module') == 'baby_arcus.services.shared_continuity_worker':
        from baby_arcus.services.shared_continuity_worker import ContinuityWorker as Worker
    imported = time.monotonic()
    print(json.dumps({'stage': 'worker_imported', 'seconds': imported-started}), flush=True)
    worker = Worker(args.config)
    try:
        ready = worker.ready()
        report = {'ready': ready, 'import_seconds': imported-started,
                  'initialize_seconds': time.monotonic()-imported,
                  'total_seconds': time.monotonic()-started,
                  'scope': 'Single measured startup; does not certify cold-cache or pressured startup',
                  'desktop_changed': False}
        from baby_arcus.language_stream import atomic_json
        atomic_json(Path(args.output), report)
        print(json.dumps(report), flush=True)
    finally:
        worker.close()


if __name__ == '__main__':
    main()
