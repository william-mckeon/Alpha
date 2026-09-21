"""Recover an explicitly selected visual run's replay index from retained logs."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baby_arcus.visual_replay import ReplayIndex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--max-rows', type=int, default=100000)
    parser.add_argument('--max-bytes', type=int, default=1024**3)
    args = parser.parse_args()
    if not args.run.is_dir(): parser.error('Run directory does not exist')
    paths = sorted((args.run/'sessions').glob('*/experiences.jsonl'))
    index = ReplayIndex(args.run/'replay.sqlite3', args.max_rows, args.max_bytes)
    added = index.ingest(paths)
    print(json.dumps({**index.manifest(), 'added': added, 'source_files': len(paths)}, indent=2))


if __name__ == '__main__': main()
