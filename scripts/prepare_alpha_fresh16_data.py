"""Create a separate 16K reviewed-data candidate; never mutate source artifacts."""
import argparse
import collections
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baby_arcus.contracts import digest
from baby_arcus.sft_dataset import windows
from baby_arcus.tool_discovery_curriculum import records, tasks
from baby_arcus.data_staging import StagingStore


def prepare(source, output):
    from arcus.tokenizer import get_tokenizer
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = get_tokenizer('o200k_base')
    counts, rejected = collections.Counter(), []
    store = StagingStore(output/'review.sqlite')
    batch, batches, groups = [], [], {}
    target = (output/'records.jsonl').open('w', encoding='utf-8')
    def admit(record):
        identity = digest(record)
        try:
            split = record['split']
            if groups.setdefault(record['group'], split) != split:
                raise ValueError('Group crosses splits')
            window_count = target_count = maximum = 0
            for window in windows(record, tokenizer, 16384):
                window_count += 1
                target_count += sum(window['mask'][1:])
                maximum = max(maximum, len(window['ids'])-1)
            if not window_count:
                raise ValueError('No supervised targets')
        except ValueError as exc:
            rejected.append({'sha256': identity, 'reason': str(exc), 'source': record['source']})
            return
        target.write(json.dumps(record)+'\n')
        counts[split+'_records'] += 1
        counts[split+'_windows'] += window_count
        counts[split+'_target_tokens'] += target_count
        counts['max_input_tokens'] = max(counts['max_input_tokens'], maximum)
        batch.append(record)
        if len(batch) == 20:
            batches.append(store.stage(batch.copy())); batch.clear()
    try:
        with (source/'packed64/records.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                admit(json.loads(line))
        for split in ('training', 'validation', 'test'):
            for row in records(split): admit(row)
        if batch: batches.append(store.stage(batch))
        manifest = json.loads((source/'combined-coding-manifest.json').read_text())
        source_batch = store.stage_sources(manifest)
        report = {'complete': True, 'approved': False, 'context_tokens': 16384,
                  'counts': dict(counts), 'rejected': rejected, 'sft_batches': batches,
                  'coding_batch': source_batch,
                  'limitations': ['Imported external calls remain context only pending semantic mapping.',
                                  'Source-code benchmark overlap is not yet cleared.',
                                  'Unseen names test schema use, not generalization to all tool semantics.']}
        (output/'manifest.json').write_text(json.dumps(report, indent=2))
        (output/'tool-holdouts.json').write_text(json.dumps({s: list(tasks(s)) for s in ('validation', 'test')}, indent=2))
        print(json.dumps({'counts': dict(counts), 'rejected': len(rejected), 'approved': False}))
    finally:
        target.close(); store.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); prepare(args.source, args.output)
