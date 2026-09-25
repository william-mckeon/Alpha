"""Read-only structural review; does not approve data or enable training."""
import collections
import json
from pathlib import Path


def review(root):
    counts = collections.Counter()
    groups = collections.defaultdict(set)
    tools = collections.Counter()
    samples = {}
    with (root / 'packed64/records.jsonl').open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            source = row['source']
            if source.startswith('alpha:synthetic-tool-mechanics-v1:'):
                source = 'alpha:synthetic-tool-mechanics-v1'
            key = f"{source} / {row['split']}"
            counts[key] += 1
            groups[row['group']].add(row['split'])
            if len(samples.setdefault(key, [])) < 3:
                samples[key].append(line_number)
            for message in row['messages']:
                if message.get('role') == 'tool':
                    tools[f"{source} tool-context messages"] += 1
                if message.get('role') == 'assistant' and message.get('train', True):
                    tools[f"{source} supervised assistant messages"] += 1
    return {'approved': False, 'training_enabled': False,
            'records_by_source_split': dict(counts), 'message_counts': dict(tools),
            'groups_crossing_splits': {k: sorted(v) for k, v in groups.items() if len(v) > 1},
            'sample_record_line_numbers': samples,
            'limitations': ['Structural inventory only; not semantic or benchmark contamination clearance.']}


if __name__ == '__main__':
    root = Path('runs/test2/fresh64-dataset-v2')
    result = review(root)
    (root / 'review-inventory.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))

