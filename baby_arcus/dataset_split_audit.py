"""Stable repository grouping and cross-source content leakage checks."""
import hashlib
import re


def content_hash(text):
    return hashlib.sha256(re.sub(r'\s+', ' ', text).strip().encode()).hexdigest()


def split_for(group):
    value = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
    return 'test' if value < 5 else 'validation' if value < 15 else 'training'


def audit(rows, heldout_groups=()):
    groups, contents = {}, {}
    for row in rows:
        group, split = row['group'], row['split']
        if split not in ('training', 'validation', 'test'):
            raise ValueError('Unknown split')
        if group in heldout_groups and split == 'training':
            raise ValueError('Held-out repository in training: ' + group)
        if group in groups and groups[group] != split:
            raise ValueError('Repository group crosses splits')
        groups[group] = split
        text = row.get('text') or '\n'.join(m['content'] for m in row['messages'])
        key = content_hash(text)
        if key in contents and contents[key] != split:
            raise ValueError('Duplicate content crosses splits')
        contents[key] = split
    return {'groups': len(groups), 'unique_contents': len(contents), 'passed': True,
            'limitation': 'Whitespace-normalized exact matching, not semantic near-duplicate detection'}
