"""Stage verified source-derived tool drills without approving or training them."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baby_arcus.contracts import digest
from baby_arcus.data_manifest import file_digest
from baby_arcus.data_staging import StagingStore
from baby_arcus.language_stream import atomic_json
from baby_arcus.sft_source_lessons import extract, demonstrate, ADAPTER


def prepare(source, output, tokenizer, fixture=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    base = Path(__file__).resolve().parents[1] / 'runs' / 'test2'
    if base not in output.parents or output.exists():
        raise ValueError('Use a fresh output directory under runs/test2')
    output.mkdir(parents=True)
    report = {'adapter': ADAPTER, 'source_sha256': file_digest(source), 'input_rows': 0,
              'candidate_edits': 0, 'accepted_lessons': 0, 'assistant_targets': 0,
              'target_tokens': 0, 'quarantined': 0, 'duplicates': 0, 'reasons': {},
              'batches': [], 'fixture': fixture, 'approved': False, 'complete': False,
              'original_repository_tasks_solved': False, 'model_loaded': False}
    candidates, parents = [], {}
    def find(key):
        parents.setdefault(key, key)
        if parents[key] != key:
            parents[key] = find(parents[key])
        return parents[key]
    def union(a, b):
        a, b = find(a), find(b)
        parents[max(a, b)] = min(a, b)
    def reject(reason):
        report['quarantined'] += 1
        report['reasons'][reason] = report['reasons'].get(reason, 0) + 1
    # Bounded streaming scan, retaining only small edit candidates. No source code execution.
    with source.open(encoding='utf-8') as stream:
        while True:
            line = stream.readline(1048577)
            if not line:
                break
            if len(line) > 1048576:
                raise ValueError('Oversized source row')
            row = json.loads(line)
            report['input_rows'] += 1
            try:
                spec = extract(row)
                session = row.get('meta', {}).get('session_id') or row.get('session_id')
                if not isinstance(session, str) or not 1 <= len(session) <= 200:
                    raise ValueError('source_session_required')
            except (ValueError, KeyError, TypeError) as exc:
                reject(str(exc)[:200])
                continue
            if len(candidates) >= 10000:
                raise ValueError('Candidate lesson limit exceeded')
            signature = digest({k: spec[k] for k in ('kind', 'old', 'new')})
            session_hash = digest(session)
            # Sessions sharing duplicate lessons stay in the same held-out cohort.
            union('session:' + session_hash, 'content:' + signature)
            candidates.append((spec, digest(row), session_hash, signature))
    report['candidate_edits'] = len(candidates)
    if file_digest(source) != report['source_sha256']:
        raise ValueError('Source changed during preparation')
    seen = set()
    prepared = []
    store = StagingStore(output / 'staging.sqlite', fixture=fixture)
    try:
        for index, (spec, row_hash, session_hash, signature) in enumerate(candidates):
            if signature in seen:
                report['duplicates'] += 1
                continue
            seen.add(signature)
            cluster = find('session:' + session_hash)
            group = 'opencode-lesson:' + digest([report['source_sha256'], cluster])
            split = 'training'  # Final held-out split is assigned across accepted clusters below.
            provenance = {'input_sha256': row_hash, 'lesson': {
                'source_file_sha256': report['source_sha256'], 'source_session_sha256': session_hash}}
            try:
                records, evidence = demonstrate(spec, output / ('workspace-' + str(index)), tokenizer,
                    ('fixture:' if fixture else '') + ADAPTER + ':' + row_hash,
                    group, split, provenance)
            except (ValueError, KeyError, TypeError) as exc:
                reject('lesson:' + str(exc)[:200])
                continue
            prepared.append((index, group))
            report['accepted_lessons'] += 1
            report['assistant_targets'] += len(records)
            report['target_tokens'] += evidence['packing']['target_tokens']
            atomic_json(output / ('lesson-' + str(index) + '.json'), {'records': records, **evidence})
        groups = sorted({group for _, group in prepared}, key=digest)
        # Avoid an empty validation cohort when there are independent lesson groups.
        validation = set(groups[:max(1, (len(groups)+9)//10)]) if len(groups) > 1 else set()
        report['split_counts'] = {'training': 0, 'validation': 0}
        report['split_target_tokens'] = {'training': 0, 'validation': 0}
        for index, group in prepared:
            path = output / ('lesson-' + str(index) + '.json')
            evidence = json.loads(path.read_text(encoding='utf-8'))
            split = 'validation' if group in validation else 'training'
            records = evidence['records']
            for record in records: record['split'] = split
            from baby_arcus.sft_dataset import packing_report
            evidence['packing'] = packing_report(records, tokenizer)
            batch = store.stage(records)
            report['batches'].append({'id': batch, 'split': split, 'records': len(records),
                                      'lesson_index': index, 'group': group})
            report['split_counts'][split] += len(records)
            report['split_target_tokens'][split] += evidence['packing']['target_tokens']
            atomic_json(path, evidence)
        report['complete'] = True
    finally:
        store.close()
        atomic_json(output / 'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--encoding', default='o200k_base')
    args = parser.parse_args()
    from baby_arcus.runtime_contract import require_container
    require_container()
    from arcus.tokenizer import get_tokenizer
    print(json.dumps(prepare(args.source, args.output, get_tokenizer(args.encoding))))
