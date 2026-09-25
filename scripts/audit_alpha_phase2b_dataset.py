"""Read-only verification of a built Phase 2B review pack."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.data_manifest import file_digest, validate_manifest
from baby_arcus.data_staging import StagingStore
from baby_arcus.dataset_split_audit import audit


def verify(root):
    root=Path(root)
    report=json.loads((root/'report.json').read_text())
    records=[]
    store=StagingStore(root/'staging.sqlite',readonly=True)
    try:
        for identity in report['batches']:
            records.extend(store.get(identity)['records'])
        if report['corpus_batch']:
            manifest=store.get(report['corpus_batch'])['manifest']
            validate_manifest(manifest)
            for entry in manifest['files']:
                path=root/entry['path']
                if file_digest(path)!=entry['sha256']: raise ValueError('Corpus hash mismatch')
                for line in path.read_text(encoding='utf-8').splitlines():
                    row=json.loads(line)
                    if row['split']!=entry['split']: raise ValueError('Corpus split mismatch')
                    records.append(row)
    finally: store.close()
    config=json.loads((root/'source-config.json').read_text())
    from baby_arcus.contracts import digest
    if digest(config)!=report['source_config_sha256']: raise ValueError('Source config changed')
    return audit(records,config.get('heldout_groups',[]))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('root')
    print(json.dumps(verify(parser.parse_args().root),indent=2))
