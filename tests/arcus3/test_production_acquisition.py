"""A failed acquisition must not consume source rows; later batches must resume."""
import hashlib
import json
import sqlite3
import tempfile
import unittest
from contextlib import ExitStack, closing
from pathlib import Path
from unittest.mock import patch

from arcus3.checkpoint import digest
from arcus3.config import read
from arcus3.production_data import prepare


class ProductionAcquisitionTests(unittest.TestCase):
    def fixture(self, root, empty_math=False):
        donor=root/'donor';(donor/'files').mkdir(parents=True)
        (donor/'files/tokenizer.json').write_text('{}')
        seed=root/'seed';seed.mkdir()
        (seed/'manifest.json').write_text('{"shards": []}')
        exclusions=root/'exclusions.json';exclusions.write_text('[]')
        policy={**read('configs/arcus3/production.json'), 'batch_input_tokens':100000,
                'minimum_free_bytes':0, 'initial_data':str(seed)}
        sources=[]
        for category in policy['mixture']:
            path=root/(category+'.jsonl')
            count=0 if empty_math and category=='math' else 15 if category=='local' else 100
            rows=[{'text':category+str(i),'split':'train', 'max_stars_repo_licenses':['MIT']}
                  for i in range(count)]
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            sources.append({'category':category,'local':True,
                            'files':[{'path':str(path),'sha256':digest(path)}]})
        catalog=root/'catalog.json';catalog.write_text(json.dumps({'reviewed':True,'sources':sources}))
        return donor,catalog,exclusions,policy

    def prepared(self, root, name, fixture):
        donor,catalog,exclusions,policy=fixture
        def encoded(tokenizer, record, context, forbidden):
            key=hashlib.sha256(record['text'].encode()).hexdigest()
            return [{'input_ids':[1]*1000,'labels':[-100]+[1]*999,'sha256':key}]
        with ExitStack() as stack:
            stack.enter_context(patch('arcus3.tokenizer_contract.load_tokenizer',return_value=object()))
            stack.enter_context(patch('arcus3.tokenizer_contract.contract',return_value={'context_tokens':8192}))
            stack.enter_context(patch('arcus3.production_data.encode_record',side_effect=encoded))
            return prepare(root/'cache'/name,donor,catalog,exclusions,policy,root/'cache',fs=object())

    def test_next_batch_advances_public_sources_and_counts_local_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture=self.fixture(root)
            self.prepared(root,'batch-1',fixture);self.prepared(root,'batch-2',fixture)
            rows=lambda name:[json.loads(line) for line in (root/'cache'/name/'train-00000.jsonl').read_text().splitlines()]
            first,second=rows('batch-1'),rows('batch-2')
            old={r['sha256'] for r in first if r['source']!='local'}
            self.assertFalse(old & {r['sha256'] for r in second if r['source']!='local'})
            for category in ('general','code','math','instruction_tools'):
                before=[r['origin']['row'] for r in first if r['source']==category]
                after=[r['origin']['row'] for r in second if r['source']==category]
                self.assertEqual(min(after),max(before)+1)
            report=read(root/'cache/batch-2/provenance.json')
            self.assertEqual(report['repeated_input_tokens']['local'],5000)
            self.assertEqual(sum(report['source_input_tokens'].values()),100000)

    def test_failed_quota_rolls_back_all_acquisition_positions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture=self.fixture(root,empty_math=True)
            with self.assertRaisesRegex(RuntimeError,'quota: math'):
                self.prepared(root,'batch-incomplete',fixture)
            with closing(sqlite3.connect(root/'cache/acquisition.sqlite')) as db:
                for table in ('cursor','seen','batches'):
                    self.assertEqual(db.execute('SELECT COUNT(*) FROM '+table).fetchone()[0],0)
            self.assertFalse((root/'cache/batch-incomplete/manifest.json').exists())
            self.assertFalse(read(root/'cache/batch-incomplete/incomplete.json')['source_positions_committed'])
