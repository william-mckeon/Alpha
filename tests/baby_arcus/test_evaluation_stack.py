"""Read-only post-run audit; BABY_EVALUATION_TEST_CONFIG selects explicit evidence.

Config: token_file, endpoints, source_endpoints, before_file, prefix_file,
qualification_file. Before captures both stacks before frozen review; prefix is
the evaluator's durable receipt recovered after the deliberate interruption.
"""
import json
import os
from pathlib import Path
import unittest
from baby_arcus.qualify import clients_for,read_token


@unittest.skipUnless(os.environ.get('BABY_EVALUATION_TEST_CONFIG'),'Requires a completed frozen GPU review')
class FrozenEvaluationStackTests(unittest.TestCase):
    def setUp(self):
        config=json.loads(Path(os.environ['BABY_EVALUATION_TEST_CONFIG']).read_text())
        token=read_token(config['token_file'])
        self.clients=clients_for(token,config['endpoints'])
        self.source=clients_for(token,config['source_endpoints'])
        self.before=json.loads(Path(config['before_file']).read_text())
        self.prefix=json.loads(Path(config['prefix_file']).read_text())
        self.evidence=json.loads(Path(config['qualification_file']).read_text())
        self.status=self.clients['controller'].request('GET','/v1/status')

    def test_complete_population_retains_recovered_prefix(self):
        run=self.status['run']
        self.assertTrue(self.evidence['qualification']['passed'])
        self.assertEqual(run['run_id'],self.evidence['run']['run_id'])
        self.assertEqual(run['status'],'completed')
        job=run['evaluation_job']
        self.assertTrue(job['complete'])
        self.assertEqual(job['batch_id'],self.prefix['batch_id'])
        receipt=self.clients['evaluator'].request('GET','/v1/evaluations/'+job['batch_id'])
        self.assertTrue(receipt['complete'])
        self.assertEqual(receipt['checkpoint_id'],run['checkpoint_id'])
        rows=receipt['provenance']
        self.assertGreater(len(self.prefix['provenance']),0)
        self.assertEqual(rows[:len(self.prefix['provenance'])],self.prefix['provenance'])
        self.assertEqual(len(rows),400)
        self.assertEqual(len({r['episode_id'] for r in rows}),400)
        self.assertEqual({(r['family'],r['index']) for r in rows},
                         {(f,i) for f in ('switch_delivery','clue_search')
                          for i in range(job['start_index'],job['start_index']+200)})
        for family in ('switch_delivery','clue_search'):
            self.assertEqual(receipt['results'][family]['episodes'],200)
            self.assertEqual(receipt['results'][family]['wins'],sum(r['success'] for r in rows if r['family']==family))
        index=next(i for i,b in enumerate(run['evaluation']) if b.get('batch_id')==job['batch_id'])
        report=self.clients['controller'].request('GET',f"/v1/reports/{run['run_id']}/evaluations/{index}")
        self.assertEqual(report['provenance'],rows)
        self.assertEqual(report['purpose'],'frozen-checkpoint-review')
        self.assertIs(report['gate_applied'],False)

    def test_training_source_and_released_workers_preserved(self):
        run=self.status['run']
        before=self.before['restored']['run']
        for key in ('checkpoint_id','cycles','sample_index','metrics','curriculum','gate','reserved_index','practice_index'):
            self.assertEqual(run.get(key),before.get(key),key)
        self.assertEqual(run['training_options'],before['options'])
        self.assertEqual(run['evaluation_index'],before['evaluation_index']+200)
        self.assertIsNone(self.status['resource']['lease'])
        source=self.source['controller'].request('GET','/v1/status')
        self.assertEqual(source['run'],self.before['source']['run'])
        self.assertIsNone(source['resource']['lease'])
        for clients in (self.clients,self.source):
            for name in ('inference','training'):
                self.assertFalse(clients[name].request('GET','/ready')['loaded'])
