"""Opt-in read-only verification after a restored full-size GPU continuation.

BABY_RESTORE_TEST_CONFIG names JSON containing token_file, endpoints,
source_endpoints and qualification_file. This suite never launches training.
"""
import json
import os
from pathlib import Path
import unittest
from baby_arcus.qualify import clients_for,read_token


@unittest.skipUnless(os.environ.get('BABY_RESTORE_TEST_CONFIG'),'Requires an explicit restored GPU qualification')
class RestoredStackTests(unittest.TestCase):
    def setUp(self):
        config=json.loads(Path(os.environ['BABY_RESTORE_TEST_CONFIG']).read_text())
        token=read_token(config['token_file'])
        self.restored=clients_for(token,config['endpoints'])
        self.source=clients_for(token,config['source_endpoints'])
        self.evidence=json.loads(Path(config['qualification_file']).read_text())

    def test_restored_update_has_parent_and_releases_workers(self):
        evidence=self.evidence['qualification']
        self.assertTrue(evidence['passed'])
        status=self.restored['controller'].request('GET','/v1/status')
        run=status['run']
        self.assertEqual(run['run_id'],self.evidence['run']['run_id'])
        self.assertIn(run['status'],('paused','completed'))
        self.assertIsNone(status['resource']['lease'])
        self.assertEqual(run['cycles'],evidence['target_cycles'])
        self.assertGreaterEqual(run['metrics'][-1]['samples'],1024)
        self.assertEqual(run['metrics'][-1]['updates'],evidence['previous_cycles']+1)
        root=self.restored['artifacts'].request('GET','/v1/artifacts/'+run['checkpoint_id'])['payload']
        self.assertEqual(root['metadata']['parent'],evidence['parent_checkpoint_id'])
        for name in ('inference','training'):
            self.assertFalse(self.restored[name].request('GET','/ready')['loaded'])
        report=self.restored['controller'].request('GET','/v1/reports/'+run['run_id'])
        self.assertEqual(report['checkpoint_id'],run['checkpoint_id'])

    def test_source_is_preserved_and_saved_training_settings_survive(self):
        evidence=self.evidence['qualification']
        source=self.source['controller'].request('GET','/v1/status')
        restored=self.restored['controller'].request('GET','/v1/status')
        self.assertEqual(source['run']['run_id'],evidence['source_run_id'])
        self.assertEqual(source['run']['checkpoint_id'],evidence['parent_checkpoint_id'])
        self.assertEqual(source['run']['cycles'],evidence['previous_cycles'])
        self.assertEqual(source['run']['status'],'paused')
        self.assertIsNone(source['resource']['lease'])
        for name in ('preset','seed','learning','max_steps','evaluate'):
            self.assertEqual(source['run']['options'][name],restored['run']['options'][name])
        self.assertGreater(restored['run']['sample_index'],source['run']['sample_index'])
