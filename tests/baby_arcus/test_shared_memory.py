import tempfile,unittest
from pathlib import Path
from copy import deepcopy
from baby_arcus.shared_memory import Memory,sensory_features
from baby_arcus.shared_experience import capture
from baby_arcus.services.playroom import PlayroomApplication

class SharedMemoryTests(unittest.TestCase):
    def test_prediction_error_requires_matching_time_interval(self):
        from baby_arcus.shared_causal_curriculum import transition
        before,outcome,later=transition(1,'training')[2][0]
        event={'experience':before,'outcome':outcome,'after':later}
        outcome['forecast']={'future_body':[0.0]*20,'horizon_ticks':later['tick']-before['tick']+1}
        with tempfile.TemporaryDirectory() as root:
            memory=Memory(Path(root)/'memory.sqlite3')
            try:
                memory.remember_outcome(event)
                query=deepcopy(before);query['id']='query'
                self.assertIsNone(memory.recall(query)[0]['error'])
                before['id']='matched';outcome['experience_id']='matched'
                outcome['forecast']['horizon_ticks']-=1
                memory.remember_outcome(event)
                self.assertGreater(memory.recall(query)[-1]['error'],0)
            finally:memory.close()

    def test_persistence_deduplication_and_scope(self):
        app=PlayroomApplication();self.addCleanup(app.close);row=capture(app)
        row['lesson_provenance']={'split':'training'}
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'memory.sqlite3';memory=Memory(path)
            self.assertTrue(memory.remember(row,sensory_features(row)))
            self.assertFalse(memory.remember(row,sensory_features(row)))
            with self.assertRaises(ValueError):memory.remember(row,[0.0]*72)
            memory.close();memory=Memory(path)
            query=deepcopy(row);query['id']='query';query['session']='restarted'
            self.assertEqual(len(memory.recall(query)),1)
            for key in ('entity_id','environment_id'):
                other=deepcopy(query);other[key]='different';self.assertEqual(memory.recall(other),[])
            query['lesson_provenance']={'split':'confirmation'};self.assertEqual(memory.recall(query),[])
            memory.close()

    def test_budget_and_invalid_features(self):
        app=PlayroomApplication();self.addCleanup(app.close);row=capture(app);row['lesson_provenance']={'split':'training'}
        with tempfile.TemporaryDirectory() as root:
            memory=Memory(Path(root)/'memory.sqlite3',8)
            for i in range(12):row['id']=str(i);memory.remember(row,sensory_features(row))
            row['id']='query';self.assertEqual(len(memory.recall(row)),8)
            with self.assertRaises(ValueError):memory.remember(row,[float('nan')]*72)
            memory.close()
