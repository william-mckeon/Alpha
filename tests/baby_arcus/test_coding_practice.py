import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from baby_arcus.coding_practice import run_episode
from baby_arcus.trajectory_store import TrajectoryStore
from baby_arcus.coding_tools import DEFINITIONS
from baby_arcus.tool_catalog import ToolCatalog
from baby_arcus.tool_context import ToolContext


class PracticeTests(unittest.TestCase):
    def test_cancelled_policy_does_not_touch_model(self):
        from baby_arcus.coding_policy import decide
        self.assertEqual(decide(None,None,None,[],[],cancelled=lambda:True)['status'],'cancelled')

    def test_caregiver_interrupt_prevents_tool_side_effect(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=TrajectoryStore(Path(tmp)/'trace.sqlite')
            catalog=ToolCatalog(DEFINITIONS); cancelled=threading.Event(); executed=[]
            tools=SimpleNamespace(catalog=catalog,context=ToolContext(catalog),
                                  environment=SimpleNamespace(task_name='positive_sum',task={'instruction':'practice'}),
                                  execute=lambda call:executed.append(call))
            def policy(state):
                cancelled.set()
                return {'call':{'name':'tool_search','version':1,'arguments':{'query':'read file'}}}
            try:
                result=run_episode('episode',tools,policy,store,cancelled=cancelled.is_set)
                self.assertFalse(result['solved']); self.assertEqual(executed,[])
                self.assertEqual(store.read('episode')[-1]['result']['status'],'cancelled')
                self.assertFalse(result['messages'][-2]['train'])
                store.db.execute("UPDATE events SET payload='{}' WHERE sequence=0")
                with self.assertRaises(ValueError): store.read('episode')
            finally: store.close()
