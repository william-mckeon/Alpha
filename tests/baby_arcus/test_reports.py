import unittest
from baby_arcus.reports import public_report
from baby_arcus.contracts import canonical,MAX_BODY

class ReportTests(unittest.TestCase):
    def test_http_view_bounded_without_discarding_archived_evidence(self):
        state={"run_id":"run","metrics":[{"loss":1}]*10000,"resource_history":[{"bytes":1}]*1440,
               "evaluation":[{"split":"evaluation","provenance":[{"episode_id":"a"}]*400,"episode_ids":["a"]*400}],
               "gate":{"seen":["a"]*100000},"pending_update":{"extra":{"gate":{"seen":["a"]*100000}}}}
        view=public_report(state)
        self.assertLess(len(canonical(view)),MAX_BODY)
        self.assertEqual(view["evaluation"][0]["provenance_count"],400)
        self.assertEqual(len(state["evaluation"][0]["provenance"]),400)
        self.assertEqual(len(view["metrics"]),80)
        self.assertTrue(view["has_pending_update"])
