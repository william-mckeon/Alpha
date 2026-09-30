import json
import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timezone, timedelta
from scripts.run_arcus3_phase8_continuous import may_continue

class ContinuousTests(unittest.TestCase):
    def test_only_window_expiration_allows_handoff(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);session=root/'child';session.mkdir()
            (session/'session-policy.json').write_text(json.dumps({'stop_at':datetime.now(timezone.utc).isoformat()}))
            self.assertTrue(may_continue(root,session,{'reason':'window_or_user_pause'}))
            self.assertFalse(may_continue(root,session,{'reason':'stage_complete'}))
            self.assertFalse(may_continue(root,session,{'reason':'worker_failure'}))
            (session/'pause-training').touch()
            self.assertFalse(may_continue(root,session,{'reason':'window_or_user_pause'}))
            (session/'pause-training').unlink()
            (session/'session-policy.json').write_text(json.dumps({'stop_at':(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()}))
            self.assertFalse(may_continue(root,session,{'reason':'window_or_user_pause'}))
