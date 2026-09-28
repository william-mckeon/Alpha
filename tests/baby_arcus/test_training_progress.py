import contextlib
import io
import sys
import unittest
from baby_arcus.training_progress import progress_line
from scripts.run_alpha_fresh_40000 import run_logged


class ProgressTests(unittest.TestCase):
    def test_saved_and_transient_steps_are_distinct(self):
        line=progress_line(420,40000,404,16,8.,2.5,'sft')
        self.assertIn('420/40,000',line)
        self.assertIn('saved=404',line)
        self.assertIn('2.00 steps/s',line)
        self.assertIn('excludes setup/evaluation',line)

    def test_output_and_failure_reach_both_destinations(self):
        durable=io.StringIO();visible=io.StringIO()
        with contextlib.redirect_stdout(visible):
            result=run_logged([sys.executable,'-u','-c',
                              "import sys; print('progress',flush=True); print('failure',file=sys.stderr); sys.exit(7)"],durable)
        self.assertEqual(result.returncode,7)
        self.assertEqual(durable.getvalue(),visible.getvalue())
        self.assertIn('progress',visible.getvalue())
        self.assertIn('failure',visible.getvalue())
