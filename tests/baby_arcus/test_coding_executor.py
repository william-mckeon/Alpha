import unittest
from unittest.mock import patch
from baby_arcus.services.coding_executor import Application, execute


class ExecutorTests(unittest.TestCase):
    def test_command_and_path_injection_rejected(self):
        app=Application()
        with patch('baby_arcus.services.coding_executor.subprocess.Popen') as spawn:
            with self.assertRaises(ValueError): app('POST','/execute',{'task':'positive_sum','source':'pass','command':'anything'})
            with self.assertRaises(ValueError): execute('../../escape','pass')
            with self.assertRaises(ValueError): execute('positive_sum','x'*12001)
            spawn.assert_not_called()

    def test_concurrent_executor_request_rejected(self):
        app=Application(); app.lock.acquire()
        try:
            status,_=app('POST','/execute',{'task':'positive_sum','source':'pass'})
            self.assertEqual(status,409)
        finally: app.lock.release()
