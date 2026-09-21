import tempfile
import unittest
from unittest.mock import patch
from baby_arcus.conversation_store import ConversationStore
from baby_arcus.contracts import ContractError
class StoreTests(unittest.TestCase):
    def test_failure_and_capacity_preserve_queue(self):
        with tempfile.TemporaryDirectory() as root:
            store=ConversationStore(root)
            first={"request_id":"one","sender":"you","text":"Hello"}
            store.send(first,False)
            with patch("baby_arcus.conversation_store.os.replace",side_effect=OSError("disk")):
                with self.assertRaises(OSError):store.release(True)
            self.assertEqual(store.rows[0]["status"],"queued")
            self.assertEqual(ConversationStore(root).rows[0]["status"],"queued")
            for i in range(1,500):store.send({"request_id":str(i),"sender":"you","text":"test"},False)
            with self.assertRaises(ContractError):store.send({"request_id":"full","sender":"you","text":"test"},False)
            self.assertEqual(len(store.rows),500)
            self.assertEqual(store.send(first,False)["request_id"],"one")

