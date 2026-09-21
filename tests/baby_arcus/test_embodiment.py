import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from baby_arcus.embodiment import Embodiment
from baby_arcus.embodiment_store import EmbodimentStore
from baby_arcus.contracts import ContractError
from baby_arcus.artifacts import Conflict

class EmbodimentTests(unittest.TestCase):
    def test_pose_without_room(self):
        body = Embodiment()
        identity = body.entity_id
        body.target_posture = "lying"
        for _ in range(10): body.step()
        self.assertEqual(body.snapshot()["posture"], "lying")
        self.assertEqual(body.snapshot()["joints"]["front_left"]["knee"], 0)
        self.assertNotIn("x", body.snapshot())
        self.assertEqual(body.entity_id, identity)

    def test_restart_and_failed_save_preserve_record(self):
        with tempfile.TemporaryDirectory() as root:
            store = EmbodimentStore(root)
            body = store.load()
            body.target_posture = "lying"
            body.step()
            store.save(body)
            expected = body.record()
            body.height = .5
            with patch("baby_arcus.embodiment_store.os.replace", side_effect=OSError("disk")):
                with self.assertRaises(OSError): store.save(body)
            store.close()
            reopened = EmbodimentStore(root)
            self.assertEqual(reopened.load().record(), expected)
            reopened.close()

    def test_exclusive_ownership_and_corruption_not_replaced(self):
        with tempfile.TemporaryDirectory() as root:
            store = EmbodimentStore(root)
            with self.assertRaises(Conflict): EmbodimentStore(root)
            Path(root, "body.json").write_text('{}')
            with self.assertRaises(ContractError): store.load()
            self.assertEqual(Path(root, "body.json").read_text(), '{}')
            store.close()
