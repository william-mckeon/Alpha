import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import os
from baby_arcus.embodiment_store import EmbodimentStore
class MigrationTests(unittest.TestCase):
    def test_transient_replace_lock_retries_without_changing_identity(self):
        with tempfile.TemporaryDirectory() as root:
            store=EmbodimentStore(root)
            try:
                body=store.load();body.head_yaw=.5
                replace=os.replace;attempts=[]
                def temporarily_locked(*args):
                    attempts.append(1)
                    if len(attempts)<3:raise PermissionError("temporary lock")
                    return replace(*args)
                with patch("baby_arcus.embodiment_store.os.replace",side_effect=temporarily_locked),patch("baby_arcus.embodiment_store.time.sleep"):
                    store.save(body)
                self.assertEqual(store.load().record(),body.record())
                self.assertEqual(len(attempts),3)
            finally:store.close()
    def test_persistent_lock_fails_and_retains_previous_body(self):
        with tempfile.TemporaryDirectory() as root:
            store=EmbodimentStore(root)
            try:
                body=store.load();before=store.path.read_bytes();body.head_yaw=.5
                with patch("baby_arcus.embodiment_store.os.replace",side_effect=PermissionError("locked")) as replace,patch("baby_arcus.embodiment_store.time.sleep"):
                    with self.assertRaises(PermissionError):store.save(body)
                    self.assertEqual(replace.call_count,6)
                self.assertEqual(store.path.read_bytes(),before)
            finally:store.close()
    def test_v1_preserved_and_backed_up(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root,"body.json")
            raw=json.dumps({"version":1,"entity_id":"arcus-original","name":"Arcus Alpha","facing":"right",
                "height":.25,"target_posture":"lying","radius":.42}).encode()
            path.write_bytes(raw)
            store=EmbodimentStore(root)
            try:
                body=store.load()
                self.assertEqual(body.entity_id,"arcus-original")
                self.assertEqual(body.record()["version"],3)
                self.assertEqual(Path(root,"body.v1.backup.json").read_bytes(),raw)
                self.assertEqual(len(body.joint_positions),12)
            finally:store.close()
