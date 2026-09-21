import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import unittest
from baby_arcus.artifacts import ArtifactStore, Conflict, CorruptArtifact
from baby_arcus.contracts import ContractError, canonical

class ArtifactTests(unittest.TestCase):
    def test_corrupt_manifest_is_integrity_failure(self):
        with tempfile.TemporaryDirectory() as root:
            store = ArtifactStore(root)
            manifest = store.publish("r","test",{"value":1})
            Path(root,manifest["artifact_id"]+".json").write_bytes(b'{')
            with self.assertRaises(CorruptArtifact):
                store.read(manifest["artifact_id"])

    def test_idempotent_and_concurrent(self):
        with tempfile.TemporaryDirectory() as root:
            store = ArtifactStore(root)
            with ThreadPoolExecutor(4) as executor:
                manifests = list(executor.map(lambda _:store.publish("r","test",{"hello":"world"}),range(4)))
            self.assertTrue(all(m == manifests[0] for m in manifests))
            self.assertEqual(store.read(manifests[0]["artifact_id"])["payload"],{"hello":"world"})
            with self.assertRaises(Conflict):
                store.publish("r","test",{"different":True})
            self.assertEqual(ArtifactStore(root).publish("r","test",{"hello":"world"}),manifests[0])

    def test_corruption_and_traversal(self):
        with tempfile.TemporaryDirectory() as root:
            store = ArtifactStore(root)
            manifest = store.publish("r","test",{"value":1})
            Path(root,manifest["artifact_id"]+".blob").write_bytes(b"corrupt")
            with self.assertRaises(CorruptArtifact):
                store.read(manifest["artifact_id"])
            with self.assertRaises(CorruptArtifact):
                store.publish("r","test",{"value":1})
            with self.assertRaises(ContractError):
                store.read("../secrets")

    def test_manifest_last_partial_publish_recovers(self):
        with tempfile.TemporaryDirectory() as root:
            store = ArtifactStore(root)
            original = store._atomic
            def interrupt(path,data):
                if path.suffix == ".json":
                    raise OSError("simulated interruption")
                return original(path,data)
            store._atomic = interrupt
            with self.assertRaises(OSError):
                store.publish("r","test",{"value":1})
            blobs = list(Path(root).glob("*.blob"))
            self.assertEqual(len(blobs),1)
            with self.assertRaises(KeyError):
                store.read(blobs[0].stem)
            manifest = ArtifactStore(root).publish("r","test",{"value":1})
            self.assertEqual(ArtifactStore(root).read(manifest["artifact_id"])["payload"],{"value":1})
