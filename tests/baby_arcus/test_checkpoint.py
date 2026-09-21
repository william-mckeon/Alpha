import tempfile
import unittest
from pathlib import Path
from baby_arcus.binary_artifacts import Repository
from baby_arcus.services.artifacts import ArtifactApplication
from baby_arcus.artifacts import CorruptArtifact

class BinaryTests(unittest.TestCase):
    def test_deadline_before_manifest_leaves_no_published_root(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/"data"
            path.write_bytes(b"one chunk")
            kinds=[]
            class Local:
                def request(self,method,route,body):
                    kinds.append(body["kind"])
                    return {"artifact_id":"chunk"}
            calls=0
            def heartbeat():
                nonlocal calls
                calls+=1
                if calls==2:
                    raise TimeoutError("deadline")
            with self.assertRaises(TimeoutError):
                Repository(Local()).put_file(path,heartbeat=heartbeat)
            self.assertEqual(kinds,["checkpoint_chunk"])

    def test_chunk_roundtrip_and_corruption_preserves_previous_target(self):
        with tempfile.TemporaryDirectory() as root:
            base=Path(root)
            app=ArtifactApplication(base/"store")
            class Local:
                def request(self,method,path,body=None):
                    return app(method,path,body)[1]
            repo=Repository(Local())
            source=base/"source.bin"
            source.write_bytes(bytes(range(256))*2100)
            ref=repo.put_file(source,{"purpose":"experience","checkpoint_id":"p"})
            target=base/"target.bin"
            manifest=repo.get_file(ref,target)
            self.assertEqual(source.read_bytes(),target.read_bytes())
            self.assertEqual(len(manifest["chunks"]),3)
            before=target.read_bytes()
            (base/"store"/(manifest["chunks"][0]+".blob")).write_bytes(b"corrupt")
            with self.assertRaises(CorruptArtifact):
                repo.get_file(ref,target)
            self.assertEqual(target.read_bytes(),before)
