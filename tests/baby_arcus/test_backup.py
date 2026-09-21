"""Offline snapshot integrity, portability, ownership and publication checks."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from baby_arcus.backup import SERVICES,export_snapshot,verify_snapshot,restore_snapshot,publish_directory
from baby_arcus.contracts import ContractError
from baby_arcus.process_lock import ProcessLock
from baby_arcus.artifacts import Conflict


class BackupTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.sources={name:self.root/"sources"/name for name in SERVICES}
        for path in self.sources.values():
            path.mkdir(parents=True)
        self.controller=self.sources["controller"]
        (self.controller/"run.json").write_text(json.dumps({"status":"paused","run_id":"run","checkpoint_id":"accepted"}))
        (self.controller/"resource.json").write_text('{"lease":null}')
        ProcessLock(self.controller/"controller.lock").close()
        (self.sources["training"]/"candidate.pt").write_bytes(b"binary\0weights"*1000)
        (self.sources["artifacts"]/"report.json").write_text('{"retained":true}')
        self.archive=self.root/"snapshot.zip"

    def test_roundtrip_every_byte_and_no_overwrite(self):
        result=export_snapshot(self.sources,self.archive,True)
        manifest=verify_snapshot(self.archive)
        self.assertEqual(result["files"],4)
        restored=self.root/"restored"
        restore_snapshot(self.archive,restored)
        for name in manifest["files"]:
            self.assertEqual((restored/name).read_bytes(),(self.root/"sources"/name).read_bytes())
        self.assertFalse((restored/"controller/controller.lock").exists())
        for name in SERVICES:
            self.assertTrue((restored/name).is_dir())
        with self.assertRaises(FileExistsError):
            restore_snapshot(self.archive,restored)
        with self.assertRaises(FileExistsError):
            export_snapshot(self.sources,self.archive,True)

    def test_requires_offline_settled_and_exclusive_ownership(self):
        with self.assertRaises(ContractError):
            export_snapshot(self.sources,self.archive)
        owner=ProcessLock(self.controller/"controller.lock")
        try:
            with self.assertRaises(Conflict):
                export_snapshot(self.sources,self.archive,True)
        finally:
            owner.close()
        (self.controller/"resource.json").write_text('{"lease":{"owner":"training"}}')
        with self.assertRaises(ContractError):
            export_snapshot(self.sources,self.archive,True)
        self.assertFalse(self.archive.exists())

    def rewrite(self,extra=None,corrupt=False):
        export_snapshot(self.sources,self.archive,True)
        with zipfile.ZipFile(self.archive) as source:
            entries={name:source.read(name) for name in source.namelist()}
        manifest=json.loads(entries["manifest.json"])
        if corrupt:
            entries["training/candidate.pt"]=b"X"*len(entries["training/candidate.pt"])
        for name in extra or []:
            entries[name]=b"test"
            manifest["files"][name]={"size":4,"sha256":hashlib.sha256(b"test").hexdigest()}
        entries["manifest.json"]=json.dumps(manifest).encode()
        altered=self.root/"altered.zip"
        with zipfile.ZipFile(altered,"w") as target:
            for name,data in entries.items():
                target.writestr(name,data)
        return altered

    def test_corruption_never_publishes_partial_restore(self):
        altered=self.rewrite(corrupt=True)
        with self.assertRaises(ContractError):
            verify_snapshot(altered)
        restored=self.root/"restored"
        with self.assertRaises(ContractError):
            restore_snapshot(altered,restored)
        self.assertFalse(restored.exists())
        self.assertFalse(list(self.root.glob('.baby-restore-*')))

    def test_unsafe_and_nonportable_paths_rejected(self):
        for names in [["training/../../escape"],["training/CON.txt"],["training/foo."],["training/A","training/a"],["training/foo","training/foo/bar"]]:
            with self.subTest(names=names):
                if self.archive.exists():
                    self.archive.unlink()
                altered=self.rewrite(extra=names)
                with self.assertRaises(ContractError):
                    restore_snapshot(altered,self.root/"restored")
                self.assertFalse((self.root/"restored").exists())

    def test_publish_refuses_even_existing_empty_directory(self):
        source=self.root/"staging"
        target=self.root/"target"
        source.mkdir()
        target.mkdir()
        with self.assertRaises(FileExistsError):
            publish_directory(source,target)
        self.assertTrue(source.exists())
