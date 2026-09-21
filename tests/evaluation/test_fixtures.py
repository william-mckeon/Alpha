import stat
import tempfile
import unittest
from unittest.mock import Mock, patch
from pathlib import Path
import zipfile

from evaluation.fixtures import extract_snapshot, sha256, snapshot_archive, validate_archive


class FixtureTests(unittest.TestCase):
    def test_basic_permissions_preserved_without_special_privilege_bits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "input.zip"
            info = zipfile.ZipInfo("category/file.txt")
            info.external_attr = (stat.S_IFREG | stat.S_ISUID | 0o640) << 16
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(info, "fixture")
            with patch("evaluation.fixtures.os.chmod") as chmod:
                extract_snapshot(archive_path, root / "trial", expected_sha256=sha256(archive_path))
            chmod.assert_called_once_with(root / "trial/category/file.txt", 0o640)
    def archive(self, root, names):
        path = root / "input.zip"
        with zipfile.ZipFile(path, "w") as archive:
            for name in names:
                archive.writestr(name, "fixture")
        return path

    def test_hash_checked_snapshot_and_fresh_trial_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.archive(root, ["category/file.txt"])
            target = root / "snap.zip"
            evidence = snapshot_archive(source, target, source_url="https://storage.mcpmark.ai/filesystem/category.zip", source_revision="a" * 40)
            output = root / "trial"
            self.assertEqual(extract_snapshot(target, output, expected_sha256=evidence["archive_sha256"])["files"], 1)
            self.assertEqual((output / "category/file.txt").read_text(), "fixture")
            with self.assertRaises(ValueError):
                extract_snapshot(target, output, expected_sha256=evidence["archive_sha256"])

    def test_unsafe_archives_fail_before_any_extraction(self):
        for names in (["../outside"], ["/absolute"], ["C:/drive"], ["A", "a"], ["CON.txt"], ["file."] , ["file", "file/child"]):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = self.archive(root, names)
                with self.assertRaises(ValueError):
                    extract_snapshot(source, root / "trial", expected_sha256=sha256(source))
                self.assertFalse((root / "trial").exists())

    def test_raw_backslash_and_encrypted_entries_rejected(self):
        # zipfile normalizes backslashes when writing on Windows; test raw metadata.
        for name, flags in (("back\\slash", 0), ("file", 1)):
            info = zipfile.ZipInfo("file")
            info.filename, info.flag_bits = name, flags
            archive = Mock()
            archive.infolist.return_value = [info]
            with self.assertRaises(ValueError):
                validate_archive(archive)

    def test_symlinks_and_hash_size_mismatches_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.archive(root, ["file"])
            with self.assertRaises(ValueError):
                extract_snapshot(source, root / "wrong", expected_sha256="0" * 64)
            with self.assertRaises(ValueError):
                extract_snapshot(source, root / "large", expected_sha256=sha256(source), max_bytes=1)
            info = zipfile.ZipInfo("link")
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr(info, "outside")
            with self.assertRaises(ValueError):
                extract_snapshot(source, root / "link", expected_sha256=sha256(source))
