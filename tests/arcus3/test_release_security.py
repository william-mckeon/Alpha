import json
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

from arcus3.release import AUTHORIZATION, digest, read_spec, validate_spec, verify_package, write_manifest


def spec(status="ready", label="alpha3.2.1"):
    return {
        "schema": "arcus3-private-release-v2",
        "model_label": label,
        "repo_id": "Islanderintel/Alpha-3.2.1",
        "private": True,
        "authorization": AUTHORIZATION,
        "status": status,
        "release_kind": "test-control",
        "package_kind": "inference-only",
        "unique_parameters": 2013403142,
        "context_tokens": 8192,
        "model_card": "docs/ALPHA_3_2_1_MODEL_CARD.md",
        "source": {
            "kind": "expanded-checkpoint",
            "parent_manifest_sha256": "a" * 64,
            "checkpoint_manifest_sha256": "b" * 64,
            "training_updates": 5,
            "input_tokens": 100,
            "target_tokens": 80,
        },
    }


class ReleaseSecurityTests(unittest.TestCase):
    def test_private_repository_is_bound_to_model_label(self):
        release = spec()
        release["repo_id"] = "someone/public-or-wrong"
        with self.assertRaisesRegex(ValueError, "repository"):
            validate_spec(release)
        release = spec()
        release["private"] = False
        with self.assertRaisesRegex(ValueError, "private"):
            validate_spec(release)

    def test_pending_release_cannot_package(self):
        release = spec(status="awaiting-evaluation")
        validate_spec(release, require_ready=False)
        with self.assertRaisesRegex(ValueError, "not ready"):
            validate_spec(release)

    def test_model_card_is_bound_to_the_release_label(self):
        release = spec()
        release["model_card"] = ".env"
        with self.assertRaisesRegex(ValueError, "approved release artifact"):
            validate_spec(release)

    def test_release_metadata_rejects_unapproved_secret_fields(self):
        release = spec();release["hf_token"] = "secret"
        with self.assertRaisesRegex(ValueError, "unapproved fields"):
            validate_spec(release)
        release = spec();release["source"]["private_note"] = "do not upload"
        with self.assertRaisesRegex(ValueError, "unapproved fields"):
            validate_spec(release)

    def test_checked_in_specs_are_well_formed_and_locked_as_expected(self):
        root = Path(__file__).resolve().parents[2]
        ready = read_spec(root / "configs/arcus3/alpha_3_2_0_release.json")
        self.assertEqual(ready["source"]["training_updates"], 11008)
        for name in ("alpha_3_2_1_release.json", "alpha_3_2_2_release.json"):
            pending = read_spec(root / "configs/arcus3" / name, require_ready=False)
            self.assertNotEqual(pending["status"], "ready")
            with self.assertRaises(ValueError):
                read_spec(root / "configs/arcus3" / name)
        future = read_spec(root / "configs/arcus3/alpha_3_2_2_release.json", require_ready=False)
        ready = copy.deepcopy(future);ready["status"] = "ready"
        ready["source"].update(checkpoint_manifest_sha256="c"*64, training_updates=1,
                               input_tokens=1, target_tokens=1,
                               adaptation_config_sha256="d"*64,
                               learning_rate_schedule_sha256="e"*64)
        validate_spec(ready)
        ready["source"]["learning_rate_schedule_sha256"] = None
        with self.assertRaisesRegex(ValueError, "learning_rate_schedule_sha256"):
            validate_spec(ready)

    def test_manifest_covers_every_inference_file_and_detects_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            (package / "model.safetensors").write_bytes(b"weights")
            (package / "config.json").write_text("{}", encoding="utf-8")
            (package / "verification.json").write_text(json.dumps({
                "complete": True, "release": spec(),
                "unique_parameters": 2013403142,
                "source": {"parent_manifest_sha256": "a" * 64,
                           "checkpoint_manifest_sha256": "b" * 64,
                           "updates": 5, "input_tokens": 100, "target_tokens": 80},
                "parity": {"logits": {"exact": True}},
            }), encoding="utf-8")
            manifest = write_manifest(package, spec())
            self.assertEqual(set(manifest["files"]), {
                "config.json", "model.safetensors", "verification.json"
            })
            verify_package(package)
            (package / "config.json").write_text('{"changed":true}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "tamper"):
                verify_package(package)

    def test_training_state_and_unlisted_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            (package / "model.safetensors").write_bytes(b"weights")
            (package / "state.pt").write_bytes(b"private optimizer state")
            with self.assertRaisesRegex(ValueError, "Non-inference|Private training"):
                write_manifest(package, spec())

    def test_manifest_digest_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "file"
            path.write_bytes(b"alpha")
            before = digest(path)
            path.write_bytes(b"beta")
            self.assertNotEqual(before, digest(path))

    def test_remote_verification_requires_private_immutable_hashes(self):
        from scripts.publish_alpha_3 import verify_remote
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            (package / "model.safetensors").write_bytes(b"weights")
            (package / "config.json").write_bytes(b"config")
            (package / "manifest.json").write_bytes(b"manifest")
            hashes = {name: digest(package / name) for name in
                      ("model.safetensors", "config.json", "manifest.json")}
            api = NS(model_info=lambda *args, **kwargs: NS(private=True, siblings=[
                NS(rfilename="model.safetensors", lfs=NS(sha256=hashes["model.safetensors"])),
                NS(rfilename="config.json", lfs=None), NS(rfilename="manifest.json", lfs=None),
            ]))
            manifest = {"release": spec(), "files": {
                "model.safetensors": {"sha256": hashes["model.safetensors"], "bytes": 7},
                "config.json": {"sha256": hashes["config.json"], "bytes": 6},
            }}
            def download(repo, name, revision):
                self.assertEqual(revision, "immutable")
                return str(package / name)
            receipt = verify_remote(api, spec()["repo_id"], package, manifest,
                                    "immutable", download)
            self.assertTrue(receipt["private"])
            self.assertEqual(set(receipt["files"]), {
                "model.safetensors", "config.json", "manifest.json"
            })
            api.model_info = lambda *args, **kwargs: NS(private=True, siblings=[
                NS(rfilename="model.safetensors", lfs=NS(sha256=hashes["model.safetensors"])),
                NS(rfilename="config.json", lfs=None), NS(rfilename="manifest.json", lfs=None),
                NS(rfilename="optimizer.pt", lfs=None),
            ])
            with self.assertRaisesRegex(ValueError, "outside the inference manifest"):
                verify_remote(api, spec()["repo_id"], package, manifest, "immutable", download)


if __name__ == "__main__":
    unittest.main()
