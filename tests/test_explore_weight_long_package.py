"""WEIGHT1_LONG package checks use only small temporary files, never SUMO."""
import json
import signal
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import explore_weight_long_package as package
import focused_package as original


class LongPackageTest(unittest.TestCase):
    def test_separate_archive_is_checked_and_excludes_source_models_and_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ("README.md", "requirements.txt", "PROJECT_PROGRESS.md", "docs/WEIGHT1_LONG_실행안내.md",
                     "src/explore_weight_long.py", "notebooks/WEIGHT1_Long_Colab.ipynb", "config/test.json",
                     "data/test.csv", "tests/test_example.py", "results/source/models/seed_42/model.pt",
                     "client_secret_private.json")
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n")
            with patch.object(package, "PROJECT", root), patch.object(original, "PROJECT", root), \
                    patch.object(original, "check_before_packing") as check, \
                    patch.object(original, "git_state", return_value={"git_commit": "test", "git_uncommitted_changes": True}):
                result = package.build_package(root / "results/colab/FutureTraffic_WEIGHT1_Long.zip")
                check.assert_called_once_with(False)
            archive = Path(result["path"])
            manifest = package.validate_long_archive(archive)
            self.assertEqual(manifest["experiment"], "WEIGHT1_LONG")
            self.assertFalse(manifest["source_models_included"])
            self.assertNotIn("client_secret_private.json", manifest["files"])
            self.assertFalse(any(name.startswith("results/") for name in manifest["files"]))
            self.assertIn("PROJECT_PROGRESS.md", manifest["files"])
            self.assertIn("docs/WEIGHT1_LONG_실행안내.md", manifest["files"])
            self.assertEqual(len(manifest["files"]), result["file_count"])
            destination = root / "extracted"
            destination.mkdir()
            (destination / "preserve.txt").write_text("keep")
            package.extract_verified(archive, destination)
            self.assertEqual((destination / "preserve.txt").read_text(), "keep")
            self.assertEqual((destination / "src/explore_weight_long.py").read_text(), "{}\n")

    def test_old_archive_names_are_protected_before_any_packing(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in package.PROTECTED_NAMES:
                target = Path(directory) / name
                target.write_bytes(b"original")
                with self.assertRaises(ValueError), patch.object(original, "check_before_packing") as check:
                    package.build_package(target)
                check.assert_not_called()
                self.assertEqual(target.read_bytes(), b"original")

    def test_original_package_kind_cannot_be_mistaken_for_long_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "not_long.zip"
            import hashlib
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("README.md", b"test")
                z.writestr("manifest.json", json.dumps({"kind": package.KIND, "experiment": "WEIGHT1",
                    "files": {"README.md": hashlib.sha256(b"test").hexdigest()}}))
            with self.assertRaises(ValueError):
                package.validate_long_archive(archive)


class CleanupTest(unittest.TestCase):
    def test_cleanup_refuses_macos_and_unrelated_commands(self):
        with patch.object(package, "sys_platform", return_value="darwin"), patch.object(package.os, "kill") as kill:
            with self.assertRaises(RuntimeError):
                package.stop_owned_process_tree(2001, Path("/expected"))
            kill.assert_not_called()
        record = {"pid": 2001, "state": "S", "started": 1, "group": 2001}
        with patch.object(package, "sys_platform", return_value="linux"), patch.object(Path, "is_dir", return_value=True), \
                patch.object(package, "_proc_record", return_value=record), \
                patch.object(Path, "read_bytes", return_value=b"python\0unrelated.py\0--out\0/expected\0"), \
                patch.object(package.os, "kill") as kill:
            with self.assertRaises(ValueError):
                package.stop_owned_process_tree(2001, Path("/expected"))
            kill.assert_not_called()

    def test_cleanup_interrupts_coordinator_then_escalates_known_descendants(self):
        root = {"pid": 2001, "state": "S", "started": 1, "group": 2001}
        children = {2001: root, 2002: {"pid": 2002, "state": "S", "started": 2, "group": 2002}}
        active = [True]

        def signal_groups(identities, sig):
            self.assertEqual(set(identities), {2001, 2002})
            if sig == signal.SIGKILL:
                active[0] = False

        with patch.object(package, "sys_platform", return_value="linux"), patch.object(Path, "is_dir", return_value=True), \
                patch.object(package, "_proc_record", return_value=root), \
                patch.object(package, "_descendants", return_value=children), \
                patch.object(package, "_still_live", side_effect=lambda identities: identities if active[0] else {}), \
                patch.object(Path, "read_bytes", return_value=b"python\0src/explore_weight_long.py\0run\0--out\0/expected\0"), \
                patch.object(package.os, "kill") as kill, \
                patch.object(package, "_signal_groups", side_effect=signal_groups) as groups:
            result = package.stop_owned_process_tree(2001, Path("/expected"), graceful_seconds=0, terminate_seconds=0)
            self.assertEqual(result["status"], "stopped")
            self.assertEqual(result["known_processes"], 2)
            kill.assert_called_once_with(2001, signal.SIGINT)
            self.assertEqual([c.args[1] for c in groups.call_args_list], [signal.SIGTERM, signal.SIGKILL])

    def test_signal_groups_never_kills_notebook_group(self):
        live = {2001: {"group": 2001}, 2002: {"group": 2002}, 2003: {"group": 900}}
        with patch.object(package, "_still_live", return_value=live), \
                patch.object(package.os, "getpgrp", return_value=900), \
                patch.object(package.os, "killpg") as killpg, patch.object(package.os, "kill") as kill:
            package._signal_groups(live, signal.SIGTERM)
            self.assertEqual({c.args for c in killpg.call_args_list}, {(2001, signal.SIGTERM), (2002, signal.SIGTERM)})
            kill.assert_called_once_with(2003, signal.SIGTERM)

    def test_reused_process_identifiers_are_not_owned(self):
        with patch.object(package, "_proc_record", return_value={"pid": 2001, "started": 99, "state": "S"}):
            self.assertEqual(package._still_live({2001: {"started": 1}}), {})


if __name__ == "__main__":
    unittest.main()
