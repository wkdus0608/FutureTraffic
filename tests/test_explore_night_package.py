"""WEIGHT1_NIGHT package checks use only small temporary files, never SUMO."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import explore_night_package as package
import focused_package as original


class NightPackageTest(unittest.TestCase):
    def test_archive_has_the_night_files_and_no_models_results_or_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ("README.md", "requirements.txt", "PROJECT_PROGRESS.md", "docs/NIGHT_실행안내.md",
                     *package.REQUIRED_FILES, "src/explore_weight_package.py", "data/test.csv",
                     "results/explore/weight_resume_v1/models/seed_42/model.pt", "client_secret_private.json")
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n")
            with patch.object(package, "PROJECT", root), patch.object(original, "PROJECT", root), \
                    patch.object(original, "check_before_packing") as check, \
                    patch.object(original, "git_state", return_value={"git_commit": "test", "git_uncommitted_changes": True}):
                result = package.build_package(root / "results/colab/FutureTraffic_NIGHT.zip")
                check.assert_called_once_with(False)
            manifest = package.validate_night_archive(Path(result["path"]))
            self.assertEqual(manifest["experiment"], "WEIGHT1_NIGHT")
            self.assertFalse(manifest["source_models_included"])
            self.assertNotIn("client_secret_private.json", manifest["files"])
            self.assertFalse(any(name.startswith("results/") for name in manifest["files"]))
            for name in package.REQUIRED_FILES + package.EXTRA_FILES:
                self.assertIn(name, manifest["files"])
            destination = root / "extracted"
            destination.mkdir()
            package.extract_verified(Path(result["path"]), destination)
            self.assertEqual((destination / "src/explore_night.py").read_text(), "{}\n")

    def test_old_archives_are_protected_before_packing(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in package.PROTECTED_NAMES:
                target = Path(directory) / name
                target.write_bytes(b"original")
                with self.assertRaises(ValueError), patch.object(original, "check_before_packing") as check:
                    package.build_package(target)
                check.assert_not_called()
                self.assertEqual(target.read_bytes(), b"original")

    def test_missing_trainer_cannot_produce_an_incomplete_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("README.md", "PROJECT_PROGRESS.md", "docs/NIGHT_실행안내.md", "src/explore_night.py"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("x")
            target = root / "results/colab/FutureTraffic_NIGHT.zip"
            with patch.object(package, "PROJECT", root), patch.object(original, "PROJECT", root), \
                    patch.object(original, "check_before_packing"), \
                    patch.object(original, "git_state", return_value={"git_commit": "t", "git_uncommitted_changes": False}):
                with self.assertRaises(FileNotFoundError):
                    package.build_package(target)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
