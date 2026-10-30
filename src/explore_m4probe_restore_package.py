"""Build the separate M4_PROBE_RESTORE_V1 Colab ZIP (no model weights, no results).

Earlier WEIGHT1 / LONG / RESUME archives and notebooks are never touched.  Building this ZIP
does not run SUMO or any training.  The ZIP is verified member by member before extraction.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path

from explore_weight_package import KIND, _safe_name, extract_verified, validate_archive

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results/colab/FutureTraffic_M4PROBE_RESTORE.zip"
EXPERIMENT = "M4_PROBE_RESTORE_V1"
EXTRA_FILES = ("docs/M4PROBE_복원진단_재개.md",)
REQUIRED_FILES = ("src/explore_m4probe_continue.py", "src/explore_m4probe_diagnostics.py", "src/explore_m4probe.py",
                  "config/explore_m4probe_v1.json", "notebooks/M4PROBE_RESTORE_Colab.ipynb",
                  "tests/test_explore_m4probe_continue.py", "tests/test_explore_m4probe_diagnostics.py")
PROTECTED_NAMES = {"FutureTraffic_WEIGHT1.zip", "FutureTraffic_WEIGHT1_Long.zip", "FutureTraffic_WEIGHT1_Resume.zip",
                   "FutureTraffic_focused.zip", "FutureTraffic_network12.zip", "FutureTraffic_NIGHT.zip", "FutureTraffic_FIXED24.zip", "FutureTraffic_M4PROBE.zip"}


def validate_night_archive(archive: Path) -> dict:
    manifest = validate_archive(archive)
    if manifest.get("experiment") != EXPERIMENT or manifest.get("source_models_included") is not False:
        raise ValueError("This is not a M4_PROBE_RESTORE_V1 package without models or results.")
    return manifest


def build_package(out: Path = OUT) -> dict:
    import focused_package as original

    out = Path(out).resolve()
    if out.name in PROTECTED_NAMES:
        raise ValueError("M4_PROBE_RESTORE_V1 must not overwrite a previous experiment archive.")
    original.check_before_packing(False)
    paths = set(original.files())
    paths.update(PROJECT / name for name in EXTRA_FILES)
    for name in REQUIRED_FILES:
        if PROJECT / name not in paths:
            raise FileNotFoundError(f"The M4_PROBE_RESTORE_V1 implementation is missing from the package: {name}")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required M4_PROBE_RESTORE_V1 package input is missing: {path}")
    manifest = {"kind": KIND, "experiment": EXPERIMENT,
                "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                **original.git_state(), "files": {}, "source_models_included": False,
                "runtime_defaults": {"compute_budget": None, "wall_time_limit": None, "episode_compute_deadline": None},
                "source": "Resumes the existing M4 probe; scientific protocol unchanged and no new time budget"}
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".m4probe_restore_package_", suffix=".zip", dir=out.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for path in sorted(paths):
                relative = path.relative_to(PROJECT).as_posix()
                _safe_name(relative)
                data = path.read_bytes()
                manifest["files"][relative] = hashlib.sha256(data).hexdigest()
                z.writestr(relative, data)
            z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
        validate_night_archive(Path(temporary))
        os.replace(temporary, out)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"path": str(out), "bytes": out.stat().st_size,
            "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "file_count": len(manifest["files"]),
            "experiment": EXPERIMENT, "source_models_included": False,
            "git_commit": manifest["git_commit"], "git_uncommitted_changes": manifest["git_uncommitted_changes"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--verify", type=Path, help="Only verify an existing M4_PROBE_RESTORE_V1 archive")
    args = parser.parse_args()
    if args.verify:
        manifest = validate_night_archive(args.verify)
        print(json.dumps({"verified": str(args.verify), "file_count": len(manifest["files"])}, ensure_ascii=False))
    else:
        print(json.dumps(build_package(args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
