"""Build and verify the separate WEIGHT1 Colab archive.

The existing focused archive and notebook are never modified. The archive has
an exact, SHA-256-checked file manifest and can be extracted over an existing
WEIGHT1 checkout without deleting that directory or following symlinks.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results/colab/FutureTraffic_WEIGHT1.zip"
EXTRA_FILES = ("PROJECT_PROGRESS.md", "docs/WEIGHT1_실행안내.md")
KIND = "WEIGHT1_PACKAGE_V1"


def _safe_name(name: str) -> str:
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise ValueError(f"Unsafe archive member name: {name!r}")
    path = PurePosixPath(name)
    if not path.parts or path.is_absolute() or any(p in (".", "..") for p in path.parts) or path.as_posix() != name:
        raise ValueError(f"Unsafe archive member path: {name!r}")
    return name


def validate_archive(archive: Path) -> dict:
    """Check the entire archive before extracting a byte to the destination."""
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or names.count("manifest.json") != 1:
            raise ValueError("Archive member names are duplicated or the manifest is absent.")
        manifest = json.loads(z.read("manifest.json"))
        if manifest.get("kind") != KIND or not isinstance(manifest.get("files"), dict):
            raise ValueError("This is not a complete WEIGHT1 package manifest.")
        expected = manifest["files"]
        if set(names) != set(expected) | {"manifest.json"} or not expected or "manifest.json" in expected:
            raise ValueError("Archive payload and manifest file list differ.")
        for info in z.infolist():
            _safe_name(info.filename)
            if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"Directories/symlinks are not package payload files: {info.filename}")
            if info.filename == "manifest.json":
                continue
            wanted = expected[info.filename]
            if not isinstance(wanted, str) or len(wanted) != 64 or any(c not in "0123456789abcdef" for c in wanted):
                raise ValueError(f"Invalid SHA-256 in manifest: {info.filename}")
            actual = hashlib.sha256(z.read(info.filename)).hexdigest()
            if actual != wanted:
                raise ValueError(f"Archive SHA-256 mismatch: {info.filename}")
    return manifest


def _destination(root: Path, relative: str) -> Path:
    target = root / _safe_name(relative)
    for part in (target, *target.parents):
        if part.is_symlink():
            raise ValueError(f"Extraction refuses an existing symlink: {part}")
        if part == root:
            break
    return target


def extract_verified(archive: Path, destination: Path) -> dict:
    """Preserve existing extra files and replace only verified package members."""
    manifest = validate_archive(Path(archive))
    root = Path(destination).expanduser().absolute()
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise ValueError("The extraction destination must be an ordinary directory.")
    root.mkdir(parents=True, exist_ok=True)
    # Check all destination paths before replacing any existing package file.
    targets = {name: _destination(root, name) for name in (*manifest["files"], "manifest.json")}
    with zipfile.ZipFile(archive) as z:
        for name, target in targets.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            _destination(root, name)
            data = z.read(name)
            fd, temporary = tempfile.mkstemp(prefix=".weight1_extract_", dir=target.parent)
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(data)
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
    for name, wanted in manifest["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != wanted:
            raise ValueError(f"Extracted file SHA-256 mismatch: {name}")
    return manifest


def build_package(out: Path = OUT) -> dict:
    # Reuse the original raw-data and three locked-environment checks. Its OUT
    # and main() are intentionally not used, preserving the original archive.
    import focused_package as original

    original.check_before_packing(False)
    paths = set(original.files())
    paths.update(PROJECT / name for name in EXTRA_FILES)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required WEIGHT1 package input is missing: {path}")
    manifest = {"kind": KIND, "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                **original.git_state(), "files": {},
                "experiment": "WEIGHT1", "budget_hours": {"total": 48, "initial": 6}}
    out = Path(out).resolve()
    if out.name in {"FutureTraffic_focused.zip", "FutureTraffic_network12.zip"}:
        raise ValueError("WEIGHT1 must not overwrite an existing experiment archive.")
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".weight1_package_", suffix=".zip", dir=out.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for path in sorted(paths):
                relative = path.relative_to(PROJECT).as_posix()
                _safe_name(relative)
                payload = path.read_bytes()
                manifest["files"][relative] = hashlib.sha256(payload).hexdigest()
                z.writestr(relative, payload)
            z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
        validate_archive(Path(temporary))
        os.replace(temporary, out)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"path": str(out), "bytes": out.stat().st_size,
            "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "file_count": len(manifest["files"]),
            "git_commit": manifest["git_commit"], "git_uncommitted_changes": manifest["git_uncommitted_changes"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--verify", type=Path, help="Verify an existing WEIGHT1 archive without rebuilding it")
    args = parser.parse_args()
    if args.verify:
        manifest = validate_archive(args.verify)
        print(json.dumps({"verified": str(args.verify), "file_count": len(manifest["files"])}, ensure_ascii=False))
    else:
        print(json.dumps(build_package(args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
