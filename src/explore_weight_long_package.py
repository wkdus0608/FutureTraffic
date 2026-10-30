"""Build the separate WEIGHT1_LONG Colab package, without source model weights.

This only creates a ZIP. It does not prepare or run the long experiment. The
existing WEIGHT1 archive, its stopped experiment, and the original notebooks
remain unchanged. Drive source checkpoints are read at Colab runtime only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import signal
import tempfile
import time
import zipfile
from pathlib import Path

from explore_weight_package import KIND, _safe_name, extract_verified, validate_archive

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results/colab/FutureTraffic_WEIGHT1_Long.zip"
EXPERIMENT = "WEIGHT1_LONG"
EXTRA_FILES = ("PROJECT_PROGRESS.md", "docs/WEIGHT1_LONG_실행안내.md")
REQUIRED_FILES = ("src/explore_weight_long.py", "notebooks/WEIGHT1_Long_Colab.ipynb")
PROTECTED_NAMES = {"FutureTraffic_focused.zip", "FutureTraffic_network12.zip", "FutureTraffic_WEIGHT1.zip"}


def _proc_record(pid: int) -> dict | None:
    """Linux process identity including start tick, so reused PIDs are not killed."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        fields = text[text.rindex(")") + 2:].split()
        return {"pid": pid, "state": fields[0], "ppid": int(fields[1]),
                "group": int(fields[2]), "started": int(fields[19])}
    except (OSError, ValueError, IndexError):
        return None


def _descendants(pid: int) -> dict[int, dict]:
    records = {}
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            record = _proc_record(int(path.name))
            if record is not None:
                records[record["pid"]] = record
    members = {pid}
    while True:
        more = {p for p, r in records.items() if r["ppid"] in members}
        if more <= members:
            break
        members |= more
    return {p: records[p] for p in members if p in records}


def _still_live(identities: dict[int, dict]) -> dict[int, dict]:
    live = {}
    for pid, old in identities.items():
        now = _proc_record(pid)
        if now is not None and now["started"] == old["started"] and now["state"] != "Z":
            live[pid] = now
    return live


def _signal_groups(identities: dict[int, dict], sig: int) -> None:
    """Only groups with a live, identity-matched descendant may be signalled."""
    live = _still_live(identities)
    protected = os.getpgrp()
    groups = {record["group"] for record in live.values() if record["group"] != protected}
    for group in groups:
        try:
            os.killpg(group, sig)
        except ProcessLookupError:
            pass
    for pid, record in live.items():
        if record["group"] == protected:
            # Do not signal the notebook kernel's entire process group.
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass


def stop_owned_process_tree(pid: int, expected_out: Path, *, graceful_seconds: float = 30,
                            terminate_seconds: float = 5) -> dict:
    """Interrupt this Colab job, then TERM/KILL its known worker/SUMO groups.

    No process-name-wide kill is used. Validate the coordinator command and its
    exact output argument first, and record descendant start ticks before any
    signal. SIGINT lets the coordinator reconcile its budget and checkpoints.
    This helper is for Colab/Linux; it never starts an experiment.
    """
    if sys_platform() != "linux" or not Path("/proc").is_dir():
        raise RuntimeError("WEIGHT1_LONG process stopping is Colab/Linux only.")
    if pid <= 1 or any(not math.isfinite(x) or x < 0 for x in (graceful_seconds, terminate_seconds)):
        raise ValueError("Invalid process identifier or cleanup deadline.")
    root = _proc_record(pid)
    if root is None or root["state"] == "Z":
        return {"status": "not_running", "pid": pid}
    arguments = Path(f"/proc/{pid}/cmdline").read_bytes().rstrip(b"\0").split(b"\0")
    arguments = [x.decode("utf-8", errors="replace") for x in arguments]
    try:
        output = arguments[arguments.index("--out") + 1]
    except (ValueError, IndexError) as error:
        raise ValueError("Refusing to stop an unrelated process.") from error
    if (not any(Path(a).name == "explore_weight_long.py" for a in arguments)
            or Path(output).absolute() != Path(expected_out).absolute()):
        raise ValueError("Refusing to stop a process with a different script/output.")
    identities = _descendants(pid)
    identities[pid] = root
    try:
        os.kill(pid, signal.SIGINT)
    except ProcessLookupError:
        pass
    until = time.monotonic() + graceful_seconds
    while _still_live(identities) and time.monotonic() < until:
        current = _proc_record(pid)
        if current is not None and current["started"] == root["started"]:
            identities.update(_descendants(pid))
        time.sleep(.2)
    if _still_live(identities):
        _signal_groups(identities, signal.SIGTERM)
        until = time.monotonic() + terminate_seconds
        while _still_live(identities) and time.monotonic() < until:
            time.sleep(.1)
    if _still_live(identities):
        _signal_groups(identities, signal.SIGKILL)
        until = time.monotonic() + 2
        while _still_live(identities) and time.monotonic() < until:
            time.sleep(.1)
    remaining = sorted(_still_live(identities))
    if remaining:
        raise RuntimeError(f"Owned long-run processes did not stop: {remaining}")
    return {"status": "stopped", "pid": pid, "known_processes": len(identities)}


def sys_platform() -> str:
    # A small seam for process-cleanup unit tests without launching any process.
    import sys
    return sys.platform


def validate_long_archive(archive: Path) -> dict:
    manifest = validate_archive(archive)
    if manifest.get("experiment") != EXPERIMENT or manifest.get("source_models_included") is not False:
        raise ValueError("This is not a WEIGHT1_LONG package without source checkpoints.")
    return manifest


def build_package(out: Path = OUT) -> dict:
    import focused_package as original

    out = Path(out).resolve()
    if out.name in PROTECTED_NAMES:
        raise ValueError("WEIGHT1_LONG must not overwrite a previous experiment archive.")
    original.check_before_packing(False)
    paths = set(original.files())
    paths.update(PROJECT / name for name in EXTRA_FILES)
    for name in REQUIRED_FILES:
        if PROJECT / name not in paths:
            raise FileNotFoundError(f"The long-run implementation is missing from the package: {name}")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required WEIGHT1_LONG package input is missing: {path}")
    manifest = {"kind": KIND, "experiment": EXPERIMENT,
                "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                **original.git_state(), "files": {}, "source_models_included": False,
                "runtime_defaults": {"wall_hours": 24, "compute_hours": 96, "workers": "auto"},
                "source": "Read-only existing Drive WEIGHT1 pilot, validated when the experiment is prepared"}
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".weight1_long_package_", suffix=".zip", dir=out.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for path in sorted(paths):
                relative = path.relative_to(PROJECT).as_posix()
                _safe_name(relative)
                # Only the original source/config/data/test/notebook allowlist
                # and two explicit reference documents are packaged. No Drive
                # results, model checkpoints, or root credential files are read.
                data = path.read_bytes()
                manifest["files"][relative] = hashlib.sha256(data).hexdigest()
                z.writestr(relative, data)
            z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
        validate_long_archive(Path(temporary))
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
    parser.add_argument("--verify", type=Path, help="Only verify an existing long-run archive")
    args = parser.parse_args()
    if args.verify:
        manifest = validate_long_archive(args.verify)
        print(json.dumps({"verified": str(args.verify), "file_count": len(manifest["files"])}, ensure_ascii=False))
    else:
        print(json.dumps(build_package(args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
