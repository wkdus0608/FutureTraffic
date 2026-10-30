"""ARENA3 2단계 Colab 묶음(FutureTraffic_ARENA3_S2.zip) — 노트북 세 개(S2A 이득 회귀, S2B 선택 모방, S2C 판정·확인)가 같은 묶음을 쓴다.

1단계 묶음(FutureTraffic_ARENA3.zip)·노트북·결과는 건드리지 않는다. 묶기만 하고 SUMO·학습은 돌리지 않는다. 모델·결과는 넣지 않는다.
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

from explore_weight_package import KIND, _safe_name, validate_archive
from arena3_package import PROTECTED_NAMES as ARENA3_PROTECTED

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results/colab/FutureTraffic_ARENA3_S2.zip"
EXPERIMENT = "ARENA3_S2_V1"
NOTEBOOKS = ("ARENA3_S2A_advantage.ipynb", "ARENA3_S2B_policy.ipynb", "ARENA3_S2C_judge.ipynb")
EXTRA_FILES = ("docs/ARENA3_설계_20261027.md", "docs/ARENA3_2단계_실행안내.md")
REQUIRED_FILES = ("src/arena3_common.py", "src/arena3_referee.py", "src/arena3_bprime.py", "src/arena3_g1.py",
                  "src/arena3_colab.py", "src/arena3_stage2_data.py", "src/arena3_stage2_train.py",
                  "src/arena3_stage2_eval.py", "src/arena3_stage2_package.py", "config/arena3_v1.json",
                  "config/arena3_stage2_v1.json", "tests/test_arena3.py", "tests/test_arena3_stage2_data.py",
                  "tests/test_arena3_stage2_train.py", "tests/test_arena3_stage2_eval.py",
                  *(f"notebooks/{n}" for n in NOTEBOOKS))
PROTECTED_NAMES = {*ARENA3_PROTECTED, "FutureTraffic_ARENA3.zip"}


def validate_stage2_archive(archive: Path) -> dict:
    manifest = validate_archive(archive)
    if manifest.get("experiment") != EXPERIMENT or manifest.get("source_models_included") is not False:
        raise ValueError("ARENA3_S2_V1 묶음이 아니거나 모델·결과가 들어 있습니다.")
    for name in REQUIRED_FILES:
        if name not in manifest["files"]:
            raise ValueError(f"ARENA3 2단계 묶음에 필요한 파일이 없습니다: {name}")
    return manifest


def build_package(out: Path = OUT) -> dict:
    import focused_package as original

    out = Path(out).resolve()
    if out.name in PROTECTED_NAMES:
        raise ValueError("이전 실험 묶음을 덮어쓸 수 없습니다.")
    original.check_before_packing(False)
    paths = set(original.files())
    paths.update(PROJECT / name for name in EXTRA_FILES)
    for name in (*REQUIRED_FILES, *EXTRA_FILES):
        if PROJECT / name not in paths or not (PROJECT / name).is_file():
            raise FileNotFoundError(f"ARENA3 2단계 묶음 입력이 없습니다: {name}")
    sha = lambda p: hashlib.sha256((PROJECT / p).read_bytes()).hexdigest()
    lock = json.loads((PROJECT / "config/arena3_v1.json").read_text())
    manifest = {"kind": KIND, "experiment": EXPERIMENT,
                "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                **original.git_state(), "files": {}, "source_models_included": False,
                "lock_sha256": sha("config/arena3_v1.json"), "stage2_config_sha256": sha("config/arena3_stage2_v1.json"),
                "environment_fingerprint": lock["environment"]["fingerprint"], "notebooks": list(NOTEBOOKS)}
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".arena3_s2_package_", suffix=".zip", dir=out.parent)
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
        validate_stage2_archive(Path(temporary))
        os.replace(temporary, out)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"path": str(out), "bytes": out.stat().st_size, "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "file_count": len(manifest["files"]), "experiment": EXPERIMENT, "lock_sha256": manifest["lock_sha256"],
            "stage2_config_sha256": manifest["stage2_config_sha256"],
            "git_commit": manifest["git_commit"], "git_uncommitted_changes": manifest["git_uncommitted_changes"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--verify", type=Path, help="이미 만든 묶음만 검사")
    args = parser.parse_args()
    if args.verify:
        manifest = validate_stage2_archive(args.verify)
        print(json.dumps({"verified": str(args.verify), "file_count": len(manifest["files"])}, ensure_ascii=False))
    else:
        print(json.dumps(build_package(args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
