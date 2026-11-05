"""Build the separate JOINT2 Colab ZIP and notebook (no models, no results).

Earlier archives and notebooks are never touched.  Building does not run SUMO.  The ZIP is verified member by member
before extraction (same checker as WEIGHT1/FIXED24).  The 12-map data still comes from FutureTraffic_network12.zip on Drive.
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

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results/colab/FutureTraffic_JOINT2.zip"
NOTEBOOK = PROJECT / "notebooks/JOINT2_Colab.ipynb"
EXPERIMENT = "JOINT2_V1"
EXTRA_FILES = ("docs/JOINT2_실행안내.md", "results/explore/joint2_20261031/사전등록.md")
REQUIRED_FILES = ("src/explore_joint2.py", "src/explore_joint2_package.py", "src/explore_pred.py", "src/explore_coord12.py",
                  "src/network_eval.py", "notebooks/JOINT2_Colab.ipynb", "tests/test_explore_joint2.py")
PROTECTED_NAMES = {"FutureTraffic_WEIGHT1.zip", "FutureTraffic_WEIGHT1_Long.zip", "FutureTraffic_WEIGHT1_Resume.zip",
                   "FutureTraffic_focused.zip", "FutureTraffic_network12.zip", "FutureTraffic_NIGHT.zip",
                   "FutureTraffic_FIXED24.zip", "FutureTraffic_ARENA3.zip", "FutureTraffic_ARENA3_S2.zip",
                   "FutureTraffic_ARENA3_PPO.zip"}

INTRO = """# 0단계 JOINT2 — 석천사거리 + 동쪽 이웃을 함께 예측 제어해 여지 재기

학습 없는 진단입니다. 12개 지도 검증 2일 46조각 × seed 5개 × 정책 4개(보정 고정신호, 기존 P2, E, J2)를 돌리고,
실행 전에 정한 기준(`사전등록.md`)으로 'J2가 P2보다 도로망 지연을 줄이나'를 판정합니다.

1. **런타임 유형: CPU, 높은 RAM(CPU 8개)** 을 고르고 **런타임 → 모두 실행**을 누르세요.
2. Drive 접근을 허용하세요. 검사 → 작은 실행(학습 날짜 한 조각) → 본 실행 → 판정 순서로 자동 진행합니다.
3. 본 실행은 약 6~8시간 걸립니다. 끊기면 다시 **모두 실행**하면 끝난 실행은 건너뜁니다.
4. 마지막 칸에 판정이 나오면 "끝났어"라고 알려 주세요. 결과: Drive `FutureTraffic/explore_joint2/`.
"""

CELLS = [
    """from google.colab import drive
from pathlib import Path
import os

drive.mount("/content/drive")
DRIVE_ROOT = Path("/content/drive/MyDrive/FutureTraffic")
ARCHIVE = DRIVE_ROOT / "FutureTraffic_JOINT2.zip"
NET12_ZIP = DRIVE_ROOT / "FutureTraffic_network12.zip"
REFERENCE = DRIVE_ROOT / "explore_pred" / "network12"
OUT = DRIVE_ROOT / "explore_joint2"
WORKSPACE = Path("/content/FutureTraffic_JOINT2")
for path in (ARCHIVE, NET12_ZIP, REFERENCE / "9d3b7504064b"):
    assert path.exists(), f"필요한 파일이 없습니다: {path}"
CPUS = os.cpu_count() or 1
assert CPUS >= 8, f"CPU가 {CPUS}개입니다. 런타임 유형에서 '높은 RAM'(CPU 8개)을 고른 뒤 다시 실행하세요."
print("CPU 개수:", CPUS, "| 결과 폴더:", OUT)""",
    """# ZIP 전체의 파일 목록·SHA-256을 확인한 뒤 검증된 전용 추출기를 사용합니다.
import hashlib, json, stat, sys, types, zipfile
from pathlib import PurePosixPath

with zipfile.ZipFile(ARCHIVE) as archive:
    names = archive.namelist()
    assert len(names) == len(set(names)) and names.count("manifest.json") == 1, "ZIP 파일 목록 오류"
    package_manifest = json.loads(archive.read("manifest.json"))
    assert package_manifest.get("kind") == "WEIGHT1_PACKAGE_V1" and package_manifest.get("experiment") == "JOINT2_V1", "JOINT2 전용 ZIP이 아닙니다"
    expected = package_manifest["files"]
    assert set(names) == set(expected) | {"manifest.json"}, "ZIP과 manifest 목록 불일치"
    for info in archive.infolist():
        name, path = info.filename, PurePosixPath(info.filename)
        assert path.parts and not path.is_absolute() and ".." not in path.parts and path.as_posix() == name, "잘못된 ZIP 경로"
        assert "\\\\" not in name and not info.is_dir() and not stat.S_ISLNK(info.external_attr >> 16), "잘못된 ZIP 파일 형식"
        if name != "manifest.json":
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected[name], f"SHA-256 불일치: {name}"
    source_code = archive.read("src/explore_weight_package.py").decode("utf-8")
bootstrap = types.ModuleType("joint2_verified_package")
bootstrap.__file__ = str(WORKSPACE / "src/explore_weight_package.py")
exec(compile(source_code, bootstrap.__file__, "exec"), bootstrap.__dict__)
package_manifest = bootstrap.extract_verified(ARCHIVE, WORKSPACE)
print(f"검증·추출 완료: {len(package_manifest['files'])}개 파일 → {WORKSPACE} | 커밋 {package_manifest['git_commit']}")

with zipfile.ZipFile(NET12_ZIP) as z:
    z.extractall(WORKSPACE)
m12 = json.loads((WORKSPACE / "network12_manifest.json").read_text())
bad = [p for p, h in m12["files"].items() if hashlib.sha256((WORKSPACE / p).read_bytes()).hexdigest() != h]
assert not bad, bad
print(f"12개 지도 묶음: 파일 {len(m12['files'])}개 | 배경 {m12['background']:.0%} | 유턴 {m12['uturn']}")
os.chdir(WORKSPACE)""",
    """import subprocess, importlib.metadata
os.environ["FOCUSED_RULESET"], os.environ["FOCUSED_NORTH_EXT"] = "pet1", "240"
WANT = {"eclipse-sumo": "1.27.1", "sumo-rl": "1.4.5"}
def versions():
    out = {}
    for name in WANT:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out
if versions() != WANT:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=WORKSPACE, check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "eclipse-sumo==1.27.1"], cwd=WORKSPACE, check=True)
assert versions() == WANT, f"패키지 버전이 맥과 다릅니다: {versions()}"
sys.path.insert(0, str(WORKSPACE / "src"))
import network_rl, network_eval
assert network_rl._code_fingerprint() == m12["network12_fingerprint"] == "711875f6c939", "12개 지도 코드 지문이 다릅니다"
assert network_eval.evaluation_fingerprint() == "9d3b7504064b", "12개 지도 평가 지문이 기존 P2와 다릅니다"
print("패키지·지문 확인:", versions())""",
    """import signal
ENV = {**os.environ, "FOCUSED_RULESET": "pet1", "FOCUSED_NORTH_EXT": "240", "PYTHONUNBUFFERED": "1",
       "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
def run_live(args, keep=None):
    process = subprocess.Popen([sys.executable, "-u", *args], cwd=WORKSPACE, env=ENV,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    tail = []
    try:
        for line in process.stdout:
            tail = (tail + [line])[-40:]
            if keep is None or any(k in line for k in keep):
                print(line.rstrip(), flush=True)
        code = process.wait()
    except KeyboardInterrupt:
        os.killpg(process.pid, signal.SIGTERM)
        raise
    if code != 0:
        print("".join(tail))
        raise RuntimeError(f"작업이 중단됐습니다 (종료 코드 {code}). 위의 원인을 확인해 주세요.")

# Colab에는 이름이 tests인 다른 패키지가 깔려 있어 'tests.test_explore_joint2'로 부르면 그쪽을 찾다 실패한다(11/01).
# tests 폴더에서 파일을 직접 찾게 한다(다른 노트북과 같은 방식).
run_live(["-m", "unittest", "discover", "-s", "tests", "-p", "test_explore_joint2.py"])
print("단위 시험 통과")""",
    """# 작은 실행: 학습 날짜 08-11 18시 한 조각에서 E·J2를 돌려 이웃 제어 규칙(녹색 순서·길이 범위)과 점수 계산을 확인합니다(약 10분).
SMOKE = Path("/content/joint2_smoke")
run_live(["src/explore_joint2.py", "smoke", "--out", str(SMOKE)], keep=("{", "}", '"', "Error", "Traceback", "실패"))
smoke = json.loads((SMOKE / "smoke.json").read_text())
assert smoke["j2_check"]["ok"], smoke["j2_check"]
print("작은 실행 통과:", smoke["j2_check"])""",
    """# 본 실행: 46조각 × seed 5개 × 정책 4개 = 920번(끝난 실행은 건너뜀)
run_live(["src/explore_joint2.py", "run", "--out", str(OUT), "--workers", str(CPUS)],
         keep=("JOINT2", "/", "Error", "Traceback"))""",
    """# 판정(사전등록.md): 재현 점검(이번 P2 = 기존 P2) → J2 대 P2
run_live(["src/explore_joint2.py", "report", "--out", str(OUT), "--reference", str(REFERENCE)])
rep = json.loads((OUT / "report.json").read_text())
print("재현 점검:", rep["reproduction_p2"])
print("판정:", rep.get("verdict"))""",
]


def write_notebook(path: Path = NOTEBOOK) -> Path:
    def cell(kind, text):
        lines = text.splitlines(keepends=True)
        c = {"cell_type": kind, "metadata": {}, "source": lines}
        if kind == "code":
            c.update({"execution_count": None, "outputs": []})
        return c
    nb = {"nbformat": 4, "nbformat_minor": 0,
          "metadata": {"colab": {"provenance": [], "machine_shape": "hm"},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
          "cells": [cell("markdown", INTRO)] + [cell("code", c) for c in CELLS]}
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
    return path


def build_package(out: Path = OUT) -> dict:
    import focused_package as original

    out = Path(out).resolve()
    if out.name in PROTECTED_NAMES:
        raise ValueError("JOINT2 must not overwrite a previous experiment archive.")
    write_notebook()
    original.check_before_packing(False)
    paths = set(original.files())
    paths.update(PROJECT / name for name in EXTRA_FILES)
    for name in REQUIRED_FILES:
        if PROJECT / name not in paths:
            raise FileNotFoundError(f"The JOINT2 implementation is missing from the package: {name}")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required JOINT2 package input is missing: {path}")
    manifest = {"kind": KIND, "experiment": EXPERIMENT,
                "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                **original.git_state(), "files": {}, "source_models_included": False,
                "source": "12-map data from FutureTraffic_network12.zip on Drive; earlier P2 runs read only for the reproduction check"}
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".joint2_package_", suffix=".zip", dir=out.parent)
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
        if validate_archive(Path(temporary)).get("experiment") != EXPERIMENT:
            raise ValueError("JOINT2 archive validation failed.")
        os.replace(temporary, out)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"path": str(out), "bytes": out.stat().st_size, "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "file_count": len(manifest["files"]), "experiment": EXPERIMENT,
            "git_commit": manifest["git_commit"], "git_uncommitted_changes": manifest["git_uncommitted_changes"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--notebook-only", action="store_true")
    args = parser.parse_args()
    if args.notebook_only:
        print(write_notebook())
    else:
        print(json.dumps(build_package(args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
