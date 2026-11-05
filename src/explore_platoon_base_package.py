"""PBASE Colab 묶음(FutureTraffic_PBASE.zip)과 노트북(notebooks/PBASE_Colab.ipynb)을 만든다.

이전 묶음·노트북·결과는 건드리지 않는다. 묶기만 하고 SUMO는 돌리지 않는다. 모델·결과는 넣지 않는다.
  .venv/bin/python src/explore_platoon_base_package.py            # 노트북을 만들고 묶는다
  .venv/bin/python src/explore_platoon_base_package.py --verify <zip>
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
from explore_platoon0_package import CELLS as P0_CELLS
from explore_platoon1_package import PROTECTED_NAMES as P1_PROTECTED

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results/colab/FutureTraffic_PBASE.zip"
NOTEBOOK = PROJECT / "notebooks/PBASE_Colab.ipynb"
EXPERIMENT = "PBASE_V1"
EXTRA_FILES = ("results/explore/platoon_base_20261101/측정계획.md", "results/explore/platoon0_20261031/사전등록.md")
REQUIRED_FILES = ("src/explore_platoon_base.py", "src/explore_platoon_base_package.py", "src/explore_platoon0.py",
                  "src/explore_ceiling.py", "src/arena3_common.py", "src/arena3_referee.py", "src/arena3_colab.py",
                  "src/explore_pred.py", "src/explore_weight_core.py", "src/explore_weight_episode.py",
                  "src/focused_platoon.py", "config/arena3_v1.json", "config/explore_platoon0_v1.json",
                  "config/east_platoon.json", "data/derived/east_upstream_204822.csv",
                  "tests/test_explore_platoon_base.py", "tests/test_explore_platoon0.py", "notebooks/PBASE_Colab.ipynb")
PROTECTED_NAMES = {*P1_PROTECTED, "FutureTraffic_PLATOON1.zip"}

# 묶음 검증·패키지 칸은 검증을 마친 PLATOON0 노트북 칸에서 이름만 바꿔 쓴다.
ZIP_CELL = P0_CELLS[1].replace('"PLATOON0_V1", "PLATOON0 전용 ZIP이 아닙니다"', '"PBASE_V1", "PBASE 전용 ZIP이 아닙니다"') \
    .replace("platoon0_verified_package", "pbase_verified_package") \
    .replace("f\"| PLATOON0 설정 {package_manifest['platoon0_config_sha256'][:12]} | 사전 등록 {package_manifest['prereg_sha256'][:12]}\")",
             "f\"| PLATOON0 설정 {package_manifest['platoon0_config_sha256'][:12]} | 측정 계획 {package_manifest['plan_sha256'][:12]}\")")
PACKAGES_CELL = P0_CELLS[2]
assert '"PBASE_V1"' in ZIP_CELL and "plan_sha256" in ZIP_CELL, "ZIP 칸 바꾸기 실패"
assert "EXPECTED" in PACKAGES_CELL

INTRO = """# PBASE — 무리 도착 세계에서 고정신호·P0 재기 (학습 없음, 약 30~40분)

PLATOON0·PLATOON1은 무리 도착 세계에서 계획기 M4·TM4만 기준으로 썼습니다. 그래서 **실제 운영 신호(보정 고정신호)** 와
**단순 예측 제어 P0** 를 같은 84상황에서 재서 학생 PT1과 한 표로 비교합니다. 판정은 없고, 측정 방법은 실행 전에 고정했습니다
(`results/explore/platoon_base_20261101/측정계획.md`).

1. 런타임 유형 **CPU + 높은 RAM(CPU 8개)** 을 고르고 **런타임 → 모두 실행**을 누르세요. Drive 접근을 허용하세요.
2. 순서: 묶음 검증 → 패키지 → 시험 → 작은 실행(4판, 약 3분) → 본 실행(252판, 약 30분) → 비교표.
3. 본 실행은 백그라운드로 돕니다. 끊기면 다시 **모두 실행**하세요. 끝난 판은 건너뜁니다.
4. 마지막 칸에 표가 나오면 "끝났어"라고 알려 주세요. 결과: Drive `FutureTraffic/explore_platoon_base/`.
"""

SETUP = '''from google.colab import drive
from pathlib import Path
import os, sys, json

drive.mount("/content/drive")
DRIVE = Path("/content/drive/MyDrive/FutureTraffic")
ARCHIVE = DRIVE / "FutureTraffic_PBASE.zip"
PLATOON0 = DRIVE / "explore_platoon0"                  # 무리 도착 세계 M4·TM4·OM4·P1M4, 무작위 세계 M4·OM4(읽기만)
PLATOON1 = DRIVE / "explore_platoon1"                  # 학생 PT1 평가(읽기만)
ARENA3 = DRIVE / "arena3" / "referee" / "baseline"     # 무작위 세계 P0(읽기만)
OUT = DRIVE / "explore_platoon_base"
WORKSPACE = Path("/content/FutureTraffic_PBASE")
MIN_CPUS = 8
for path in (ARCHIVE, PLATOON0 / "experiment.json", PLATOON0 / "runs" / "P" / "TM4", PLATOON1 / "eval" / "PT1" / "experiment.json",
             ARENA3 / "runs" / "P0"):
    assert path.exists(), f"필요한 파일이 없습니다: {path}"
CPUS = os.cpu_count() or 1
print("CPU 개수:", CPUS, "| 결과 폴더:", OUT)
assert CPUS >= MIN_CPUS, (f"CPU가 {CPUS}개입니다. 런타임 → 런타임 유형 변경에서 CPU와 '높은 RAM'을 고른 뒤 다시 실행하세요.")'''

TESTS_CELL = '''sys.path.insert(0, str(WORKSPACE / "src"))
import arena3_colab as colab
from IPython.display import display, Markdown

for name in ("test_explore_platoon_base.py", "test_explore_platoon0.py"):
    colab.run_live(["-m", "unittest", "discover", "-s", "tests", "-p", name], WORKSPACE)
print("실행 전 검사(SUMO 없이) 통과")'''

SMOKE_CELL = '''# 1) 작은 실행(약 3분): 첫 상황 4판(무리 도착 FIXC·P0, 무작위 FIXC·P0).
#    - 다른 제어와 같은 차·같은 넘겨받기인가(PLATOON0 M4와 대조), 무작위 세계 P0가 ARENA3 기준선 P0와 같은가.
SMOKE = OUT / "smoke"
previous = colab.read_json(SMOKE / "smoke.json")
if previous and previous.get("ok"):
    print("이전 작은 실행이 통과해 건너뜁니다.")
else:
    colab.run_live(["src/explore_platoon_base.py", "smoke", "--out", str(SMOKE), "--work", "/content/pb_smoke",
                    "--workers", "4", "--platoon0", str(PLATOON0), "--arena3", str(ARENA3)], WORKSPACE, check=False)
s = colab.read_json(SMOKE / "smoke.json")
assert s and s.get("ok"), f"작은 실행 점검 실패 — 이 칸의 출력을 저에게 보내 주세요: {s and s.get('checks')}"
print("작은 실행 통과:", s["checks"])'''

RUN_CELL = '''# 2) 본 실행(252판 = 무리 도착 FIXC·P0 + 무작위 FIXC, 각 84판). 백그라운드로 돌아 셀을 멈춰도 계속 돕니다.
#    런타임이 끊기면 처음 네 칸 → 이 칸을 다시 실행하세요. 끝난 판은 건너뜁니다.
assert "colab" in globals(), "처음 네 칸(1~4)을 먼저 실행하세요. 런타임이 끊기면 변수와 작업 폴더가 사라집니다."
LOGS = OUT / "logs"

def describe(name):
    done = len(list((OUT / "runs").glob("*/*/*/cache.json")))
    failed = len(list((OUT / "runs").glob("*/*/*.failed.json")))
    return f"끝난 판 {done}/252, 실패 기록 {failed}"

pids = colab.launch({"pbase_run": ["src/explore_platoon_base.py", "run", "--out", str(OUT), "--work", "/content/pb",
                                   "--workers", str(CPUS)]}, WORKSPACE, LOGS)
code = colab.watch(pids, LOGS, 300, describe)
assert code == 0, "본 실행에서 오류가 났습니다. 위 로그를 저에게 보내 주세요(다시 실행하면 끝난 판은 건너뜁니다)."'''

REPORT_CELL = '''# 3) 비교표(판정 없음): 보정 고정신호·P0·M4·TM4·PT1을 같은 84상황에서. 숫자는 맥에서 다시 계산합니다.
colab.run_live(["src/explore_platoon_base.py", "report", "--out", str(OUT), "--platoon0", str(PLATOON0),
                "--platoon1", str(PLATOON1), "--arena3", str(ARENA3)], WORKSPACE, check=False)
if (OUT / "summary.md").exists():
    display(Markdown((OUT / "summary.md").read_text()))'''


def _cell(kind: str, text: str) -> dict:
    out = {"cell_type": kind, "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}
    if kind == "code":
        out.update(execution_count=None, outputs=[])
    return out


def notebook() -> dict:
    cells = [_cell("markdown", INTRO)] + [_cell("code", c) for c in (SETUP, ZIP_CELL, PACKAGES_CELL, TESTS_CELL,
                                                                    SMOKE_CELL, RUN_CELL, REPORT_CELL)]
    return {"cells": cells, "metadata": {"colab": {"provenance": [], "machine_shape": "hm"},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


def write_notebook(path: Path = NOTEBOOK) -> str:
    nb = notebook()
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            compile("".join(c["source"]), f"{path.name}[{i}]", "exec")
    text = json.dumps(nb, ensure_ascii=False, indent=1) + "\n"
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def validate_pbase_archive(archive: Path) -> dict:
    manifest = validate_archive(archive)
    if manifest.get("experiment") != EXPERIMENT or manifest.get("source_models_included") is not False:
        raise ValueError("PBASE 묶음이 아니거나 모델·결과가 들어 있습니다.")
    for name in REQUIRED_FILES + EXTRA_FILES:
        if name not in manifest["files"]:
            raise ValueError(f"PBASE 묶음에 필요한 파일이 없습니다: {name}")
    return manifest


def build_package(out: Path = OUT) -> dict:
    import focused_package as original

    out = Path(out).resolve()
    if out.name in PROTECTED_NAMES:
        raise ValueError("이전 실험 묶음을 덮어쓸 수 없습니다.")
    original.check_before_packing(False)
    notebook_sha = write_notebook()
    paths = set(original.files())
    paths.update(PROJECT / name for name in EXTRA_FILES)
    for name in (*REQUIRED_FILES, *EXTRA_FILES):
        if PROJECT / name not in paths or not (PROJECT / name).is_file():
            raise FileNotFoundError(f"PBASE 묶음 입력이 없습니다: {name}")
    sha = lambda p: hashlib.sha256((PROJECT / p).read_bytes()).hexdigest()  # noqa: E731
    lock = json.loads((PROJECT / "config/arena3_v1.json").read_text())
    manifest = {"kind": KIND, "experiment": EXPERIMENT,
                "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                **original.git_state(), "files": {}, "source_models_included": False,
                "lock_sha256": sha("config/arena3_v1.json"), "platoon0_config_sha256": sha("config/explore_platoon0_v1.json"),
                "plan_sha256": sha("results/explore/platoon_base_20261101/측정계획.md"),
                "environment_fingerprint": lock["environment"]["fingerprint"], "notebook_sha256": notebook_sha}
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pbase_package_", suffix=".zip", dir=out.parent)
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
        validate_pbase_archive(Path(temporary))
        os.replace(temporary, out)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"path": str(out), "bytes": out.stat().st_size, "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "file_count": len(manifest["files"]), "experiment": EXPERIMENT, "notebook_sha256": notebook_sha,
            "plan_sha256": manifest["plan_sha256"], "git_commit": manifest["git_commit"],
            "git_uncommitted_changes": manifest["git_uncommitted_changes"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--verify", type=Path, help="이미 만든 묶음만 검사")
    args = parser.parse_args()
    if args.verify:
        manifest = validate_pbase_archive(args.verify)
        print(json.dumps({"verified": str(args.verify), "file_count": len(manifest["files"])}, ensure_ascii=False))
    else:
        print(json.dumps(build_package(args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
