"""RELAX0 Colab 묶음(FutureTraffic_RELAX0.zip)과 노트북 3개(notebooks/RELAX0_R·P1·P2.ipynb)를 만든다.

이전 묶음·노트북·결과는 건드리지 않는다. 묶기만 하고 SUMO는 돌리지 않는다. 모델·결과는 넣지 않는다.
  .venv/bin/python src/explore_relax0_package.py            # 노트북을 만들고 묶는다
  .venv/bin/python src/explore_relax0_package.py --verify <zip>
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

from explore_platoon_base_package import PACKAGES_CELL, PROTECTED_NAMES as PB_PROTECTED, ZIP_CELL as PB_ZIP_CELL
from explore_weight_package import KIND, _safe_name, validate_archive

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results/colab/FutureTraffic_RELAX0.zip"
NOTEBOOKS = {"R": PROJECT / "notebooks/RELAX0_R.ipynb", "P1": PROJECT / "notebooks/RELAX0_P1.ipynb", "P2": PROJECT / "notebooks/RELAX0_P2.ipynb"}
EXPERIMENT = "RELAX0_V1"
PLAN = "results/explore/rule_relax_20261101/측정계획.md"
EXTRA_FILES = (PLAN, "results/explore/platoon0_20261031/사전등록.md")
REQUIRED_FILES = ("src/explore_relax0.py", "src/explore_relax0_package.py", "src/explore_platoon_base.py", "src/explore_platoon_base_package.py",
                  "src/explore_platoon0.py", "src/explore_ceiling.py", "src/arena3_common.py", "src/arena3_referee.py", "src/arena3_colab.py",
                  "src/explore_pred.py", "src/explore_weight_core.py", "src/explore_weight_episode.py", "src/focused_platoon.py",
                  "config/arena3_v1.json", "config/explore_platoon0_v1.json", "config/east_platoon.json", "data/derived/east_upstream_204822.csv",
                  "tests/test_explore_relax0.py", "tests/test_explore_platoon0.py", *(p.relative_to(PROJECT).as_posix() for p in NOTEBOOKS.values()))
PROTECTED_NAMES = {*PB_PROTECTED, "FutureTraffic_PBASE.zip", "FutureTraffic_PLATOON1C.zip"}

# 묶음 검증·패키지 칸은 검증을 마친 PBASE 노트북 칸에서 이름만 바꿔 쓴다.
ZIP_CELL = PB_ZIP_CELL.replace('"PBASE_V1", "PBASE 전용 ZIP이 아닙니다"', '"RELAX0_V1", "RELAX0 전용 ZIP이 아닙니다"') \
    .replace("pbase_verified_package", "relax0_verified_package")
assert '"RELAX0_V1"' in ZIP_CELL and "plan_sha256" in ZIP_CELL, "ZIP 칸 바꾸기 실패"

WHAT = {"R": ("무작위 도착 세계", "A·B·C·D", 336, "4~6시간"), "P1": ("무리 도착 세계", "A·B", 168, "3~4시간"), "P2": ("무리 도착 세계", "C·D", 168, "6~9시간")}
PAIRS = {"R": "R/A,R/B,R/C,R/D", "P1": "P/A,P/B", "P2": "P/C,P/D"}

INTRO = """# RELAX0 {name} — 신호 규칙을 줄이면 얼마나 줄어드나 ({world}, 규칙 {variants}, 학습 없음)

학습 없는 계획기로, 같은 84상황에서 규칙만 바꿔 잽니다. 황색·전적색은 그대로입니다. 측정 방법은
`results/explore/rule_relax_20261101/측정계획.md`에 적어 두었습니다. 판정은 없습니다.

노트북은 3개(R·P1·P2)이고 **런타임을 하나씩 따로** 씁니다. 순서는 상관없습니다.

1. 런타임 유형 **CPU + 높은 RAM(CPU 8개)** 을 고르고 **런타임 → 모두 실행**을 누르세요. Drive 접근을 허용하세요.
2. 순서: 묶음 검증 → 패키지 → 시험 → 작은 실행(4판, 약 10분) → 본 실행({runs}판, CPU 8개 기준 약 {hours}, 어림) → 비교표.
3. 본 실행은 백그라운드로 돕니다. 끊기면 다시 **모두 실행**하세요. 끝난 판은 건너뜁니다.
4. 세 노트북이 끝나면 "끝났어"라고 알려 주세요. 결과: Drive `FutureTraffic/explore_relax0/`.
"""

SETUP = '''from google.colab import drive
from pathlib import Path
import os, sys, json

drive.mount("/content/drive")
NAME = "{name}"                                        # 이 노트북이 맡은 부분(R·P1·P2)
PAIRS = "{pairs}"
RUNS = {runs}
DRIVE = Path("/content/drive/MyDrive/FutureTraffic")
ARCHIVE = DRIVE / "FutureTraffic_RELAX0.zip"
PLATOON0 = DRIVE / "explore_platoon0"                  # 현재 규칙의 M4·TM4(읽기만)
PBASE = DRIVE / "explore_platoon_base"                 # 고정신호·P0(읽기만)
ARENA3 = DRIVE / "arena3" / "referee" / "baseline"     # 무작위 세계 P0(읽기만)
ROOT_OUT = DRIVE / "explore_relax0"
OUT = ROOT_OUT / NAME
WORKSPACE = Path("/content/FutureTraffic_RELAX0")
MIN_CPUS = 8
for path in (ARCHIVE, PLATOON0 / "runs" / "P" / "TM4", PLATOON0 / "runs" / "R" / "M4", PBASE / "runs" / "P" / "FIXC", PBASE / "runs" / "R" / "FIXC",
             ARENA3 / "runs" / "P0"):
    assert path.exists(), f"필요한 파일이 없습니다: {{path}}"
CPUS = os.cpu_count() or 1
print("CPU 개수:", CPUS, "| 결과 폴더:", OUT, "| 맡은 부분:", PAIRS)
assert CPUS >= MIN_CPUS, (f"CPU가 {{CPUS}}개입니다. 런타임 → 런타임 유형 변경에서 CPU와 '높은 RAM'을 고른 뒤 다시 실행하세요.")'''

TESTS_CELL = '''sys.path.insert(0, str(WORKSPACE / "src"))
import arena3_colab as colab
from IPython.display import display, Markdown

for name in ("test_explore_relax0.py", "test_explore_platoon0.py"):
    colab.run_live(["-m", "unittest", "discover", "-s", "tests", "-p", name], WORKSPACE)
print("실행 전 검사(SUMO 없이) 통과")'''

SMOKE_CELL = '''# 1) 작은 실행(약 10분): 첫 상황 4판(두 세계의 현재 규칙·A).
#    - 규칙을 안 바꾼 판이 기존 결과와 신호까지 같은가, A에서 좌회전 녹색이 실제로 짧아졌는가.
SMOKE = OUT / "smoke"
previous = colab.read_json(SMOKE / "smoke.json")
if previous and previous.get("ok"):
    print("이전 작은 실행이 통과해 건너뜁니다.")
else:
    colab.run_live(["src/explore_relax0.py", "smoke", "--out", str(SMOKE), "--work", "/content/rx_smoke",
                    "--workers", "4", "--platoon0", str(PLATOON0)], WORKSPACE, check=False)
s = colab.read_json(SMOKE / "smoke.json")
assert s and s.get("ok"), f"작은 실행 점검 실패 — 이 칸의 출력을 저에게 보내 주세요: {s and s.get('checks')}"
print("작은 실행 통과:", s["checks"])'''

RUN_CELL = '''# 2) 본 실행. 백그라운드로 돌아 셀을 멈춰도 계속 돕니다. 런타임이 끊기면 처음 네 칸 → 이 칸을 다시 실행하세요. 끝난 판은 건너뜁니다.
assert "colab" in globals(), "처음 네 칸(1~4)을 먼저 실행하세요. 런타임이 끊기면 변수와 작업 폴더가 사라집니다."
LOGS = OUT / "logs"

def describe(name):
    done = len(list((OUT / "runs").glob("*/*/*/cache.json")))
    failed = len(list((OUT / "runs").glob("*/*/*.failed.json")))
    return f"끝난 판 {done}/{RUNS}, 실패 기록 {failed}"

pids = colab.launch({f"relax0_{NAME}": ["src/explore_relax0.py", "run", "--out", str(OUT), "--work", "/content/rx",
                                        "--workers", str(CPUS), "--pairs", PAIRS]}, WORKSPACE, LOGS)
code = colab.watch(pids, LOGS, 300, describe)
assert code == 0, "본 실행에서 오류가 났습니다. 위 로그를 저에게 보내 주세요(다시 실행하면 끝난 판은 건너뜁니다)."'''

REPORT_CELL = '''# 3) 비교표(판정 없음). 끝난 규칙만 나옵니다. 숫자는 맥에서 다시 계산합니다.
colab.run_live(["src/explore_relax0.py", "report", "--out", str(ROOT_OUT), "--platoon0", str(PLATOON0),
                "--pbase", str(PBASE), "--arena3", str(ARENA3)], WORKSPACE, check=False)
if (ROOT_OUT / "summary.md").exists():
    display(Markdown((ROOT_OUT / "summary.md").read_text()))'''


def _cell(kind: str, text: str) -> dict:
    out = {"cell_type": kind, "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}
    if kind == "code":
        out.update(execution_count=None, outputs=[])
    return out


def notebook(name: str) -> dict:
    world, variants, runs, hours = WHAT[name]
    cells = [_cell("markdown", INTRO.format(name=name, world=world, variants=variants, runs=runs, hours=hours))]
    setup = SETUP.format(name=name, pairs=PAIRS[name], runs=runs)
    cells += [_cell("code", c) for c in (setup, ZIP_CELL, PACKAGES_CELL, TESTS_CELL, SMOKE_CELL, RUN_CELL, REPORT_CELL)]
    return {"cells": cells, "metadata": {"colab": {"provenance": [], "machine_shape": "hm"},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


def write_notebooks() -> dict[str, str]:
    out = {}
    for name, path in NOTEBOOKS.items():
        nb = notebook(name)
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] == "code":
                compile("".join(c["source"]), f"{path.name}[{i}]", "exec")
        text = json.dumps(nb, ensure_ascii=False, indent=1) + "\n"
        path.write_text(text)
        out[path.name] = hashlib.sha256(text.encode()).hexdigest()
    return out


def validate_relax0_archive(archive: Path) -> dict:
    manifest = validate_archive(archive)
    if manifest.get("experiment") != EXPERIMENT or manifest.get("source_models_included") is not False:
        raise ValueError("RELAX0 묶음이 아니거나 모델·결과가 들어 있습니다.")
    for name in REQUIRED_FILES + EXTRA_FILES:
        if name not in manifest["files"]:
            raise ValueError(f"RELAX0 묶음에 필요한 파일이 없습니다: {name}")
    return manifest


def build_package(out: Path = OUT) -> dict:
    import focused_package as original

    out = Path(out).resolve()
    if out.name in PROTECTED_NAMES:
        raise ValueError("이전 실험 묶음을 덮어쓸 수 없습니다.")
    original.check_before_packing(False)
    notebook_sha = write_notebooks()
    paths = set(original.files())
    paths.update(PROJECT / name for name in EXTRA_FILES)
    for name in (*REQUIRED_FILES, *EXTRA_FILES):
        if PROJECT / name not in paths or not (PROJECT / name).is_file():
            raise FileNotFoundError(f"RELAX0 묶음 입력이 없습니다: {name}")
    sha = lambda p: hashlib.sha256((PROJECT / p).read_bytes()).hexdigest()  # noqa: E731
    lock = json.loads((PROJECT / "config/arena3_v1.json").read_text())
    manifest = {"kind": KIND, "experiment": EXPERIMENT,
                "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                **original.git_state(), "files": {}, "source_models_included": False,
                "lock_sha256": sha("config/arena3_v1.json"), "platoon0_config_sha256": sha("config/explore_platoon0_v1.json"),
                "plan_sha256": sha(PLAN), "environment_fingerprint": lock["environment"]["fingerprint"], "notebook_sha256": notebook_sha}
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".relax0_package_", suffix=".zip", dir=out.parent)
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
        validate_relax0_archive(Path(temporary))
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
        manifest = validate_relax0_archive(args.verify)
        print(json.dumps({"verified": str(args.verify), "file_count": len(manifest["files"])}, ensure_ascii=False))
    else:
        print(json.dumps(build_package(args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
