"""PLATOON1C Colab 묶음(FutureTraffic_PLATOON1C.zip)과 노트북 네 개(문턱 t09·t20·t30 평가, 판정·확인)를 만든다.

이전 묶음·노트북·결과는 건드리지 않는다. 묶기만 하고 SUMO는 돌리지 않는다. 모델·결과는 넣지 않는다.
  .venv/bin/python src/explore_platoon1c_package.py            # 노트북을 만들고 묶는다
  .venv/bin/python src/explore_platoon1c_package.py --verify <zip>
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
from explore_platoon_base_package import PROTECTED_NAMES as PBASE_PROTECTED

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results/colab/FutureTraffic_PLATOON1C.zip"
EXPERIMENT = "PLATOON1C_V1"
CANDIDATES = {"t09": ("0.9", "약 20%"), "t20": ("2.0", "약 10%"), "t30": ("3.0", "약 5%")}
NOTEBOOKS = {**{c: PROJECT / f"notebooks/PLATOON1C_{c}.ipynb" for c in CANDIDATES}, "judge": PROJECT / "notebooks/PLATOON1C_judge.ipynb"}
EXTRA_FILES = ("results/explore/platoon1c_20261101/사전등록.md", "results/explore/platoon1_20261101/사전등록.md",
               "results/explore/platoon0_20261031/사전등록.md", "results/explore/platoon_base_20261101/측정계획.md")
REQUIRED_FILES = ("src/explore_platoon1c.py", "src/explore_platoon1c_package.py", "src/explore_platoon1.py",
                  "src/explore_platoon1_learn.py", "src/explore_platoon_base.py", "src/explore_platoon0.py", "src/explore_ceiling.py",
                  "src/arena3_bprime.py", "src/arena3_stage2_eval.py", "src/arena3_common.py", "src/arena3_referee.py",
                  "src/arena3_colab.py", "src/explore_pred.py", "src/explore_weight_core.py", "src/explore_weight_episode.py",
                  "src/explore_rollout.py", "src/explore_m4probe_pending.py", "src/focused_platoon.py", "config/arena3_v1.json",
                  "config/explore_platoon0_v1.json", "config/explore_platoon1_v1.json", "config/explore_platoon1c_v1.json",
                  "config/arena3_pi_r1.json", "config/east_platoon.json", "data/derived/east_upstream_204822.csv",
                  "tests/test_explore_platoon1c.py", "tests/test_explore_platoon1.py",
                  *(p.relative_to(PROJECT).as_posix() for p in NOTEBOOKS.values()))
PROTECTED_NAMES = {*PBASE_PROTECTED, "FutureTraffic_PBASE.zip"}

# 묶음 검증·패키지 칸은 검증을 마친 PLATOON0 노트북 칸에서 이름만 바꿔 쓴다.
ZIP_CELL = P0_CELLS[1].replace('"PLATOON0_V1", "PLATOON0 전용 ZIP이 아닙니다"', '"PLATOON1C_V1", "PLATOON1C 전용 ZIP이 아닙니다"') \
    .replace("platoon0_verified_package", "platoon1c_verified_package") \
    .replace("f\"| PLATOON0 설정 {package_manifest['platoon0_config_sha256'][:12]} | 사전 등록 {package_manifest['prereg_sha256'][:12]}\")",
             "f\"| PLATOON1C 설정 {package_manifest['platoon1c_config_sha256'][:12]} | 사전 등록 {package_manifest['prereg_sha256'][:12]}\")")
PACKAGES_CELL = P0_CELLS[2]
assert '"PLATOON1C_V1"' in ZIP_CELL and "platoon1c_config_sha256" in ZIP_CELL, "ZIP 칸 바꾸기 실패"
assert "EXPECTED" in PACKAGES_CELL

SETUP = '''from google.colab import drive
from pathlib import Path
import os, sys, json

drive.mount("/content/drive")
DRIVE = Path("/content/drive/MyDrive/FutureTraffic")
ARCHIVE = DRIVE / "FutureTraffic_PLATOON1C.zip"
PLATOON0 = DRIVE / "explore_platoon0"          # 무리 도착 세계 M4·TM4·OM4·P1M4(읽기만)
PLATOON1 = DRIVE / "explore_platoon1"          # 학생 3개와 PT1 평가(읽기만)
PBASE = DRIVE / "explore_platoon_base"         # 무리 도착 세계 고정신호·P0(읽기만)
STUDENTS = PLATOON1 / "students"
OUT = DRIVE / "explore_platoon1c"
WORKSPACE = Path("/content/FutureTraffic_PLATOON1C")
MIN_CPUS = 8
for path in (ARCHIVE, PLATOON0 / "experiment.json", PLATOON0 / "runs" / "P" / "TM4", PLATOON1 / "eval" / "PT1" / "experiment.json",
             PBASE / "runs" / "P" / "FIXC", *(STUDENTS / f"seed_{s}" / "model.pt" for s in (42, 43, 44))):
    assert path.exists(), f"필요한 파일이 없습니다: {path}"
CPUS = os.cpu_count() or 1
'''

TESTS_CELL = '''sys.path.insert(0, str(WORKSPACE / "src"))
import arena3_colab as colab
from IPython.display import display, Markdown

for name in ("test_explore_platoon1c.py", "test_explore_platoon1.py"):
    colab.run_live(["-m", "unittest", "discover", "-s", "tests", "-p", name], WORKSPACE)
print("실행 전 검사(SUMO 없이) 통과")'''

EVAL_INTRO = """# 1단계 PLATOON1C — '이득이 확실할 때만 예측 제어와 다르게 고르기': 문턱 {c}

**무엇을 하나:** PT1은 결정의 40%를 예측 제어 TM4와 다르게 골라 TM4보다 1.03초 길었습니다. 이번에는 **같은 학생 3개**를 그대로 쓰고,
예측 이득이 문턱 이상일 때만 다르게 고릅니다. 새 수집·재학습은 없고 SUMO 평가만 합니다. 기준은 실행 전에 고정했습니다
(`results/explore/platoon1c_20261101/사전등록.md`).

**이 파일 = 문턱 τ = {tau} (학습 자료에서 바꾸는 비율 {rate}).**

**이 파일 하나 = 런타임 하나입니다.** `PLATOON1C_t09`, `PLATOON1C_t20`, `PLATOON1C_t30` 세 파일을 하나씩 열어(런타임 유형 **CPU + 높은 RAM**)
각각 **런타임 → 모두 실행**을 누르세요. 동시에 돌리면 각 2~3시간, 하나씩 돌려도 됩니다.

순서: 묶음 검증 → 패키지 → 시험 → 작은 실행(3판, 약 5분) → 평가(252판, 백그라운드).
끊기면 다시 **모두 실행**하면 끝난 판은 건너뜁니다. 세 파일이 모두 끝나면 `PLATOON1C_judge`를 실행하세요.
"""

SMOKE_CELL = '''# 1) 작은 실행(약 5분): 첫 평가 상황 3판(난수 42 모델).
#    - 문턱을 아주 높이면 PLATOON0의 TM4와 같은가, 문턱을 0에 가깝게 두면 PLATOON1의 PT1과 같은가, 이 문턱은 PT1보다 덜 바꾸는가.
SMOKE = OUT / "smoke" / CANDIDATE
previous = colab.read_json(SMOKE / "smoke.json")
if previous and previous.get("ok"):
    print("이전 작은 실행이 통과해 건너뜁니다.")
else:
    colab.run_live(["src/explore_platoon1c.py", "smoke", "--candidate", CANDIDATE, "--students", str(STUDENTS), "--out", str(SMOKE),
                    "--work", "/content/pc_smoke", "--workers", "3", "--platoon0", str(PLATOON0), "--platoon1", str(PLATOON1)],
                   WORKSPACE, check=False)
s = colab.read_json(SMOKE / "smoke.json")
assert s and s.get("ok"), f"작은 실행 점검 실패 — 이 칸의 출력을 저에게 보내 주세요: {s and s.get('checks')}"
print("작은 실행 통과:", s["checks"])
print("이 상황에서 TM4와 다르게 고른 횟수:", s["deviations"])'''

EVAL_CELL = '''# 2) 평가: 모델 3개 × 84상황 = 252판(백그라운드). 셀을 멈춰도 계속 돕니다.
#    런타임이 끊기면 처음 네 칸 → 이 칸을 다시 실행하세요. 끝난 판은 건너뜁니다.
assert "colab" in globals(), "처음 네 칸(1~4)을 먼저 실행하세요. 런타임이 끊기면 변수와 작업 폴더가 사라집니다."
EVAL = OUT / "eval" / CANDIDATE
LOGS = OUT / "logs"


def describe(name):
    done = len(list((EVAL / "runs").glob(f"{CANDIDATE}_m*/*/cache.json")))
    failed = len(list((EVAL / "runs").glob(f"{CANDIDATE}_m*/*.failed.json")))
    return f"끝난 판 {done}/252, 실패 기록 {failed}"


pids = colab.launch({f"evaluate_{CANDIDATE}": ["src/explore_platoon1c.py", "evaluate", "--candidate", CANDIDATE, "--students", str(STUDENTS),
                                               "--out", str(EVAL), "--work", f"/content/pc_{CANDIDATE}", "--workers", str(CPUS)]},
                    WORKSPACE, LOGS)
code = colab.watch(pids, LOGS, 600, describe)
assert code == 0, "평가에서 오류가 났습니다. 위 로그를 저에게 보내 주세요(다시 실행하면 끝난 판은 건너뜁니다)."
print(describe(None), "— 세 파일이 모두 끝나면 PLATOON1C_judge를 실행하세요.")'''

PROGRESS_CELL = '''# 3) 세 문턱의 진행(언제든 다시 실행해 볼 수 있습니다).
for c in ("t09", "t20", "t30"):
    done = len(list((OUT / "eval" / c / "runs").glob(f"{c}_m*/*/cache.json")))
    print(c, f"{done}/252")'''

JUDGE_INTRO = """# 1단계 PLATOON1C — 판정과 확인

`PLATOON1C_t09`, `PLATOON1C_t20`, `PLATOON1C_t30`의 252판이 모두 끝난 뒤 **런타임 하나**(CPU + 높은 RAM)에서 **모두 실행**합니다.
1. 판정(몇 분, SUMO 없음): 문턱마다 **학습의 몫 L = 후보 − TM4**가 0.3초 이상·하한 > 0·양수 날짜 5개 이상이고 ARENA3 판정을 통과하는지 봅니다.
   고정신호·P0와 비교한 표도 함께 나옵니다.
2. 확인(약 4~5시간): 통과한 문턱 중 가장 좋은 하나만 아직 쓰지 않은 확인 84상황에서 잽니다. 통과가 없으면 SUMO를 돌리지 않고 끝납니다.
"""

JUDGE_TAIL = '''print("CPU 개수:", CPUS, "| 결과 폴더:", OUT)
if CPUS < MIN_CPUS:
    print(f"주의: CPU가 {CPUS}개입니다. 확인을 돌리게 되면 오래 걸립니다. 가능하면 CPU + 높은 RAM 런타임을 쓰세요.")
counts = {c: len(list((OUT / "eval" / c / "runs").glob(f"{c}_m*/*/cache.json"))) for c in ("t09", "t20", "t30")}
assert set(counts.values()) == {252}, f"아직 끝나지 않은 평가가 있습니다: {counts}"'''

JUDGE_CELL = '''# 1) 판정(SUMO 없음, 몇 분). 숫자는 맥에서 다시 계산합니다.
assert "colab" in globals(), "처음 네 칸(1~4)을 먼저 실행하세요."
code = colab.run_live(["src/explore_platoon1c.py", "judge", "--eval", str(OUT / "eval"), "--platoon0", str(PLATOON0),
                       "--platoon1", str(PLATOON1), "--pbase", str(PBASE), "--out", str(OUT / "judge")], WORKSPACE, check=False)
assert code == 0, "판정에서 오류가 났습니다. 위 출력을 저에게 보내 주세요."
display(Markdown((OUT / "judge" / "summary.md").read_text()))'''

CONFIRM_CELL = '''# 2) 확인(확인 84상황, 무리 도착 세계): 1)이 고른 문턱이 있을 때만 M4·TM4·후보를 돌립니다(약 4~5시간, 백그라운드).
assert "colab" in globals(), "처음 네 칸(1~4)을 먼저 실행하세요."
LOGS = OUT / "logs"
pids = colab.launch({"confirm_PT1C": ["src/explore_platoon1c.py", "confirm", "--judge", str(OUT / "judge"), "--students", str(STUDENTS),
                                      "--out", str(OUT / "judge" / "confirm"), "--work", "/content/pc_confirm", "--workers", str(CPUS)]},
                    WORKSPACE, LOGS)
code = colab.watch(pids, LOGS, 600)
result = colab.read_json(OUT / "judge" / "confirm" / "confirmation.json")
print("확인 결과:", result and result.get("status"))
if (OUT / "judge" / "confirm" / "summary.md").exists():
    display(Markdown((OUT / "judge" / "confirm" / "summary.md").read_text()))'''


def _cell(kind: str, text: str) -> dict:
    out = {"cell_type": kind, "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}
    if kind == "code":
        out.update(execution_count=None, outputs=[])
    return out


def _notebook(cells: list[dict]) -> dict:
    return {"cells": cells, "metadata": {"colab": {"provenance": [], "machine_shape": "hm"},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


def eval_notebook(candidate: str) -> dict:
    tau, rate = CANDIDATES[candidate]
    first = SETUP + f'''CANDIDATE = "{candidate}"      # 이 파일 = 문턱 τ = {tau}(바꾸지 마세요)
print("CPU 개수:", CPUS, "| 후보:", CANDIDATE, "| 결과 폴더:", OUT / "eval" / CANDIDATE)
assert CPUS >= MIN_CPUS, (f"CPU가 {{CPUS}}개입니다. 런타임 → 런타임 유형 변경에서 CPU와 '높은 RAM'을 고른 뒤 다시 실행하세요.")'''
    return _notebook([_cell("markdown", EVAL_INTRO.format(c=candidate, tau=tau, rate=rate)), _cell("code", first), _cell("code", ZIP_CELL),
                      _cell("code", PACKAGES_CELL), _cell("code", TESTS_CELL), _cell("code", SMOKE_CELL), _cell("code", EVAL_CELL),
                      _cell("code", PROGRESS_CELL)])


def judge_notebook() -> dict:
    return _notebook([_cell("markdown", JUDGE_INTRO), _cell("code", SETUP + JUDGE_TAIL), _cell("code", ZIP_CELL),
                      _cell("code", PACKAGES_CELL), _cell("code", TESTS_CELL), _cell("code", JUDGE_CELL), _cell("code", CONFIRM_CELL)])


def write_notebooks() -> dict[str, str]:
    books = {**{c: eval_notebook(c) for c in CANDIDATES}, "judge": judge_notebook()}
    shas = {}
    for key, nb in books.items():
        path = NOTEBOOKS[key]
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] == "code":
                compile("".join(c["source"]), f"{path.name}[{i}]", "exec")
        text = json.dumps(nb, ensure_ascii=False, indent=1) + "\n"
        path.write_text(text)
        shas[path.name] = hashlib.sha256(text.encode()).hexdigest()
    return shas


def validate_platoon1c_archive(archive: Path) -> dict:
    manifest = validate_archive(archive)
    if manifest.get("experiment") != EXPERIMENT or manifest.get("source_models_included") is not False:
        raise ValueError("PLATOON1C 묶음이 아니거나 모델·결과가 들어 있습니다.")
    for name in REQUIRED_FILES + EXTRA_FILES:
        if name not in manifest["files"]:
            raise ValueError(f"PLATOON1C 묶음에 필요한 파일이 없습니다: {name}")
    return manifest


def build_package(out: Path = OUT) -> dict:
    import focused_package as original

    out = Path(out).resolve()
    if out.name in PROTECTED_NAMES:
        raise ValueError("이전 실험 묶음을 덮어쓸 수 없습니다.")
    original.check_before_packing(False)
    notebook_shas = write_notebooks()
    paths = set(original.files())
    paths.update(PROJECT / name for name in EXTRA_FILES)
    for name in (*REQUIRED_FILES, *EXTRA_FILES):
        if PROJECT / name not in paths or not (PROJECT / name).is_file():
            raise FileNotFoundError(f"PLATOON1C 묶음 입력이 없습니다: {name}")
    sha = lambda p: hashlib.sha256((PROJECT / p).read_bytes()).hexdigest()  # noqa: E731
    lock = json.loads((PROJECT / "config/arena3_v1.json").read_text())
    manifest = {"kind": KIND, "experiment": EXPERIMENT,
                "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                **original.git_state(), "files": {}, "source_models_included": False,
                "lock_sha256": sha("config/arena3_v1.json"), "platoon1c_config_sha256": sha("config/explore_platoon1c_v1.json"),
                "prereg_sha256": sha("results/explore/platoon1c_20261101/사전등록.md"),
                "environment_fingerprint": lock["environment"]["fingerprint"], "notebook_sha256": notebook_shas}
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".platoon1c_package_", suffix=".zip", dir=out.parent)
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
        validate_platoon1c_archive(Path(temporary))
        os.replace(temporary, out)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"path": str(out), "bytes": out.stat().st_size, "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "file_count": len(manifest["files"]), "experiment": EXPERIMENT, "notebook_sha256": notebook_shas,
            "platoon1c_config_sha256": manifest["platoon1c_config_sha256"], "prereg_sha256": manifest["prereg_sha256"],
            "git_commit": manifest["git_commit"], "git_uncommitted_changes": manifest["git_uncommitted_changes"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--verify", type=Path, help="이미 만든 묶음만 검사")
    args = parser.parse_args()
    if args.verify:
        manifest = validate_platoon1c_archive(args.verify)
        print(json.dumps({"verified": str(args.verify), "file_count": len(manifest["files"])}, ensure_ascii=False))
    else:
        print(json.dumps(build_package(args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
