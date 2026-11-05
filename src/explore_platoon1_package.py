"""PLATOON1 Colab 묶음(FutureTraffic_PLATOON1.zip)과 노트북 네 개(자료 조각 0·1·2, 학습·평가·판정)를 만든다.

이전 묶음·노트북·결과는 건드리지 않는다. 묶기만 하고 SUMO는 돌리지 않는다. 모델·결과는 넣지 않는다.
  .venv/bin/python src/explore_platoon1_package.py            # 노트북을 만들고 묶는다
  .venv/bin/python src/explore_platoon1_package.py --verify <zip>
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
from explore_platoon0_package import CELLS as P0_CELLS, PROTECTED_NAMES as P0_PROTECTED

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results/colab/FutureTraffic_PLATOON1.zip"
EXPERIMENT = "PLATOON1_V1"
SHARDS = 3
NOTEBOOKS = {"data0": PROJECT / "notebooks/PLATOON1_data.ipynb", "data1": PROJECT / "notebooks/PLATOON1_data_shard1.ipynb",
             "data2": PROJECT / "notebooks/PLATOON1_data_shard2.ipynb", "learn": PROJECT / "notebooks/PLATOON1_learn.ipynb"}
EXTRA_FILES = ("docs/PLATOON1_실행안내.md", "results/explore/platoon1_20261101/사전등록.md",
               "results/explore/platoon0_20261031/사전등록.md")
REQUIRED_FILES = ("src/explore_platoon1.py", "src/explore_platoon1_learn.py", "src/explore_platoon1_package.py",
                  "src/explore_platoon0.py", "src/explore_ceiling.py", "src/arena3_bprime.py", "src/arena3_stage2_eval.py",
                  "src/arena3_common.py", "src/arena3_referee.py", "src/arena3_colab.py", "src/explore_pred.py",
                  "src/explore_weight_core.py", "src/explore_weight_episode.py", "src/explore_rollout.py",
                  "src/explore_m4probe_pending.py", "src/focused_platoon.py", "config/arena3_v1.json",
                  "config/explore_platoon0_v1.json", "config/explore_platoon1_v1.json", "config/arena3_pi_r1.json",
                  "config/east_platoon.json", "data/derived/east_upstream_204822.csv", "tests/test_explore_platoon1.py",
                  "tests/test_explore_platoon0.py", "tests/test_arena3.py",
                  *(p.relative_to(PROJECT).as_posix() for p in NOTEBOOKS.values()))
PROTECTED_NAMES = {*P0_PROTECTED, "FutureTraffic_PLATOON0.zip"}

# 묶음 검증·패키지 칸은 검증을 마친 PLATOON0 노트북 칸에서 이름만 바꿔 쓴다.
ZIP_CELL = P0_CELLS[1].replace('"PLATOON0_V1", "PLATOON0 전용 ZIP이 아닙니다"', '"PLATOON1_V1", "PLATOON1 전용 ZIP이 아닙니다"') \
    .replace("platoon0_verified_package", "platoon1_verified_package") \
    .replace("f\"| PLATOON0 설정 {package_manifest['platoon0_config_sha256'][:12]} | 사전 등록 {package_manifest['prereg_sha256'][:12]}\")",
             "f\"| PLATOON1 설정 {package_manifest['platoon1_config_sha256'][:12]} | 사전 등록 {package_manifest['prereg_sha256'][:12]}\")")
PACKAGES_CELL = P0_CELLS[2]
assert '"PLATOON1_V1"' in ZIP_CELL and "platoon1_config_sha256" in ZIP_CELL, "ZIP 칸 바꾸기 실패"
assert "EXPECTED" in PACKAGES_CELL

SETUP = '''from google.colab import drive
from pathlib import Path
import os, sys, json

drive.mount("/content/drive")
DRIVE = Path("/content/drive/MyDrive/FutureTraffic")
ARCHIVE = DRIVE / "FutureTraffic_PLATOON1.zip"
PLATOON0 = DRIVE / "explore_platoon0"          # PLATOON0 결과(무리 도착 세계 M4·TM4·OM4·P1M4, 읽기만)
OUT = DRIVE / "explore_platoon1"
WORKSPACE = Path("/content/FutureTraffic_PLATOON1")
MIN_CPUS = 8
for path in (ARCHIVE, PLATOON0 / "experiment.json", PLATOON0 / "runs" / "P" / "TM4"):
    assert path.exists(), f"필요한 파일이 없습니다: {path}"
CPUS = os.cpu_count() or 1
'''

TESTS_CELL = '''sys.path.insert(0, str(WORKSPACE / "src"))
import arena3_colab as colab
from IPython.display import display, Markdown

for name in ("test_explore_platoon1.py", "test_explore_platoon0.py", "test_arena3.py"):
    colab.run_live(["-m", "unittest", "discover", "-s", "tests", "-p", name], WORKSPACE)
print("실행 전 검사(SUMO 없이) 통과")'''

DATA_INTRO = """# 1단계 PLATOON1 — 무리 도착 세계에서 정책 반복 1라운드: 자료 모으기, 조각 {k} / 3

PLATOON0에서 동쪽 차가 이웃 신호(204822)에 맞춰 무리로 오는 세계에서는 이웃 시간표를 쓰는 계획기(TM4)가 M4보다 3.82초
짧았습니다. 이번에는 그 정보를 보는 **강화학습(정책 반복)** 학생이 TM4보다 더 짧아지는지 봅니다. 기준은 실행 전에 고정했습니다
(`results/explore/platoon1_20261101/사전등록.md`).
- TM4가 학습 날짜의 무리 도착 상황 64판을 운전합니다. 고를 수 있는 순간의 25%에서 상태를 저장하고, **허용된 녹색을 모두**
  무리 규칙으로 만든 표본 미래 8개로 300초씩 굴려 봅니다(뒤는 TM4가 이어 운전).
- 다음 노트북 `PLATOON1_learn`에서 신경망이 '어느 녹색이 더 나은가'를 배우고 같은 심판으로 잽니다.

**이 파일 하나 = 런타임 하나입니다.** 조각 0 = `PLATOON1_data.ipynb`, 조각 1 = `PLATOON1_data_shard1.ipynb`,
조각 2 = `PLATOON1_data_shard2.ipynb`. 세 파일을 하나씩 열어(런타임 유형 **CPU + 높은 RAM**) 각각 **런타임 → 모두 실행**을 누르세요.
첫 칸의 `SHARD`는 이미 {k}로 정해져 있으니 바꾸지 마세요. 같은 조각을 두 런타임에서 동시에 돌리지 마세요.

순서: 묶음 검증 → 패키지 → 시험 → 실행 전 점검(진짜 SUMO, 약 20~30분) → 자료 모으기(백그라운드, 약 11~16시간) → 진행 확인.
끊기면 다시 **모두 실행**하면 끝난 판은 건너뜁니다. 마지막 칸에 **64/64**가 나오면 `PLATOON1_learn`을 실행하세요.
"""

SMOKE_CELL = '''# 1) 실행 전 점검(약 20~30분): 무리 도착 학습 상황 하나에서
#    - TM4를 그냥 돌린 것과, 상태를 저장·롤아웃하며 돌린 것이 소수점까지 같은가,
#    - 학습 전 학생(신경망 출력 0)이 TM4와 같은가, 평가 상황에서 PLATOON0의 TM4와 같은가,
#    - 롤아웃 복원·TM4 기록 넘기기가 맞는가, 동쪽 표본 미래가 204822 녹색에 놓이는가, 판당 시간이 얼마인가.
SMOKE = OUT / "smoke" / f"shard{SHARD}"
previous = colab.read_json(SMOKE / "smoke.json")
if previous and previous.get("passed"):
    print("이전 점검이 통과해 건너뜁니다. 판당 약", round(previous["estimate_cpu_hours_per_episode"], 2), "CPU-시간")
else:
    colab.run_live(["src/explore_platoon1.py", "smoke", "--out", str(SMOKE), "--work", "/content/pt1_smoke",
                    "--workers", str(CPUS), "--platoon0", str(PLATOON0)], WORKSPACE, check=False)
s = colab.read_json(SMOKE / "smoke.json")
assert s and s.get("passed"), f"실행 전 점검 실패 — 이 칸의 출력을 저에게 보내 주세요: {s and s.get('checks')}"
print("점검 통과:", s["checks"])
print("판당 약", round(s["estimate_cpu_hours_per_episode"], 2), "CPU-시간, 64판 합 약", round(s["estimate_cpu_hours_all"]), "CPU-시간")'''

COLLECT_CELL = '''# 2) 자료 모으기: 이 런타임 몫(64판 중 SHARD번째부터 3칸씩). 백그라운드로 돌아 셀을 멈춰도 계속 돕니다.
#    런타임이 끊기면 처음 네 칸 → 이 칸을 다시 실행하세요. 끝난 판은 건너뜁니다.
assert "colab" in globals(), "처음 네 칸(1~4)을 먼저 실행하세요. 런타임이 끊기면 변수와 작업 폴더가 사라집니다."
DATA = OUT / "data"
LOGS = OUT / "logs"
NAME = f"collect_shard{SHARD}"


def describe(name):
    s = colab.read_json(DATA / "status" / f"{name}.json")
    if not s:
        return None
    return (f"끝난 판 {s['done']}/{s['total']}, 롤아웃 중 {len(s['inflight'])}판, 실패 {len(s['failed'])}, "
            f"복원 불일치 {s['restore_mismatched']}/{s['restore_checked']}")


pids = colab.launch({NAME: ["src/explore_platoon1.py", "collect", "--out", str(DATA), "--work", "/content/pt1",
                            "--shard", str(SHARD), "--shards", str(SHARDS), "--workers", str(CPUS)]}, WORKSPACE, LOGS)
code = colab.watch(pids, LOGS, 600, describe)
assert code == 0, "자료 모으기에서 오류가 났습니다. 위 로그를 저에게 보내 주세요(다시 실행하면 끝난 판은 건너뜁니다)."'''

PROGRESS_CELL = '''# 3) 모든 런타임의 진행(언제든 다시 실행해 볼 수 있습니다).
for f in sorted((OUT / "data" / "status").glob("collect_shard*.json")):
    s = colab.read_json(f) or {}
    print(f.stem, f"{s.get('done')}/{s.get('total')}", "| 실패", len(s.get("failed", [])),
          "| 복원 불일치", f"{s.get('restore_mismatched')}/{s.get('restore_checked')}")
done = len(list((OUT / "data" / "runs" / "PT1_DATA").glob("*/cache.json")))
print(f"끝난 판 {done}/64")
if done == 64:
    print("64판이 모두 끝났습니다. PLATOON1_learn 노트북을 실행하세요.")'''

LEARN_INTRO = """# 1단계 PLATOON1 — 학생 학습·무리 도착 세계 심판 평가·판정

`PLATOON1_data` 세 조각의 64판이 모두 끝난 뒤 **런타임 하나**(CPU + 높은 RAM)에서 **모두 실행**합니다.
1. 날짜 보류 진단(참고, 약 10분): 학습에 안 쓴 날짜에서 '더 나은 녹색'을 얼마나 맞히는지 봅니다. 판정·선택에는 쓰지 않습니다.
2. 학생 학습(약 5분): 같은 자료로 난수 42·43·44 세 신경망을 학습합니다. 입력은 관측 167칸(ARENA3 119칸 + 시간표 관측 48칸)입니다.
3. 심판 평가: 모델 3개 × 평가 84상황 = 252판(백그라운드, CPU 8개 약 3시간). 기준 M4·TM4·OM4·P1M4는 PLATOON0 결과를 씁니다.
4. 판정(사전 등록): PT1이 M4 대비 ARENA3 판정을 통과하고, **학습의 몫 L = PT1 − TM4**가 0.3초 이상·하한 > 0·양수 날짜 5개 이상이면 통과.
5. 확인: 4가 통과했을 때만 확인 84상황에서 M4·TM4·PT1을 한 번 더 잽니다(약 4~5시간). 통과 못 하면 SUMO를 돌리지 않고 끝납니다.
"""

LEARN_SETUP_TAIL = '''print("CPU 개수:", CPUS, "| 결과 폴더:", OUT)
if CPUS < MIN_CPUS:
    print(f"주의: CPU가 {CPUS}개입니다. 평가가 오래 걸립니다. 가능하면 CPU + 높은 RAM 런타임을 쓰세요.")
done = len(list((OUT / "data" / "runs" / "PT1_DATA").glob("*/cache.json")))
assert done == 64, f"자료가 64판 중 {done}판만 끝났습니다. PLATOON1_data 세 조각을 먼저 끝내세요."'''

OFFLINE_CELL = '''# 1) 날짜 보류 진단(참고, SUMO 없음).
OFFLINE = OUT / "offline"
if not (OFFLINE / "offline_report.md").exists():
    colab.run_live(["src/explore_platoon1.py", "offline", "--data", str(OUT / "data"), "--out", str(OFFLINE)], WORKSPACE)
display(Markdown((OFFLINE / "offline_report.md").read_text()))'''

TRAIN_CELL = '''# 2) 학생 학습(난수 42·43·44, SUMO 없음). 이미 있으면 같은 자료로 학습했는지만 확인하고 건너뜁니다.
assert "colab" in globals(), "처음 네 칸(1~4)을 먼저 실행하세요."
colab.run_live(["src/explore_platoon1.py", "train", "--data", str(OUT / "data"), "--out", str(OUT / "students")], WORKSPACE)'''

EVAL_CELL = '''# 3) 심판 평가: 모델 3개 × 84상황 = 252판(백그라운드). 끊겨도 다시 실행하면 끝난 판은 건너뜁니다.
assert "colab" in globals(), "처음 네 칸(1~4)을 먼저 실행하세요. 런타임이 끊기면 변수와 작업 폴더가 사라집니다."
missing = [s for s in (42, 43, 44) if not (OUT / "students" / f"seed_{s}" / "model.pt").exists()]
assert not missing, f"학생 모델이 없습니다(난수 {missing}). 2)를 먼저 실행하세요."
LOGS = OUT / "logs"


def describe(name):
    done = len(list((OUT / "eval" / "PT1" / "runs").glob("PT1_m*/*/cache.json")))
    failed = len(list((OUT / "eval" / "PT1" / "runs").glob("PT1_m*/*.failed.json")))
    return f"끝난 판 {done}/252, 실패 기록 {failed}"


pids = colab.launch({"evaluate_PT1": ["src/explore_platoon1.py", "evaluate", "--students", str(OUT / "students"),
                                      "--out", str(OUT / "eval" / "PT1"), "--work", "/content/pt1_eval",
                                      "--workers", str(CPUS)]}, WORKSPACE, LOGS)
code = colab.watch(pids, LOGS, 600, describe)
assert code == 0, "평가에서 오류가 났습니다. 위 로그를 저에게 보내 주세요(다시 실행하면 끝난 판은 건너뜁니다)."'''

JUDGE_CELL = '''# 4) 판정(SUMO 없음, 몇 분): 기준은 PLATOON0의 무리 도착 세계 실행.
assert "colab" in globals(), "처음 네 칸(1~4)을 먼저 실행하세요."
code = colab.run_live(["src/explore_platoon1.py", "judge", "--eval", str(OUT / "eval" / "PT1"), "--platoon0", str(PLATOON0),
                       "--out", str(OUT / "judge")], WORKSPACE, check=False)
assert code == 0, "아직 평가가 끝나지 않았습니다(위 '아직 끝나지 않은 실행'). 평가가 끝난 뒤 다시 실행하세요."
display(Markdown((OUT / "judge" / "summary.md").read_text()))'''

CONFIRM_CELL = '''# 5) 확인(확인 84상황, 무리 도착 세계): 4)의 주 판정이 통과했을 때만 M4·TM4·PT1을 돌립니다(약 4~5시간, 백그라운드).
assert "colab" in globals(), "처음 네 칸(1~4)을 먼저 실행하세요."
LOGS = OUT / "logs"
pids = colab.launch({"confirm_PT1": ["src/explore_platoon1.py", "confirm", "--judge", str(OUT / "judge"),
                                     "--students", str(OUT / "students"), "--out", str(OUT / "judge" / "confirm"),
                                     "--work", "/content/pt1_confirm", "--workers", str(CPUS)]}, WORKSPACE, LOGS)
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


def data_notebook(k: int) -> dict:
    first = SETUP + f'''SHARD = {k}          # 이 파일 = 조각 {k}(바꾸지 마세요)
SHARDS = {SHARDS}         # 모든 조각 파일에서 같은 값
assert 0 <= SHARD < SHARDS
print("CPU 개수:", CPUS, "| 조각:", SHARD, "/", SHARDS, "| 결과 폴더:", OUT / "data")
assert CPUS >= MIN_CPUS, (f"CPU가 {{CPUS}}개입니다. 런타임 → 런타임 유형 변경에서 CPU와 '높은 RAM'을 고른 뒤 다시 실행하세요. "
                          "(롤아웃을 CPU 수만큼 동시에 돌립니다. CPU 2개면 4배 오래 걸려 24시간 한도에 걸립니다.)")'''
    return _notebook([_cell("markdown", DATA_INTRO.format(k=k)), _cell("code", first), _cell("code", ZIP_CELL),
                      _cell("code", PACKAGES_CELL), _cell("code", TESTS_CELL), _cell("code", SMOKE_CELL),
                      _cell("code", COLLECT_CELL), _cell("code", PROGRESS_CELL)])


def learn_notebook() -> dict:
    return _notebook([_cell("markdown", LEARN_INTRO), _cell("code", SETUP + LEARN_SETUP_TAIL), _cell("code", ZIP_CELL),
                      _cell("code", PACKAGES_CELL), _cell("code", TESTS_CELL), _cell("code", OFFLINE_CELL),
                      _cell("code", TRAIN_CELL), _cell("code", EVAL_CELL), _cell("code", JUDGE_CELL),
                      _cell("code", CONFIRM_CELL)])


def write_notebooks() -> dict[str, str]:
    books = {"data0": data_notebook(0), "data1": data_notebook(1), "data2": data_notebook(2), "learn": learn_notebook()}
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


def validate_platoon1_archive(archive: Path) -> dict:
    manifest = validate_archive(archive)
    if manifest.get("experiment") != EXPERIMENT or manifest.get("source_models_included") is not False:
        raise ValueError("PLATOON1 묶음이 아니거나 모델·결과가 들어 있습니다.")
    for name in REQUIRED_FILES + EXTRA_FILES:
        if name not in manifest["files"]:
            raise ValueError(f"PLATOON1 묶음에 필요한 파일이 없습니다: {name}")
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
            raise FileNotFoundError(f"PLATOON1 묶음 입력이 없습니다: {name}")
    sha = lambda p: hashlib.sha256((PROJECT / p).read_bytes()).hexdigest()  # noqa: E731
    lock = json.loads((PROJECT / "config/arena3_v1.json").read_text())
    manifest = {"kind": KIND, "experiment": EXPERIMENT,
                "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                **original.git_state(), "files": {}, "source_models_included": False,
                "lock_sha256": sha("config/arena3_v1.json"), "platoon1_config_sha256": sha("config/explore_platoon1_v1.json"),
                "prereg_sha256": sha("results/explore/platoon1_20261101/사전등록.md"),
                "environment_fingerprint": lock["environment"]["fingerprint"], "notebook_sha256": notebook_shas}
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".platoon1_package_", suffix=".zip", dir=out.parent)
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
        validate_platoon1_archive(Path(temporary))
        os.replace(temporary, out)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"path": str(out), "bytes": out.stat().st_size, "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "file_count": len(manifest["files"]), "experiment": EXPERIMENT, "notebook_sha256": notebook_shas,
            "platoon1_config_sha256": manifest["platoon1_config_sha256"], "prereg_sha256": manifest["prereg_sha256"],
            "git_commit": manifest["git_commit"], "git_uncommitted_changes": manifest["git_uncommitted_changes"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--verify", type=Path, help="이미 만든 묶음만 검사")
    args = parser.parse_args()
    if args.verify:
        manifest = validate_platoon1_archive(args.verify)
        print(json.dumps({"verified": str(args.verify), "file_count": len(manifest["files"])}, ensure_ascii=False))
    else:
        print(json.dumps(build_package(args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
