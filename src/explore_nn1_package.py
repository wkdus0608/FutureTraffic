"""NN1 Colab 묶음(FutureTraffic_NN1.zip)과 노트북 5개(세계별 기본·자료 강화 + 판정)를 만든다.

이전 묶음·노트북·결과는 건드리지 않는다. 묶기만 하고 SUMO는 돌리지 않는다. 실험 모델·결과는 넣지 않는다(점검용 맥 참조 모델 2개만 넣는다).
  .venv/bin/python src/explore_nn1_package.py            # 노트북을 만들고 묶는다
  .venv/bin/python src/explore_nn1_package.py --verify <zip>
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

from explore_label0_package import PACKAGES_CELL, PROTECTED_NAMES as L0_PROTECTED, ZIP_CELL as L0_ZIP_CELL
from explore_weight_package import KIND, _safe_name, validate_archive

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results/colab/FutureTraffic_NN1.zip"
PARTS = ("P_base", "P_aug", "R_base", "R_aug")
NOTEBOOKS = {**{v: PROJECT / f"notebooks/NN1_{v}.ipynb" for v in PARTS}, "judge": PROJECT / "notebooks/NN1_judge.ipynb"}
EXPERIMENT = "NN1_V1"
PREREG = "results/explore/nn1_20261102/사전등록.md"
REFERENCE = tuple(f"results/explore/nn1_20261102/smoke_reference/{x}/{name}" for x in "PR" for name in ("model.pt", "manifest.json", "free_run.json"))
EXTRA_FILES = (PREREG, *REFERENCE, "results/explore/label0_20261102/사전등록.md", "results/explore/platoon1c_20261101/사전등록.md", "results/explore/platoon1_20261101/사전등록.md",
               "results/explore/platoon0_20261031/사전등록.md", "results/explore/platoon_base_20261101/측정계획.md")
REQUIRED_FILES = ("src/explore_nn1.py", "src/explore_nn1_package.py", "src/explore_label0.py", "src/explore_platoon1c.py", "src/explore_platoon1.py", "src/explore_platoon1_learn.py",
                  "src/explore_platoon_base.py", "src/explore_platoon0.py", "src/explore_ceiling.py", "src/arena3_pi.py", "src/arena3_pi_learn.py", "src/arena3_g1.py",
                  "src/arena3_bprime.py", "src/arena3_stage2_eval.py", "src/arena3_stage2_train.py", "src/arena3_stage2_data.py", "src/arena3_common.py", "src/arena3_referee.py",
                  "src/arena3_colab.py", "src/explore_pred.py", "src/explore_weight_core.py", "src/explore_weight_episode.py", "src/explore_rollout.py",
                  "src/explore_m4probe_pending.py", "src/focused_platoon.py", "config/arena3_v1.json", "config/arena3_stage2_v1.json", "config/explore_platoon0_v1.json",
                  "config/explore_platoon1_v1.json", "config/explore_platoon1c_v1.json", "config/arena3_pi_r1.json", "config/explore_label0_v1.json", "config/explore_nn1_v1.json",
                  "config/east_platoon.json", "data/derived/east_upstream_204822.csv", "tests/test_explore_nn1.py",
                  *(p.relative_to(PROJECT).as_posix() for p in NOTEBOOKS.values()))
PROTECTED_NAMES = {*L0_PROTECTED, "FutureTraffic_LABEL0.zip"}

ZIP_CELL = L0_ZIP_CELL.replace('"LABEL0_V1", "LABEL0 전용 ZIP이 아닙니다"', '"NN1_V1", "NN1 전용 ZIP이 아닙니다"').replace("label0_verified_package", "nn1_verified_package")
assert '"NN1_V1"' in ZIP_CELL and "plan_sha256" in ZIP_CELL, "ZIP 칸 바꾸기 실패"

WORLD = {"P": {"name": "무리 도착 세계", "data": "explore_platoon1/data", "teacher": "explore_platoon1/students/seed_42", "evaluation": "explore_platoon0",
               "base": "N1,N2,N4", "aug": "N5", "reference_model": "N1 난수 42", "teacher_name": "t20(TM4 + PT1 학생 + 문턱 2.0)"},
         "R": {"name": "무작위 도착 세계", "data": "arena3/pi/R1/data", "teacher": "arena3/stage2/S2A_train/seed_42/pretrained", "evaluation": "arena3/referee/baseline",
               "base": "Rn,RnS", "aug": "Rn2", "reference_model": "Rn 난수 42", "teacher_name": "S2A_pre 난수 42"}}
HOURS = {"P_base": ("756판", "약 2.5시간"), "R_base": ("504판", "약 1.5시간"), "P_aug": ("기록 192판 + 검증 288판 + 평가 252판", "약 4시간"), "R_aug": ("기록 192판 + 검증 288판 + 평가 252판", "약 3시간")}

BASE_INTRO = """# NN1 {part} — 계획기 없는 신경망 모델 재기 ({world})

계획서 1단계입니다. 신경망 **{cands}** 를 학습 6일로 만들어 같은 84상황에서 잽니다. 실행 때 계획기를 부르면 멈추고, 제어 60분과 꼬리 15분을 모두 신경망이 맡습니다.
방법과 읽는 법은 `results/explore/nn1_20261102/사전등록.md`에 고정했습니다. 이 노트북의 값은 흉내와 첫 학생까지이고 강화학습 라운드는 아직 아닙니다.

노트북은 넷(P_base·P_aug·R_base·R_aug)과 판정 하나입니다. **넷을 런타임 하나씩 따로** 씁니다. 순서는 상관없고 동시에 돌려도 됩니다.

1. 런타임 유형 **CPU + 높은 RAM(CPU 8개)** 을 고르고 **런타임 → 모두 실행**을 누르세요. GPU는 필요 없습니다. Drive 접근을 허용하세요.
2. 순서: 묶음 검증 → 패키지 → 시험 → 모델 학습(약 2분) → 경로 점검(3판, 약 5분) → 본 평가({runs}, CPU 8개 기준 {hours}, 어림).
3. 본 평가는 백그라운드로 돕니다. 끊기면 다시 **모두 실행**하세요. 끝난 판은 건너뜁니다.
4. 넷이 끝나면 `NN1_judge`를 실행하고 "끝났어"라고 알려 주세요. 결과: Drive `FutureTraffic/explore_nn1/`.
"""

AUG_INTRO = """# NN1 {part} — 자료를 늘려 흉내 다시 배우기 ({world})

계획서 1단계의 자료 강화 후보 **{cands}** 입니다. 선생님({teacher})이 운전한 판의 모든 결정을 기록하고, 이어 신경망이 운전하며 선생님이 라벨만 붙이는 라운드를 3번 돕니다.
어느 라운드를 쓸지는 검증 날짜의 한 시간 결과로 정하고, 고른 모델을 84상황에서 잽니다. 방법과 읽는 법은 `results/explore/nn1_20261102/사전등록.md`에 고정했습니다.

노트북은 넷(P_base·P_aug·R_base·R_aug)과 판정 하나입니다. **넷을 런타임 하나씩 따로** 씁니다. 순서는 상관없고 동시에 돌려도 됩니다.

1. 런타임 유형 **CPU + 높은 RAM(CPU 8개)** 을 고르고 **런타임 → 모두 실행**을 누르세요. GPU는 필요 없습니다. Drive 접근을 허용하세요.
2. 순서: 묶음 검증 → 패키지 → 시험 → 기록 점검(3판, 약 10분) → 본 실행({runs}, CPU 8개 기준 {hours}, 어림).
3. 본 실행은 백그라운드로 돕니다. 끊기면 다시 **모두 실행**하세요. 끝난 판과 학습은 건너뜁니다.
4. 넷이 끝나면 `NN1_judge`를 실행하고 "끝났어"라고 알려 주세요. 결과: Drive `FutureTraffic/explore_nn1/`.
"""

SETUP = '''from google.colab import drive
from pathlib import Path
import os, sys, json

drive.mount("/content/drive")
WORLD, PART = "{world}", "{kind}"
DRIVE = Path("/content/drive/MyDrive/FutureTraffic")
ARCHIVE = DRIVE / "FutureTraffic_NN1.zip"
DATA = DRIVE / "{data}"                                # 기존 자료(읽기만)
TEACHER = DRIVE / "{teacher}"                          # 선생님 모델(읽기만)
EVALUATION = DRIVE / "{evaluation}"                    # 평가 지문 대조(읽기만)
PLATOON1C = next((p for p in (DRIVE / "explore_platoon1c", DRIVE / "explore_platoon1c (1)") if (p / "eval" / "t20" / "models.json").exists()), None)
TEACHER_RUN = {teacher_run}
OUT = DRIVE / "explore_nn1" / WORLD / PART
WORKSPACE = Path("/content/FutureTraffic_NN1")
REFERENCE = WORKSPACE / "results/explore/nn1_20261102/smoke_reference" / WORLD      # 맥에서 만든 점검용 모델
MIN_CPUS = 8
for path in (ARCHIVE, DATA / "experiment.json", TEACHER / "model.pt", EVALUATION / "experiment.json", TEACHER_RUN):
    assert path.exists(), f"필요한 파일이 없습니다: {{path}}"
CPUS = os.cpu_count() or 1
print("CPU 개수:", CPUS, "| 세계:", WORLD, "| 결과 폴더:", OUT)
assert CPUS >= MIN_CPUS, (f"CPU가 {{CPUS}}개입니다. 런타임 → 런타임 유형 변경에서 CPU와 '높은 RAM'을 고른 뒤 다시 실행하세요.")'''

JUDGE_INTRO = """# NN1 판정 — 출발 모델 7종 비교

노트북 넷(P_base·P_aug·R_base·R_aug)이 끝난 뒤 **모두 실행**하세요. SUMO는 돌리지 않으며 CPU 수는 상관없습니다. 몇 분 걸립니다.
읽는 법은 `results/explore/nn1_20261102/사전등록.md`에 고정했습니다. 숫자는 맥에서 다시 계산합니다.
"""

JUDGE_SETUP = '''from google.colab import drive
from pathlib import Path
import os, sys, json

drive.mount("/content/drive")
DRIVE = Path("/content/drive/MyDrive/FutureTraffic")
ARCHIVE = DRIVE / "FutureTraffic_NN1.zip"
ROOT_OUT = DRIVE / "explore_nn1"
A3 = DRIVE / "arena3"
PLATOON1C = next((p for p in (DRIVE / "explore_platoon1c", DRIVE / "explore_platoon1c (1)") if (p / "eval" / "t20" / "models.json").exists()), None)
WORKSPACE = Path("/content/FutureTraffic_NN1")
assert ARCHIVE.exists() and PLATOON1C is not None, "필요한 파일이 없습니다."'''

TESTS_CELL = '''sys.path.insert(0, str(WORKSPACE / "src"))
import arena3_colab as colab
from IPython.display import display, Markdown

colab.run_live(["-m", "unittest", "discover", "-s", "tests", "-p", "test_explore_nn1.py"], WORKSPACE)
print("실행 전 검사(SUMO 없이) 통과")'''

TRAIN_CELL = '''# 1) 모델 학습(약 2분). 이미 있으면 다시 만들지 않습니다.
args = ["src/explore_nn1.py", "train", "--world", WORLD, "--data", str(DATA), "--out", str(OUT / "models")]
if WORLD == "P":
    args += ["--teacher", str(TEACHER)]
colab.run_live(args, WORKSPACE)
print("모델 학습 끝")'''

PATHS_CELL = '''# 2) 경로 점검(약 5분): 맥 참조 모델을 기존 경로와 계획기 없는 경로로 돌려 신호·입력이 같은지, 계획기 호출이 0인지, 맥과 값이 같은지 봅니다.
CHECK = OUT / "check_paths"
previous = colab.read_json(CHECK / "check_paths.json")
if previous and previous.get("ok"):
    print("이전 점검이 통과해 건너뜁니다.")
else:
    colab.run_live(["src/explore_nn1.py", "check-paths", "--world", WORLD, "--model", str(REFERENCE), "--out", str(CHECK), "--work", "/content/n1_paths",
                    "--workers", "3", "--situations", "1", "--evaluation", str(EVALUATION)], WORKSPACE, check=False)
r = colab.read_json(CHECK / "check_paths.json")
assert r and r.get("ok"), f"점검 실패 — 이 칸의 출력을 저에게 보내 주세요: {r and r.get('checks')}"
print("경로 점검 통과:", r["checks"], "| 판당 초", r["seconds"], "| 꼬리 신경망 − 꼬리 P0:", r["tail_nn_minus_p0_s"])'''

BASE_RUN_CELL = '''# 3) 본 평가. 백그라운드로 돌아 셀을 멈춰도 계속 돕니다.
#    런타임이 끊기면 처음 네 칸 → 이 칸을 다시 실행하세요. 끝난 판은 건너뜁니다.
assert "colab" in globals(), "처음 네 칸(1~4)을 먼저 실행하세요. 런타임이 끊기면 변수와 작업 폴더가 사라집니다."
CANDIDATES = "{cands}"
TOTAL = len(CANDIDATES.split(",")) * 3 * 84
LOGS = OUT / "logs"

def describe(name):
    done = len(list((OUT / "eval").glob("*/runs/*/*/cache.json")))
    failed = len(list((OUT / "eval").glob("*/runs/*/*.failed.json")))
    return f"끝난 판 {{done}}/{{TOTAL}}, 실패 기록 {{failed}}"

pids = colab.launch({{f"nn1_{{WORLD}}_base": ["src/explore_nn1.py", "evaluate", "--world", WORLD, "--candidates", CANDIDATES, "--models", str(OUT / "models"),
                                            "--out", str(OUT / "eval"), "--work", "/content/n1", "--workers", str(CPUS)]}}, WORKSPACE, LOGS)
code = colab.watch(pids, LOGS, 300, describe)
assert code == 0, "본 평가에서 오류가 났습니다. 위 로그를 저에게 보내 주세요(다시 실행하면 끝난 판은 건너뜁니다)."
print("본 평가 끝. 노트북 넷이 끝나면 NN1_judge를 실행하세요.")'''

LOGS_CELL = '''# 1) 기록 점검(약 10분): 선생님 운전 기록이 저장된 선생님 실행과 같은지, 신경망 운전 기록의 신호가 계획기 없는 실행과 같은지 봅니다.
CHECK = OUT / "check_logs"
previous = colab.read_json(CHECK / "check_logs.json")
if previous and previous.get("ok"):
    print("이전 점검이 통과해 건너뜁니다.")
else:
    colab.run_live(["src/explore_nn1.py", "check-logs", "--world", WORLD, "--teacher", str(TEACHER), "--teacher-run", str(TEACHER_RUN), "--model", str(REFERENCE),
                    "--out", str(CHECK), "--work", "/content/n1_logs", "--workers", "3", "--evaluation", str(EVALUATION)], WORKSPACE, check=False)
r = colab.read_json(CHECK / "check_logs.json")
assert r and r.get("ok"), f"점검 실패 — 이 칸의 출력을 저에게 보내 주세요: {r and r.get('checks')}"
print("기록 점검 통과:", r["checks"], "| 판당 초", r["seconds"], "| 결정 수", r["decisions"])'''

AUG_RUN_CELL = '''# 2) 본 실행: 라운드 0~3 기록·학습 → 검증으로 라운드 고르기 → 84상황 평가. 백그라운드로 돌아 셀을 멈춰도 계속 돕니다.
#    런타임이 끊기면 처음 네 칸 → 이 칸을 다시 실행하세요. 끝난 판과 학습은 건너뜁니다.
assert "colab" in globals(), "처음 네 칸(1~4)을 먼저 실행하세요. 런타임이 끊기면 변수와 작업 폴더가 사라집니다."

def describe(name):
    logged = len(list((OUT / "runs").glob("*/*/cache.json")))
    rounds = len(list((OUT / "models").glob("r*/training.json")))
    valid = len(list((OUT / "valid" / "runs").glob("*/*/cache.json")))
    evaluated = len(list((OUT / "eval").glob("*/runs/*/*/cache.json")))
    failed = len(list((OUT / "runs").glob("*/*.failed.json"))) + len(list((OUT / "valid" / "runs").glob("*/*.failed.json"))) + len(list((OUT / "eval").glob("*/runs/*/*.failed.json")))
    return f"기록 {{logged}}/192, 학습한 라운드 {{rounds}}/4, 검증 {{valid}}/288, 평가 {{evaluated}}/252, 실패 기록 {{failed}}"

pids = colab.launch({{f"nn1_{{WORLD}}_aug": ["src/explore_nn1.py", "augment", "--world", WORLD, "--teacher", str(TEACHER), "--out", str(OUT), "--work", "/content/n1a",
                                           "--workers", str(CPUS)]}}, WORKSPACE, OUT / "logs_console")
code = colab.watch(pids, OUT / "logs_console", 300, describe)
assert code == 0, "본 실행에서 오류가 났습니다. 위 로그를 저에게 보내 주세요(다시 실행하면 끝난 판은 건너뜁니다)."
print("본 실행 끝. 고른 라운드:", colab.read_json(OUT / "selected.json")["selected_round"], "| 노트북 넷이 끝나면 NN1_judge를 실행하세요.")'''

JUDGE_CELL = '''# 판정(SUMO 없음). 아직 안 끝난 후보가 있으면 표 아래에 나옵니다.
code = colab.run_live(["src/explore_nn1.py", "judge", "--root", str(ROOT_OUT), "--platoon0", str(DRIVE / "explore_platoon0"), "--platoon1", str(DRIVE / "explore_platoon1"),
                       "--platoon1c", str(PLATOON1C), "--pbase", str(DRIVE / "explore_platoon_base"), "--baseline", str(A3 / "referee" / "baseline"),
                       "--stage2", str(A3 / "stage2"), "--out", str(ROOT_OUT / "judge")], WORKSPACE, check=False)
if (ROOT_OUT / "judge" / "summary.md").exists():
    display(Markdown((ROOT_OUT / "judge" / "summary.md").read_text()))
print("판정 코드:", code, "(0이면 후보 7개가 모두 끝났습니다)")'''


def _cell(kind: str, text: str) -> dict:
    out = {"cell_type": kind, "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}
    if kind == "code":
        out.update(execution_count=None, outputs=[])
    return out


def _wrap(cells: list[dict]) -> dict:
    return {"cells": cells, "metadata": {"colab": {"provenance": [], "machine_shape": "hm"}, "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


def notebook(name: str) -> dict:
    if name == "judge":
        return _wrap([_cell("markdown", JUDGE_INTRO)] + [_cell("code", c) for c in (JUDGE_SETUP, ZIP_CELL, PACKAGES_CELL, TESTS_CELL, JUDGE_CELL)])
    world, kind = name.split("_")
    spec = WORLD[world]
    stored = 'PLATOON1C / "eval" / "t20" / "runs" / "t20_m0"' if world == "P" else 'DRIVE / "arena3" / "stage2" / "eval" / "S2A_pre" / "runs" / "S2A_pre_m0"'
    setup = SETUP.format(world=world, kind=kind, data=spec["data"], teacher=spec["teacher"], evaluation=spec["evaluation"], teacher_run=stored)
    runs, hours = HOURS[name]
    if kind == "base":
        intro = BASE_INTRO.format(part=name, world=spec["name"], cands=spec["base"].replace(",", "·"), runs=runs, hours=hours)
        steps = (setup, ZIP_CELL, PACKAGES_CELL, TESTS_CELL, TRAIN_CELL, PATHS_CELL, BASE_RUN_CELL.format(cands=spec["base"]))
    else:
        intro = AUG_INTRO.format(part=name, world=spec["name"], cands=spec["aug"], teacher=spec["teacher_name"], runs=runs, hours=hours)
        steps = (setup, ZIP_CELL, PACKAGES_CELL, TESTS_CELL, LOGS_CELL, AUG_RUN_CELL.format())
    return _wrap([_cell("markdown", intro)] + [_cell("code", c) for c in steps])


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


def validate_nn1_archive(archive: Path) -> dict:
    manifest = validate_archive(archive)
    if manifest.get("experiment") != EXPERIMENT or manifest.get("source_models_included") is not False:
        raise ValueError("NN1 묶음이 아니거나 실험 모델·결과가 들어 있습니다.")
    for name in REQUIRED_FILES + EXTRA_FILES:
        if name not in manifest["files"]:
            raise ValueError(f"NN1 묶음에 필요한 파일이 없습니다: {name}")
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
            raise FileNotFoundError(f"NN1 묶음 입력이 없습니다: {name}")
    sha = lambda p: hashlib.sha256((PROJECT / p).read_bytes()).hexdigest()  # noqa: E731
    lock = json.loads((PROJECT / "config/arena3_v1.json").read_text())
    manifest = {"kind": KIND, "experiment": EXPERIMENT, "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **original.git_state(), "files": {},
                "source_models_included": False, "smoke_reference_models": 2, "lock_sha256": sha("config/arena3_v1.json"), "platoon0_config_sha256": sha("config/explore_platoon0_v1.json"),
                "plan_sha256": sha(PREREG), "nn1_config_sha256": sha("config/explore_nn1_v1.json"), "environment_fingerprint": lock["environment"]["fingerprint"], "notebook_sha256": notebook_sha}
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".nn1_package_", suffix=".zip", dir=out.parent)
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
        validate_nn1_archive(Path(temporary))
        os.replace(temporary, out)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"path": str(out), "bytes": out.stat().st_size, "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "file_count": len(manifest["files"]), "experiment": EXPERIMENT,
            "notebook_sha256": notebook_sha, "plan_sha256": manifest["plan_sha256"], "git_commit": manifest["git_commit"], "git_uncommitted_changes": manifest["git_uncommitted_changes"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--verify", type=Path, help="이미 만든 묶음만 검사")
    args = parser.parse_args()
    if args.verify:
        manifest = validate_nn1_archive(args.verify)
        print(json.dumps({"verified": str(args.verify), "file_count": len(manifest["files"])}, ensure_ascii=False))
    else:
        print(json.dumps(build_package(args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
