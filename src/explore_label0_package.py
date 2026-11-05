"""LABEL0 Colab 묶음(FutureTraffic_LABEL0.zip)과 노트북 5개(후보 4개 + 판정)를 만든다.

이전 묶음·노트북·결과는 건드리지 않는다. 묶기만 하고 SUMO는 돌리지 않는다. 모델·결과는 넣지 않는다.
  .venv/bin/python src/explore_label0_package.py            # 노트북을 만들고 묶는다
  .venv/bin/python src/explore_label0_package.py --verify <zip>
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

from explore_relax0_package import PACKAGES_CELL, PROTECTED_NAMES as RX_PROTECTED, ZIP_CELL as RX_ZIP_CELL
from explore_weight_package import KIND, _safe_name, validate_archive

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results/colab/FutureTraffic_LABEL0.zip"
VARIANTS = ("P_w0", "P_w25", "R_w0", "R_w25")
NOTEBOOKS = {**{v: PROJECT / f"notebooks/LABEL0_{v}.ipynb" for v in VARIANTS}, "judge": PROJECT / "notebooks/LABEL0_judge.ipynb"}
EXPERIMENT = "LABEL0_V1"
PREREG = "results/explore/label0_20261102/사전등록.md"
EXTRA_FILES = (PREREG, "results/explore/platoon1c_20261101/사전등록.md", "results/explore/platoon1_20261101/사전등록.md",
               "results/explore/platoon0_20261031/사전등록.md", "results/explore/platoon_base_20261101/측정계획.md")
REQUIRED_FILES = ("src/explore_label0.py", "src/explore_label0_package.py", "src/explore_platoon1c.py", "src/explore_platoon1.py", "src/explore_platoon1_learn.py",
                  "src/explore_platoon_base.py", "src/explore_platoon0.py", "src/explore_ceiling.py", "src/arena3_pi.py", "src/arena3_pi_learn.py",
                  "src/arena3_bprime.py", "src/arena3_stage2_eval.py", "src/arena3_common.py", "src/arena3_referee.py", "src/arena3_colab.py",
                  "src/explore_pred.py", "src/explore_weight_core.py", "src/explore_weight_episode.py", "src/explore_rollout.py",
                  "src/explore_m4probe_pending.py", "src/focused_platoon.py", "config/arena3_v1.json", "config/explore_platoon0_v1.json",
                  "config/explore_platoon1_v1.json", "config/explore_platoon1c_v1.json", "config/arena3_pi_r1.json", "config/explore_label0_v1.json",
                  "config/east_platoon.json", "data/derived/east_upstream_204822.csv", "tests/test_explore_label0.py",
                  *(p.relative_to(PROJECT).as_posix() for p in NOTEBOOKS.values()))
PROTECTED_NAMES = {*RX_PROTECTED, "FutureTraffic_RELAX0.zip"}

ZIP_CELL = RX_ZIP_CELL.replace('"RELAX0_V1", "RELAX0 전용 ZIP이 아닙니다"', '"LABEL0_V1", "LABEL0 전용 ZIP이 아닙니다"') \
    .replace("relax0_verified_package", "label0_verified_package")
assert '"LABEL0_V1"' in ZIP_CELL and "plan_sha256" in ZIP_CELL, "ZIP 칸 바꾸기 실패"

WORLD = {"P": ("무리 도착 세계", "PT1", "explore_platoon1/data", "explore_platoon1/students", "explore_platoon1/eval/PT1", "2~3시간"),
         "R": ("무작위 도착 세계", "R1A", "arena3/pi/R1/data", "arena3/pi/R1/students", "arena3/pi/R1/eval/R1A", "1.5~2시간")}

INTRO = """# LABEL0 {variant} — 5분 점수의 끝 순간 항 무게를 {weight}으로 바꾼 학생 ({world})

5분 점수 = 5분 지연 + 끝 순간 멈춘 차 × 무게입니다. 지금 무게 75 대신 {weight}으로 라벨을 다시 계산해 학생을 다시 학습하고, 한 시간 SUMO로 잽니다.
자료·신경망·학습 설정은 {old}와 같고 새 굴림은 없습니다. 방법과 읽는 법은 `results/explore/label0_20261102/사전등록.md`에 고정했습니다.

노트북은 후보 4개(P_w0·P_w25·R_w0·R_w25)와 판정 1개입니다. **후보마다 런타임을 하나씩 따로** 씁니다. 순서는 상관없습니다.

1. 런타임 유형 **CPU + 높은 RAM(CPU 8개)** 을 고르고 **런타임 → 모두 실행**을 누르세요. Drive 접근을 허용하세요.
2. 순서: 묶음 검증 → 패키지 → 시험 → 학생 학습(약 5분) → 작은 실행(약 10분) → 본 평가(252판, CPU 8개 기준 약 {hours}, 어림).
3. 본 평가는 백그라운드로 돕니다. 끊기면 다시 **모두 실행**하세요. 끝난 판은 건너뜁니다.
4. 후보 4개가 끝나면 `LABEL0_judge`를 실행하고 "끝났어"라고 알려 주세요. 결과: Drive `FutureTraffic/explore_label0/`.
"""

SETUP = '''from google.colab import drive
from pathlib import Path
import os, sys, json

drive.mount("/content/drive")
VARIANT = "{variant}"
DRIVE = Path("/content/drive/MyDrive/FutureTraffic")
ARCHIVE = DRIVE / "FutureTraffic_LABEL0.zip"
DATA = DRIVE / "{data}"                                # 기존 자료(읽기만)
OLD_STUDENTS = DRIVE / "{students}"                    # 무게 75 학생(읽기만)
OLD_EVAL = DRIVE / "{evaluation}"                      # 무게 75 평가(읽기만)
PLATOON0 = DRIVE / "explore_platoon0"                  # 평가 지문 대조(읽기만)
OUT = DRIVE / "explore_label0" / VARIANT
WORKSPACE = Path("/content/FutureTraffic_LABEL0")
MIN_CPUS = 8
for path in (ARCHIVE, DATA / "experiment.json", OLD_STUDENTS / "seed_42" / "model.pt", OLD_EVAL / "models.json", PLATOON0 / "experiment.json"):
    assert path.exists(), f"필요한 파일이 없습니다: {{path}}"
CPUS = os.cpu_count() or 1
print("CPU 개수:", CPUS, "| 후보:", VARIANT, "| 결과 폴더:", OUT)
assert CPUS >= MIN_CPUS, (f"CPU가 {{CPUS}}개입니다. 런타임 → 런타임 유형 변경에서 CPU와 '높은 RAM'을 고른 뒤 다시 실행하세요.")'''

JUDGE_INTRO = """# LABEL0 판정 — 끝 순간 항을 고치면 좋아졌나

후보 4개(P_w0·P_w25·R_w0·R_w25)의 평가가 끝난 뒤 **모두 실행**하세요. SUMO는 돌리지 않으며 CPU 수는 상관없습니다. 몇 분 걸립니다.
읽는 법은 `results/explore/label0_20261102/사전등록.md`에 고정했습니다. 숫자는 맥에서 다시 계산합니다.
"""

JUDGE_SETUP = '''from google.colab import drive
from pathlib import Path
import os, sys, json

drive.mount("/content/drive")
DRIVE = Path("/content/drive/MyDrive/FutureTraffic")
ARCHIVE = DRIVE / "FutureTraffic_LABEL0.zip"
ROOT_OUT = DRIVE / "explore_label0"
A3 = DRIVE / "arena3"
WORKSPACE = Path("/content/FutureTraffic_LABEL0")
assert ARCHIVE.exists(), f"필요한 파일이 없습니다: {ARCHIVE}"'''

TESTS_CELL = '''sys.path.insert(0, str(WORKSPACE / "src"))
import arena3_colab as colab
from IPython.display import display, Markdown

colab.run_live(["-m", "unittest", "discover", "-s", "tests", "-p", "test_explore_label0.py"], WORKSPACE)
print("실행 전 검사(SUMO 없이) 통과")'''

TRAIN_CELL = '''# 1) 학생 학습(약 5분). 이미 있으면 다시 쓰지 않습니다.
colab.run_live(["src/explore_label0.py", "train", "--variant", VARIANT, "--data", str(DATA), "--out", str(OUT / "students")], WORKSPACE)
print("학생 학습 끝")'''

SMOKE_CELL = '''# 2) 작은 실행(약 10분): 무게 75 라벨이 원래 자료와 같은지, 후보 라벨이 달라졌는지, 학생 한 판이 온전한지 봅니다.
SMOKE = OUT / "smoke"
previous = colab.read_json(SMOKE / "smoke.json")
if previous and previous.get("ok"):
    print("이전 작은 실행이 통과해 건너뜁니다.")
else:
    colab.run_live(["src/explore_label0.py", "smoke", "--variant", VARIANT, "--data", str(DATA), "--students", str(OUT / "students"),
                    "--reference-students", str(OLD_STUDENTS), "--reference-eval", str(OLD_EVAL), "--out", str(SMOKE),
                    "--work", "/content/l0_smoke", "--evaluation", str(PLATOON0)], WORKSPACE, check=False)
s = colab.read_json(SMOKE / "smoke.json")
assert s and s.get("ok"), f"작은 실행 점검 실패 — 이 칸의 출력을 저에게 보내 주세요: {s and s.get('checks')}"
print("작은 실행 통과:", s["checks"], "| 지연", s["delay_s"], "(무게 75:", s["delay_w75_s"], ") | 무게 75 재학습 일치:", s["retrain_w75_same_model"])'''

EVAL_CELL = '''# 3) 본 평가(모델 3개 × 84상황 = 252판). 백그라운드로 돌아 셀을 멈춰도 계속 돕니다.
#    런타임이 끊기면 처음 네 칸 → 이 칸을 다시 실행하세요. 끝난 판은 건너뜁니다.
assert "colab" in globals(), "처음 네 칸(1~4)을 먼저 실행하세요. 런타임이 끊기면 변수와 작업 폴더가 사라집니다."
LOGS = OUT / "logs"
EVAL = OUT / "eval"

def describe(name):
    done = len(list((EVAL / "runs").glob("*/*/cache.json")))
    failed = len(list((EVAL / "runs").glob("*/*.failed.json")))
    return f"끝난 판 {done}/252, 실패 기록 {failed}"

pids = colab.launch({f"label0_{VARIANT}": ["src/explore_label0.py", "evaluate", "--variant", VARIANT, "--students", str(OUT / "students"),
                                           "--out", str(EVAL), "--work", "/content/l0", "--workers", str(CPUS)]}, WORKSPACE, LOGS)
code = colab.watch(pids, LOGS, 300, describe)
assert code == 0, "본 평가에서 오류가 났습니다. 위 로그를 저에게 보내 주세요(다시 실행하면 끝난 판은 건너뜁니다)."
print("본 평가 끝. 후보 4개가 끝나면 LABEL0_judge를 실행하세요.")'''

JUDGE_CELL = '''# 판정(SUMO 없음). 아직 안 끝난 후보가 있으면 표 아래에 나옵니다.
code = colab.run_live(["src/explore_label0.py", "judge", "--root", str(ROOT_OUT), "--platoon0", str(DRIVE / "explore_platoon0"),
                       "--platoon1", str(DRIVE / "explore_platoon1"), "--pi", str(A3 / "pi" / "R1"), "--stage2", str(A3 / "stage2"),
                       "--baseline", str(A3 / "referee" / "baseline"), "--pbase", str(DRIVE / "explore_platoon_base"),
                       "--out", str(ROOT_OUT / "judge")], WORKSPACE, check=False)
if (ROOT_OUT / "judge" / "summary.md").exists():
    display(Markdown((ROOT_OUT / "judge" / "summary.md").read_text()))
print("판정 코드:", code, "(0이면 네 후보가 모두 끝났습니다)")'''


def _cell(kind: str, text: str) -> dict:
    out = {"cell_type": kind, "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}
    if kind == "code":
        out.update(execution_count=None, outputs=[])
    return out


def _wrap(cells: list[dict]) -> dict:
    return {"cells": cells, "metadata": {"colab": {"provenance": [], "machine_shape": "hm"},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


def notebook(name: str) -> dict:
    if name == "judge":
        return _wrap([_cell("markdown", JUDGE_INTRO)] + [_cell("code", c) for c in (JUDGE_SETUP, ZIP_CELL, PACKAGES_CELL, TESTS_CELL, JUDGE_CELL)])
    world, old, data, students, evaluation, hours = WORLD[name[0]]
    intro = INTRO.format(variant=name, weight=name.split("_w")[1], world=world, old=old, hours=hours)
    setup = SETUP.format(variant=name, data=data, students=students, evaluation=evaluation)
    return _wrap([_cell("markdown", intro)] + [_cell("code", c) for c in (setup, ZIP_CELL, PACKAGES_CELL, TESTS_CELL, TRAIN_CELL, SMOKE_CELL, EVAL_CELL)])


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


def validate_label0_archive(archive: Path) -> dict:
    manifest = validate_archive(archive)
    if manifest.get("experiment") != EXPERIMENT or manifest.get("source_models_included") is not False:
        raise ValueError("LABEL0 묶음이 아니거나 모델·결과가 들어 있습니다.")
    for name in REQUIRED_FILES + EXTRA_FILES:
        if name not in manifest["files"]:
            raise ValueError(f"LABEL0 묶음에 필요한 파일이 없습니다: {name}")
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
            raise FileNotFoundError(f"LABEL0 묶음 입력이 없습니다: {name}")
    sha = lambda p: hashlib.sha256((PROJECT / p).read_bytes()).hexdigest()  # noqa: E731
    lock = json.loads((PROJECT / "config/arena3_v1.json").read_text())
    manifest = {"kind": KIND, "experiment": EXPERIMENT,
                "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                **original.git_state(), "files": {}, "source_models_included": False,
                "lock_sha256": sha("config/arena3_v1.json"), "platoon0_config_sha256": sha("config/explore_platoon0_v1.json"),
                "plan_sha256": sha(PREREG), "label0_config_sha256": sha("config/explore_label0_v1.json"),
                "environment_fingerprint": lock["environment"]["fingerprint"], "notebook_sha256": notebook_sha}
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".label0_package_", suffix=".zip", dir=out.parent)
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
        validate_label0_archive(Path(temporary))
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
        manifest = validate_label0_archive(args.verify)
        print(json.dumps({"verified": str(args.verify), "file_count": len(manifest["files"])}, ensure_ascii=False))
    else:
        print(json.dumps(build_package(args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
