"""NN3 Colab 묶음(FutureTraffic_NN3.zip)과 노트북 6개(준비 + 라운드 1 조각 4개 + 판정)를 만든다.

이전 묶음·노트북·결과는 건드리지 않는다. 묶기만 하고 SUMO는 돌리지 않는다. 실험 모델·결과는 넣지 않는다(모델은 Drive의 NN1 결과를 해시로 대조해 읽는다).
  .venv/bin/python src/explore_nn3_package.py            # 노트북을 만들고 묶는다
  .venv/bin/python src/explore_nn3_package.py --verify <zip>
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

from explore_nn12_package import PACKAGES_CELL, PROTECTED_NAMES as NN12_PROTECTED, ZIP_CELL as NN12_ZIP_CELL
from explore_weight_package import KIND, _safe_name, validate_archive

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results/colab/FutureTraffic_NN3.zip"
SHARDS = 4
NOTEBOOKS = {"prep": PROJECT / "notebooks/NN3_P_prep.ipynb", **{f"R1_shard{s}": PROJECT / f"notebooks/NN3_P_R1_shard{s}.ipynb" for s in range(SHARDS)}, "judge": PROJECT / "notebooks/NN3_P_R1_judge.ipynb"}
EXPERIMENT = "NN3_V1"
PREREG = "results/explore/nn3_20261103/사전등록.md"
REFERENCE = "results/explore/nn3_20261103/smoke_reference/check.json"
EXTRA_FILES = (PREREG, REFERENCE, "results/explore/nn12_20261103/사전등록.md", "results/explore/nn1_20261102/사전등록.md", "results/explore/label0_20261102/사전등록.md", "results/explore/platoon1c_20261101/사전등록.md", "results/explore/platoon1_20261101/사전등록.md",
               "results/explore/platoon0_20261031/사전등록.md", "results/explore/platoon_base_20261101/측정계획.md")
REQUIRED_FILES = ("src/explore_nn3.py", "src/explore_nn3_package.py", "src/explore_nn12_package.py", "src/explore_nn1.py", "src/explore_nn1_package.py", "src/explore_label0.py", "src/explore_label0_package.py", "src/explore_platoon1c.py", "src/explore_platoon1.py", "src/explore_platoon1_learn.py",
                  "src/explore_platoon_base.py", "src/explore_platoon0.py", "src/explore_ceiling.py", "src/arena3_pi.py", "src/arena3_pi_learn.py", "src/arena3_g1.py", "src/arena3_bprime.py", "src/arena3_stage2_eval.py", "src/arena3_stage2_train.py", "src/arena3_stage2_data.py",
                  "src/arena3_common.py", "src/arena3_referee.py", "src/arena3_colab.py", "src/explore_pred.py", "src/explore_weight_core.py", "src/explore_weight_episode.py", "src/explore_rollout.py", "src/explore_m4probe_pending.py", "src/focused_platoon.py",
                  "config/arena3_v1.json", "config/arena3_stage2_v1.json", "config/explore_platoon0_v1.json", "config/explore_platoon1_v1.json", "config/explore_platoon1c_v1.json", "config/arena3_pi_r1.json", "config/explore_label0_v1.json", "config/explore_nn1_v1.json",
                  "config/explore_nn12_v1.json", "config/explore_nn3_v1.json", "config/east_platoon.json", "data/derived/east_upstream_204822.csv", "tests/test_explore_nn3.py", *(p.relative_to(PROJECT).as_posix() for p in NOTEBOOKS.values()))
PROTECTED_NAMES = {*NN12_PROTECTED, "FutureTraffic_NN12.zip", "FutureTraffic_NN12_fix1.zip"}

ZIP_CELL = NN12_ZIP_CELL.replace('"NN12_V1", "NN12 전용 ZIP이 아닙니다"', '"NN3_V1", "NN3 전용 ZIP이 아닙니다"').replace("nn12_verified_package", "nn3_verified_package")
assert '"NN3_V1"' in ZIP_CELL and "plan_sha256" in ZIP_CELL, "ZIP 칸 바꾸기 실패"

PREP_INTRO = """# NN3 준비 — 출발 모델 고르기(먼저 이것부터)

계획서 3단계 앞부분입니다. 12개 지도에서 N5가 뒤진 곳은 학습하지 않은 새벽·심야였습니다. 그 시간대의 기록을 더한 **N6**를 만들어 N5와 견주고, 강화학습 라운드의 출발 모델을 규칙대로 고릅니다. 이 단계는 흉내이고 강화학습이 아닙니다.
방법과 읽는 법은 `results/explore/nn3_20261103/사전등록.md`에 고정했습니다.

1. 런타임 유형 **CPU + 높은 RAM(CPU 8개)** 을 고르고 **런타임 → 모두 실행**을 누르세요. GPU는 필요 없습니다. Drive 접근을 허용하세요.
2. 순서: 묶음 검증 → 패키지 → 시험 → 실행 전 점검(약 10분) → 본 실행(기록 48판 + 검증 156판, 약 1~1.5시간, 어림).
3. 끊기면 다시 **모두 실행**하세요. 끝난 판은 건너뜁니다.
4. 끝나면 `NN3_P_R1_shard0`~`shard3` 네 개를 **런타임 하나씩** 시작하세요. 결과: Drive `FutureTraffic/explore_nn3/prep/`.
"""

ROUND_INTRO = """# NN3 라운드 1 — 조각 {shard} (네 조각을 런타임 하나씩)

계획서 3A단계 1라운드입니다. 출발 모델이 직접 운전하며 고를 수 있는 순간의 25%에서 허용 녹색을 모두 같은 미래로 굴려 비교하고, 학생이 '확실한 곳만' 바꾸도록 배웁니다. 이 부분이 강화학습입니다.
방법과 읽는 법은 `results/explore/nn3_20261103/사전등록.md`에 고정했습니다.

`NN3_P_prep`이 끝난 뒤 `shard0`~`shard3`을 **각각 런타임 하나씩** 씁니다. 순서는 상관없고 동시에 돌리면 됩니다. 넷을 다 돌려야 다음 단계로 넘어갑니다.

1. 런타임 유형 **CPU + 높은 RAM(CPU 8개)** 을 고르고 **런타임 → 모두 실행**을 누르세요.
2. 이 노트북 하나가 끝까지 알아서 돕니다: 자료 15판(약 8시간) → 다른 조각 기다림 → 학생 학습 → 거르기·문턱 → 확인(약 2시간). 합쳐 약 10시간(어림). 런타임 사이는 Drive의 상태 파일로 맞춥니다.
3. 끊기면 그 노트북만 다시 **모두 실행**하세요. 끝난 부분은 건너뜁니다. 다른 조각은 기다립니다.
4. 네 조각이 모두 끝나면 `NN3_P_R1_judge`를 실행하고 "끝났어"라고 알려 주세요. 결과: Drive `FutureTraffic/explore_nn3/r1/`.
"""

COMMON_SETUP = '''from google.colab import drive
from pathlib import Path
import os, sys, json

drive.mount("/content/drive")
DRIVE = Path("/content/drive/MyDrive/FutureTraffic")
ARCHIVE = DRIVE / "FutureTraffic_NN3.zip"
WORKSPACE = Path("/content/FutureTraffic_NN3")
NN1 = DRIVE / "explore_nn1" / "P" / "aug"                     # N5 모델·기록(읽기만, 해시 대조)
TEACHER = DRIVE / "explore_platoon1" / "students" / "seed_42"  # 선생님 t20의 학생(읽기만, 해시 대조)
PREP = DRIVE / "explore_nn3" / "prep"
MIN_CPUS = 8
for path in (ARCHIVE, NN1 / "models" / "r1" / "seed_42" / "model.pt", TEACHER):
    assert path.exists(), f"필요한 파일이 없습니다: {path}"
CPUS = os.cpu_count() or 1
print("CPU 개수:", CPUS, "| 결과 폴더:", DRIVE / "explore_nn3")
assert CPUS >= MIN_CPUS, (f"CPU가 {CPUS}개입니다. 런타임 → 런타임 유형 변경에서 CPU와 '높은 RAM'을 고른 뒤 다시 실행하세요.")'''

TESTS_CELL = '''os.chdir(WORKSPACE)
sys.path.insert(0, str(WORKSPACE / "src"))
import arena3_colab as colab
from IPython.display import display, Markdown

colab.run_live(["-m", "unittest", "discover", "-s", "tests", "-p", "test_explore_nn3.py"], WORKSPACE)
print("실행 전 검사(SUMO 없이) 통과")'''

CHECK_CELL = '''# 1) 실행 전 점검(약 10분): 상태 저장·굴림을 해도 본 판 신호가 같은지, 입력에 미래가 새지 않는지, 새벽·심야에서 끝까지 도는지 봅니다.
CHECK = PREP / "check"
previous = colab.read_json(CHECK / "check.json")
if previous and previous.get("ok"):
    print("이전 점검이 통과해 건너뜁니다.")
else:
    colab.run_live(["src/explore_nn3.py", "check", "--models", str(NN1 / "models" / "r1"), "--out", str(CHECK), "--work", "/content/n3c", "--workers", str(CPUS)], WORKSPACE, check=False)
r = colab.read_json(CHECK / "check.json")
assert r and r.get("ok"), f"점검 실패 — 본 실행을 하지 않습니다. 이 칸의 출력을 저에게 보내 주세요: {r}"
print("점검 통과 | 맥과 같음:", r.get("same_as_mac"), "| 굴림 1회 평균 초:", round(r["rollout_seconds"]["mean"], 1))'''

PREP_RUN_CELL = '''# 2) 본 실행(남은 판만). 백그라운드로 돌아 셀을 멈춰도 계속 돕니다. 끊기면 처음부터 다시 모두 실행하세요.
LOGS = PREP / "logs"

def describe(name):
    low = len(list((PREP / "runs").glob("r*/*/metrics.json"))) if (PREP / "runs").exists() else 0
    valid = len(list((PREP / "valid" / "runs").glob("*/*/metrics.json"))) if (PREP / "valid" / "runs").exists() else 0
    return f"기록 {low}/48, 검증 {valid}/156"

pids = colab.launch({"nn3_prep": ["src/explore_nn3.py", "prep", "--nn1", str(NN1), "--teacher", str(TEACHER), "--out", str(PREP), "--work", "/content/n3p", "--workers", str(CPUS)]}, WORKSPACE, LOGS)
code = colab.watch(pids, LOGS, 120, describe)
assert code == 0, "본 실행에서 오류가 났습니다. 위 로그를 저에게 보내 주세요(다시 실행하면 끝난 판은 건너뜁니다)."
s = json.loads((PREP / "start.json").read_text())
print("출발 모델:", s["start"], "| 값:", {k: round(v, 2) for k, v in s["means"].items()})
print("이제 NN3_P_R1_shard0~3을 런타임 하나씩 시작하세요.")'''

ROUND_RUN_CELL = '''# 자료 → 학생 → 거르기 → 문턱 → 확인. 백그라운드로 돌아 셀을 멈춰도 계속 돕니다. 끊기면 처음부터 다시 모두 실행하세요.
SHARD = {shard}
OUT = DRIVE / "explore_nn3" / "r1"
LOGS = OUT / "logs" / f"shard{{SHARD}}"
assert (PREP / "start.json").exists(), "NN3_P_prep을 먼저 끝내세요."

def describe(name):
    status = OUT / "data" / "status" / f"collect_shard{{SHARD}}.json"
    s = colab.read_json(status) or {{}}
    stages = [x for x in ("data", "screen", "thresholds", "final") if (OUT / "status" / f"{{x}}_shard{{SHARD}}.json").exists()]
    return f"자료 {{s.get('done', 0)}}/{{s.get('total', 15)}}판, 실패 {{len(s.get('failed', []))}}, 끝난 단계 {{stages}}"

pids = colab.launch({{f"nn3_r1_shard{{SHARD}}": ["src/explore_nn3.py", "round", "--prep", str(PREP), "--out", str(OUT), "--work", "/content/n3r", "--shard", str(SHARD), "--shards", "{shards}", "--workers", str(CPUS)]}}, WORKSPACE, LOGS)
code = colab.watch(pids, LOGS, 300, describe)
assert code == 0, "오류가 났습니다. 위 로그를 저에게 보내 주세요(다시 실행하면 끝난 부분은 건너뜁니다)."
print("이 조각 끝. 네 조각이 모두 끝나면 NN3_P_R1_judge를 실행하세요.")'''

JUDGE_INTRO = """# NN3 라운드 1 판정

네 조각이 끝난 뒤 **모두 실행**하세요. SUMO는 돌리지 않으며 CPU 수는 상관없습니다. 작은 파일을 많이 읽어 10~30분쯤 걸릴 수 있습니다.
읽는 법은 `results/explore/nn3_20261103/사전등록.md`에 고정했습니다. 숫자는 맥에서 다시 계산합니다.
"""

JUDGE_SETUP = '''from google.colab import drive
from pathlib import Path
import os, sys, json

drive.mount("/content/drive")
DRIVE = Path("/content/drive/MyDrive/FutureTraffic")
ARCHIVE = DRIVE / "FutureTraffic_NN3.zip"
ROOT_OUT = DRIVE / "explore_nn3"
WORKSPACE = Path("/content/FutureTraffic_NN3")
assert ARCHIVE.exists() and (ROOT_OUT / "prep" / "start.json").exists(), "필요한 파일이 없습니다."'''

JUDGE_CELL = '''# 판정(SUMO 없음). 아직 안 끝났으면 남은 부분을 알려 줍니다.
code = colab.run_live(["src/explore_nn3.py", "judge", "--root", str(ROOT_OUT), "--out", str(ROOT_OUT / "judge")], WORKSPACE, check=False)
if (ROOT_OUT / "judge" / "summary.md").exists():
    display(Markdown((ROOT_OUT / "judge" / "summary.md").read_text()))
print("판정 코드:", code, "(0이면 라운드 1이 끝났습니다)")'''


def _cell(kind: str, text: str) -> dict:
    out = {"cell_type": kind, "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}
    if kind == "code":
        out.update(execution_count=None, outputs=[])
    return out


def _wrap(cells: list[dict]) -> dict:
    return {"cells": cells, "metadata": {"colab": {"provenance": [], "machine_shape": "hm"}, "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}


def notebook(name: str) -> dict:
    if name == "prep":
        steps = (COMMON_SETUP, ZIP_CELL, PACKAGES_CELL, TESTS_CELL, CHECK_CELL, PREP_RUN_CELL)
        return _wrap([_cell("markdown", PREP_INTRO)] + [_cell("code", c) for c in steps])
    if name == "judge":
        return _wrap([_cell("markdown", JUDGE_INTRO)] + [_cell("code", c) for c in (JUDGE_SETUP, ZIP_CELL, PACKAGES_CELL, TESTS_CELL, JUDGE_CELL)])
    shard = int(name.removeprefix("R1_shard"))
    steps = (COMMON_SETUP, ZIP_CELL, PACKAGES_CELL, TESTS_CELL, ROUND_RUN_CELL.format(shard=shard, shards=SHARDS))
    return _wrap([_cell("markdown", ROUND_INTRO.format(shard=shard))] + [_cell("code", c) for c in steps])


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


def validate_nn3_archive(archive: Path) -> dict:
    manifest = validate_archive(archive)
    if manifest.get("experiment") != EXPERIMENT or manifest.get("source_models_included") is not False:
        raise ValueError("NN3 묶음이 아니거나 모델·결과가 들어 있습니다.")
    for name in REQUIRED_FILES + EXTRA_FILES:
        if name not in manifest["files"]:
            raise ValueError(f"NN3 묶음에 필요한 파일이 없습니다: {name}")
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
            raise FileNotFoundError(f"NN3 묶음 입력이 없습니다: {name}")
    sha = lambda p: hashlib.sha256((PROJECT / p).read_bytes()).hexdigest()  # noqa: E731
    lock = json.loads((PROJECT / "config/arena3_v1.json").read_text())
    manifest = {"kind": KIND, "experiment": EXPERIMENT, "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **original.git_state(), "files": {}, "source_models_included": False, "lock_sha256": sha("config/arena3_v1.json"),
                "platoon0_config_sha256": sha("config/explore_platoon0_v1.json"), "plan_sha256": sha(PREREG), "nn3_config_sha256": sha("config/explore_nn3_v1.json"), "environment_fingerprint": lock["environment"]["fingerprint"], "notebook_sha256": notebook_sha}
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".nn3_package_", suffix=".zip", dir=out.parent)
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
        validate_nn3_archive(Path(temporary))
        os.replace(temporary, out)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"path": str(out), "bytes": out.stat().st_size, "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "file_count": len(manifest["files"]), "experiment": EXPERIMENT, "notebook_sha256": notebook_sha, "plan_sha256": manifest["plan_sha256"],
            "git_commit": manifest["git_commit"], "git_uncommitted_changes": manifest["git_uncommitted_changes"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--verify", type=Path, help="이미 만든 묶음만 검사")
    args = parser.parse_args()
    if args.verify:
        manifest = validate_nn3_archive(args.verify)
        print(json.dumps({"verified": str(args.verify), "file_count": len(manifest["files"])}, ensure_ascii=False))
    else:
        print(json.dumps(build_package(args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
