"""TUNE1 Colab 묶음(FutureTraffic_TUNE1.zip)과 노트북 8개(칸 6개 + 기준 + 판정)를 만든다.

이전 묶음·노트북·결과는 건드리지 않는다. 묶기만 하고 SUMO는 돌리지 않는다. 실험 모델·결과는 넣지 않는다(N5는 Drive의 NN3 준비 결과를 해시로 대조해 쓴다).
  .venv/bin/python src/explore_tune_package.py            # 노트북을 만들고 묶는다
  .venv/bin/python src/explore_tune_package.py --verify <zip>
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

import explore_nn3r3_package as prev
from explore_nn12_package import PACKAGES_CELL
from explore_weight_package import KIND, _safe_name, validate_archive

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results/colab/FutureTraffic_TUNE1.zip"
CELLS = {"DQN_64": ("DQN", "64-64"), "DQN_256": ("DQN", "256-256"), "DQN_64x3": ("DQN", "64-64-64"), "PPO_64": ("PPO", "64-64"), "PPO_256": ("PPO", "256-256"), "PPO_64x3": ("PPO", "64-64-64")}
NOTEBOOKS = {**{c: PROJECT / f"notebooks/TUNE1_{c}.ipynb" for c in CELLS}, "baseline": PROJECT / "notebooks/TUNE1_baseline.ipynb", "judge": PROJECT / "notebooks/TUNE1_judge.ipynb"}
EXPERIMENT = "TUNE1_V1"
PREREG = "results/explore/tune1_20261105/사전등록.md"
REFERENCE = "results/explore/tune1_20261105/smoke_reference/check.json"
EXTRA_FILES = (PREREG, REFERENCE, *prev.EXTRA_FILES)
REQUIRED_FILES = ("src/explore_tune.py", "src/explore_tune_package.py", "src/explore_nn3r3.py", "src/explore_nn3r3_package.py", "src/arena3_g1.py", "src/arena3_ppo.py", "config/explore_tune1_v1.json", "config/explore_nn3r3_v1.json", "tests/test_explore_tune.py", *prev.REQUIRED_FILES, *(p.relative_to(PROJECT).as_posix() for p in NOTEBOOKS.values()))
PROTECTED_NAMES = {*prev.PROTECTED_NAMES, "FutureTraffic_NN3R3.zip"}

ZIP_CELL = prev.ZIP_CELL.replace('"NN3R3_V1", "NN3R3 전용 ZIP이 아닙니다"', '"TUNE1_V1", "TUNE1 전용 ZIP이 아닙니다"').replace("nn3r3_verified_package", "tune1_verified_package")
assert '"TUNE1_V1"' in ZIP_CELL and "plan_sha256" in ZIP_CELL, "ZIP 칸 바꾸기 실패"

CELL_INTRO = """# TUNE1 {cell} — 무리 세계에서 처음부터 {algo}, 은닉층 {size}

선생님·계획기 없이 빈 신경망에서 학습합니다. 난수 3개 × 600판을 학습한 뒤, 모델마다 84상황을 잽니다. 방법과 읽는 법은 `results/explore/tune1_20261105/사전등록.md`에 고정했습니다.

칸 6개(`TUNE1_DQN_*`, `TUNE1_PPO_*`)는 **노트북 하나에 런타임 하나**입니다. 동시에 돌려도 되고, 런타임이 비는 대로 차례로 돌려도 됩니다.

1. 런타임 유형 **CPU + 높은 RAM(CPU 8개)** 을 고르고 **런타임 → 모두 실행**을 누르세요. GPU는 필요 없습니다.
2. 이 노트북 하나가 끝까지 돕니다: 점검(약 10분) → 학습(약 7~9시간, 어림) → 평가(약 1시간, 어림).
3. 끊기면 그 노트북만 다시 **모두 실행**하세요. 학습은 마지막 체크포인트부터, 평가는 남은 판부터 이어 갑니다.
4. 칸 6개와 `TUNE1_baseline`이 모두 끝나면 `TUNE1_judge`를 실행하고 "끝났어"라고 알려 주세요. 결과: Drive `FutureTraffic/explore_tune/tune1/`.
"""

BASELINE_INTRO = """# TUNE1 기준 — 같은 84상황의 고정신호와 N5

칸들과 비교할 기준입니다. 학습은 없습니다. 언제 돌려도 되고 약 1시간(어림) 걸립니다.

1. 런타임 유형 **CPU + 높은 RAM(CPU 8개)** 을 고르고 **런타임 → 모두 실행**을 누르세요.
2. 끊기면 다시 **모두 실행**하세요. 끝난 판은 건너뜁니다.
"""

JUDGE_INTRO = """# TUNE1 판정

칸 6개와 `TUNE1_baseline`이 끝난 뒤 **모두 실행**하세요. SUMO는 돌리지 않으며 CPU 수는 상관없습니다. 작은 파일을 많이 읽어 10~30분쯤 걸릴 수 있습니다.
읽는 법은 `results/explore/tune1_20261105/사전등록.md`에 고정했습니다. 숫자는 맥에서 다시 계산합니다.
"""

SETUP = '''from google.colab import drive
from pathlib import Path
import os, sys, json, time

drive.mount("/content/drive")
DRIVE = Path("/content/drive/MyDrive/FutureTraffic")
ARCHIVE = DRIVE / "FutureTraffic_TUNE1.zip"
WORKSPACE = Path("/content/FutureTraffic_TUNE1")
PREP = DRIVE / "explore_nn3" / "prep"                  # N5(읽기만, 해시 대조)
OUT = DRIVE / "explore_tune" / "tune1"
CELL = "{cell}"
MIN_CPUS = 8
for path in (ARCHIVE, PREP / "start" / "seed_42" / "model.pt", OUT / "baseline" / "runs"):
    assert path.exists(), f"필요한 파일·폴더가 없습니다: {{path}}"
CPUS = os.cpu_count() or 1
print("CPU 개수:", CPUS, "| 칸:", CELL, "| 결과 폴더:", OUT)
assert CPUS >= MIN_CPUS, (f"CPU가 {{CPUS}}개입니다. 런타임 → 런타임 유형 변경에서 CPU와 '높은 RAM'을 고른 뒤 다시 실행하세요.")'''

TESTS_CELL = '''os.chdir(WORKSPACE)
sys.path.insert(0, str(WORKSPACE / "src"))
import arena3_colab as colab
from IPython.display import display, Markdown

colab.run_live(["-m", "unittest", "discover", "-s", "tests", "-p", "test_explore_tune.py"], WORKSPACE)
print("실행 전 검사(SUMO 없이) 통과")'''

CHECK_CELL = '''# 실행 전 점검(약 10분): 작은 학습 → 저장 모델 평가 → 신호 대조. 통과해야 본 학습으로 갑니다.
CHECK = OUT / CELL / "check"
result = CHECK / "check.json"
if not (colab.read_json(result) or {}).get("ok"):
    colab.run_live(["src/explore_tune.py", "check", "--cell", CELL, "--prep", str(PREP), "--out", str(CHECK), "--work", "/content/tc"], WORKSPACE, check=False)
r = colab.read_json(result)
assert r and r.get("ok"), f"점검 실패 — 본 실행을 하지 않습니다. 이 칸의 출력을 저에게 보내 주세요: {r}"
print("점검 통과 | 맥과 같음:", r.get("same_as_mac"), "| 평가 한 판 초:", round(r["eval_seconds"]), "| 작은 학습 작업시간:", round(r["train"]["cpu_hours"], 3))'''

TRAIN_CELL = '''# 본 학습: 난수 3개 × 판 2개 동시. 백그라운드로 돌아 셀을 멈춰도 계속 돕니다. 끊기면 처음부터 다시 모두 실행하세요.
colab.run_live(["src/explore_tune.py", "prepare", "--cell", CELL, "--out", str(OUT)], WORKSPACE)
LOGS = OUT / CELL / "logs"
TRAIN = OUT / CELL / "train"

def describe(name):
    s = colab.read_json(TRAIN / f"seed_{name.rsplit('_', 1)[1]}" / "status.json") or {}
    return f"{s.get('episodes', 0)}/{s.get('budget', 600)}판, 결정 {s.get('decisions', 0)}, 최근 지연 {s.get('recent_delay_s')}, 버림 {s.get('discarded', 0)}, 작업시간 {s.get('cpu_hours', 0):.1f}h"

jobs = {f"train_{seed}": ["src/explore_tune.py", "train", "--cell", CELL, "--seed", str(seed), "--out", str(OUT), "--work", f"/content/tt{seed}"] for seed in (42, 43, 44)}
pids = colab.launch(jobs, WORKSPACE, LOGS)
code = colab.watch(pids, LOGS, 300, describe)
assert code == 0, "오류가 났습니다. 위 로그를 저에게 보내 주세요(다시 실행하면 체크포인트부터 이어 갑니다)."
print("학습 끝.")'''

EVAL_CELL = '''# 평가: 최종 모델 3개 × 84상황 + 곡선 모델 9개 × 12상황(360판). 끝난 판은 건너뜁니다.
def describe_eval(name, folder=OUT / CELL / "eval" / "runs"):
    return f"끝난 판 {len(list(folder.glob('*/*/metrics.json')))}/360"

pids = colab.launch({"eval": ["src/explore_tune.py", "evaluate", "--cell", CELL, "--out", str(OUT), "--work", "/content/te", "--workers", str(CPUS)]}, WORKSPACE, LOGS)
code = colab.watch(pids, LOGS, 300, describe_eval)
assert code == 0, "평가에서 오류가 났습니다. 위 로그를 저에게 보내 주세요(다시 실행하면 끝난 판은 건너뜁니다)."
print("이 칸 끝. 칸 6개와 TUNE1_baseline이 끝나면 TUNE1_judge를 실행하세요.")'''

BASELINE_CELL = '''# 고정신호 84판 + N5 252판. 끝난 판은 건너뜁니다.
LOGS = OUT / "baseline" / "logs"
def describe(name, folder=OUT / "baseline" / "runs"):
    return f"끝난 판 {len(list(folder.glob('*/*/metrics.json')))}/336"

pids = colab.launch({"baseline": ["src/explore_tune.py", "baseline", "--prep", str(PREP), "--out", str(OUT), "--work", "/content/tb", "--workers", str(CPUS)]}, WORKSPACE, LOGS)
code = colab.watch(pids, LOGS, 300, describe)
assert code == 0, "오류가 났습니다. 위 로그를 저에게 보내 주세요(다시 실행하면 끝난 판은 건너뜁니다)."
print("기준 끝.")'''

JUDGE_SETUP = '''from google.colab import drive
from pathlib import Path
import os, sys, json

drive.mount("/content/drive")
DRIVE = Path("/content/drive/MyDrive/FutureTraffic")
ARCHIVE = DRIVE / "FutureTraffic_TUNE1.zip"
OUT = DRIVE / "explore_tune" / "tune1"
WORKSPACE = Path("/content/FutureTraffic_TUNE1")
assert ARCHIVE.exists() and OUT.exists(), "필요한 파일이 없습니다."'''

JUDGE_CELL = '''# 판정(SUMO 없음). 아직 안 끝났으면 남은 부분을 알려 줍니다.
code = colab.run_live(["src/explore_tune.py", "judge", "--root", str(OUT), "--out", str(OUT / "judge")], WORKSPACE, check=False)
if (OUT / "judge" / "summary.md").exists():
    display(Markdown((OUT / "judge" / "summary.md").read_text()))
print("판정 코드:", code, "(0이면 TUNE1이 끝났습니다)")'''


def _cell(kind: str, text: str) -> dict:
    return prev._cell(kind, text)


def notebook(name: str) -> dict:
    if name == "judge":
        return prev.prev._wrap([_cell("markdown", JUDGE_INTRO)] + [_cell("code", c) for c in (JUDGE_SETUP, ZIP_CELL, PACKAGES_CELL, TESTS_CELL, JUDGE_CELL)])
    if name == "baseline":
        return prev.prev._wrap([_cell("markdown", BASELINE_INTRO)] + [_cell("code", c) for c in (SETUP.format(cell="baseline"), ZIP_CELL, PACKAGES_CELL, TESTS_CELL, BASELINE_CELL)])
    algo, size = CELLS[name]
    return prev.prev._wrap([_cell("markdown", CELL_INTRO.format(cell=name, algo=algo, size=size))] + [_cell("code", c) for c in (SETUP.format(cell=name), ZIP_CELL, PACKAGES_CELL, TESTS_CELL, CHECK_CELL, TRAIN_CELL, EVAL_CELL)])


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


def validate_package(archive: Path) -> dict:
    manifest = validate_archive(archive)
    if manifest.get("experiment") != EXPERIMENT or manifest.get("source_models_included") is not False:
        raise ValueError("TUNE1 묶음이 아니거나 모델·결과가 들어 있습니다.")
    for name in REQUIRED_FILES + EXTRA_FILES:
        if name not in manifest["files"]:
            raise ValueError(f"TUNE1 묶음에 필요한 파일이 없습니다: {name}")
    return manifest


def build_package(out: Path = OUT) -> dict:
    import focused_package as original

    out = Path(out).resolve()
    if out.name in PROTECTED_NAMES:
        raise ValueError("이전 실험 묶음을 덮어쓸 수 없습니다.")
    original.check_before_packing(False)
    notebook_sha = write_notebooks()
    paths = set(original.files())
    paths.update(PROJECT / name for name in (*EXTRA_FILES, *REQUIRED_FILES))
    for name in (*REQUIRED_FILES, *EXTRA_FILES):
        if not (PROJECT / name).is_file():
            raise FileNotFoundError(f"TUNE1 묶음 입력이 없습니다: {name}")
    sha = lambda p: hashlib.sha256((PROJECT / p).read_bytes()).hexdigest()  # noqa: E731
    lock = json.loads((PROJECT / "config/arena3_v1.json").read_text())
    manifest = {"kind": KIND, "experiment": EXPERIMENT, "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **original.git_state(), "files": {}, "source_models_included": False, "lock_sha256": sha("config/arena3_v1.json"),
                "platoon0_config_sha256": sha("config/explore_platoon0_v1.json"), "plan_sha256": sha(PREREG), "tune1_config_sha256": sha("config/explore_tune1_v1.json"), "environment_fingerprint": lock["environment"]["fingerprint"], "notebook_sha256": notebook_sha}
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".tune1_package_", suffix=".zip", dir=out.parent)
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
        validate_package(Path(temporary))
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
        manifest = validate_package(args.verify)
        print(json.dumps({"verified": str(args.verify), "file_count": len(manifest["files"])}, ensure_ascii=False))
    else:
        print(json.dumps(build_package(args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
