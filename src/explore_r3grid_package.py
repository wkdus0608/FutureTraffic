"""R3GRID Colab 묶음(FutureTraffic_R3GRID.zip)과 노트북 5개(조각 4개 + 판정)를 만든다.

이전 묶음·노트북·결과는 건드리지 않는다. 묶기만 하고 SUMO는 돌리지 않는다. 실험 모델·결과는 넣지 않는다(자료·모델은 Drive의 NN3 결과를 해시로 대조해 쓴다).
  .venv/bin/python src/explore_r3grid_package.py            # 노트북을 만들고 묶는다
  .venv/bin/python src/explore_r3grid_package.py --verify <zip>
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
OUT = PROJECT / "results/colab/FutureTraffic_R3GRID.zip"
SHARDS = 4
NOTEBOOKS = {**{f"shard{s}": PROJECT / f"notebooks/R3GRID_shard{s}.ipynb" for s in range(SHARDS)}, "judge": PROJECT / "notebooks/R3GRID_judge.ipynb"}
EXPERIMENT = "R3GRID_V1"
PREREG = "results/explore/r3grid_20261105/사전등록.md"
EXTRA_FILES = (PREREG, *prev.EXTRA_FILES)
REQUIRED_FILES = ("src/explore_r3grid.py", "src/explore_r3grid_package.py", "src/explore_nn3r3.py", "src/explore_nn3r3_package.py", "config/explore_r3grid_v1.json", "config/explore_tune1_v1.json", "config/explore_nn3r3_v1.json", "tests/test_explore_r3grid.py", *prev.REQUIRED_FILES, *(p.relative_to(PROJECT).as_posix() for p in NOTEBOOKS.values()))
PROTECTED_NAMES = {*prev.PROTECTED_NAMES, "FutureTraffic_NN3R3.zip", "FutureTraffic_TUNE1.zip"}

ZIP_CELL = prev.ZIP_CELL.replace('"NN3R3_V1", "NN3R3 전용 ZIP이 아닙니다"', '"R3GRID_V1", "R3GRID 전용 ZIP이 아닙니다"').replace("nn3r3_verified_package", "r3grid_verified_package")
assert '"R3GRID_V1"' in ZIP_CELL and "plan_sha256" in ZIP_CELL, "ZIP 칸 바꾸기 실패"

INTRO = """# R3GRID — 조각 {shard} (네 조각을 런타임 하나씩)

라운드 3 학생만 은닉층·학습률·배치·에포크 54조합으로 다시 학습하고, 앞 정책에 쌓아 한 시간 SUMO로 잽니다. 새 굴림은 없습니다.
24상황으로 거른 뒤 상위 3개를 84상황에서 라운드 3 정책과 비교합니다. 방법과 읽는 법은 `results/explore/r3grid_20261105/사전등록.md`에 고정했습니다.

`shard0`~`shard3`을 **각각 런타임 하나씩** 씁니다. **동시에 시작해도 됩니다.** 조각 0이 점검(약 10분)과 학생 학습(약 30분, 어림)을 끝내면 나머지가 이어서 돕니다.

1. 런타임 유형 **CPU + 높은 RAM(CPU 8개)** 을 고르고 **런타임 → 모두 실행**을 누르세요. GPU는 필요 없습니다.
2. 이 노트북 하나가 끝까지 돕니다: 점검 → 학생 학습 → 거르기(3,816판) → 확인(1,344판). 합쳐 약 5~6시간(어림).
3. 끊기면 그 노트북만 다시 **모두 실행**하세요. 끝난 부분은 건너뜁니다. 다른 조각은 기다립니다.
4. 네 조각이 모두 끝나면 `R3GRID_judge`를 실행하고 "끝났어"라고 알려 주세요. 결과: Drive `FutureTraffic/explore_tune/r3grid/`.
"""

SETUP = '''from google.colab import drive
from pathlib import Path
import os, sys, json, time

drive.mount("/content/drive")
DRIVE = Path("/content/drive/MyDrive/FutureTraffic")
ARCHIVE = DRIVE / "FutureTraffic_R3GRID.zip"
WORKSPACE = Path("/content/FutureTraffic_R3GRID")
N3 = DRIVE / "explore_nn3"                             # 자료·모델·저장 결과(읽기만, 해시 대조)
OUT = DRIVE / "explore_tune" / "r3grid"
SHARD = {shard}
MIN_CPUS = 8
for path in (ARCHIVE, N3 / "prep" / "start.json", N3 / "r3" / "students" / "done.json", N3 / "r3" / "eval" / "runs" / "recent_s10_m0", N3 / "r2" / "data" / "runs" / "DATA", OUT / "status"):
    assert path.exists(), f"필요한 파일·폴더가 없습니다: {{path}}"
CPUS = os.cpu_count() or 1
print("CPU 개수:", CPUS, "| 조각:", SHARD, "| 결과 폴더:", OUT)
assert CPUS >= MIN_CPUS, (f"CPU가 {{CPUS}}개입니다. 런타임 → 런타임 유형 변경에서 CPU와 '높은 RAM'을 고른 뒤 다시 실행하세요.")'''

TESTS_CELL = '''os.chdir(WORKSPACE)
sys.path.insert(0, str(WORKSPACE / "src"))
import arena3_colab as colab
from IPython.display import display, Markdown

colab.run_live(["-m", "unittest", "discover", "-s", "tests", "-p", "test_explore_r3grid.py"], WORKSPACE)
print("실행 전 검사(SUMO 없이) 통과")'''

CHECK_CELL = '''# 실행 전 점검: 조각 0이 돌리고(약 10분), 다른 조각은 결과를 기다립니다. 통과해야 본 실행으로 갑니다.
CHECK = OUT / "check"
result = CHECK / "check.json"
if SHARD == 0 and not (colab.read_json(result) or {}).get("ok"):
    colab.run_live(["src/explore_r3grid.py", "check", "--n3", str(N3), "--out", str(CHECK), "--work", "/content/rc"], WORKSPACE, check=False)
while not result.exists():
    print("조각 0의 점검을 기다립니다."); time.sleep(60)
r = colab.read_json(result)
assert r and r.get("ok"), f"점검 실패 — 본 실행을 하지 않습니다. 이 칸의 출력을 저에게 보내 주세요: {r}"
print("점검 통과 | 저장 결과와 같음:", r["same_as_stored"], "| 값이 있는 3층 학생이 바꾼 결정:", r["active_changed_decisions"])'''

RUN_CELL = '''# 학생 → 거르기 → 확인. 백그라운드로 돌아 셀을 멈춰도 계속 돕니다. 끊기면 처음부터 다시 모두 실행하세요.
LOGS = OUT / "logs" / f"shard{SHARD}"

def describe(name):
    students = (OUT / "students" / "done.json").exists()
    stages = [x for x in ("screen", "final") if (OUT / "status" / f"{x}_shard{SHARD}.json").exists()]
    return f"학생 끝: {students}, 끝난 단계 {stages} | " + " / ".join(colab.tail(LOGS / f"{name}.log", 1))

pids = colab.launch({f"r3grid_shard{SHARD}": ["src/explore_r3grid.py", "run", "--n3", str(N3), "--out", str(OUT), "--work", "/content/rg", "--shard", str(SHARD), "--shards", "4", "--workers", str(CPUS)]}, WORKSPACE, LOGS)
code = colab.watch(pids, LOGS, 300, describe)
assert code == 0, "오류가 났습니다. 위 로그를 저에게 보내 주세요(다시 실행하면 끝난 부분은 건너뜁니다)."
print("이 조각 끝. 네 조각이 모두 끝나면 R3GRID_judge를 실행하세요.")'''

JUDGE_INTRO = """# R3GRID 판정

네 조각이 끝난 뒤 **모두 실행**하세요. SUMO는 돌리지 않으며 CPU 수는 상관없습니다. 작은 파일을 많이 읽어 30분쯤 걸릴 수 있습니다.
읽는 법은 `results/explore/r3grid_20261105/사전등록.md`에 고정했습니다. 숫자는 맥에서 다시 계산합니다.
"""

JUDGE_SETUP = '''from google.colab import drive
from pathlib import Path
import os, sys, json

drive.mount("/content/drive")
DRIVE = Path("/content/drive/MyDrive/FutureTraffic")
ARCHIVE = DRIVE / "FutureTraffic_R3GRID.zip"
N3 = DRIVE / "explore_nn3"
OUT = DRIVE / "explore_tune" / "r3grid"
WORKSPACE = Path("/content/FutureTraffic_R3GRID")
assert ARCHIVE.exists() and (OUT / "students" / "done.json").exists(), "필요한 파일이 없습니다."'''

JUDGE_CELL = '''# 판정(SUMO 없음). 아직 안 끝났으면 남은 부분을 알려 줍니다.
code = colab.run_live(["src/explore_r3grid.py", "judge", "--n3", str(N3), "--root", str(OUT), "--out", str(OUT / "judge")], WORKSPACE, check=False)
if (OUT / "judge" / "summary.md").exists():
    display(Markdown((OUT / "judge" / "summary.md").read_text()))
print("판정 코드:", code, "(0이면 R3GRID가 끝났습니다)")'''


def notebook(name: str) -> dict:
    wrap, cell = prev.prev._wrap, prev._cell
    if name == "judge":
        return wrap([cell("markdown", JUDGE_INTRO)] + [cell("code", c) for c in (JUDGE_SETUP, ZIP_CELL, PACKAGES_CELL, TESTS_CELL, JUDGE_CELL)])
    shard = int(name.removeprefix("shard"))
    return wrap([cell("markdown", INTRO.format(shard=shard))] + [cell("code", c) for c in (SETUP.format(shard=shard), ZIP_CELL, PACKAGES_CELL, TESTS_CELL, CHECK_CELL, RUN_CELL)])


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
        raise ValueError("R3GRID 묶음이 아니거나 모델·결과가 들어 있습니다.")
    for name in REQUIRED_FILES + EXTRA_FILES:
        if name not in manifest["files"]:
            raise ValueError(f"R3GRID 묶음에 필요한 파일이 없습니다: {name}")
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
            raise FileNotFoundError(f"R3GRID 묶음 입력이 없습니다: {name}")
    sha = lambda p: hashlib.sha256((PROJECT / p).read_bytes()).hexdigest()  # noqa: E731
    lock = json.loads((PROJECT / "config/arena3_v1.json").read_text())
    manifest = {"kind": KIND, "experiment": EXPERIMENT, "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **original.git_state(), "files": {}, "source_models_included": False, "lock_sha256": sha("config/arena3_v1.json"),
                "platoon0_config_sha256": sha("config/explore_platoon0_v1.json"), "plan_sha256": sha(PREREG), "r3grid_config_sha256": sha("config/explore_r3grid_v1.json"), "environment_fingerprint": lock["environment"]["fingerprint"], "notebook_sha256": notebook_sha}
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".r3grid_package_", suffix=".zip", dir=out.parent)
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
