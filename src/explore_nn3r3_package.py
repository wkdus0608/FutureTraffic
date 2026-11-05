"""NN3R3 Colab 묶음(FutureTraffic_NN3R3.zip)과 노트북 6개(라운드 3 조각 4개 + 판정 + 12개 지도)를 만든다.

이전 묶음·노트북·결과는 건드리지 않는다. 묶기만 하고 SUMO는 돌리지 않는다. 실험 모델·결과는 넣지 않는다(모델은 Drive의 NN3 결과를 해시로 대조해 쓴다).
12개 지도 자료는 Drive의 FutureTraffic_network12.zip을 그대로 쓴다.
  .venv/bin/python src/explore_nn3r3_package.py            # 노트북을 만들고 묶는다
  .venv/bin/python src/explore_nn3r3_package.py --verify <zip>
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

import explore_nn12_package as nn12_pkg
import explore_nn3r2_package as prev
from explore_nn12_package import NET12_CELL, PACKAGES_CELL
from explore_weight_package import KIND, _safe_name, validate_archive

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results/colab/FutureTraffic_NN3R3.zip"
SHARDS = 4
NOTEBOOKS = {**{f"R3_shard{s}": PROJECT / f"notebooks/NN3_P_R3_shard{s}.ipynb" for s in range(SHARDS)}, "judge": PROJECT / "notebooks/NN3_P_R3_judge.ipynb", "map": PROJECT / "notebooks/NN3_12map.ipynb"}
EXPERIMENT = "NN3R3_V1"
PREREG = "results/explore/nn3r3_20261105/사전등록.md"
REFERENCE = "results/explore/nn3r3_20261105/smoke_reference/check.json"
EXTRA_FILES = (PREREG, REFERENCE, *prev.EXTRA_FILES, *nn12_pkg.EXTRA_FILES)
REQUIRED_FILES = ("src/explore_nn3r3.py", "src/explore_nn3r3_package.py", "src/explore_nn3map.py", "config/explore_nn3r3_v1.json", "tests/test_explore_nn3r3.py", "tests/test_explore_nn3map.py", *prev.REQUIRED_FILES, *nn12_pkg.REQUIRED_FILES, *(p.relative_to(PROJECT).as_posix() for p in NOTEBOOKS.values()))
PROTECTED_NAMES = {*prev.PROTECTED_NAMES, "FutureTraffic_NN3R2.zip"}

ZIP_CELL = prev.ZIP_CELL.replace('"NN3R2_V1", "NN3R2 전용 ZIP이 아닙니다"', '"NN3R3_V1", "NN3R3 전용 ZIP이 아닙니다"').replace("nn3r2_verified_package", "nn3r3_verified_package")
assert '"NN3R3_V1"' in ZIP_CELL and "plan_sha256" in ZIP_CELL, "ZIP 칸 바꾸기 실패"

ROUND_INTRO = """# NN3 라운드 3 — 조각 {shard} (네 조각을 런타임 하나씩)

라운드 2 정책(N5 + 학생 2개)이 직접 운전하며 고를 수 있는 순간의 25%에서 허용 녹색을 모두 같은 미래로 300초 굴려 비교하고, 학생을 한 번 더 쌓습니다.
새 자료 120판에 앞 라운드 자료를 합친 세 가지(120·240·300판)의 효과를 한 시간 결과로 직접 비교합니다. 같은 상황의 고정신호도 함께 잽니다.
방법과 읽는 법은 `results/explore/nn3r3_20261105/사전등록.md`에 고정했습니다.

`shard0`~`shard3`을 **각각 런타임 하나씩** 씁니다. **동시에 시작해도 됩니다.** 조각 0이 실행 전 점검(약 10분)을 끝내면 나머지가 이어서 돕니다.

1. 런타임 유형 **CPU + 높은 RAM(CPU 8개)** 을 고르고 **런타임 → 모두 실행**을 누르세요. GPU는 필요 없습니다.
2. 이 노트북 하나가 끝까지 알아서 돕니다: 점검 → 자료 30판 → 다른 조각 기다림 → 학생 학습 → 거르기 → 확인. 라운드 2가 약 13시간 걸렸습니다.
3. 끊기면 그 노트북만 다시 **모두 실행**하세요. 끝난 부분은 건너뜁니다. 다른 조각은 기다립니다.
4. 네 조각이 모두 끝나면 `NN3_P_R3_judge`를 실행하고, 이어서 `NN3_12map`을 **다시** 모두 실행한 뒤 "끝났어"라고 알려 주세요. 결과: Drive `FutureTraffic/explore_nn3/r3/`.

12개 지도 평가(`NN3_12map`)는 다른 노트북입니다. 런타임이 하나 더 있으면 지금 함께 돌려도 라운드 3에 영향을 주지 않습니다.
"""

COMMON_SETUP = '''from google.colab import drive
from pathlib import Path
import os, sys, json, time

drive.mount("/content/drive")
DRIVE = Path("/content/drive/MyDrive/FutureTraffic")
ARCHIVE = DRIVE / "FutureTraffic_NN3R3.zip"
WORKSPACE = Path("/content/FutureTraffic_NN3R3")
N3 = DRIVE / "explore_nn3"
PREP, R1, R2, OUT = N3 / "prep", N3 / "r1", N3 / "r2", N3 / "r3"       # prep·r1·r2는 읽기만(해시 대조)
MIN_CPUS = 8
for path in (ARCHIVE, PREP / "start.json", R1 / "students" / "h300w25" / "seed_42" / "model.pt", R2 / "students" / "done.json", R2 / "students" / "pool" / "seed_42" / "model.pt", OUT / "status"):
    assert path.exists(), f"필요한 파일·폴더가 없습니다: {path}"
CPUS = os.cpu_count() or 1
print("CPU 개수:", CPUS, "| 결과 폴더:", OUT)
assert CPUS >= MIN_CPUS, (f"CPU가 {CPUS}개입니다. 런타임 → 런타임 유형 변경에서 CPU와 '높은 RAM'을 고른 뒤 다시 실행하세요.")'''

TESTS_CELL = '''os.chdir(WORKSPACE)
sys.path.insert(0, str(WORKSPACE / "src"))
import arena3_colab as colab
from IPython.display import display, Markdown

colab.run_live(["-m", "unittest", "discover", "-s", "tests", "-p", "test_explore_nn3r3.py"], WORKSPACE)
print("실행 전 검사(SUMO 없이) 통과")'''

CHECK_CELL = '''# 실행 전 점검: 조각 0이 돌리고(약 10분), 다른 조각은 결과를 기다립니다. 통과해야 본 실행으로 갑니다.
SHARD = {shard}
CHECK = OUT / "check"
result = CHECK / "check.json"
if SHARD == 0 and not (colab.read_json(result) or {{}}).get("ok"):
    colab.run_live(["src/explore_nn3r3.py", "check", "--prep", str(PREP), "--r1-students", str(R1 / "students"), "--r2-students", str(R2 / "students"), "--out", str(CHECK), "--work", "/content/n3c", "--workers", str(CPUS)], WORKSPACE, check=False)
while not result.exists():
    print("조각 0의 점검을 기다립니다."); time.sleep(60)
r = colab.read_json(result)
assert r and r.get("ok"), f"점검 실패 — 본 실행을 하지 않습니다. 이 칸의 출력을 저에게 보내 주세요: {{r}}"
print("점검 통과 | 맥과 같음:", r.get("same_as_mac"), "| 굴림 1회 평균 초:", round(r["rollout_seconds"]["mean"], 1), "| 앞 정책이 N5와 다른 결정:", round(r["change_share"], 3))'''

ROUND_RUN_CELL = '''# 자료 → 학생 → 거르기 → 확인. 백그라운드로 돌아 셀을 멈춰도 계속 돕니다. 끊기면 처음부터 다시 모두 실행하세요.
LOGS = OUT / "logs" / f"shard{{SHARD}}"

def describe(name):
    s = colab.read_json(OUT / "data" / "status" / f"collect_shard{{SHARD}}.json") or {{}}
    stages = [x for x in ("data", "screen", "final") if (OUT / "status" / f"{{x}}_shard{{SHARD}}.json").exists()]
    return f"자료 {{s.get('done', 0)}}/{{s.get('total', 30)}}판, 실패 {{len(s.get('failed', []))}}, 끝난 단계 {{stages}}"

pids = colab.launch({{f"nn3r3_shard{{SHARD}}": ["src/explore_nn3r3.py", "round", "--prep", str(PREP), "--r1", str(R1), "--r2", str(R2), "--out", str(OUT), "--work", "/content/n3r", "--shard", str(SHARD), "--shards", "{shards}", "--workers", str(CPUS)]}}, WORKSPACE, LOGS)
code = colab.watch(pids, LOGS, 300, describe)
assert code == 0, "오류가 났습니다. 위 로그를 저에게 보내 주세요(다시 실행하면 끝난 부분은 건너뜁니다)."
print("이 조각 끝. 네 조각이 모두 끝나면 NN3_P_R3_judge를 실행하세요.")'''

JUDGE_INTRO = """# NN3 라운드 3 판정

네 조각이 끝난 뒤 **모두 실행**하세요. SUMO는 돌리지 않으며 CPU 수는 상관없습니다. 작은 파일을 많이 읽어 10~30분쯤 걸릴 수 있습니다.
읽는 법은 `results/explore/nn3r3_20261105/사전등록.md`에 고정했습니다. 숫자는 맥에서 다시 계산합니다.
끝나면 `NN3_12map`을 다시 모두 실행하세요. 채택이면 라운드 3 정책도 12개 지도에서 잽니다.
"""

JUDGE_SETUP = '''from google.colab import drive
from pathlib import Path
import os, sys, json

drive.mount("/content/drive")
DRIVE = Path("/content/drive/MyDrive/FutureTraffic")
ARCHIVE = DRIVE / "FutureTraffic_NN3R3.zip"
OUT = DRIVE / "explore_nn3" / "r3"
WORKSPACE = Path("/content/FutureTraffic_NN3R3")
assert ARCHIVE.exists() and (OUT / "students" / "done.json").exists(), "필요한 파일이 없습니다."'''

JUDGE_CELL = '''# 판정(SUMO 없음). 아직 안 끝났으면 남은 부분을 알려 줍니다.
code = colab.run_live(["src/explore_nn3r3.py", "judge", "--r3", str(OUT), "--out", str(OUT / "judge")], WORKSPACE, check=False)
if (OUT / "judge" / "summary.md").exists():
    display(Markdown((OUT / "judge" / "summary.md").read_text()))
print("판정 코드:", code, "(0이면 라운드 3이 끝났습니다)")'''

MAP_INTRO = """# NN3 12개 지도 평가 — 라운드 정책이 실제 신호가 있는 지도에도 남나

라운드 2 정책(N5 + 학생 2개)을 NN12와 같은 46조각 × seed 5개(230판)에서 돌려 N5와 짝으로 비교합니다. 학습은 없고, 라운드 3 후보 고르기에도 쓰지 않습니다.
라운드 3이 채택되면 라운드 3 정책(230판)도 잽니다. 방법과 읽는 법은 `results/explore/nn3r3_20261105/사전등록.md`에 고정했습니다.

**이 노트북은 두 번 실행합니다.**

1. **지금, 또는 런타임이 비는 대로:** 런타임 유형 **CPU + 높은 RAM(CPU 8개)** 을 고르고 **런타임 → 모두 실행**. 묶음 검증 → 12개 지도 자료 → 시험 → 점검(약 5분) → 라운드 2 정책 230판(약 1~2시간, 어림) → 임시 판정.
2. **`NN3_P_R3_judge`가 끝난 뒤:** 다시 **모두 실행**. 끝난 판은 건너뛰고, 채택이면 라운드 3 정책 230판을 돌린 뒤 최종 판정을 냅니다. 미채택이면 라운드 2 정책만으로 판정합니다.

끊기면 다시 **모두 실행**하세요. 끝나면 "끝났어"라고 알려 주세요. 결과: Drive `FutureTraffic/explore_nn3/map/`.
"""

MAP_SETUP = '''from google.colab import drive
from pathlib import Path
import os, sys, json, zipfile, hashlib

drive.mount("/content/drive")
DRIVE = Path("/content/drive/MyDrive/FutureTraffic")
ARCHIVE = DRIVE / "FutureTraffic_NN3R3.zip"
NET12_ZIP = DRIVE / "FutureTraffic_network12.zip"
WORKSPACE = Path("/content/FutureTraffic_NN3R3")
N3 = DRIVE / "explore_nn3"
OUT = N3 / "map"
NN12, JOINT2 = DRIVE / "explore_nn12", DRIVE / "explore_joint2"       # 읽기만
R3_JUDGE = N3 / "r3" / "judge" / "judge.json"
MIN_CPUS = 8
for path in (ARCHIVE, NET12_ZIP, N3 / "r2" / "students" / "pool" / "seed_42" / "model.pt", NN12 / "N5" / "runs", NN12 / "TM4" / "runs", JOINT2, OUT / "R2W" / "runs"):
    assert path.exists(), f"필요한 파일·폴더가 없습니다: {path}"
CPUS = os.cpu_count() or 1
print("CPU 개수:", CPUS, "| 결과 폴더:", OUT)
assert CPUS >= MIN_CPUS, (f"CPU가 {CPUS}개입니다. 런타임 → 런타임 유형 변경에서 CPU와 '높은 RAM'을 고른 뒤 다시 실행하세요.")'''

MAP_TESTS_CELL = '''sys.path.insert(0, str(WORKSPACE / "src"))
import arena3_colab as colab
from IPython.display import display, Markdown

colab.run_live(["-m", "unittest", "discover", "-s", "tests", "-p", "test_explore_nn3map.py"], WORKSPACE)
print("실행 전 검사(SUMO 없이) 통과")'''

MAP_CHECK_CELL = '''# 실행 전 점검(약 5분): 학생을 뺀 이번 코드가 저장된 NN12 N5 판과 소수점까지 같은지, 라운드 2 정책 한 판이 건강한지 봅니다.
CHECK = OUT / "check"
result = CHECK / "check.json"
if not (colab.read_json(result) or {}).get("ok"):
    colab.run_live(["src/explore_nn3map.py", "check", "--n3", str(N3), "--stored-n5", str(NN12 / "N5"), "--out", str(CHECK), "--work", "/content/n3mc"], WORKSPACE, check=False)
r = colab.read_json(result)
assert r and r.get("ok"), f"점검 실패 — 본 실행을 하지 않습니다. 이 칸의 출력을 저에게 보내 주세요: {r and r.get('checks')}"
print("점검 통과:", r["checks"], "| 판 초", r["seconds"])'''

MAP_RUN_CELL = '''# 본 실행(남은 판만). 라운드 3 판정이 채택이면 라운드 3 정책도 돕니다. 백그라운드로 돌아 셀을 멈춰도 계속 돕니다.
judge = colab.read_json(R3_JUDGE) or {}
adopted = judge.get("status") == "done" and judge["reading"]["adopted"]
policies = ["R2W", *(["R3W"] if adopted else [])]
print("이번에 돌릴 정책:", policies, "| 라운드 3 판정:", judge.get("status", "아직 없음"), "| 채택:", adopted)

for policy in policies:
    logs = OUT / "logs" / policy
    def describe(name, folder=OUT / policy / "runs"):
        done = len(list(folder.glob("*/metrics.json")))
        return f"끝난 판 {done}/230, 실패 기록 {len(list(folder.glob('*.failed.json')))}"
    args = ["src/explore_nn3map.py", "run", "--policy", policy, "--n3", str(N3), "--out", str(OUT / policy), "--work", f"/content/n3m_{policy}", "--workers", str(CPUS)]
    if policy == "R3W":
        args += ["--r3-judge", str(R3_JUDGE)]
    pids = colab.launch({f"nn3map_{policy}": args}, WORKSPACE, logs)
    code = colab.watch(pids, logs, 300, describe)
    assert code == 0, "본 실행에서 오류가 났습니다. 위 로그를 저에게 보내 주세요(다시 실행하면 끝난 판은 건너뜁니다)."
    print(policy, "끝.", describe(policy))'''

MAP_JUDGE_CELL = '''# 판정(SUMO 없음). 작은 파일을 많이 읽어 10~30분쯤 걸릴 수 있습니다.
args = ["src/explore_nn3map.py", "judge", "--root", str(OUT), "--nn12", str(NN12), "--joint2", str(JOINT2), "--out", str(OUT / "judge")]
if R3_JUDGE.exists():
    args += ["--r3-judge", str(R3_JUDGE)]
code = colab.run_live(args, WORKSPACE, check=False)
if (OUT / "judge" / "summary.md").exists():
    display(Markdown((OUT / "judge" / "summary.md").read_text()))
print("판정 코드:", code, "(0이면 끝)")
if not R3_JUDGE.exists():
    print("라운드 3 판정이 아직 없어 임시 판정입니다. NN3_P_R3_judge가 끝난 뒤 이 노트북을 다시 모두 실행하세요.")'''


def _cell(kind: str, text: str) -> dict:
    return prev._cell(kind, text)


def notebook(name: str) -> dict:
    if name == "judge":
        return prev._wrap([_cell("markdown", JUDGE_INTRO)] + [_cell("code", c) for c in (JUDGE_SETUP, ZIP_CELL, PACKAGES_CELL, TESTS_CELL, JUDGE_CELL)])
    if name == "map":
        return prev._wrap([_cell("markdown", MAP_INTRO)] + [_cell("code", c) for c in (MAP_SETUP, ZIP_CELL, NET12_CELL, PACKAGES_CELL, MAP_TESTS_CELL, MAP_CHECK_CELL, MAP_RUN_CELL, MAP_JUDGE_CELL)])
    shard = int(name.removeprefix("R3_shard"))
    steps = (COMMON_SETUP, ZIP_CELL, PACKAGES_CELL, TESTS_CELL, CHECK_CELL.format(shard=shard), ROUND_RUN_CELL.format(shards=SHARDS))
    return prev._wrap([_cell("markdown", ROUND_INTRO.format(shard=shard))] + [_cell("code", c) for c in steps])


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
        raise ValueError("NN3R3 묶음이 아니거나 모델·결과가 들어 있습니다.")
    for name in REQUIRED_FILES + EXTRA_FILES:
        if name not in manifest["files"]:
            raise ValueError(f"NN3R3 묶음에 필요한 파일이 없습니다: {name}")
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
            raise FileNotFoundError(f"NN3R3 묶음 입력이 없습니다: {name}")
    sha = lambda p: hashlib.sha256((PROJECT / p).read_bytes()).hexdigest()  # noqa: E731
    lock = json.loads((PROJECT / "config/arena3_v1.json").read_text())
    manifest = {"kind": KIND, "experiment": EXPERIMENT, "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **original.git_state(), "files": {}, "source_models_included": False, "lock_sha256": sha("config/arena3_v1.json"),
                "platoon0_config_sha256": sha("config/explore_platoon0_v1.json"), "plan_sha256": sha(PREREG), "nn3r3_config_sha256": sha("config/explore_nn3r3_v1.json"), "environment_fingerprint": lock["environment"]["fingerprint"], "notebook_sha256": notebook_sha}
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".nn3r3_package_", suffix=".zip", dir=out.parent)
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
