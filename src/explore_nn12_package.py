"""NN12 정정 묶음(FutureTraffic_NN12_fix1.zip)과 노트북 3개(N5·TM4 + 판정)를 만든다. 사전 등록 정정 기록: 실패한 23시 20판만 고친 코드로 돌린다.

처음 묶음(FutureTraffic_NN12.zip)·노트북 5개는 그대로 두고 쓰지 않는다. 이전 묶음·노트북·결과는 건드리지 않는다. 묶기만 하고 SUMO는 돌리지 않는다. 모델·결과는 넣지 않는다(모델은 Drive의 NN1 결과를 해시로 대조해 읽는다).
12개 지도 자료는 Drive의 FutureTraffic_network12.zip을 그대로 쓴다.
  .venv/bin/python src/explore_nn12_package.py            # 노트북을 만들고 묶는다
  .venv/bin/python src/explore_nn12_package.py --verify <zip>
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

from explore_nn1_package import PACKAGES_CELL, PROTECTED_NAMES as NN1_PROTECTED, ZIP_CELL as NN1_ZIP_CELL
from explore_weight_package import KIND, _safe_name, validate_archive

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results/colab/FutureTraffic_NN12_fix1.zip"
POLICIES = ("N5", "TM4")
NOTEBOOKS = {**{v: PROJECT / f"notebooks/NN12_{v}_fix1.ipynb" for v in POLICIES}, "judge": PROJECT / "notebooks/NN12_judge_fix1.ipynb"}
EXPERIMENT = "NN12_V1"
PREREG = "results/explore/nn12_20261103/사전등록.md"
EXTRA_FILES = (PREREG, "results/explore/nn1_20261102/사전등록.md", "results/explore/label0_20261102/사전등록.md", "results/explore/platoon1c_20261101/사전등록.md", "results/explore/platoon1_20261101/사전등록.md",
               "results/explore/platoon0_20261031/사전등록.md", "results/explore/platoon_base_20261101/측정계획.md")
REQUIRED_FILES = ("src/explore_nn12.py", "src/explore_nn12_package.py", "src/explore_nn1.py", "src/explore_joint2.py", "src/explore_coord12.py", "src/explore_pred.py", "src/explore_platoon0.py", "src/explore_platoon1.py",
                  "src/explore_weight_core.py", "src/arena3_common.py", "src/arena3_colab.py", "src/network_eval.py", "src/network_rl.py", "src/focused_platoon.py", "config/explore_nn12_v1.json", "config/explore_nn1_v1.json",
                  "config/east_platoon.json", "config/network12_capacity.json", "config/network12_environment.json", "config/network12_windows.json", "data/derived/east_upstream_204822.csv", "tests/test_explore_nn12.py",
                  *(p.relative_to(PROJECT).as_posix() for p in NOTEBOOKS.values()))
PROTECTED_NAMES = {*NN1_PROTECTED, "FutureTraffic_NN1.zip", "FutureTraffic_NN12.zip"}

ZIP_CELL = NN1_ZIP_CELL.replace('"NN1_V1", "NN1 전용 ZIP이 아닙니다"', '"NN12_V1", "NN12 전용 ZIP이 아닙니다"').replace("nn1_verified_package", "nn12_verified_package")
assert '"NN12_V1"' in ZIP_CELL and "plan_sha256" in ZIP_CELL, "ZIP 칸 바꾸기 실패"

SPEC = {"N5": {"title": "N5(무리 세계에서 배운 신경망)", "models": "explore_nn1/P/aug/models/r1"},
        "TM4": {"title": "TM4(N5의 짝 계획기, 이웃 시간표 + 좌회전 4배)", "models": None}}

INTRO = """# NN12 {policy} 정정 — 못 돈 23시 조각만 다시: {title}

처음 실행에서 이 정책의 23시 조각 10판이 기록용 대조 코드의 오류로 멈췄습니다. 끝난 220판은 그대로 쓰고, **멈춘 10판만** 고친 코드로 돌립니다. 판단·입력·평가 코드는 바뀌지 않았습니다.
자세한 내용은 `results/explore/nn12_20261103/사전등록.md`의 정정 기록에 있습니다.

정정 노트북은 둘(N5·TM4)과 판정 하나입니다. **둘을 런타임 하나씩 따로** 씁니다. 순서는 상관없고 동시에 돌려도 됩니다. Rn·M4는 다시 돌리지 않습니다.

1. 런타임 유형 **CPU + 높은 RAM(CPU 8개)** 을 고르고 **런타임 → 모두 실행**을 누르세요. GPU는 필요 없습니다. Drive 접근을 허용하세요.
2. 순서: 묶음 검증 → 12개 지도 자료 → 패키지 → 시험 → 끝난 판 하나가 소수점까지 같은지 점검(약 3분) → 본 실행(10판, 약 5분, 어림).
3. 끊기면 다시 **모두 실행**하세요. 끝난 판은 건너뜁니다.
4. 둘이 끝나면 `NN12_judge_fix1`을 실행하고 "끝났어"라고 알려 주세요. 처음 노트북(`NN12_N5`, `NN12_TM4`, `NN12_judge`)은 쓰지 마세요.
"""

SETUP = '''from google.colab import drive
from pathlib import Path
import os, sys, json

drive.mount("/content/drive")
POLICY = "{policy}"
DRIVE = Path("/content/drive/MyDrive/FutureTraffic")
ARCHIVE = DRIVE / "FutureTraffic_NN12_fix1.zip"
NET12_ZIP = DRIVE / "FutureTraffic_network12.zip"
MODELS = {models}                                       # 신경망 모델(읽기만, 해시 대조)
OUT = DRIVE / "explore_nn12" / POLICY
WORKSPACE = Path("/content/FutureTraffic_NN12_fix1")
MIN_CPUS = 8
for path in (ARCHIVE, NET12_ZIP, OUT / "runs", *([MODELS / "seed_42" / "model.pt"] if MODELS else [])):
    assert path.exists(), f"필요한 파일이 없습니다: {{path}}"
CPUS = os.cpu_count() or 1
print("CPU 개수:", CPUS, "| 정책:", POLICY, "| 결과 폴더:", OUT)
assert CPUS >= MIN_CPUS, (f"CPU가 {{CPUS}}개입니다. 런타임 → 런타임 유형 변경에서 CPU와 '높은 RAM'을 고른 뒤 다시 실행하세요.")'''

NET12_CELL = '''# 12개 지도 자료(기존 묶음 그대로). 코드는 들어 있지 않고 수요·지도만 있습니다.
with zipfile.ZipFile(NET12_ZIP) as z:
    z.extractall(WORKSPACE)
m12 = json.loads((WORKSPACE / "network12_manifest.json").read_text())
bad = [p for p, h in m12["files"].items() if hashlib.sha256((WORKSPACE / p).read_bytes()).hexdigest() != h]
assert not bad, f"12개 지도 자료 해시 불일치: {bad[:3]}"
assert m12["network12_fingerprint"] == "711875f6c939", "12개 지도 지문이 다릅니다"
print(f"12개 지도 자료: 파일 {len(m12['files'])}개 | 배경 {m12['background']:.0%} | 유턴 {m12['uturn']}")
os.chdir(WORKSPACE)'''

TESTS_CELL = '''sys.path.insert(0, str(WORKSPACE / "src"))
import arena3_colab as colab
from IPython.display import display, Markdown

colab.run_live(["-m", "unittest", "discover", "-s", "tests", "-p", "test_explore_nn12.py"], WORKSPACE)
print("실행 전 검사(SUMO 없이) 통과")'''

RECHECK_CELL = '''# 1) 실행 전 점검(약 3분): 이미 끝난 판 하나를 고친 코드로 다시 돌려 저장된 결과와 소수점까지 같은지 봅니다. 결과 값은 보여 주지 않습니다.
CHECK = OUT / "check_fix1"
previous = colab.read_json(CHECK / f"recheck_{POLICY}.json")
if previous and previous.get("ok"):
    print("이전 점검이 통과해 건너뜁니다.")
else:
    args = ["src/explore_nn12.py", "recheck", "--policy", POLICY, "--out", str(OUT), "--work", "/content/n12c"]
    if MODELS:
        args += ["--models", str(MODELS)]
    colab.run_live(args, WORKSPACE, check=False)
r = colab.read_json(CHECK / f"recheck_{POLICY}.json")
assert r and r.get("ok"), f"점검 실패 — 본 실행을 하지 않습니다. 이 칸의 출력을 저에게 보내 주세요: {r and r.get('same')}"
print("점검 통과:", r["same"], "| 판 초", r["seconds"])'''

RUN_CELL = '''# 2) 본 실행(남은 판만). 끝난 220판은 건너뛰고 23시 조각 10판을 돌립니다. 백그라운드로 돌아 셀을 멈춰도 계속 돕니다.
#    런타임이 끊기면 처음 다섯 칸 → 이 칸을 다시 실행하세요.
assert "colab" in globals(), "처음 다섯 칸(1~5)을 먼저 실행하세요. 런타임이 끊기면 변수와 작업 폴더가 사라집니다."
LOGS = OUT / "logs_fix1"

def describe(name):
    done = len(list((OUT / "runs").glob("*/metrics.json")))
    still = [p for p in (OUT / "runs").glob("*.failed.json") if not (OUT / "runs" / p.name.replace(".failed.json", "") / "metrics.json").exists()]
    return f"끝난 판 {done}/230, 아직 못 끝낸 실패 {len(still)}"

args = ["src/explore_nn12.py", "run", "--policy", POLICY, "--out", str(OUT), "--work", "/content/n12", "--workers", str(CPUS)]
if MODELS:
    args += ["--models", str(MODELS)]
pids = colab.launch({f"nn12_{POLICY}": args}, WORKSPACE, LOGS)
code = colab.watch(pids, LOGS, 60, describe)
assert code == 0, "본 실행에서 오류가 났습니다. 위 로그를 저에게 보내 주세요(다시 실행하면 끝난 판은 건너뜁니다)."
print("본 실행 끝.", describe(POLICY), "| N5·TM4가 모두 끝나면 NN12_judge_fix1을 실행하세요.")'''

JUDGE_INTRO = """# NN12 판정 — 신경망만 모델의 12개 지도 결과

정정 노트북 둘(`NN12_N5_fix1`, `NN12_TM4_fix1`)이 끝난 뒤 **모두 실행**하세요. 처음 노트북 `NN12_judge`는 쓰지 마세요. Rn·M4의 끝난 판은 그대로 읽습니다. SUMO는 돌리지 않으며 CPU 수는 상관없습니다. 작은 파일을 많이 읽어 10~30분쯤 걸릴 수 있습니다.
읽는 법은 `results/explore/nn12_20261103/사전등록.md`에 고정했습니다. 숫자는 맥에서 다시 계산합니다.
"""

JUDGE_SETUP = '''from google.colab import drive
from pathlib import Path
import os, sys, json

drive.mount("/content/drive")
DRIVE = Path("/content/drive/MyDrive/FutureTraffic")
ARCHIVE = DRIVE / "FutureTraffic_NN12_fix1.zip"
ROOT_OUT = DRIVE / "explore_nn12"
WORKSPACE = Path("/content/FutureTraffic_NN12_fix1")
assert ARCHIVE.exists() and (DRIVE / "explore_joint2" / "9d3b7504064b").exists(), "필요한 파일이 없습니다."'''

JUDGE_CD = 'os.chdir(WORKSPACE)'

JUDGE_CELL = '''# 판정(SUMO 없음). 아직 안 끝난 정책이 있으면 남은 판 수가 나옵니다.
code = colab.run_live(["src/explore_nn12.py", "judge", "--root", str(ROOT_OUT), "--joint2", str(DRIVE / "explore_joint2"), "--pred", str(DRIVE / "explore_pred" / "network12"),
                       "--b12", str(DRIVE / "explore_12map"), "--out", str(ROOT_OUT / "judge")], WORKSPACE, check=False)
if (ROOT_OUT / "judge" / "summary.md").exists():
    display(Markdown((ROOT_OUT / "judge" / "summary.md").read_text()))
print("판정 코드:", code, "(0이면 정책 4개가 모두 끝났습니다)")'''


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
        return _wrap([_cell("markdown", JUDGE_INTRO)] + [_cell("code", c) for c in (JUDGE_SETUP, ZIP_CELL, JUDGE_CD, PACKAGES_CELL, TESTS_CELL, JUDGE_CELL)])
    spec = SPEC[name]
    models = f'DRIVE / "{spec["models"]}"' if spec["models"] else "None"
    intro = INTRO.format(policy=name, title=spec["title"])
    steps = (SETUP.format(policy=name, models=models), ZIP_CELL, NET12_CELL, PACKAGES_CELL, TESTS_CELL, RECHECK_CELL, RUN_CELL)
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


def validate_nn12_archive(archive: Path) -> dict:
    manifest = validate_archive(archive)
    if manifest.get("experiment") != EXPERIMENT or manifest.get("source_models_included") is not False:
        raise ValueError("NN12 묶음이 아니거나 모델·결과가 들어 있습니다.")
    for name in REQUIRED_FILES + EXTRA_FILES:
        if name not in manifest["files"]:
            raise ValueError(f"NN12 묶음에 필요한 파일이 없습니다: {name}")
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
            raise FileNotFoundError(f"NN12 묶음 입력이 없습니다: {name}")
    sha = lambda p: hashlib.sha256((PROJECT / p).read_bytes()).hexdigest()  # noqa: E731
    lock = json.loads((PROJECT / "config/arena3_v1.json").read_text())
    manifest = {"kind": KIND, "experiment": EXPERIMENT, "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **original.git_state(), "files": {},
                "source_models_included": False, "lock_sha256": sha("config/arena3_v1.json"), "platoon0_config_sha256": sha("config/explore_platoon0_v1.json"), "plan_sha256": sha(PREREG),
                "nn12_config_sha256": sha("config/explore_nn12_v1.json"), "environment_fingerprint": lock["environment"]["fingerprint"], "notebook_sha256": notebook_sha}
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".nn12_package_", suffix=".zip", dir=out.parent)
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
        validate_nn12_archive(Path(temporary))
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
        manifest = validate_nn12_archive(args.verify)
        print(json.dumps({"verified": str(args.verify), "file_count": len(manifest["files"])}, ensure_ascii=False))
    else:
        print(json.dumps(build_package(args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
