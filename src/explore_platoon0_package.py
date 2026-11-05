"""PLATOON0 Colab 묶음(FutureTraffic_PLATOON0.zip)과 노트북(notebooks/PLATOON0_Colab.ipynb)을 만든다.

이전 묶음·노트북·결과는 건드리지 않는다. 묶기만 하고 SUMO는 돌리지 않는다. 모델·결과는 넣지 않는다.
  .venv/bin/python src/explore_platoon0_package.py            # 노트북을 만들고 묶는다
  .venv/bin/python src/explore_platoon0_package.py --verify <zip>
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
from arena3_pi_package import PROTECTED_NAMES as PI_PROTECTED

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results/colab/FutureTraffic_PLATOON0.zip"
NOTEBOOK = PROJECT / "notebooks/PLATOON0_Colab.ipynb"
EXPERIMENT = "PLATOON0_V1"
EXTRA_FILES = ("docs/PLATOON0_실행안내.md", "results/explore/platoon0_20261031/사전등록.md")
REQUIRED_FILES = ("src/explore_platoon0.py", "src/explore_platoon0_report.py", "src/explore_platoon0_package.py",
                  "src/arena3_common.py", "src/arena3_referee.py", "src/arena3_colab.py", "src/explore_ceiling.py",
                  "src/explore_pred.py", "src/explore_weight_core.py", "src/explore_weight_episode.py",
                  "src/focused_platoon.py", "config/arena3_v1.json", "config/explore_platoon0_v1.json",
                  "config/east_platoon.json", "data/derived/east_upstream_204822.csv",
                  "tests/test_explore_platoon0.py", "tests/test_arena3.py", "notebooks/PLATOON0_Colab.ipynb")
PROTECTED_NAMES = {*PI_PROTECTED, "FutureTraffic_ARENA3_PI.zip"}

INTRO = """# 0단계 PLATOON0 — 동쪽 무리 도착 경기장의 여지 재기 (학습 없음)

지금 단독 교차로 경기장은 차가 15분 안에서 무작위로 옵니다. 그래서 앞으로 올 차를 알아도 쓸모가 없었습니다(111·238절).
이 노트북은 같은 차의 동쪽 출발 시각만 이웃 신호(204822)의 실제 녹색에 맞춘 경기장을 만듭니다. 그 경기장에서 계획기가
앞으로 올 차를 정확히 알거나(OM4) 이웃 시간표로 예측하면(TM4) M4보다 얼마나 줄일 수 있는지 잽니다.
판정 기준은 실행 전에 고정했습니다(`results/explore/platoon0_20261031/사전등록.md`).

1. 런타임 유형 **CPU + 높은 RAM(CPU 8개)** 을 고르고 **런타임 → 모두 실행**을 누르세요. Drive 접근을 허용하세요.
2. 순서: 묶음 검증 → 패키지 → 시험 → 작은 실행(약 10~15분: 차량 파일 84개 점검 + 6판) → 본 실행(504판, 약 4~5시간) → 판정.
3. 본 실행은 백그라운드로 돕니다. 셀을 멈추거나 창을 닫아도 런타임이 살아 있으면 계속 돕니다.
   끊기면 다시 **모두 실행**하세요. 끝난 판은 건너뜁니다.
4. 마지막 칸에 판정이 나오면 "끝났어"라고 알려 주세요. 결과: Drive `FutureTraffic/explore_platoon0/`.
"""

CELLS = [
    '''from google.colab import drive
from pathlib import Path
import os, sys, json

drive.mount("/content/drive")
DRIVE = Path("/content/drive/MyDrive/FutureTraffic")
ARCHIVE = DRIVE / "FutureTraffic_PLATOON0.zip"
REFERENCE = DRIVE / "arena3" / "referee" / "baseline"        # ARENA3 심판 기준선(무작위 세계 M4 재현 확인용, 읽기만)
OUT = DRIVE / "explore_platoon0"
WORKSPACE = Path("/content/FutureTraffic_PLATOON0")
MIN_CPUS = 8
for path in (ARCHIVE, REFERENCE / "runs" / "M4"):
    assert path.exists(), f"필요한 파일이 없습니다: {path}"
CPUS = os.cpu_count() or 1
print("CPU 개수:", CPUS, "| 결과 폴더:", OUT)
assert CPUS >= MIN_CPUS, (f"CPU가 {CPUS}개입니다. 런타임 → 런타임 유형 변경에서 CPU와 '높은 RAM'을 고른 뒤 다시 실행하세요. "
                          "(CPU 2개면 4배 오래 걸려 24시간 한도에 걸립니다.)")''',
    '''# ZIP 전체의 파일 목록·SHA-256을 확인한 뒤 검증된 전용 추출기를 사용합니다.
import hashlib, stat, types, zipfile
from pathlib import PurePosixPath

with zipfile.ZipFile(ARCHIVE) as archive:
    names = archive.namelist()
    assert len(names) == len(set(names)) and names.count("manifest.json") == 1, "ZIP 파일 목록 오류"
    package_manifest = json.loads(archive.read("manifest.json"))
    assert package_manifest.get("kind") == "WEIGHT1_PACKAGE_V1", "검증할 수 없는 ZIP 형식입니다"
    assert package_manifest.get("experiment") == "PLATOON0_V1", "PLATOON0 전용 ZIP이 아닙니다"
    assert package_manifest.get("source_models_included") is False, "모델·결과가 든 묶음은 쓰지 않습니다"
    expected = package_manifest["files"]
    assert expected and "manifest.json" not in expected and set(names) == set(expected) | {"manifest.json"}, "ZIP과 manifest 목록 불일치"
    for info in archive.infolist():
        name, path = info.filename, PurePosixPath(info.filename)
        assert path.parts and not path.is_absolute() and ".." not in path.parts and path.as_posix() == name, "잘못된 ZIP 경로"
        assert "\\\\" not in name and not info.is_dir() and not stat.S_ISLNK(info.external_attr >> 16), "잘못된 ZIP 파일 형식"
        if name != "manifest.json":
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected[name], f"SHA-256 불일치: {name}"
    source_code = archive.read("src/explore_weight_package.py").decode("utf-8")

bootstrap = types.ModuleType("platoon0_verified_package")
bootstrap.__file__ = str(WORKSPACE / "src/explore_weight_package.py")
exec(compile(source_code, bootstrap.__file__, "exec"), bootstrap.__dict__)
package_manifest = bootstrap.extract_verified(ARCHIVE, WORKSPACE)
print(f"검증·추출 완료: {len(package_manifest['files'])}개 파일 → {WORKSPACE} | 잠금 {package_manifest['lock_sha256'][:12]} "
      f"| PLATOON0 설정 {package_manifest['platoon0_config_sha256'][:12]} | 사전 등록 {package_manifest['prereg_sha256'][:12]}")''',
    '''import subprocess, importlib.metadata

os.environ["FOCUSED_RULESET"] = "pet1"
os.environ["FOCUSED_NORTH_EXT"] = "240"
# ARENA3 심판 기준선과 같은 평가 지문이 되도록 같은 패키지 버전을 씁니다.
EXPECTED = {"numpy": "2.1.3", "torch": "2.11.0+cpu", "stable-baselines3": "2.9.0",
            "sumo-rl": "1.4.5", "eclipse-sumo": "1.27.1"}

def installed_versions():
    versions = {}
    for name in EXPECTED:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions

before = installed_versions()
if before == EXPECTED:
    print("이전 실험과 패키지 버전이 같아 재설치를 건너뜁니다.")
else:
    print("이전 실험과 같은 패키지 버전을 준비합니다:", before)
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=WORKSPACE, check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "numpy==2.1.3", "stable-baselines3==2.9.0", "eclipse-sumo==1.27.1"],
                   cwd=WORKSPACE, check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "torch==2.11.0+cpu", "--index-url", "https://download.pytorch.org/whl/cpu"],
                   cwd=WORKSPACE, check=True)
after = installed_versions()
assert after == EXPECTED, f"패키지 버전이 이전 실험과 다릅니다(평가 지문이 달라져 기준선과 비교할 수 없어 멈춥니다): {after}"
print("실행 패키지 검증 통과:", after)''',
    '''sys.path.insert(0, str(WORKSPACE / "src"))
import arena3_colab as colab
from IPython.display import display, Markdown

colab.run_live(["-m", "unittest", "discover", "-s", "tests", "-p", "test_explore_platoon0.py"], WORKSPACE)
colab.run_live(["-m", "unittest", "discover", "-s", "tests", "-p", "test_arena3.py"], WORKSPACE)
print("실행 전 검사(SUMO 없이) 통과")''',
    '''# 1) 작은 실행(약 10~15분): 84상황의 무리 도착 차량 파일 점검(SUMO 없음) + 첫 상황 6판.
#    - 감싼 계획기(P0M4)가 M4를 소수점까지 그대로 재현하는가, 시간표·정확한 도착이 들어가는가,
#    - 무작위 세계 M4가 ARENA3 심판 기준선과 같은가. 결과는 이 노트북 전용 점검 폴더에만 씁니다.
SMOKE = OUT / "smoke"
previous = colab.read_json(SMOKE / "smoke.json")
if previous and previous.get("ok"):
    print("이전 작은 실행이 통과해 건너뜁니다.")
else:
    colab.run_live(["src/explore_platoon0.py", "smoke", "--out", str(SMOKE), "--work", "/content/p0_smoke",
                    "--workers", str(CPUS), "--reference", str(REFERENCE)], WORKSPACE, check=False)
s = colab.read_json(SMOKE / "smoke.json")
assert s and s.get("ok"), f"작은 실행 점검 실패 — 이 칸의 출력을 저에게 보내 주세요: {s and s.get('checks')}"
print("작은 실행 통과:", s["checks"])
print("무리 도착 세계 동쪽 차:", s["world"]["changed_P"], "| 첫 상황 차량당 지연(참고만, 판정 아님):", s["delay_s"])''',
    '''# 2) 본 실행(504판 = 무리 도착 4제어 × 84 + 무작위 2제어 × 84). 백그라운드로 돌아 셀을 멈춰도 계속 돕니다.
#    런타임이 끊기면 처음 네 칸 → 이 칸을 다시 실행하세요. 끝난 판은 건너뜁니다.
assert "colab" in globals(), "처음 네 칸(1~4)을 먼저 실행하세요. 런타임이 끊기면 변수와 작업 폴더가 사라집니다."
LOGS = OUT / "logs"

def describe(name):
    done = len(list((OUT / "runs").glob("*/*/*/cache.json")))
    failed = len(list((OUT / "runs").glob("*/*/*.failed.json")))
    return f"끝난 판 {done}/504, 실패 기록 {failed}"

pids = colab.launch({"platoon0_run": ["src/explore_platoon0.py", "run", "--out", str(OUT), "--work", "/content/p0",
                                      "--workers", str(CPUS)]}, WORKSPACE, LOGS)
code = colab.watch(pids, LOGS, 600, describe)
assert code == 0, "본 실행에서 오류가 났습니다. 위 로그를 저에게 보내 주세요(다시 실행하면 끝난 판은 건너뜁니다)."''',
    '''# 3) 판정(사전 등록 그대로): 무리 도착 세계 OM4 − M4 ≥ 2.0초이고 t95 하한 > 0 → 여지 있음, < 1.0초 → 없음, 그 밖 불분명.
colab.run_live(["src/explore_platoon0_report.py", "--out", str(OUT), "--reference", str(REFERENCE)], WORKSPACE, check=False)
rep = colab.read_json(OUT / "report.json")
print("판정:", rep and rep.get("verdict_ko"), "| 상태:", rep and rep.get("status"))
if (OUT / "summary.md").exists():
    display(Markdown((OUT / "summary.md").read_text()))''',
]


def notebook() -> dict:
    def cell(kind, text):
        out = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
        if kind == "code":
            out.update(execution_count=None, outputs=[])
        return out

    return {"cells": [cell("markdown", INTRO)] + [cell("code", c) for c in CELLS],
            "metadata": {"colab": {"provenance": [], "machine_shape": "hm"},
                         "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


def write_notebook(path: Path = NOTEBOOK) -> str:
    nb = notebook()
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            compile("".join(c["source"]), f"{path.name}[{i}]", "exec")
    text = json.dumps(nb, ensure_ascii=False, indent=1) + "\n"
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def validate_platoon0_archive(archive: Path) -> dict:
    manifest = validate_archive(archive)
    if manifest.get("experiment") != EXPERIMENT or manifest.get("source_models_included") is not False:
        raise ValueError("PLATOON0 묶음이 아니거나 모델·결과가 들어 있습니다.")
    for name in REQUIRED_FILES + EXTRA_FILES:
        if name not in manifest["files"]:
            raise ValueError(f"PLATOON0 묶음에 필요한 파일이 없습니다: {name}")
    return manifest


def build_package(out: Path = OUT) -> dict:
    import focused_package as original

    out = Path(out).resolve()
    if out.name in PROTECTED_NAMES:
        raise ValueError("이전 실험 묶음을 덮어쓸 수 없습니다.")
    original.check_before_packing(False)
    notebook_sha = write_notebook()
    paths = set(original.files())
    paths.update(PROJECT / name for name in EXTRA_FILES)
    for name in (*REQUIRED_FILES, *EXTRA_FILES):
        if PROJECT / name not in paths or not (PROJECT / name).is_file():
            raise FileNotFoundError(f"PLATOON0 묶음 입력이 없습니다: {name}")
    sha = lambda p: hashlib.sha256((PROJECT / p).read_bytes()).hexdigest()  # noqa: E731
    lock = json.loads((PROJECT / "config/arena3_v1.json").read_text())
    manifest = {"kind": KIND, "experiment": EXPERIMENT,
                "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                **original.git_state(), "files": {}, "source_models_included": False,
                "lock_sha256": sha("config/arena3_v1.json"), "platoon0_config_sha256": sha("config/explore_platoon0_v1.json"),
                "prereg_sha256": sha("results/explore/platoon0_20261031/사전등록.md"),
                "environment_fingerprint": lock["environment"]["fingerprint"], "notebook_sha256": notebook_sha}
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".platoon0_package_", suffix=".zip", dir=out.parent)
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
        validate_platoon0_archive(Path(temporary))
        os.replace(temporary, out)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"path": str(out), "bytes": out.stat().st_size, "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "file_count": len(manifest["files"]), "experiment": EXPERIMENT, "notebook_sha256": notebook_sha,
            "platoon0_config_sha256": manifest["platoon0_config_sha256"], "prereg_sha256": manifest["prereg_sha256"],
            "git_commit": manifest["git_commit"], "git_uncommitted_changes": manifest["git_uncommitted_changes"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--verify", type=Path, help="이미 만든 묶음만 검사")
    args = parser.parse_args()
    if args.verify:
        manifest = validate_platoon0_archive(args.verify)
        print(json.dumps({"verified": str(args.verify), "file_count": len(manifest["files"])}, ensure_ascii=False))
    else:
        print(json.dumps(build_package(args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
