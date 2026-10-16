"""Colab에서 쓸 프로젝트 묶음(zip)을 만든다. Colab에서 풀면 이 맥과 같은 폴더 구조가 된다.

담는 것: README·requirements·config·src·tests·data(raw·derived)·notebooks, 그리고 재현성 비교용 기준 결과
(이 맥에서 돌린 2022-08-11 하루 점검 결과). 파일별 SHA-256과 Git 커밋을 manifest.json에 적는다.

사용: .venv/bin/python src/focused_package.py
결과: results/colab/FutureTraffic_focused.zip → Google Drive의 MyDrive/FutureTraffic/에 올린다.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results/colab/FutureTraffic_focused.zip"
REFERENCE = PROJECT / "results/focused_204820/day/check/20220811/result.json"
INCLUDE = ["README.md", "requirements.txt", "config", "src", "tests", "data", "notebooks"]
# 잠근 환경 지문(변형 이름 → 지문). 비어 있으면 아직 잠그지 않은 것이다(v6 개발 중).
# v5는 환경 변수 FOCUSED_RULESET=none FOCUSED_NORTH_EXT=0일 때의 변형이고, 코드가 바뀌어 v5의 지문(0b1cad3a2d35)과는 다르다.
# 잠글 때 여기에 {"none": "...", "pet1": "...", "pet1_5": "..."}를 적으면 묶기 전에 다르면 멈춘다.
# 2026-10-15 사용자 확정으로 잠갔다(docs/실험설정_v6.md). pet1 = 주 결과, pet1_5 = 민감도, none = 규칙 없는 비교.
LOCKED_ENVIRONMENTS: dict[str, str] = {"none": "84ba298fb9ba", "pet1": "b20888514ea3", "pet1_5": "f31f84aa9924"}


def check_before_packing(new_environment: bool) -> None:
    """원본 자료가 검사값(data/raw/manifest.json)과 같은지, 환경 지문이 잠근 값과 같은지 확인한다.

    10-08에 잠근 코드와 원본 지도에 공부용 주석이 들어가 지문·검사값이 바뀐 일이 있었다(주석만이라도 바뀐다).
    """
    import os
    sys.path.insert(0, str(PROJECT / "src"))
    from focused_data import environment_fingerprint
    raw = json.loads((PROJECT / "data/raw/manifest.json").read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    bad = [name for name, v in raw["network"].items() if sha(PROJECT / "data/raw/network" / name) != v["sha256"]]
    turn = raw["turn_counts"]
    if sha(PROJECT / "data/raw/turn_counts" / turn["block_file"]) != turn["block_sha256"]:
        bad.append(turn["block_file"])
    if bad:
        raise SystemExit(f"원본 자료가 검사값과 다릅니다: {bad}. 원본으로 되돌린 뒤 다시 묶으세요.")
    old = os.environ.get("FOCUSED_RULESET")
    fps = {}
    try:
        for name in ("none", "pet1", "pet1_5"):
            os.environ["FOCUSED_RULESET"] = name
            fps[name] = environment_fingerprint()
    finally:
        if old is None:
            os.environ.pop("FOCUSED_RULESET", None)
        else:
            os.environ["FOCUSED_RULESET"] = old
    wrong = {n: (fps[n], want) for n, want in LOCKED_ENVIRONMENTS.items() if fps.get(n) != want}
    if wrong and not new_environment:
        raise SystemExit(f"환경 지문이 잠근 값과 다릅니다: {wrong}. 잠근 파일이 바뀌었는지 `git status`로 보세요. "
                         "주석만 바꿔도 지문이 바뀝니다.")
    print(f"원본 자료 검사 통과, 환경 지문 {fps}" + ("" if LOCKED_ENVIRONMENTS else " (아직 잠그지 않음)"))


def files() -> list[Path]:
    out = []
    for name in INCLUDE:
        path = PROJECT / name
        items = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        out += [p for p in items if "__pycache__" not in p.parts and p.name != ".DS_Store"]
    return out


def git_state() -> dict:
    run = lambda *a: subprocess.run(["git", *a], cwd=PROJECT, capture_output=True, text=True).stdout.strip()
    return {"git_commit": run("rev-parse", "--short", "HEAD"), "git_uncommitted_changes": bool(run("status", "--porcelain"))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Colab용 프로젝트 묶음을 만든다.")
    parser.add_argument("--new-environment", action="store_true", help="환경 지문이 잠근 값과 달라도 묶는다(새 판일 때만)")
    check_before_packing(parser.parse_args().new_environment)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"created": dt.datetime.now().isoformat(timespec="seconds"), **git_state(), "files": {}}
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in files():
            rel = path.relative_to(PROJECT).as_posix()
            data = path.read_bytes()
            manifest["files"][rel] = hashlib.sha256(data).hexdigest()
            z.writestr(rel, data)
        reference = "reference/day_check_20220811_mac.json"   # 맥에서 돌린 결과(Colab 결과와 비교용)
        data = REFERENCE.read_bytes()
        manifest["files"][reference] = hashlib.sha256(data).hexdigest()
        z.writestr(reference, data)
        z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=1))
    print(f"{OUT} ({OUT.stat().st_size / 1e6:.1f}MB, 파일 {len(manifest['files'])}개, 커밋 {manifest['git_commit']}"
          f"{' + 커밋 안 된 변경' if manifest['git_uncommitted_changes'] else ''})")


if __name__ == "__main__":
    main()
