"""Colab에서 쓸 프로젝트 묶음(zip)을 만든다. Colab에서 풀면 이 맥과 같은 폴더 구조가 된다.

담는 것: README·requirements·config·src·tests·data(raw·derived)·notebooks, 그리고 재현성 비교용 기준 결과
(이 맥에서 돌린 2022-08-11 하루 점검 결과). 파일별 SHA-256과 Git 커밋을 manifest.json에 적는다.

사용: .venv/bin/python src/focused_package.py
결과: results/colab/FutureTraffic_focused.zip → Google Drive의 MyDrive/FutureTraffic/에 올린다.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results/colab/FutureTraffic_focused.zip"
REFERENCE = PROJECT / "results/focused_204820/day/check/20220811/result.json"
INCLUDE = ["README.md", "requirements.txt", "config", "src", "tests", "data", "notebooks"]


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
