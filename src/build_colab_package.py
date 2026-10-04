"""Colab 본 실험용 묶음(focused_package.zip)을 만든다.

코드(src), 검사(tests), 설치 목록, 날짜 분할, 석천사거리 기록, 19일 신호, AI Hub 지도·차종 정의를
같은 상대 경로로 담는다. Colab에서 /content/FutureTraffic에 풀면 로컬과 같은 구조가 된다.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results/focused_204820/focused_package.zip"


def main() -> None:
    files = []
    files += sorted((PROJECT / "src").glob("*.py"))
    files += [PROJECT / "tests/test_focused.py", PROJECT / "tests/test_constraints.py",
              PROJECT / "tests/test_signal_timeline.py",
              PROJECT / "requirements.txt", PROJECT / "config/focused_split.json", PROJECT / "config/driver_model.json"]
    data = PROJECT / "results/focused_204820"
    # 갈래 기록(branch_inflow)은 write_routes가 길주로 서쪽 접근로 차량을 나눌 때 쓴다.
    files += [data / "counts_204820.csv", data / "signal_capacity_check.json", data / "branch_inflow_477748652_01.csv"]
    files += sorted((data / "signals_all").rglob("*.xml"))
    network_dir = next((PROJECT / "Sample").rglob("1200012n.net.xml")).parent
    files += [network_dir / "1200012n.net.xml", network_dir / "exp.rou.xml"]
    manifest = {}
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in files:
            rel = f.relative_to(PROJECT).as_posix()
            z.write(f, rel)
            manifest[rel] = hashlib.sha256(f.read_bytes()).hexdigest()
        z.writestr("PACKAGE_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=1))
    print(json.dumps({"zip": str(OUT), "files": len(files), "megabytes": round(OUT.stat().st_size / 1e6, 1)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
