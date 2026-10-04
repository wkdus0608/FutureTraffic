"""Sample 전체의 회전교통량 시간대를 SUMO 시나리오로 일괄 변환한다."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

try:
    # ``python src/prepare_all_scenarios.py``로 실행할 때 사용한다.
    from prepare_sumo_scenario import scenario_id_from_prefix
except ModuleNotFoundError:
    # 테스트에서 ``src.prepare_all_scenarios``로 불러올 때 사용한다.
    from src.prepare_sumo_scenario import scenario_id_from_prefix


VEHICLE_CODES = {"p", "b", "t", "m"}
# AI Hub 파일명 앞 6자리는 자료 묶음이다. 900012 = 부천 교차로_all(도로망 전체).
# 101xx(단독 교차로)·301xx(일부 묶음) 파일은 한두 교차로의 기록만 있어
# 도로망 전체 시나리오로 쓰면 안 되고, 날짜·시각이 같아 출력 폴더도 겹친다.
ALL_INTERSECTIONS_PREFIX = "900012"


def discover_complete_prefixes(sample_root: Path) -> list[str]:
    """p·b·t·m 네 파일이 모두 있는 회전교통량 시간대를 찾는다."""
    found: dict[str, set[str]] = defaultdict(set)
    pattern = re.compile(r"^(.*)([pbtm])\.xml$")
    for path in sample_root.rglob("*.xml"):
        match = pattern.match(path.name)
        if (match and re.search(r"\d{14}$", match.group(1))
                and match.group(1).startswith(ALL_INTERSECTIONS_PREFIX)):
            found[match.group(1)].add(match.group(2))
    prefixes = sorted(prefix for prefix, codes in found.items() if codes == VEHICLE_CODES)
    scenario_ids = [scenario_id_from_prefix(prefix) for prefix in prefixes]
    duplicates = {s for s in scenario_ids if scenario_ids.count(s) > 1}
    if duplicates:
        raise ValueError(f"같은 시나리오 ID가 여러 파일에서 나왔습니다: {sorted(duplicates)[:5]}")
    return prefixes


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="다운로드한 회전교통량 전체를 시간대별 SUMO 시나리오로 변환합니다."
    )
    parser.add_argument("--sample-root", type=Path, default=project_root / "Sample")
    parser.add_argument(
        "--processed-root", type=Path, default=project_root / "data" / "processed"
    )
    parser.add_argument("--max-mismatch-rate", type=float, default=0.10)
    parser.add_argument(
        "--limit", type=int,
        help="처음 몇 개만 시험할 때 사용합니다. 생략하면 발견한 전체를 처리합니다.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_root = args.sample_root.expanduser().resolve()
    processed_root = args.processed_root.expanduser().resolve()
    # 네 차종 파일이 모두 있는 시간대만 처리 대상으로 선택한다.
    prefixes = discover_complete_prefixes(sample_root)
    if args.limit is not None:
        prefixes = prefixes[: args.limit]
    if not prefixes:
        raise FileNotFoundError("p·b·t·m 파일이 모두 있는 시간대를 찾지 못했습니다.")

    # 실제 시나리오 생성은 단일 처리 스크립트에 맡기고, 이 파일은 반복만 담당한다.
    worker = Path(__file__).with_name("prepare_sumo_scenario.py")
    records: list[dict[str, str | int]] = []
    for index, prefix in enumerate(prefixes, start=1):
        scenario_id = scenario_id_from_prefix(prefix)
        output_dir = processed_root / scenario_id
        print(f"[{index}/{len(prefixes)}] {scenario_id} 전처리")
        completed = subprocess.run(
            [
                sys.executable,
                str(worker),
                "--sample-root", str(sample_root),
                "--turn-prefix", prefix,
                "--output-dir", str(output_dir),
                "--max-mismatch-rate", str(args.max_mismatch_rate),
            ],
            check=False,
        )
        records.append(
            {
                "scenario_id": scenario_id,
                "turn_prefix": prefix,
                "return_code": completed.returncode,
                "status": "prepared" if completed.returncode == 0 else "excluded_or_failed",
            }
        )

    # 전체 시간대의 성공,실패 결과를 한 파일에 남긴다.
    manifest = {
        "discovered_complete_scenarios": len(prefixes),
        "prepared_scenarios": sum(row["return_code"] == 0 for row in records),
        "excluded_or_failed_scenarios": sum(row["return_code"] != 0 for row in records),
        "records": records,
    }
    processed_root.mkdir(parents=True, exist_ok=True)
    manifest_path = processed_root / "preparation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"전체 전처리 기록: {manifest_path}")


if __name__ == "__main__":
    main()
