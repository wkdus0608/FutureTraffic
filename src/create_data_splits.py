from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="전처리된 시나리오를 날짜 단위로 학습·검증·테스트에 분리합니다."
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=project_root / "data" / "processed",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def split_dates(dates: list[str], seed: int) -> dict[str, set[str]]:
    """같은 날짜가 서로 다른 분할에 섞이지 않도록 날짜 목록을 나눈다."""
    if len(dates) < 3:
        raise ValueError(
            "날짜 누수를 막으려면 서로 다른 날짜가 최소 3개 필요합니다. "
            f"현재: {len(dates)}개"
        )
    shuffled = sorted(dates)
    random.Random(seed).shuffle(shuffled)

    # 날짜가 적어도 검증과 테스트에 각각 하루 이상 들어가게 한다.
    validation_count = max(1, round(len(shuffled) * 0.15))
    test_count = max(1, round(len(shuffled) * 0.15))
    while len(shuffled) - validation_count - test_count < 1:
        if validation_count >= test_count and validation_count > 1:
            validation_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            raise ValueError("학습 날짜를 한 개 이상 확보할 수 없습니다.")

    return {
        "test": set(shuffled[:test_count]),
        "validation": set(shuffled[test_count : test_count + validation_count]),
        "train": set(shuffled[test_count + validation_count :]),
    }


def main() -> None:
    args = parse_args()
    processed_root = args.processed_root.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output
        else processed_root / "splits.json"
    )

    scenarios: list[tuple[Path, dict[str, object]]] = []
    for metadata_path in sorted(processed_root.glob("*/scenario_metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("data_quality_status") == "excluded_mismatch_above_threshold":
            continue
        scenarios.append((metadata_path.parent, metadata))

    dates = sorted({str(metadata["source_date"]) for _, metadata in scenarios})
    date_splits = split_dates(dates, args.seed)
    split_rows: dict[str, list[str]] = {name: [] for name in date_splits}
    for scenario_dir, metadata in scenarios:
        source_date = str(metadata["source_date"])
        for split_name, split_dates_set in date_splits.items():
            if source_date in split_dates_set:
                split_rows[split_name].append(
                    str(scenario_dir.relative_to(output.parent))
                )
                break

    payload = {
        "version": 1,
        "split_unit": "calendar_date",
        "seed": args.seed,
        "dates": {name: sorted(values) for name, values in date_splits.items()},
        "splits": {name: sorted(values) for name, values in split_rows.items()},
        "note": "전처리·설정 선택 전에 날짜 단위로 분리해 테스트 누수를 방지합니다.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"분할 파일: {output}")
    for name in ("train", "validation", "test"):
        print(
            f"{name}: 날짜 {len(date_splits[name])}개, "
            f"시나리오 {len(split_rows[name])}개"
        )


if __name__ == "__main__":
    main()
