from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import xml.etree.ElementTree as ET
from pathlib import Path


VEHICLE_TYPES = {
    "p": ("passenger", "p_"),
    "b": ("bus", "b_"),
    "t": ("truck", "t_"),
    "m": ("motorcycle", "m_"),
}
DEFAULT_MAX_MISMATCH_RATE = 0.10


def find_one(root: Path, filename: str) -> Path:
    matches = list(root.rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"{filename!r} 파일을 하나만 찾을 수 있어야 합니다. 발견: {len(matches)}개"
        )
    return matches[0]


def find_sumo_home() -> Path:
    configured = os.environ.get("SUMO_HOME")
    if configured:
        return Path(configured).expanduser().resolve()

    candidate = Path(sysconfig.get_paths()["purelib"]) / "sumo"
    if candidate.exists():
        return candidate.resolve()

    raise RuntimeError(
        "SUMO_HOME을 찾지 못했습니다. 프로젝트 가상환경에서 실행해 주세요: "
        ".venv/bin/python src/prepare_sumo_scenario.py"
    )


def read_time_range(turn_files: list[Path]) -> tuple[float, float]:
    begins: list[float] = []
    ends: list[float] = []
    for turn_file in turn_files:
        root = ET.parse(turn_file).getroot()
        for interval in root.findall(".//interval"):
            begins.append(float(interval.attrib["begin"]))
            ends.append(float(interval.attrib["end"]))
    if not begins:
        raise ValueError("회전교통량 XML에서 interval을 찾지 못했습니다.")
    return min(begins), max(ends)


def scenario_id_from_prefix(turn_prefix: str) -> str:
    """AI Hub 파일명의 날짜·시간 부분을 ``YYYYMMDD_HHMM``으로 바꾼다."""
    match = re.search(r"(\d{8})(\d{4})\d{2}$", turn_prefix)
    if not match:
        raise ValueError(
            "turn-prefix 끝에는 YYYYMMDDHHMMSS가 있어야 합니다: "
            f"{turn_prefix}"
        )
    return f"{match.group(1)}_{match.group(2)}"


def run_route_sampler(
    route_sampler: Path,
    routes: Path,
    turn_file: Path,
    output_file: Path,
    mismatch_file: Path,
    vehicle_type: str,
    prefix: str,
    begin: float,
    end: float,
    sumo_home: Path,
) -> None:
    command = [
        sys.executable,
        str(route_sampler),
        "-r",
        str(routes),
        "-t",
        str(turn_file),
        "-o",
        str(output_file),
        "--mismatch-output",
        str(mismatch_file),
        "--prefix",
        prefix,
        "--attributes",
        f'type="{vehicle_type}"',
        "--begin",
        str(int(begin)),
        "--end",
        str(int(end)),
        "--interval",
        "60",
        "--seed",
        "42",
    ]
    environment = os.environ.copy()
    environment["SUMO_HOME"] = str(sumo_home)
    subprocess.run(command, check=True, env=environment)


def merge_and_rebase_routes(
    vehicle_types_file: Path,
    sampled_files: list[Path],
    output_file: Path,
    begin: float,
) -> int:
    output_root = ET.Element("routes")

    type_root = ET.parse(vehicle_types_file).getroot()
    for vehicle_type in type_root.findall("vType"):
        output_root.append(copy.deepcopy(vehicle_type))

    vehicles: list[ET.Element] = []
    for sampled_file in sampled_files:
        sampled_root = ET.parse(sampled_file).getroot()
        for vehicle in sampled_root.findall("vehicle"):
            copied = copy.deepcopy(vehicle)
            copied.attrib["depart"] = f"{float(copied.attrib['depart']) - begin:.2f}"
            vehicles.append(copied)

    vehicles.sort(key=lambda vehicle: float(vehicle.attrib["depart"]))
    output_root.extend(vehicles)
    tree = ET.ElementTree(output_root)
    ET.indent(tree, space="  ")
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    return len(vehicles)


def write_config(
    output_dir: Path, route_filename: str, simulation_duration_sec: int
) -> Path:
    config = output_dir / "scenario.sumocfg"
    config.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="network.net.xml"/>
        <route-files value="%s"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="%s"/>
    </time>
    <report>
        <no-step-log value="true"/>
        <duration-log.statistics value="true"/>
    </report>
</configuration>
"""
        % (route_filename, simulation_duration_sec),
        encoding="utf-8",
    )
    return config


def read_mismatch_rows(
    mismatch_files: dict[str, Path],
    source_begin: float,
) -> list[dict[str, int | float | str]]:
    """routeSampler의 mismatch XML을 표 형태로 바꾼다.

    measured_count는 원본 회전교통량 관측 수이고 deficit은 그중 후보 경로로
    만족시키지 못한 관측 수이다. 이 수치는 차량 대수가 아니라 회전 이동 관측
    건수라는 점을 보고서의 이름에도 명시한다.
    """
    rows: list[dict[str, int | float | str]] = []
    for vehicle_type, mismatch_file in mismatch_files.items():
        root = ET.parse(mismatch_file).getroot()
        for interval in root.findall(".//interval"):
            begin = float(interval.attrib["begin"])
            end = float(interval.attrib["end"])
            for relation in interval.findall("edgeRelation"):
                measured = int(float(relation.attrib.get("measuredCount", "0")))
                deficit = int(float(relation.attrib.get("deficit", "0")))
                rows.append(
                    {
                        "vehicle_type": vehicle_type,
                        "source_begin_sec": begin,
                        "source_end_sec": end,
                        "simulation_begin_sec": begin - source_begin,
                        "simulation_end_sec": end - source_begin,
                        "from_edge": relation.attrib["from"],
                        "to_edge": relation.attrib["to"],
                        "original_turn_count": measured,
                        "generated_turn_count": measured - deficit,
                        "missing_turn_count": deficit,
                        "mismatch_rate": deficit / measured if measured else 0.0,
                        "geh": float(relation.attrib.get("GEH", "0")),
                    }
                )
    return rows


def write_data_quality_report(
    output_dir: Path,
    mismatch_files: dict[str, Path],
    source_begin: float,
    generated_vehicle_count: int,
    max_mismatch_rate: float,
) -> dict[str, object]:
    """회전교통량 정합성 CSV와 요약 JSON을 저장한다."""
    rows = read_mismatch_rows(mismatch_files, source_begin)
    detail_path = output_dir / "data_quality_by_movement.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    detail_json_path = output_dir / "data_quality_by_movement.json"
    detail_json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    by_vehicle_type: dict[str, dict[str, float | int]] = {}
    for vehicle_type in VEHICLE_TYPES.values():
        name = vehicle_type[0]
        type_rows = [row for row in rows if row["vehicle_type"] == name]
        original = sum(int(row["original_turn_count"]) for row in type_rows)
        missing = sum(int(row["missing_turn_count"]) for row in type_rows)
        by_vehicle_type[name] = {
            "original_turn_count": original,
            "generated_turn_count": original - missing,
            "missing_turn_count": missing,
            "mismatch_rate": missing / original if original else 0.0,
        }

    original_total = sum(int(row["original_turn_count"]) for row in rows)
    missing_total = sum(int(row["missing_turn_count"]) for row in rows)
    mismatch_rate = missing_total / original_total if original_total else 0.0
    status = "usable_with_documented_mismatch"
    if mismatch_rate == 0:
        status = "usable_no_mismatch"
    elif mismatch_rate > max_mismatch_rate:
        status = "excluded_mismatch_above_threshold"

    summary: dict[str, object] = {
        "status": status,
        "max_allowed_mismatch_rate": max_mismatch_rate,
        "original_turn_count": original_total,
        "generated_turn_count": original_total - missing_total,
        "missing_turn_count": missing_total,
        "mismatch_rate": mismatch_rate,
        "generated_vehicle_count": generated_vehicle_count,
        "note": (
            "original/generated/missing_turn_count는 차량 대수가 아니라 회전 "
            "관측 수입니다. 차량 한 대의 경로가 여러 회전 관측을 만족할 수 있습니다."
        ),
        "by_vehicle_type": by_vehicle_type,
        "detail_csv": detail_path.name,
        "detail_json": detail_json_path.name,
    }
    (output_dir / "data_quality_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="AI Hub 회전교통량으로 실행 가능한 SUMO 시나리오를 만듭니다."
    )
    parser.add_argument(
        "--sample-root",
        type=Path,
        default=project_root / "Sample",
        help="AI Hub Sample 폴더",
    )
    parser.add_argument(
        "--turn-prefix",
        default="90001220220810050000",
        help="p/b/t/m 앞에 공통으로 붙은 시간대 파일명",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--max-mismatch-rate",
        type=float,
        default=DEFAULT_MAX_MISMATCH_RATE,
        help="이 비율을 넘는 시나리오는 본 실험에서 제외합니다. 기본값: 0.10",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_root = args.sample_root.expanduser().resolve()
    scenario_id = scenario_id_from_prefix(args.turn_prefix)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else (Path(__file__).resolve().parents[1] / "data" / "processed" / scenario_id)
    )
    intermediate_dir = output_dir / "intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    network = find_one(sample_root, "1200012n.net.xml")
    possible_routes = find_one(sample_root, "possible_routes.rou.xml")
    vehicle_types_file = find_one(sample_root, "exp.rou.xml")
    turn_files = {
        code: find_one(sample_root, f"{args.turn_prefix}{code}.xml")
        for code in VEHICLE_TYPES
    }

    begin, end = read_time_range(list(turn_files.values()))
    sumo_home = find_sumo_home()
    route_sampler = sumo_home / "tools" / "routeSampler.py"
    if not route_sampler.exists():
        raise FileNotFoundError(f"routeSampler.py를 찾지 못했습니다: {route_sampler}")

    sampled_files: list[Path] = []
    mismatch_files: dict[str, Path] = {}
    for code, (vehicle_type, prefix) in VEHICLE_TYPES.items():
        sampled = intermediate_dir / f"{vehicle_type}.rou.xml"
        mismatch = intermediate_dir / f"{vehicle_type}.mismatch.xml"
        print(f"[{vehicle_type}] 회전교통량을 차량 경로로 변환합니다.")
        run_route_sampler(
            route_sampler=route_sampler,
            routes=possible_routes,
            turn_file=turn_files[code],
            output_file=sampled,
            mismatch_file=mismatch,
            vehicle_type=vehicle_type,
            prefix=prefix,
            begin=begin,
            end=end,
            sumo_home=sumo_home,
        )
        sampled_files.append(sampled)
        mismatch_files[vehicle_type] = mismatch

    route_file = output_dir / f"traffic_{scenario_id}.rou.xml"
    vehicle_count = merge_and_rebase_routes(
        vehicle_types_file, sampled_files, route_file, begin
    )
    shutil.copy2(network, output_dir / "network.net.xml")
    # 15분 수요가 끝난 뒤 차량이 빠져나갈 15분을 추가한다.
    simulation_duration_sec = max(1800, int(end - begin) + 900)
    config = write_config(output_dir, route_file.name, simulation_duration_sec)
    quality = write_data_quality_report(
        output_dir=output_dir,
        mismatch_files=mismatch_files,
        source_begin=begin,
        generated_vehicle_count=vehicle_count,
        max_mismatch_rate=args.max_mismatch_rate,
    )
    metadata = {
        "scenario_id": scenario_id,
        "source_turn_prefix": args.turn_prefix,
        "source_date": scenario_id.split("_")[0],
        "source_time": scenario_id.split("_")[1],
        "source_begin_sec": begin,
        "source_end_sec": end,
        "demand_duration_sec": int(end - begin),
        "simulation_duration_sec": simulation_duration_sec,
        "network_file": "network.net.xml",
        "route_file": route_file.name,
        "config_file": config.name,
        "data_quality_status": quality["status"],
        "mismatch_rate": quality["mismatch_rate"],
    }
    (output_dir / "scenario_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"완료: {config}")
    print(f"생성 차량: {vehicle_count:,}대")
    print(f"원본 시간: {int(begin)}~{int(end)}초 → SUMO 시간: 0~{int(end - begin)}초")
    print(
        "회전교통량 누락: "
        f"{quality['missing_turn_count']}/{quality['original_turn_count']} "
        f"({float(quality['mismatch_rate']) * 100:.2f}%)"
    )
    print(f"데이터 품질 판정: {quality['status']}")
    if quality["status"] == "excluded_mismatch_above_threshold":
        raise RuntimeError(
            "누락률이 허용 기준을 넘었습니다. 이 시나리오는 본 실험에서 제외합니다."
        )


if __name__ == "__main__":
    main()
