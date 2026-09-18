from __future__ import annotations

import argparse
import copy
import os
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


def write_config(output_dir: Path, route_filename: str) -> Path:
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
        <end value="1800"/>
    </time>
    <report>
        <no-step-log value="true"/>
        <duration-log.statistics value="true"/>
    </report>
</configuration>
"""
        % route_filename,
        encoding="utf-8",
    )
    return config


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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "processed" / "20220810_0500",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_root = args.sample_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
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

    route_file = output_dir / "traffic_20220810_0500.rou.xml"
    vehicle_count = merge_and_rebase_routes(
        vehicle_types_file, sampled_files, route_file, begin
    )
    shutil.copy2(network, output_dir / "network.net.xml")
    config = write_config(output_dir, route_file.name)

    print(f"완료: {config}")
    print(f"생성 차량: {vehicle_count:,}대")
    print(f"원본 시간: {int(begin)}~{int(end)}초 → SUMO 시간: 0~{int(end - begin)}초")


if __name__ == "__main__":
    main()
