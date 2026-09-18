from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import sys
import sysconfig
import xml.etree.ElementTree as ET
from pathlib import Path


TARGET_TLS = "204820"


def configure_sumo_python() -> Path:
    configured = os.environ.get("SUMO_HOME")
    sumo_home = (
        Path(configured).expanduser().resolve()
        if configured
        else (Path(sysconfig.get_paths()["purelib"]) / "sumo").resolve()
    )
    tools = sumo_home / "tools"
    if not tools.exists():
        raise RuntimeError(
            "SUMO 도구를 찾지 못했습니다. 프로젝트 가상환경에서 실행해 주세요."
        )
    sys.path.insert(0, str(tools))
    return sumo_home


def find_sumo_binary(sumo_home: Path, gui: bool) -> str:
    name = "sumo-gui" if gui else "sumo"
    candidates = [Path(sys.prefix) / "bin" / name, sumo_home / "bin" / name]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(f"{name} 실행 파일을 찾지 못했습니다.")


def parse_tripinfo(path: Path) -> dict[str, float | int]:
    root = ET.parse(path).getroot()
    trips = root.findall("tripinfo")
    waits = [float(trip.attrib["waitingTime"]) for trip in trips]
    durations = [float(trip.attrib["duration"]) for trip in trips]
    losses = [float(trip.attrib["timeLoss"]) for trip in trips]
    return {
        "arrived_vehicles": len(trips),
        "average_waiting_time_sec": statistics.fmean(waits) if waits else 0.0,
        "average_travel_time_sec": statistics.fmean(durations) if durations else 0.0,
        "average_time_loss_sec": statistics.fmean(losses) if losses else 0.0,
    }


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="내장 고정신호로 SUMO를 실행하고 기준 성능을 저장합니다."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root
        / "data"
        / "processed"
        / "20220810_0500"
        / "scenario.sumocfg",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "results" / "fixed_20220810_0500",
    )
    parser.add_argument("--gui", action="store_true", help="SUMO-GUI로 실행")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = args.config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not config.exists():
        raise FileNotFoundError(
            f"설정 파일이 없습니다: {config}\n먼저 prepare_sumo_scenario.py를 실행하세요."
        )

    sumo_home = configure_sumo_python()
    import traci  # type: ignore

    sumo_binary = find_sumo_binary(sumo_home, args.gui)
    tripinfo = output_dir / "tripinfo.xml"
    summary = output_dir / "summary.xml"
    statistics_file = output_dir / "statistics.xml"
    command = [
        sumo_binary,
        "-c",
        str(config),
        "--tripinfo-output",
        str(tripinfo),
        "--summary-output",
        str(summary),
        "--statistic-output",
        str(statistics_file),
        "--tripinfo-output.write-unfinished",
        "true",
        "--no-warnings",
        "true",
    ]

    traci.start(command)
    controlled_lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(TARGET_TLS)))
    if not controlled_lanes:
        traci.close()
        raise RuntimeError(f"신호 {TARGET_TLS}의 진입 차로를 찾지 못했습니다.")

    rows: list[dict[str, float | int]] = []
    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            queue = sum(
                traci.lane.getLastStepHaltingNumber(lane_id)
                for lane_id in controlled_lanes
            )
            rows.append(
                {
                    "time_sec": traci.simulation.getTime(),
                    "queue_vehicles": queue,
                    "signal_phase": traci.trafficlight.getPhase(TARGET_TLS),
                }
            )
    finally:
        traci.close()

    step_metrics = output_dir / "step_metrics.csv"
    with step_metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["time_sec", "queue_vehicles", "signal_phase"]
        )
        writer.writeheader()
        writer.writerows(rows)

    metrics = parse_tripinfo(tripinfo)
    metrics["target_tls"] = TARGET_TLS
    metrics["controlled_lane_count"] = len(controlled_lanes)
    metrics["max_queue_vehicles"] = max(
        (int(row["queue_vehicles"]) for row in rows), default=0
    )
    metrics["simulation_steps"] = len(rows)

    metrics_json = output_dir / "baseline_metrics.json"
    metrics_json.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "baseline_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"결과 폴더: {output_dir}")


if __name__ == "__main__":
    main()
