"""SUMO 실행 도구: SUMO 찾기, SUMO-RL 환경 만들기, SUMO 결과 파일 읽기.

예전 전체 도로망 실험의 rl_environment.py·evaluate_policies.py에서 석천사거리 실험이 실제로
쓰는 부분만 옮겼다(2026-10-05 정리). 동작은 그대로다.
"""

from __future__ import annotations

import os
import statistics
import sys
import sysconfig
import xml.etree.ElementTree as ET
from pathlib import Path

TARGET_TLS = "204820"
# 행동 하나는 5초 동안 유지된다. 대기행렬은 행동과 별개로 매초 잰다.
ACTION_INTERVAL_SEC = 5


def configure_sumo() -> None:
    """pip로 설치한 SUMO를 SUMO-RL이 찾을 수 있도록 경로를 설정한다."""
    sumo_home = Path(sysconfig.get_paths()["purelib"]) / "sumo"
    if not (sumo_home / "tools").exists():
        raise RuntimeError("SUMO를 찾지 못했습니다. 프로젝트 가상환경(.venv)에서 실행하세요.")
    os.environ.setdefault("SUMO_HOME", str(sumo_home))
    bin_dir = str(Path(sys.prefix) / "bin")
    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


def make_environment(*, net_file: Path, route_file: Path, simulation_duration_sec: int, observation_class,
                     reward_fn, seed: int = 42, fixed_ts: bool = False, additional_sumo_cmd: str | None = None,
                     record_per_second_queue: bool = False, extra_queue_edges: list[str] | None = None,
                     min_green: int = 15, max_green: int = 120, yellow_time: int = 3):
    """학습과 평가가 같이 쓰는 SUMO-RL 환경. 204820 하나만 제어 대상이다.

    fixed_ts: True이면 지도·추가 파일의 신호 프로그램대로 돌고(고정신호·감응식), False이면
        5초마다 받은 행동(녹색 현시 번호)으로 신호를 바꾼다(DQN).
    record_per_second_queue: True이면 매초 대기행렬·진입 대기·신호 상태를 기록한다(평가용).
    순간이동은 끄고(-1), 도로에 들어오지 못한 차량도 버리지 않는다(SUMO-RL 기본값).
    """
    for path in (net_file, route_file):
        if not Path(path).exists():
            raise FileNotFoundError(f"필요한 파일이 없습니다: {path}")
    configure_sumo()
    from sumo_rl import SumoEnvironment

    class PerSecondQueueEnvironment(SumoEnvironment):
        """5초 행동 사이의 1~4초 대기행렬까지 남기는 평가용 환경. 신호 선택·보상은 바꾸지 않는다."""

        def __init__(self, *args, **kwargs):
            self.per_second_queue_records: list[dict] = []
            super().__init__(*args, **kwargs)

        def reset(self, *args, **kwargs):
            self.per_second_queue_records.clear()
            return super().reset(*args, **kwargs)

        def _sumo_step(self) -> None:
            super()._sumo_step()   # SUMO를 정확히 1초 진행한다.
            traffic_signal = self.traffic_signals[TARGET_TLS]
            self.per_second_queue_records.append({
                "simulation_time_sec": float(self.sim_step),
                "local_queue_vehicles": int(traffic_signal.get_total_queued()),
                "tls_state": self.sumo.trafficlight.getRedYellowGreenState(TARGET_TLS),
                "zone_queue_vehicles": (sum(self.sumo.edge.getLastStepHaltingNumber(e) for e in extra_queue_edges)
                                        if extra_queue_edges else None),
                # 출발 시각이 지났지만 대기열 때문에 도로에 들어오지 못한 차량 수
                "pending_vehicles": len(self.sumo.simulation.getPendingVehicles()),
            })

    environment_class = PerSecondQueueEnvironment if record_per_second_queue else SumoEnvironment
    return environment_class(
        add_system_info=False, add_per_agent_info=False, observation_class=observation_class,
        net_file=str(Path(net_file).resolve()), route_file=str(Path(route_file).resolve()), out_csv_name=None,
        single_agent=True, ts_ids=[TARGET_TLS], use_gui=False, num_seconds=simulation_duration_sec,
        delta_time=ACTION_INTERVAL_SEC, yellow_time=yellow_time, min_green=min_green, max_green=max_green,
        enforce_max_green=True, reward_fn=reward_fn, sumo_seed=seed, fixed_ts=fixed_ts, sumo_warnings=False,
        additional_sumo_cmd=additional_sumo_cmd,
    )


# ------------------------------------------------------------------ SUMO 결과 파일
def sumo_output_options(run_dir: Path) -> tuple[str, Path, Path]:
    """차량별 결과(tripinfo)와 안전 통계(statistics)를 남기는 SUMO 옵션."""
    tripinfo = (run_dir / "tripinfo.xml").resolve()
    summary = (run_dir / "summary.xml").resolve()
    statistics_file = (run_dir / "statistics.xml").resolve()
    for path in (tripinfo, summary, statistics_file):
        if " " in str(path):
            raise ValueError(f"평가 출력 경로에 공백을 쓸 수 없습니다: {path}")
    command = " ".join(["--tripinfo-output", str(tripinfo), "--tripinfo-output.write-unfinished", "true",
                        "--summary-output", str(summary), "--statistic-output", str(statistics_file),
                        "--duration-log.statistics", "true", "--no-step-log", "true"])
    return command, tripinfo, statistics_file


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def parse_tripinfo(path: Path) -> dict[str, float | int]:
    """도착한 차량의 평균 값. 도착하지 못한 차량 수는 따로 센다(평균에 섞으면 좋아 보이는 착시)."""
    trips = ET.parse(path).getroot().findall("tripinfo")
    completed = [t for t in trips if float(t.attrib.get("arrival", "-1")) >= 0]
    values = lambda key: [float(t.attrib.get(key, "0")) for t in completed]
    losses, delays = values("timeLoss"), values("departDelay")
    return {
        "recorded_tripinfo_vehicles": len(trips),
        "completed_tripinfo_vehicles": len(completed),
        "unfinished_tripinfo_vehicles": len(trips) - len(completed),
        "global_mean_waiting_time_sec": _mean(values("waitingTime")),
        "global_mean_travel_time_sec": _mean(values("duration")),
        "global_mean_time_loss_sec": _mean(losses),
        "global_mean_depart_delay_sec": _mean(delays),
        "global_mean_total_delay_sec": _mean([a + b for a, b in zip(losses, delays)]),   # 진입 전 지연 포함
    }


def parse_statistics(path: Path) -> dict[str, int]:
    """SUMO 실행 전체의 투입 차량 수와 안전 통계."""
    root = ET.parse(path).getroot()
    vehicles, safety, teleports = root.find("vehicles"), root.find("safety"), root.find("teleports")
    if vehicles is None or safety is None or teleports is None:
        raise ValueError(f"SUMO 통계 파일의 형식을 확인할 수 없습니다: {path}")
    return {
        "loaded_vehicles": int(vehicles.attrib["loaded"]),
        "inserted_vehicles": int(vehicles.attrib["inserted"]),
        "running_vehicles_at_end": int(vehicles.attrib["running"]),
        "waiting_vehicles_at_end": int(vehicles.attrib["waiting"]),
        "collision_count": int(safety.attrib["collisions"]),
        "emergency_stop_count": int(safety.attrib["emergencyStops"]),
        "emergency_braking_count": int(safety.attrib["emergencyBraking"]),
        "teleport_count_from_statistics": int(teleports.attrib["total"]),
    }
