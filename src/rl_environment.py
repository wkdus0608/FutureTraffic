from __future__ import annotations

import json
import os
import sys
import sysconfig
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DIR = PROJECT_ROOT / "data" / "processed" / "20220810_0500"
TARGET_TLS = "204820"
ENVIRONMENT_VERSION = "queue_reward_yellow3_v1"

# 이 프로젝트에서 한 번의 행동은 5초 동안 유지된다.
# 예: DQN이 0번 신호를 선택하면 SUMO는 그 선택을 반영한 채 5초 진행한다.
ACTION_INTERVAL_SEC = 5

# 대기행렬은 행동 주기와 별개로 매초 측정한다.
# 행동을 5초마다 선택하더라도, 1~4초 사이의 혼잡을 빠뜨리면 안 되기 때문이다.
QUEUE_MEASUREMENT_INTERVAL_SEC = 1

# 원본 도로망에서 204820 교차로의 모든 황색 현시는 3초이다.
# DQN도 같은 3초를 사용해야 고정신호와 신호 안전시간이 같아진다.
YELLOW_TIME_SEC = 3

# SUMO-RL의 queue 보상은 현재 정지 차량 수에 마이너스를 붙인 값이다.
# 대기 차량이 0대면 0, 10대면 -10이므로 값이 클수록 혼잡이 적다.
REWARD_FUNCTION = "queue"

# AI Hub 샘플 교통량은 15분간 차량을 발생시키며,
# 뒤에 남은 차량이 빠져나가는 시간까지 포함해 총 30분을 실행한다.
SIMULATION_DURATION_SEC = 1800


def configure_sumo() -> None:
    """pip로 설치한 SUMO를 SUMO-RL이 찾을 수 있도록 경로를 설정한다."""
    sumo_home = Path(sysconfig.get_paths()["purelib"]) / "sumo"
    if not (sumo_home / "tools").exists():
        raise RuntimeError("SUMO를 찾지 못했습니다. 프로젝트 가상환경에서 실행하세요.")
    os.environ.setdefault("SUMO_HOME", str(sumo_home))
    bin_dir = str(Path(sys.prefix) / "bin")
    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


def find_route_file(scenario_dir: Path) -> Path:
    """시나리오 폴더에서 차량 경로 파일 하나를 찾는다.

    시간대가 늘어나면 파일명이 달라지므로 특정 날짜의 파일명을 코드에
    고정하지 않는다. 각 시나리오 폴더에는 ``traffic_*.rou.xml``이 정확히
    하나 있어야 한다.
    """
    candidates = sorted(scenario_dir.glob("traffic_*.rou.xml"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"{scenario_dir}에서 traffic_*.rou.xml을 하나만 찾아야 합니다. "
            f"발견: {len(candidates)}개"
        )
    return candidates[0]


def read_simulation_duration(scenario_dir: Path) -> int:
    """전처리 메타데이터에 저장된 실행시간을 읽고, 없으면 30분을 사용한다."""
    metadata_path = scenario_dir / "scenario_metadata.json"
    if not metadata_path.exists():
        return SIMULATION_DURATION_SEC
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return int(metadata.get("simulation_duration_sec", SIMULATION_DURATION_SEC))


def make_environment(
    *,
    seed: int = 42,
    use_gui: bool = False,
    output_prefix: Path | None = None,
    fixed_ts: bool = False,
    additional_sumo_cmd: str | None = None,
    record_per_second_queue: bool = False,
    scenario_dir: Path | None = None,
):
    """학습과 평가에서 공통으로 사용하는 SUMO-RL 환경을 만든다.

    주요 매개변수
    ------------
    fixed_ts:
        True이면 RL이 신호를 선택하지 않고 도로망 파일에 저장된 고정신호를 따른다.
        False이면 전달받은 action에 따라 204820 교차로의 신호를 바꾼다.
    record_per_second_queue:
        True이면 SUMO가 1초 진행될 때마다 204820 교차로의 대기행렬을 기록한다.
        학습에는 필요하지 않고, 정책을 공정하게 평가할 때만 사용한다.
    scenario_dir:
        도로망과 차량 경로가 들어 있는 폴더이다. 생략하면 현재 샘플 시간대인
        ``20220810_0500``을 사용한다. 이후 다른 날짜도 같은 형식으로 넣을 수 있다.
    """
    configure_sumo()
    from sumo_rl import SumoEnvironment

    class PerSecondQueueEnvironment(SumoEnvironment):
        """5초 행동 사이에 사라지는 1초 단위 대기행렬을 보존하는 평가 환경.

        SUMO-RL의 ``env.step(action)`` 한 번은 내부적으로 SUMO를 5번 진행한다.
        원래 코드는 다섯 번이 끝난 시점의 값만 돌려주므로 1~4초 값은 볼 수 없다.
        따라서 SUMO가 실제로 1초 진행되는 가장 안쪽 함수인 ``_sumo_step`` 뒤에서
        대기행렬을 기록한다. 신호 선택, 보상 계산, 학습 방식은 바꾸지 않는다.
        """

        def __init__(self, *args, **kwargs):
            # 매초 집계한 대기행렬 자료이다.
            self.per_second_queue_records: list[dict[str, float | int | str]] = []

            # 교차로 진입 차로에서 한 번이라도 관측된 차량과 차량별 정지시간이다.
            # 차량별 정지시간의 총합은 대기행렬을 매초 더한 차량·초와 같다.
            self.local_vehicle_ids: set[str] = set()
            self.local_passed_vehicle_ids: set[str] = set()
            self.local_vehicle_wait_seconds: dict[str, int] = {}
            super().__init__(*args, **kwargs)

        def reset(self, *args, **kwargs):
            # 새로운 평가 실행이 시작되면 이전 실행의 기록을 비운다.
            self.per_second_queue_records.clear()
            self.local_vehicle_ids.clear()
            self.local_passed_vehicle_ids.clear()
            self.local_vehicle_wait_seconds.clear()
            return super().reset(*args, **kwargs)

        def _sumo_step(self) -> None:
            # 부모 클래스가 차량을 정확히 1초 이동시키고 출발·도착 수를 갱신한다.
            super()._sumo_step()

            # 제어 교차로로 들어오는 20개 차로의 차량 ID를 중복 없이 모은다.
            traffic_signal = self.traffic_signals[TARGET_TLS]
            approaching_ids: set[str] = set()
            outgoing_ids: set[str] = set()
            queued_ids: set[str] = set()
            for lane_id in traffic_signal.lanes:
                for vehicle_id in self.sumo.lane.getLastStepVehicleIDs(lane_id):
                    approaching_ids.add(vehicle_id)
                    # SUMO의 halting 기준과 같이 속도 0.1m/s 미만을 정지로 본다.
                    if self.sumo.vehicle.getSpeed(vehicle_id) < 0.1:
                        queued_ids.add(vehicle_id)

            # 진출 차로에서 다시 확인된 차량만 "교차로 통과 차량"으로 센다.
            # 단순히 진입 차로에서 사라진 차량을 세면 텔레포트 차량까지 통과로
            # 잘못 분류할 수 있기 때문이다.
            for lane_id in traffic_signal.out_lanes:
                outgoing_ids.update(self.sumo.lane.getLastStepVehicleIDs(lane_id))

            self.local_vehicle_ids.update(approaching_ids)
            # 진출 차로가 짧으면 한 번의 1초 step 안에 차량이 진출 차로를 지나
            # 목적지에 도착할 수 있다. 그 차량은 arrived ID로 보완한다.
            arrived_ids = set(self.sumo.simulation.getArrivedIDList())
            passed_now = (outgoing_ids | arrived_ids) & self.local_vehicle_ids
            self.local_passed_vehicle_ids.update(passed_now)
            for vehicle_id in queued_ids:
                self.local_vehicle_wait_seconds[vehicle_id] = (
                    self.local_vehicle_wait_seconds.get(vehicle_id, 0)
                    + QUEUE_MEASUREMENT_INTERVAL_SEC
                )

            # get_total_queued()는 같은 진입 차로들의 정지 차량 수를 합산한다.
            # ID도 함께 저장해 나중에 차량별 대기시간을 검증할 수 있게 한다.
            self.per_second_queue_records.append(
                {
                    "simulation_time_sec": float(self.sim_step),
                    "local_queue_vehicles": int(traffic_signal.get_total_queued()),
                    "local_approaching_vehicles": len(approaching_ids),
                    "local_passed_vehicles_cumulative": len(
                        self.local_passed_vehicle_ids
                    ),
                    "approaching_vehicle_ids": "|".join(sorted(approaching_ids)),
                    "queued_vehicle_ids": "|".join(sorted(queued_ids)),
                    "passed_vehicle_ids": "|".join(sorted(passed_now)),
                }
            )

    selected_scenario_dir = (scenario_dir or SCENARIO_DIR).expanduser().resolve()
    net_file = selected_scenario_dir / "network.net.xml"
    route_file = find_route_file(selected_scenario_dir)
    simulation_duration_sec = read_simulation_duration(selected_scenario_dir)
    for path in (net_file, route_file):
        if not path.exists():
            raise FileNotFoundError(
                f"필요한 파일이 없습니다: {path}\n"
                "먼저 prepare_sumo_scenario.py를 실행하세요."
            )

    # 일반 학습에는 원래 SUMO-RL 환경을 사용한다. 평가할 때만 기록 기능이 있는
    # 하위 클래스를 선택하므로 학습 속도와 학습 동작에는 영향을 주지 않는다.
    environment_class = (
        PerSecondQueueEnvironment if record_per_second_queue else SumoEnvironment
    )

    return environment_class(
        net_file=str(net_file),
        route_file=str(route_file),
        out_csv_name=str(output_prefix) if output_prefix else None,
        single_agent=True,
        ts_ids=[TARGET_TLS],
        use_gui=use_gui,
        num_seconds=simulation_duration_sec,
        delta_time=ACTION_INTERVAL_SEC,
        yellow_time=YELLOW_TIME_SEC,
        min_green=5,
        max_green=50,
        enforce_max_green=True,
        reward_fn=REWARD_FUNCTION,
        sumo_seed=seed,
        fixed_ts=fixed_ts,
        sumo_warnings=False,
        additional_sumo_cmd=additional_sumo_cmd,
    )
