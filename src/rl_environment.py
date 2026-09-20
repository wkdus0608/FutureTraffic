from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DIR = PROJECT_ROOT / "data" / "processed" / "20220810_0500"
TARGET_TLS = "204820"

# 이 프로젝트에서 한 번의 행동은 5초 동안 유지된다.
# 예: DQN이 0번 신호를 선택하면 SUMO는 그 선택을 반영한 채 5초 진행한다.
ACTION_INTERVAL_SEC = 5

# 대기행렬은 행동 주기와 별개로 매초 측정한다.
# 행동을 5초마다 선택하더라도, 1~4초 사이의 혼잡을 빠뜨리면 안 되기 때문이다.
QUEUE_MEASUREMENT_INTERVAL_SEC = 1

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


def make_environment(
    *,
    seed: int = 42,
    use_gui: bool = False,
    output_prefix: Path | None = None,
    fixed_ts: bool = False,
    additional_sumo_cmd: str | None = None,
    record_per_second_queue: bool = False,
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
            # 각 원소는 {simulation_time_sec, local_queue_vehicles} 형태이다.
            self.per_second_queue_records: list[dict[str, float | int]] = []
            super().__init__(*args, **kwargs)

        def reset(self, *args, **kwargs):
            # 새로운 평가 실행이 시작되면 이전 실행의 기록을 비운다.
            self.per_second_queue_records.clear()
            return super().reset(*args, **kwargs)

        def _sumo_step(self) -> None:
            # 부모 클래스가 차량을 정확히 1초 이동시키고 출발·도착 수를 갱신한다.
            super()._sumo_step()

            # get_total_queued()는 제어 교차로로 들어오는 차로에서
            # 속도가 거의 0인 차량 수를 합산한다. 상태를 바꾸지 않는 조회 작업이다.
            traffic_signal = self.traffic_signals[TARGET_TLS]
            self.per_second_queue_records.append(
                {
                    "simulation_time_sec": float(self.sim_step),
                    "local_queue_vehicles": int(traffic_signal.get_total_queued()),
                }
            )

    net_file = SCENARIO_DIR / "network.net.xml"
    route_file = SCENARIO_DIR / "traffic_20220810_0500.rou.xml"
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
        num_seconds=SIMULATION_DURATION_SEC,
        delta_time=ACTION_INTERVAL_SEC,
        yellow_time=2,
        min_green=5,
        max_green=50,
        enforce_max_green=True,
        reward_fn="diff-waiting-time",
        sumo_seed=seed,
        fixed_ts=fixed_ts,
        sumo_warnings=False,
        additional_sumo_cmd=additional_sumo_cmd,
    )
