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


# 교차로 안 충돌까지 검사한다. warn은 충돌을 경고로 남기기만 하고 차량을 치우지 않는다. 기본값(teleport)이나
# remove는 부딪힌 차를 옮기거나 지워 교통을 바꾸고, 충돌을 낸 정책의 정체를 줄여 주기까지 한다. 그래서 warn으로
# 모든 정책이 같은 교통을 유지하고, 충돌은 기록 파일로 따로 판정한다(focused_eval.judge_collisions).
# 차가 그대로 움직였다는 것은 충돌이 무해하다는 증거가 아니다.
COLLISION_OPTIONS = "--collision.check-junctions true --collision.action warn"


_RULE_SIGNAL_CLASS = None


def rule_signal_class():
    """전환(i→j)마다 황색 뒤에 전적색(모든 방향 적색)을 두는 SUMO-RL 신호 객체.

    전적색 길이는 focused_data.variant()["all_red"][(i, j)]다. 없으면 0초라 SUMO-RL 기본 동작(황색 뒤 바로 녹색)과
    같다. 전환 동안 yellow_time을 '황색 + 전적색'으로 두어, 최소 녹색·관측·운영 규칙이 쓰는 값이 그대로 맞게 한다.
    전환 중(황색·전적색)에 판단 시각이 와도 신호 상태를 바꾸지 않는다(SUMO-RL 기본은 현재 녹색 상태를 다시 써서
    전환을 끊어 버린다. 황색 4초 < 판단 간격 5초일 때는 일어나지 않던 일이다).
    """
    global _RULE_SIGNAL_CLASS
    if _RULE_SIGNAL_CLASS is None:
        from sumo_rl.environment.traffic_signal import TrafficSignal

        class RuleTrafficSignal(TrafficSignal):
            def _build_phases(self):
                super()._build_phases()
                self.base_yellow = self.yellow_time
                self.all_red = {}
                self.all_red_state = {}
                self._from_green = None
                if self.env.fixed_ts:
                    return
                from focused_data import variant
                self.all_red = variant()["all_red"]
                self.all_red_state = {pair: self.all_phases[idx].state.replace("y", "r")
                                      for pair, idx in self.yellow_dict.items()}

            def set_next_phase(self, new_phase):
                if self.is_yellow:
                    self.next_action_time = self.env.sim_step + self.delta_time
                    return
                before = self.green_phase
                super().set_next_phase(new_phase)
                if self.is_yellow and self.green_phase != before:       # 전환이 시작됐다
                    self._from_green = before
                    self.yellow_time = self.base_yellow + self.all_red.get((before, self.green_phase), 0)

            def update(self):
                self.time_since_last_phase_change += 1
                if not self.is_yellow:
                    return
                t = self.time_since_last_phase_change
                if t == self.base_yellow and self.yellow_time > self.base_yellow:
                    self.sumo.trafficlight.setRedYellowGreenState(
                        self.id, self.all_red_state[(self._from_green, self.green_phase)])
                if t == self.yellow_time:
                    self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
                    self.is_yellow = False

        _RULE_SIGNAL_CLASS = RuleTrafficSignal
    return _RULE_SIGNAL_CLASS


def make_environment(*, net_file: Path, route_file: Path, simulation_duration_sec: int, observation_class,
                     reward_fn, seed: int = 42, fixed_ts: bool = False, additional_sumo_cmd: str | None = None,
                     record_per_second_queue: bool = False, extra_queue_edges: list[str] | None = None,
                     min_green: int = 15, max_green: int = 120, yellow_time: int = 3):
    """학습과 평가가 같이 쓰는 SUMO-RL 환경. 204820 하나만 제어 대상이다.

    fixed_ts: True이면 지도·추가 파일의 신호 프로그램대로 돌고(고정신호·감응식), False이면
        5초마다 받은 행동(녹색 현시 번호)으로 신호를 바꾼다(DQN).
    record_per_second_queue: True이면 매초 대기행렬·진입 대기·신호 상태를 기록한다(평가용).
    순간이동은 끄고(-1), 도로에 들어오지 못한 차량도 버리지 않는다(SUMO-RL 기본값).
    대기열이 있는 상태에서 시작하려면 reset 전에 env.warmup_states(1초마다의 실제 신호 상태)와
    env.warmup_handover(넘겨받을 녹색 상태)를 넣는다. reset이 그 시간만큼 실제 신호로 돌린 뒤,
    그 녹색이 막 켜진 상태로 넘겨받는다.
    """
    for path in (net_file, route_file):
        if not Path(path).exists():
            raise FileNotFoundError(f"필요한 파일이 없습니다: {path}")
    configure_sumo()
    from sumo_rl import SumoEnvironment
    import sumo_rl.environment.env as sumo_rl_env
    sumo_rl_env.TrafficSignal = rule_signal_class()      # 환경이 신호 객체를 만들 때 쓰는 클래스

    class FocusedEnvironment(SumoEnvironment):
        warmup_states: list[str] = []
        warmup_handover: str | None = None

        def reset(self, *args, **kwargs):
            observation, info = super().reset(*args, **kwargs)
            if not self.warmup_states:
                return observation, info
            ts = self.traffic_signals[TARGET_TLS]
            for state in self.warmup_states:          # 실제 신호로 1초씩 돌려 대기열을 만든다
                self.sumo.trafficlight.setRedYellowGreenState(TARGET_TLS, state)
                self._sumo_step()
            greens = [p.state for p in ts.green_phases]
            ts.green_phase = greens.index(self.warmup_handover)
            ts.is_yellow = False
            ts.yellow_time = getattr(ts, "base_yellow", ts.yellow_time)     # 넘겨받은 녹색은 전환 중이 아니다
            ts.time_since_last_phase_change = 0
            ts.next_action_time = self.sim_step
            self.sumo.trafficlight.setRedYellowGreenState(TARGET_TLS, self.warmup_handover)
            return self._compute_observations()[self.ts_ids[0]], self._compute_info()

    class PerSecondQueueEnvironment(FocusedEnvironment):
        """매초 신호 상태·감지 구간 정지 차량·진입 대기 차량을 남기는 평가용 환경. 신호 선택·보상은 바꾸지 않는다.

        감지 구간 도로의 정지 차량 수와 신호 상태는 SUMO '구독'으로 한 번에 받아(매초 수십 번 묻지 않고)
        하루 연속 평가를 빠르게 한다. 값은 직접 묻는 것과 같다.
        """

        def __init__(self, *args, **kwargs):
            self.per_second_queue_records: list[dict] = []
            self._subscribed = False
            super().__init__(*args, **kwargs)

        def reset(self, *args, **kwargs):
            self.per_second_queue_records.clear()
            self._subscribed = False          # SUMO를 새로 시작하므로 구독도 새로 한다
            return super().reset(*args, **kwargs)

        def _sumo_step(self) -> None:
            import traci.constants as tc
            if not self._subscribed:
                for edge in extra_queue_edges or []:
                    self.sumo.edge.subscribe(edge, [tc.LAST_STEP_VEHICLE_HALTING_NUMBER])
                self.sumo.trafficlight.subscribe(TARGET_TLS, [tc.TL_RED_YELLOW_GREEN_STATE])
                self._subscribed = True
            super()._sumo_step()   # SUMO를 정확히 1초 진행한다.
            halting = self.sumo.edge.getAllSubscriptionResults()
            self.per_second_queue_records.append({
                "simulation_time_sec": float(self.sim_step),
                "tls_state": self.sumo.trafficlight.getSubscriptionResults(TARGET_TLS)[tc.TL_RED_YELLOW_GREEN_STATE],
                "zone_queue_vehicles": (sum(v[tc.LAST_STEP_VEHICLE_HALTING_NUMBER] for v in halting.values())
                                        if extra_queue_edges else None),
                # 출발 시각이 지났지만 대기열 때문에 도로에 들어오지 못한 차량 수
                "pending_vehicles": len(self.sumo.simulation.getPendingVehicles()),
            })

    environment_class = PerSecondQueueEnvironment if record_per_second_queue else FocusedEnvironment
    return environment_class(
        add_system_info=False, add_per_agent_info=False, observation_class=observation_class,
        net_file=str(Path(net_file).resolve()), route_file=str(Path(route_file).resolve()), out_csv_name=None,
        single_agent=True, ts_ids=[TARGET_TLS], use_gui=False, num_seconds=simulation_duration_sec,
        delta_time=ACTION_INTERVAL_SEC, yellow_time=yellow_time, min_green=min_green, max_green=max_green,
        enforce_max_green=True, reward_fn=reward_fn, sumo_seed=seed, fixed_ts=fixed_ts, sumo_warnings=False,
        additional_sumo_cmd=" ".join(x for x in (COLLISION_OPTIONS, additional_sumo_cmd) if x),
    )


# ------------------------------------------------------------------ SUMO 결과 파일
def sumo_output_options(run_dir: Path) -> tuple[str, Path, Path]:
    """차량별 결과(tripinfo), 안전 통계(statistics), 충돌 기록(collisions.xml)을 남기는 SUMO 옵션."""
    tripinfo = (run_dir / "tripinfo.xml").resolve()
    statistics_file = (run_dir / "statistics.xml").resolve()
    collisions = (run_dir / "collisions.xml").resolve()
    for path in (tripinfo, statistics_file, collisions):
        if " " in str(path):
            raise ValueError(f"평가 출력 경로에 공백을 쓸 수 없습니다: {path}")
    command = " ".join(["--tripinfo-output", str(tripinfo), "--tripinfo-output.write-unfinished", "true",
                        "--statistic-output", str(statistics_file),
                        "--collision-output", str(collisions),
                        "--duration-log.statistics", "true", "--no-step-log", "true"])
    return command, tripinfo, statistics_file


def parse_collisions(path: Path) -> list[dict]:
    """충돌 기록 파일의 충돌을 모두 읽는다(시각·종류·차로·두 차량·차종·속도·앞뒤 좌표). 판정은 focused_eval.judge_collisions.

    파일이 없으면 충돌 검사가 돌지 않은 것이므로 '충돌 0'으로 보지 않고 멈춘다.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"충돌 기록 파일이 없습니다(충돌 검사가 돌지 않음): {path}")
    return [dict(c.attrib) for c in ET.parse(path).getroot().iter("collision")]


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
