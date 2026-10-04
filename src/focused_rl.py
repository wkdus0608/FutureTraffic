"""석천사거리 중심 실험의 관측·보상·운영 제약 정의 (v3).

감지 구간
    진입로마다 정지선에서 차량 경로가 시작되는 곳(상류 최대 300m, 다음 신호나 차로가
    절반 미만으로 줄기 전)까지. 이 지도는 교차로 직전 도로가 11~19m로 잘게 나뉘어
    있어서 SUMO-RL 기본 관측(진입 차로만 봄)은 정지선 앞 2~3대만 센다.
진입 대기 차량
    대기열이 경로 시작점까지 차서 출발 시각이 지났는데도 도로에 들어오지 못한 차량.
    v2까지는 관측·보상 어디에도 없어서, 짧은 진입로(석천로 58m)가 넘치면 DQN이 몰랐다.
보상
    -0.1 × (감지 구간의 정지 차량 + 진입 대기 차량)
운영 제약(SignalConstraints)
    보행자 최소 녹색, 보행자 최대 대기, 차량 최대 대기. 실제 신호 운영에서 지켜야 하는
    규칙이라 DQN이 어길 수 없게 행동을 바로잡고, 바로잡은 횟수를 따로 센다.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from gymnasium import spaces

from rl_environment import configure_sumo

configure_sumo()  # SUMO-RL을 불러오기 전에 SUMO_HOME을 설정한다.
from sumo_rl.environment.observations import ObservationFunction  # noqa: E402

from focused_data import TARGET_TLS  # noqa: E402
from focused_scenario import UPSTREAM_LIMIT_M, FocusedNetwork  # noqa: E402

VEHICLE_SPACING_M = 7.5
REWARD_SCALE = 0.1

# ------------------------------------------------------------------ 보행자 횡단 제약
# 최소 녹색 = 보행 진입 7초 + 횡단 길이 ÷ 1.0m/s (도로교통공단 신호기 설치·관리 기준).
# 보행자는 평행한 직진 신호와 같이 건넌다. 길주로를 건너는 보행자는 석천로 직진과,
# 석천로를 건너는 보행자는 길주로 직진과 같은 현시에 건넌다.
# 최대 대기 = AI Hub 실제 운영 신호(19일)에서 같은 직진 현시가 끝나고 다시 켜질 때까지
# 걸린 시간의 99% 값(석천로 직진 154초, 길주로 직진 118초)을 5초 단위로 올린 값이다.
# 실제 운영보다 보행자를 더 오래 세우지 않게 한다.
WALK_START_S = 7
WALK_SPEED_MPS = 1.0
VEHICLE_MIN_GREEN = 15
CROSSWALKS = {
    "길주로 횡단": {"links": (1, 2, 13, 14),
                "legs": (("477706392_02", "477706392_01"), ("477707013_02", "477707013_01")),
                "max_wait": 155},
    "석천로 횡단": {"links": (5, 6, 7, 8, 9, 17, 18, 19, 20, 21),
                "legs": (("477706911_01", "477706911_02_02"), ("477706883_01", "477706883_02")),
                "max_wait": 120},
}


def crossing_length(fn: FocusedNetwork, legs) -> float:
    """횡단 길이(m) = 한쪽 다리의 진입 + 진출 차로 폭 합. 두 다리 중 긴 쪽."""
    width = lambda e: sum(float(l.get("width", 3.2)) for l in fn.edges[e].findall("lane"))
    return max(width(a) + width(b) for a, b in legs)


def crosswalk_min_green(fn: FocusedNetwork | None = None) -> dict[str, int]:
    fn = fn or FocusedNetwork()
    return {name: WALK_START_S + math.ceil(crossing_length(fn, c["legs"]) / WALK_SPEED_MPS)
            for name, c in CROSSWALKS.items()}


def phase_rules(greens: list[str], pedestrians: bool = True,
                min_green: dict[str, int] | None = None) -> tuple[list[int], dict[str, list[int]]]:
    """현시별 최소 녹색(초)과, 횡단보도별로 그 횡단보도를 열어 주는 현시 번호 목록."""
    if not pedestrians:
        return [VEHICLE_MIN_GREEN] * len(greens), {}
    min_green = min_green or CROSSWALK_MIN_GREEN
    serving = {name: [i for i, g in enumerate(greens) if all(g[l] == "G" for l in c["links"])]
               for name, c in CROSSWALKS.items()}
    phase_min = [max([VEHICLE_MIN_GREEN] + [min_green[n] for n, idx in serving.items() if i in idx])
                 for i in range(len(greens))]
    return phase_min, serving


# ------------------------------------------------------------------ 감지 구간
def build_zones(fn: FocusedNetwork | None = None) -> dict[str, list[str]]:
    """진입 도로별 감지 구간의 차로 목록 (경로가 시작되는 곳까지)."""
    fn = fn or FocusedNetwork()
    zones = {}
    for approach in sorted({a for a, _ in fn.movements}):
        edges = fn._chain(approach, True, UPSTREAM_LIMIT_M) + [approach]
        zones[approach] = [lane.get("id") for e in edges for lane in fn.edges[e].findall("lane")]
    return zones


_FN = FocusedNetwork()
CROSSWALK_MIN_GREEN = crosswalk_min_green(_FN)
ZONES = build_zones(_FN)
APPROACHES = sorted(ZONES)
ZONE_LANES = [lane for lanes in ZONES.values() for lane in lanes]
# 같은 값을 더 적은 조회로 얻기 위해 도로(edge) 단위로도 묶는다.
ZONE_EDGES = {a: sorted({l.rsplit("_", 1)[0] for l in lanes}) for a, lanes in ZONES.items()}
ALL_ZONE_EDGES = [e for edges in ZONE_EDGES.values() for e in edges]


def _capacity(ts, lanes: list[str]) -> float:
    return max(1.0, sum(ts.sumo.lane.getLength(l) for l in lanes) / VEHICLE_SPACING_M)


def pending_by_approach(ts) -> dict[str, int]:
    """출발 시각이 지났지만 아직 도로에 들어오지 못한 차량 수(진입로별)."""
    cache = ts.__dict__.setdefault("_pending_approach", {})  # 신호 객체는 에피소드마다 새로 만들어진다.
    counts = dict.fromkeys(APPROACHES, 0)
    for vehicle in ts.sumo.simulation.getPendingVehicles():
        approach = cache.get(vehicle)
        if approach is None:
            approach = cache[vehicle] = next((e for e in ts.sumo.vehicle.getRoute(vehicle) if e in ZONES), "")
        if approach:
            counts[approach] += 1
    return counts


class ZoneObservation(ObservationFunction):
    """관측 = 현시 · 최소녹색 충족 · 녹색 경과 · 진입 차로 밀도/대기 · 진입로별 감지 구간의
    차량/정지 비율 · 진입 대기 비율 · 횡단보도별 보행자 대기 비율."""

    def __init__(self, ts):
        super().__init__(ts)
        self._capacity = None

    def __call__(self) -> np.ndarray:
        ts = self.ts
        if self._capacity is None:
            self._capacity = {a: _capacity(ts, lanes) for a, lanes in ZONES.items()}
        rules = getattr(ts.env, "signal_constraints", None)
        phase_min = rules.phase_min[ts.green_phase] if rules else ts.min_green
        phase = [1 if ts.green_phase == i else 0 for i in range(ts.num_green_phases)]
        min_green = [0 if ts.time_since_last_phase_change < phase_min + ts.yellow_time else 1]
        elapsed = [min(1.0, ts.time_since_last_phase_change / max(1, ts.max_green))]
        zone = []
        for approach, edges in ZONE_EDGES.items():
            vehicles = sum(ts.sumo.edge.getLastStepVehicleNumber(e) for e in edges)
            halting = sum(ts.sumo.edge.getLastStepHaltingNumber(e) for e in edges)
            zone += [min(1.0, vehicles / self._capacity[approach]), min(1.0, halting / self._capacity[approach])]
        pending = pending_by_approach(ts)
        waiting = [min(1.0, pending[a] / self._capacity[a]) for a in APPROACHES]
        ped = [min(1.0, rules.ped_wait(name, ts) / CROSSWALKS[name]["max_wait"]) if rules and rules.serving else 0.0
               for name in CROSSWALKS]
        obs = (phase + min_green + elapsed + ts.get_lanes_density() + ts.get_lanes_queue()
               + zone + waiting + ped)
        return np.array(obs, dtype=np.float32)

    def observation_space(self) -> spaces.Box:
        size = (self.ts.num_green_phases + 2 + 2 * len(self.ts.lanes) + 3 * len(ZONES) + len(CROSSWALKS))
        return spaces.Box(low=np.zeros(size, dtype=np.float32), high=np.ones(size, dtype=np.float32))


def zone_queue(ts) -> int:
    return sum(ts.sumo.edge.getLastStepHaltingNumber(e) for e in ALL_ZONE_EDGES)


def zone_queue_reward(ts) -> float:
    """보상 = -(감지 구간의 정지 차량 수 + 진입 대기 차량 수) × 0.1."""
    return -REWARD_SCALE * (zone_queue(ts) + len(ts.sumo.simulation.getPendingVehicles()))


# ------------------------------------------------------------------ 운영 제약
class SignalConstraints:
    """DQN의 선택에 실제 신호 운영 규칙을 씌운다.

    1. 현시별 최소 녹색: 보행자가 건너는 현시는 보행 최소 시간(길주로 횡단 46초,
       석천로 횡단 33초), 나머지는 15초. 그 전에는 현시를 바꿀 수 없다.
    2. 보행자 최대 대기: 각 횡단보도는 max_wait초(155초, 120초) 안에 다시 열려야 한다.
       지금 고르는 행동 뒤에도, 남은 횡단보도를 가장 빠른 순서로 열면 모두 제때 열 수
       있는지 미리 계산해(최소 녹색 때문에 바로 못 바꾸는 시간 포함) 그런 행동만 허용한다.
    3. 최대 녹색: 같은 현시를 max_green 넘게 유지하지 않는다. SUMO-RL의 자동 전환(다음
       번호 현시로 넘김)이 보행자 규칙을 모르고 끼어들지 않도록 이 규칙이 먼저 바꾼다.
    4. 차량 최대 대기: 정지선 앞 차량이 max_wait초 이상 멈춰 있으면 그 차량의 신호를
       주는 현시로 바꾼다(1~3을 어기지 않을 때만).
    DQN 선택을 바꾼 횟수를 종류별로 센다.
    """

    def __init__(self, max_wait: float = 120.0, pedestrians: bool = True):
        self.max_wait = max_wait
        self.pedestrians = pedestrians
        self.phase_min: list[int] = []
        self.serving: dict[str, list[int]] = {}
        self.decisions = self.ped_interventions = self.vehicle_interventions = self.max_green_switches = 0
        self.last_served: dict[str, float] = {}

    @property
    def interventions(self) -> int:
        return self.ped_interventions + self.vehicle_interventions + self.max_green_switches

    def bind(self, env) -> "SignalConstraints":
        """환경의 녹색 현시 목록으로 규칙을 만들고 관측 함수가 볼 수 있게 붙인다."""
        greens = [p.state for p in env.traffic_signals[TARGET_TLS].green_phases]
        self.phase_min, self.serving = phase_rules(greens, self.pedestrians)
        missing = [n for n, idx in self.serving.items() if not idx]
        if missing:
            raise ValueError(f"보행자 횡단을 열어 주는 현시가 없습니다: {missing}")
        env.signal_constraints = self
        self.reset()
        return self

    def reset(self) -> None:
        """새 에피소드: 모든 횡단보도가 시작 직전에 열렸던 것으로 본다."""
        self.last_served = dict.fromkeys(self.serving, 0.0)

    def ped_wait(self, name: str, ts) -> float:
        if ts.green_phase in self.serving[name] and not ts.is_yellow:
            return 0.0
        return ts.env.sim_step - self.last_served[name]

    # ---------------------------------------------------------- 보행자 일정 계산
    def _hold(self, ts, phase: int) -> int:
        """phase로 바꾼 뒤 다시 바꿀 수 있는 판단 시점까지의 최소 시간(황색+최소녹색, 판단 간격 단위로 올림)."""
        step = ts.env.delta_time
        return math.ceil((ts.yellow_time + self.phase_min[phase]) / step) * step

    def _lateness(self, ts, phase: int, next_switch: float, ends: dict[str, float]) -> float:
        """phase를 next_switch까지 유지한 뒤 아직 안 열린 횡단보도를 가장 빠른 순서로 열 때,
        가장 늦게 열리는 횡단보도가 max_wait를 넘는 시간(초). 0 이하이면 모두 제때 열 수 있다."""
        pending = [c for c in self.serving if phase not in self.serving[c]]
        if not pending:
            return -math.inf
        best = math.inf
        for order in itertools.permutations(pending):
            t, worst = next_switch, -math.inf
            for c in order:
                p = min(self.serving[c], key=lambda i: self._hold(ts, i))
                worst = max(worst, t + ts.yellow_time - ends[c] - CROSSWALKS[c]["max_wait"])
                t += self._hold(ts, p)
            best = min(best, worst)
        return best

    def _outcome(self, ts, current: int, action: int, now: float) -> float:
        """행동 하나를 골랐을 때의 보행자 지각 시간(위 _lateness, 지금 여는 횡단보도 포함)."""
        ends = dict(self.last_served)
        if action == current:
            return self._lateness(ts, current, now + ts.env.delta_time, ends)
        for c, idx in self.serving.items():
            if current in idx:
                ends[c] = now          # 지금 현시를 떠나는 순간부터 그 횡단보도의 대기가 시작된다.
        late_now = max([now + ts.yellow_time - ends[c] - CROSSWALKS[c]["max_wait"]
                        for c, idx in self.serving.items() if action in idx and current not in idx],
                       default=-math.inf)
        return max(late_now, self._lateness(ts, action, now + self._hold(ts, action), ends))

    def __call__(self, env, action) -> int:
        ts = env.traffic_signals[TARGET_TLS]
        now = env.sim_step
        action = int(action)
        self.decisions += 1
        current = ts.green_phase
        if not ts.is_yellow:
            for name, idx in self.serving.items():
                if current in idx:
                    self.last_served[name] = now
        # SUMO-RL의 최소녹색 검사도 현재 현시의 값으로 맞춘다.
        ts.min_green = self.phase_min[current]
        if ts.time_since_last_phase_change < ts.yellow_time + ts.min_green:
            return current
        phases = range(ts.num_green_phases)
        # 최대 녹색에 닿았으면 지금 현시를 유지하는 선택지는 없다(SUMO-RL 자동 전환보다 먼저 처리).
        at_max = ts.time_since_last_phase_change >= ts.max_green
        allowed = [p for p in phases if not (at_max and p == current)]
        lateness = {p: self._outcome(ts, current, p, now) for p in allowed}
        chosen = action
        if action not in lateness or lateness[action] > 0:
            feasible = [p for p in allowed if lateness[p] <= 0]
            in_order = sorted(feasible, key=lambda p: (p - current) % ts.num_green_phases or ts.num_green_phases)
            if action == current and at_max:
                # 최대 녹색: 순서상 다음 현시로(보행 규칙을 지키는 것 중).
                chosen = in_order[0] if in_order else min(allowed, key=lambda p: lateness[p])
                self.max_green_switches += 1
                return chosen
            # 보행 시간이 임박해 DQN 선택을 따를 수 없다. 지금 현시를 몇 초 더 끌어 봐야 곧 같은
            # 상황이 되므로, 가장 급한 횡단보도를 여는 현시로 바로 넘어간다.
            slack = sorted((CROSSWALKS[c]["max_wait"] - (now + ts.yellow_time - self.last_served[c]), c)
                           for c, idx in self.serving.items() if current not in idx)
            urgent = [p for _, c in slack for p in self.serving[c] if p in feasible]
            if urgent:
                chosen = urgent[0]
            elif in_order:
                chosen = in_order[0]
            else:
                chosen = min(allowed, key=lambda p: lateness[p])   # 이미 늦었으면 가장 덜 늦는 쪽
            self.ped_interventions += 1
            return chosen
        # 차량 최대 대기 (max_wait가 0이면 끈다). 보행자 규칙을 어기지 않는 경우에만 따른다.
        overdue_phase = [0.0] * ts.num_green_phases
        for lane in (ts.lanes if self.max_wait > 0 else []):
            for vehicle in env.sumo.lane.getLastStepVehicleIDs(lane):
                wait = env.sumo.vehicle.getWaitingTime(vehicle)
                if wait < self.max_wait:
                    continue
                for tls_id, link, _, _ in env.sumo.vehicle.getNextTLS(vehicle):
                    if tls_id != TARGET_TLS:
                        continue
                    serving = [i for i, p in enumerate(ts.green_phases) if p.state[link] in "Gg"]
                    if current in serving:
                        break  # 지금 현시가 이미 이 차량의 신호를 주고 있다.
                    for i in serving:
                        overdue_phase[i] += wait
                    break
        if max(overdue_phase) > 0:
            target = max(phases, key=lambda i: overdue_phase[i])
            if target != action and lateness.get(target, math.inf) <= 0:
                self.vehicle_interventions += 1
                return target
        return chosen


def pedestrian_service(states: list[str]) -> dict[str, dict[str, float]]:
    """매초 신호 상태로 횡단보도별 보행 녹색 길이와 보행자 대기(녹색이 다시 켜질 때까지)를 잰다.

    처음과 끝에서 잘린 녹색·대기는 길이를 알 수 없으므로 빼고 센다.
    """
    out = {}
    for name, c in CROSSWALKS.items():
        runs, gaps = [], []
        green = red = 0
        started = False   # 시작 직후의 녹색/적색은 잘렸을 수 있다.
        seen_green = False
        for s in states:
            if all(s[l] == "G" for l in c["links"]):
                if red and seen_green:
                    gaps.append(red)
                red = 0
                green += 1
            else:
                if green and started:
                    runs.append(green)
                if green:
                    seen_green = True
                green = 0
                red += 1
                started = True
        limit = c["max_wait"]
        out[name] = {"greens": len(runs), "short_greens": sum(r < CROSSWALK_MIN_GREEN[name] for r in runs),
                     "waits": len(gaps), "wait_sum_s": float(sum(gaps)), "max_wait_s": float(max(gaps, default=0)),
                     "waits_over_limit": sum(g > limit for g in gaps)}
    return out


# ------------------------------------------------------------------ 학습기
def hold_aware_dqn_class():
    """운영 제약이 행동을 바꾸는 환경을 위한 DQN (stable-baselines3 DQN을 조금 고친 것).

    1. 재생 버퍼에는 DQN이 고른 행동이 아니라 실제로 실행된 행동(info["executed_action"])을
       저장한다. 최소 녹색·보행 규칙이 행동을 바꿨는데 고른 행동으로 저장하면 행동과 결과가
       어긋나서, 어떤 선택이 좋은지 배울 수 없다(v3 첫 학습에서 확인: 판단의 85%가 무시됐다).
    2. 다음 상태가 최소 녹색 진행 중이면(관측의 최소녹색 충족 값 = 0) 현재 현시 유지만
       가능하므로, 미래 가치로 '현재 현시'의 Q값만 쓴다. 실행할 수 없는 행동의 학습되지 않은
       Q값이 목표값에 섞이지 않게 한다.
    관측 앞부분은 [현시 원-핫(n), 최소녹색 충족(1), ...] 순서여야 한다(ZoneObservation).
    """
    import torch as th
    import torch.nn.functional as F
    from stable_baselines3 import DQN

    class HoldAwareDQN(DQN):
        def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, dones, infos):
            executed = [info.get("executed_action") for info in infos]
            if all(a is not None for a in executed):
                buffer_action = np.asarray(executed, dtype=buffer_action.dtype).reshape(buffer_action.shape)
            super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)

        def allowed_next_values(self, next_q: "th.Tensor", next_obs: "th.Tensor") -> "th.Tensor":
            n = self.action_space.n
            current = next_obs[:, :n].argmax(dim=1, keepdim=True)
            holding = next_obs[:, n] < 0.5
            return th.where(holding, next_q.gather(1, current).squeeze(1), next_q.max(dim=1).values)

        def train(self, gradient_steps: int, batch_size: int = 100) -> None:
            self.policy.set_training_mode(True)
            self._update_learning_rate(self.policy.optimizer)
            losses = []
            for _ in range(gradient_steps):
                replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
                discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma
                with th.no_grad():
                    next_q = self.q_net_target(replay_data.next_observations)
                    next_values = self.allowed_next_values(next_q, replay_data.next_observations).reshape(-1, 1)
                    target = replay_data.rewards + (1 - replay_data.dones) * discounts * next_values
                current_q = th.gather(self.q_net(replay_data.observations), dim=1, index=replay_data.actions.long())
                loss = F.smooth_l1_loss(current_q, target)
                losses.append(loss.item())
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
            self._n_updates += gradient_steps
            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            self.logger.record("train/loss", np.mean(losses))

    return HoldAwareDQN
