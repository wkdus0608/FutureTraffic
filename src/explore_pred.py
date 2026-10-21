"""[추가 탐색] 예측 제어(PRED, 롤링 호라이즌 + 동적계획법) — 지문 밖, 진행 기록 98절 계획 그대로.

5초마다 앞으로 HORIZON_S초의 녹색 계획을 5초 단위로 운영 규칙 안에서 모두 따져, 이동별 줄로 매긴 점수가 가장 작은 계획의
첫 5초만 실행한다(5초 뒤 다시 짠다).
- 점수 = 계획 기간 Σ(기다리는 차 수) × 5초(한 단계의 처음·끝 줄의 평균) + 끝에 남은 차 × TERMINAL_S초
- 줄 모형: 이동마다 줄 하나. 보호 녹색(G)일 때만 줄이 줄고, 속도는 그 녹색에서 그 이동의 보호 녹색 연결이 있는 진입 차로 수
  × 차로당 SAT_HEADWAY_S초에 한 대. 녹색이 켜진 뒤 STARTUP_LOST_S초와 황색·전적색 동안은 통과가 없다. 보호 녹색이 한 번도 없는
  이동(우회전 등)은 어느 계획에서나 같으므로 점수에서 뺀다.
- 운영 규칙: 최소 녹색, 최대 녹색, 보행 최대 대기(SignalConstraints._outcome를 그대로 불러 같은 계산), 전환마다 황색 + 전적색.
  차량 최대 누적 대기는 계획에서 계산하지 않는다. 첫 단계는 늘 SignalConstraints.allowed(env) 안에서 고르므로 실행에서는 지켜진다.
- 상태 합치기: 같은 시각에 같은 신호 상태(녹색, 전환 시각, 그 전환의 황색 + 전적색, 횡단보도별 마지막 보행 녹색 시각)에 이른
  계획 가운데, 다른 계획보다 줄이 모두 같거나 많고 점수도 같거나 큰 계획은 버린다. 줄 모형은 처음 줄이 적을수록 앞으로의 줄과
  점수가 늘 같거나 적으므로, 버린 계획이 가장 좋은 계획일 수는 없다. 그래서 결과는 모든 계획을 따진 것과 같다
  (98절의 '같은 상태면 하나만'을, 결과를 바꾸지 않고 더 많이 합치도록 넓힌 것. 시험에서 모든 계획을 따진 결과와 맞춰 본다).
- 동점: 지금 녹색을 유지하는 계획, 그다음 첫 녹색 번호가 작은 계획(단순 제어와 같은 규칙).

정보 세 가지(98.2절, 한 층씩 더함):
- P0: 단순 제어와 똑같다. 감지 구간에서 멈춘 차 + 진입 대기 차(이동은 차의 경로로). 새로 오는 차는 없다고 본다.
- P1: P0 + 감지 구간 안 달리는 차는 도로 제한속도로 정지선까지 와서 그 이동의 줄 끝에 선다. 구간 밖에서 올 차는 최근
  ENTRY_WINDOW_S초 동안 진입로마다 구간에 새로 들어온 차의 비율(이동 비율도 같은 차들로)로, 구간 통과 시간 뒤부터 고르게 온다.
- P2(12개 지도만): P1 + 동쪽 이웃 신호 204822. 직진 녹색의 지금 상태와 바뀐 뒤 지난 시간, 최근 NEIGHBOR_WINDOW_S초의 켜짐·꺼짐
  길이 중앙값, 그때마다 동쪽 구간에 들어온 비율. 앞으로는 지금 상태가 중앙값 길이까지 이어진 뒤 중앙값 길이로 번갈아 온다고
  보고, 동쪽 구간 밖에서 올 차를 그 상태의 비율로 넣는다(앞으로의 시간표는 쓰지 않는다).
- 기록이 짧을 때(조각 시작 직후) 비율이 튀지 않게, 비율을 낼 때 나누는 시간은 MIN_RATE_WINDOW_S초 이상으로 둔다.

평가 코드(focused_eval·network_eval·focused_conflicts)는 고치지 않는다. 실행할 때만 focused_eval.QueueFirst를 이 파일의
제어로 바꿔 끼우고 '단순 제어' 실행을 돌린다(평가 지문 그대로, 결과 폴더 이름만 pred_<변형>). Max-Pressure(81절)와 같은 방식이다.
내다보는 길이는 check가 판단 시간만 보고 정한다(98.1: 평균 0.3초를 넘으면 90초). single·network12는 같은 코드의 check 결과만 받는다.

사용(Colab 29·30번 칸):
  FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240 python src/explore_pred.py check --out <Drive>/explore_pred/check.json
  FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240 python src/explore_pred.py single --check <check.json> --out <Drive>/explore_pred/single \
      --h1-models <H1 seed 5개>
  FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240 python src/explore_pred.py network12 --check <check.json> --out <Drive>/explore_pred/network12
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

HORIZON_S = 120               # 98.1: 내다보는 길이
FALLBACK_HORIZON_S = 90       # 98.1: 판단 한 번 평균이 TIME_LIMIT_S를 넘으면(check에서 시간으로만 정함)
TIME_LIMIT_S = 0.3
SAT_HEADWAY_S = 2.0           # 차로당 2.0초에 한 대(시간당 1,800대)
STARTUP_LOST_S = 2.0
TERMINAL_S = 75.0             # 실제 신호 주기 약 150초의 절반
ENTRY_WINDOW_S = 300.0        # P1: 구간 밖 도착 비율을 재는 길이
NEIGHBOR_WINDOW_S = 900.0     # P2: 이웃 신호 켜짐·꺼짐 길이와 그때의 도착 비율을 재는 길이
MIN_RATE_WINDOW_S = 60.0
DEFAULT_RUN_S = {True: 60.0, False: 90.0}   # P2: 지난 켜짐·꺼짐 기록이 하나도 없을 때(첫 판단 등)만 쓰는 길이
VARIANTS = ("p0", "p1", "p2")
SINGLE_VARIANTS = ("p0", "p1")
SEEDS = [42, 43, 44, 45, 46]
CHECK_DATE, CHECK_HOUR = "20220811", 18     # 학습 날짜(검증·시험 날짜가 아님)
TOL = 1e-9


def code_fingerprint() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


# ------------------------------------------------------------------ 계획(SUMO 없음)
@dataclass(frozen=True)
class SignalState:
    phase: int
    switch_time: float                 # 마지막 전환(황색)이 시작된 시각
    intergreen: float                  # 그 전환의 황색 + 전적색(넘겨받은 녹색은 황색 길이)
    served: tuple[float, ...]          # 횡단보도별 마지막 보행 녹색 시각(SignalModel.crosswalks 순서)


@dataclass
class SignalModel:
    served: list[list[tuple[int, float]]]      # 녹색마다 (이동 번호, 통과 차/초)
    phase_min: list[int]
    max_green: float
    step: float
    guard: object                              # SignalConstraints(intergreen·serving·_outcome)
    crosswalks: tuple[str, ...]
    terminal_s: float = TERMINAL_S
    startup_s: float = STARTUP_LOST_S

    def __post_init__(self):
        self._ts = SimpleNamespace(env=SimpleNamespace(delta_time=self.step))   # _outcome는 판단 간격만 읽는다
        self._rates = [dict(row) for row in self.served]

    def is_transition(self, st: SignalState, t: float) -> bool:
        return t - st.switch_time < st.intergreen

    def actions(self, st: SignalState, t: float) -> list[int]:
        """SignalConstraints._decide와 같은 규칙(차량 누적 대기는 뺌)으로 그대로 실행될 녹색."""
        tau = t - st.switch_time
        if tau < st.intergreen + self.phase_min[st.phase]:
            return [st.phase]
        allowed = [p for p in range(len(self.phase_min)) if not (tau >= self.max_green and p == st.phase)]
        served = dict(zip(self.crosswalks, st.served))
        late = {p: self.guard._outcome(self._ts, st.phase, p, t, served) for p in allowed}
        feasible = [p for p in allowed if late[p] <= 0]
        return feasible or [min(allowed, key=lambda p: late[p])]      # 이미 늦었으면 가장 덜 늦는 쪽(규칙과 같음)

    def advance(self, st: SignalState, t: float, action: int) -> tuple[SignalState, float]:
        """[t, t + step) 동안 action을 실행한 다음 상태와, 그동안 action 녹색이 차를 통과시킨 초."""
        if action != st.phase:
            st = SignalState(action, t, float(self.guard.intergreen(st.phase, action)), st.served)
        start = st.switch_time + st.intergreen + self.startup_s
        effective = max(0.0, t + self.step - max(t, start))
        t2 = t + self.step
        if not self.is_transition(st, t2):
            serving = self.guard.serving
            st = SignalState(st.phase, st.switch_time, st.intergreen,
                             tuple(t2 if st.phase in serving[c] else s for c, s in zip(self.crosswalks, st.served)))
        return st, effective


def _insert(nodes: list, cost: float, q: tuple, rank: tuple) -> None:
    """같은 신호 상태의 계획 목록에 넣는다. 줄이 모두 같거나 적고 점수도 같거나 작은 계획이 있으면 버린다(모두 같으면 동점 규칙)."""
    keep = []
    for c, q0, r in nodes:
        if c <= cost + TOL and all(a <= b + TOL for a, b in zip(q0, q)):
            if c < cost - TOL or any(a < b - TOL for a, b in zip(q0, q)) or r <= rank:
                return
        if not (cost <= c + TOL and all(a <= b + TOL for a, b in zip(q, q0))):
            keep.append((c, q0, r))
    keep.append((cost, q, rank))
    nodes[:] = keep


def step_queue(q: tuple, arriving: list[float], rate: dict[int, float], effective: float) -> tuple:
    return tuple(max(0.0, x + y - rate.get(m, 0.0) * effective) for m, (x, y) in enumerate(zip(q, arriving)))


def plan(model: SignalModel, state: SignalState, now: float, q0: list[float], arrivals: list[list[float]],
         first_actions: list[int], merge: bool = True) -> tuple[int, dict]:
    """앞으로 len(arrivals)단계의 가장 좋은 계획의 첫 녹색. arrivals[k][m] = [now + k·step, now + (k + 1)·step)에
    정지선에 닿는 이동 m의 차 수. merge=False면 합치지 않고 모든 계획을 따진다(시험용).
    돌려줌: (첫 녹색, {'cost': 점수, 'nodes': 따진 계획 단계 수, 'plan': 가장 좋은 계획의 녹색들})."""
    current, step = state.phase, model.step
    level = {state: [(0.0, tuple(q0), (False, current, ()))]}
    expanded = 0
    for k, arriving in enumerate(arrivals):
        t = now + k * step
        nxt: dict[SignalState, list] = {}
        for st, nodes in level.items():
            for a in (first_actions if k == 0 else model.actions(st, t)):
                st2, effective = model.advance(st, t, a)
                rate = model._rates[a] if effective > 0 else {}
                bucket = nxt.setdefault(st2, [])
                for cost, q, rank in nodes:
                    expanded += 1
                    q2 = step_queue(q, arriving, rate, effective)
                    seq = rank[2] + (a,)
                    new_rank = (seq[0] != current, seq[0], seq)
                    c2 = cost + step * (sum(q) + sum(q2)) / 2
                    if merge:
                        _insert(bucket, c2, q2, new_rank)
                    else:
                        bucket.append((c2, q2, new_rank))
        level = nxt
    total, rank = min(((cost + model.terminal_s * sum(q), rank) for nodes in level.values() for cost, q, rank in nodes),
                      key=lambda x: (round(x[0], 6), x[1][0], x[1][1]))
    return rank[2][0], {"cost": total, "nodes": expanded, "plan": list(rank[2])}


# ------------------------------------------------------------------ 정보(SUMO 없음)
def zone_geometry(net, zone_edges: dict[str, list[str]]) -> dict:
    """감지 구간 도로마다 정지선까지 거리(도로 끝 기준)·제한속도·길이, 진입로마다 구간 입구 → 정지선 시간(가장 먼 도로부터)."""
    zone_of, offset, speed, length, travel = {}, {}, {}, {}, {}
    for approach, edges in zone_edges.items():
        members = set(edges)
        offset[approach] = 0.0
        frontier = [approach]
        while frontier:
            e = net.getEdge(frontier.pop())
            for pred in e.getIncoming():
                pid = pred.getID()
                if pid in members and pid not in offset:
                    offset[pid] = offset[e.getID()] + e.getLength()
                    frontier.append(pid)
        far = 0.0
        for eid in edges:
            e = net.getEdge(eid)
            if eid not in offset:
                raise SystemExit(f"{eid}: 감지 구간 도로가 {approach}까지 이어지지 않습니다.")
            zone_of[eid], speed[eid], length[eid] = approach, e.getSpeed(), e.getLength()
            far = max(far, sum(net.getEdge(x).getLength() / net.getEdge(x).getSpeed()
                               for x in _path_to_stop(net, eid, approach, offset)))
        travel[approach] = far
    return {"zone_of": zone_of, "offset": offset, "speed": speed, "length": length, "travel": travel}


def _path_to_stop(net, eid: str, approach: str, offset: dict) -> list[str]:
    """eid에서 정지선 쪽 도로까지 감지 구간 안의 길(정지선까지 거리가 줄어드는 도로를 따라)."""
    path = [eid]
    while path[-1] != approach:
        e = net.getEdge(path[-1])
        nxt = [c.getID() for c in e.getOutgoing() if c.getID() in offset
               and abs(offset[c.getID()] + c.getLength() - offset[e.getID()]) < 1e-6]
        if not nxt:
            raise SystemExit(f"{eid}: 정지선까지 길을 찾지 못했습니다.")
        path.append(nxt[0])
    return path


def entry_rates(entries, now: float, first_time: float | None, window: float = ENTRY_WINDOW_S) -> dict[tuple[str, int], float]:
    """최근 window초 동안 (진입로, 이동)마다 구간에 새로 들어온 차 ÷ 초. 기록이 짧으면 MIN_RATE_WINDOW_S초로 나눈다."""
    span = max(MIN_RATE_WINDOW_S, min(window, now - (first_time if first_time is not None else now)))
    out: dict[tuple[str, int], float] = {}
    for t, approach, i, _ in entries:
        if t >= now - window:
            out[(approach, i)] = out.get((approach, i), 0.0) + 1.0 / span
    return out


def neighbor_history(durations: list[float], on: list[bool], phase: int, spent: float,
                     window: float = NEIGHBOR_WINDOW_S) -> tuple[bool, float, dict[bool, float]]:
    """이웃 신호 프로그램의 지난 현시로: (지금 직진 녹색인가, 지금 상태가 이어진 시간, 최근 window초 안에 끝난 켜짐·꺼짐 길이 중앙값).
    지금 진행 중인 덩어리는 중앙값에 넣지 않는다. 기록이 없는 상태는 DEFAULT_RUN_S."""
    cur = on[phase]
    elapsed, k = spent, phase - 1
    while k >= 0 and on[k] == cur:
        elapsed += durations[k]
        k -= 1
    runs, covered = {True: [], False: []}, elapsed
    while k >= 0 and covered < window:
        state, length = on[k], 0.0
        while k >= 0 and on[k] == state:
            length += durations[k]
            k -= 1
        runs[state].append(length)
        covered += length
    med = {s: (statistics.median(runs[s]) if runs[s] else DEFAULT_RUN_S[s]) for s in (True, False)}
    return cur, elapsed, med


def predicted_on(cur: bool, elapsed: float, med: dict[bool, float], t: float) -> bool:
    """지금부터 t초 뒤 상태: 지금 상태가 중앙값 길이까지 이어지고(이미 넘었으면 지금 바로 뒤에 바뀜), 그 뒤 중앙값 길이로
    번갈아 온다. t = 0이면 늘 지금 상태다."""
    state, left = cur, max(0.0, med[cur] - elapsed)
    while t > left:
        t -= left
        state = not state
        left = med[state]
    return state


# ------------------------------------------------------------------ SUMO에 붙는 제어
def controller_class(variant: str, horizon_s: int):
    from focused_eval import QueueFirst
    if variant not in VARIANTS:
        raise SystemExit(f"변형은 {VARIANTS} 가운데 하나: {variant}")

    class PredictiveControl(QueueFirst):
        """단순 제어와 같은 자리(5초마다 허용된 녹색 목록을 받아 하나를 돌려줌)."""
        stats: dict = {}

        def __init__(self, movement_of, greens):
            super().__init__(movement_of, greens)
            self.greens = greens
            self._model = None
            self._seen: set[str] = set()
            self._entries: deque = deque()            # (시각, 진입로, 이동 번호, 204822 직진 녹색이었나)
            self._first_time: float | None = None
            self._neighbor_log: deque = deque()       # P2: (시각, 204822 직진 녹색이었나), 판단마다
            type(self).stats = {"variant": variant, "horizon_s": horizon_s, "code": code_fingerprint(), "decisions": 0,
                                "planned": 0, "seconds": 0.0, "max_seconds": 0.0, "nodes": 0, "outside_allowed": 0}

        def _setup(self, env) -> None:
            import sumolib
            from focused_data import TARGET_TLS
            from focused_rl import CROSSWALKS, MOVEMENT_LINKS, ZONE_EDGES
            sumo = env.sumo
            lane_of = [links[0][0] if links else None for links in sumo.trafficlight.getControlledLinks(TARGET_TLS)]
            self.movements = sorted({m for p in self.served for m in p})
            self.index = {m: i for i, m in enumerate(self.movements)}
            caps = [[(self.index[m], len({lane_of[l] for l in MOVEMENT_LINKS[m] if g[l] == "G"}) / SAT_HEADWAY_S)
                     for m in self.movements if any(g[l] == "G" for l in MOVEMENT_LINKS[m])] for g in self.greens]
            ts = env.traffic_signals[TARGET_TLS]
            self._model = SignalModel(served=caps, phase_min=list(env.signal_constraints.phase_min),
                                      max_green=float(ts.max_green), step=float(env.delta_time),
                                      guard=env.signal_constraints, crosswalks=tuple(CROSSWALKS))
            self._steps = int(round(horizon_s / env.delta_time))
            self.geo = zone_geometry(sumolib.net.readNet(env._net), ZONE_EDGES)
            if variant == "p2":
                import explore_coord12 as ec
                self._east = ec.east_layout()
                self._runs = ec.program_runs(sumo, self._east["straight_links"])

        def _neighbor_on(self, sumo) -> bool:
            import explore_coord12 as ec
            return ec.green_share(sumo.trafficlight.getRedYellowGreenState(ec.NEIGHBOR), self._east["straight_links"]) >= 0.5

        def _observe(self, env) -> None:
            """매 판단(규칙이 녹색을 하나만 남겨도): 구간에 새로 들어온 차와 이웃 신호 상태를 적는다."""
            if variant == "p0":
                return
            sumo, now = env.sumo, float(env.sim_step)
            on = self._neighbor_on(sumo) if variant == "p2" else None
            fresh = self._first_time is None
            if fresh:
                self._first_time = now
            for e, approach in self.geo["zone_of"].items():
                for v in sumo.edge.getLastStepVehicleIDs(e):
                    if v in self._seen:
                        continue
                    self._seen.add(v)
                    i = self.index.get(self.movement_of.get(v))
                    if not fresh and i is not None:        # 처음 본 차는 '새로 들어온 차'가 아니다
                        self._entries.append((now, approach, i, on))
            keep = NEIGHBOR_WINDOW_S if variant == "p2" else ENTRY_WINDOW_S
            while self._entries and self._entries[0][0] < now - keep:
                self._entries.popleft()
            if variant == "p2" and not fresh:          # 이 판단이 센 '새로 들어온 차'의 앞 5초를 그 상태로 적는다(첫 판단은 세지 않음)
                self._neighbor_log.append((now, on))
                while self._neighbor_log and self._neighbor_log[0][0] <= now - NEIGHBOR_WINDOW_S:
                    self._neighbor_log.popleft()

        def queues_and_arrivals(self, env) -> tuple[list[float], list[list[float]]]:
            sumo, now, step = env.sumo, float(env.sim_step), float(env.delta_time)
            n, K, geo = len(self.movements), self._steps, self.geo
            q0 = [0.0] * n
            arrivals = [[0.0] * n for _ in range(K)]
            for e in geo["zone_of"]:
                for v in sumo.edge.getLastStepVehicleIDs(e):
                    i = self.index.get(self.movement_of.get(v))
                    if i is None:
                        continue
                    if sumo.vehicle.getSpeed(v) < 0.1:                     # 단순 제어와 같은 '기다리는 차'
                        q0[i] += 1
                    elif variant != "p0":
                        d = geo["offset"][e] + geo["length"][e] - sumo.vehicle.getLanePosition(v)
                        k = int(max(0.0, d) / geo["speed"][e] // step)
                        if k < K:
                            arrivals[k][i] += 1
            for v in sumo.simulation.getPendingVehicles():
                i = self.index.get(self.movement_of.get(v))
                if i is not None:
                    q0[i] += 1
            if variant != "p0":
                self._outside(arrivals, sumo, now, step)
            return q0, arrivals

        def _outside(self, arrivals, sumo, now: float, step: float) -> None:
            K = len(arrivals)
            east = None
            if variant == "p2":
                import explore_coord12 as ec
                east = ec.APPROACH
            for (approach, i), r in entry_rates(self._entries, now, self._first_time).items():
                if approach == east:
                    continue
                for k in range(math.ceil(self.geo["travel"][approach] / step), K):
                    arrivals[k][i] += r * step
            if east is None:
                return
            import explore_coord12 as ec
            durations, on = self._runs
            cur, elapsed, med = neighbor_history(durations, on, sumo.trafficlight.getPhase(ec.NEIGHBOR),
                                                 sumo.trafficlight.getSpentDuration(ec.NEIGHBOR))
            counts = {True: {}, False: {}}
            for t, a, i, o in self._entries:
                if a == east and t >= now - NEIGHBOR_WINDOW_S:
                    counts[o][i] = counts[o].get(i, 0) + 1
            seconds = {s: step * sum(1 for t, o in self._neighbor_log if o == s) for s in (True, False)}
            rates = {s: {i: c / max(MIN_RATE_WINDOW_S, seconds[s]) for i, c in counts[s].items()} for s in (True, False)}
            travel = self.geo["travel"][east]
            for k in range(math.ceil(travel / step), K):
                for i, r in rates[predicted_on(cur, elapsed, med, k * step - travel)].items():
                    arrivals[k][i] += r * step

        def __call__(self, env, allowed: list[int]) -> int:
            from focused_data import TARGET_TLS
            stats = type(self).stats
            stats["decisions"] += 1
            if self._model is None:
                self._setup(env)
            self._observe(env)
            if len(allowed) == 1:
                return allowed[0]
            started = time.perf_counter()
            ts = env.traffic_signals[TARGET_TLS]
            now = float(env.sim_step)
            served = env.signal_constraints._served_now(ts, now)
            state = SignalState(ts.green_phase, now - ts.time_since_last_phase_change, float(ts.yellow_time),
                                tuple(served[c] for c in self._model.crosswalks))
            q0, arrivals = self.queues_and_arrivals(env)
            action, info = plan(self._model, state, now, q0, arrivals, list(allowed))
            took = time.perf_counter() - started
            stats["planned"] += 1
            stats["seconds"] += took
            stats["max_seconds"] = max(stats["max_seconds"], took)
            stats["nodes"] += info["nodes"]
            stats["outside_allowed"] += action not in allowed
            return action

    PredictiveControl.__name__ = f"PredictiveControl_{variant}"
    return PredictiveControl


def install(variant: str, horizon_s: int):
    """이 프로세스의 단순 제어 자리에 예측 제어를 끼운다(평가 코드는 실행할 때 focused_eval.QueueFirst를 가져다 쓴다)."""
    import focused_eval
    cls = controller_class(variant, horizon_s)
    focused_eval.QueueFirst = cls
    return cls


def _add_stats(metrics_file: Path, stats: dict) -> None:
    m = json.loads(metrics_file.read_text())
    m["pred"] = {**stats, "mean_seconds": stats["seconds"] / max(1, stats["planned"])}
    tmp = metrics_file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m, ensure_ascii=False, indent=1))
    tmp.replace(metrics_file)


# ------------------------------------------------------------------ 실행
def _pred_tag(variant: str) -> str:
    return f"pred_{variant}"


def _single_job(job: dict, target: str, variant: str | None, horizon_s: int) -> str:
    """단독 모형 하루(PET 포함, focused_conflicts.measure_job 그대로). variant가 없으면(원자료 고정신호·DQN) 끼우지 않는다."""
    import focused_conflicts
    cls = install(variant, horizon_s) if variant else None
    focused_conflicts.measure_job(job, target)
    if cls is not None:
        _add_stats(Path(target) / "metrics.json", cls.stats)
    return f"{job['date']} {Path(target).name}"


def _window_job(job: dict, variant: str, horizon_s: int) -> str:
    import network_eval
    cls = install(variant, horizon_s)
    network_eval.run_window(job)
    _add_stats(Path(job["run_dir"]) / "metrics.json", cls.stats)
    return f"{job['date']} {job['hour']:02d}시 {Path(job['run_dir']).name}"


def _load_check(path: str) -> int:
    c = json.loads(Path(path).read_text())
    if not c.get("ok") or c.get("code") != code_fingerprint():
        raise SystemExit(f"{path}: 통과한 점검이 아니거나 지금 코드({code_fingerprint()})의 점검이 아닙니다({c.get('code')}).")
    return int(c["horizon_s"])


def single(args) -> None:
    import focused_conflicts as fc
    from focused_eval import CACHE, dqn_template_network, evaluation_fingerprint, planned_runs
    from focused_platoon import platoon_route_file
    from focused_scenario import day_route_file
    horizon = _load_check(args.check)
    split = json.loads((PROJECT / "config/focused_split.json").read_text())
    fp = evaluation_fingerprint()
    drive, tmp = Path(args.out) / fp, Path(args.tmp) / fp
    plan_runs = []                                  # (job, Drive 폴더, 변형)
    for dates, arrivals_list in ((split["validation"], ("random", "platoon")), (split["train"], ("random",))):
        for arrivals in arrivals_list:
            for v in SINGLE_VARIANTS:
                for j in planned_runs(tmp, dates, ["greedy"], [], 42, arrivals):
                    j["run_dir"] = j["run_dir"].replace("/greedy_eval", f"/{_pred_tag(v)}_eval")
                    plan_runs.append((j, v))
    plan_runs += [(j, None) for j in planned_runs(tmp, split["train"], ["fixed"], [], 42, "random")]   # 원자료 고정신호 13일
    if args.h1_models:
        plan_runs += [(j, None) for j in planned_runs(tmp, split["validation"], ["dqn"], args.h1_models, 42, "platoon")]
    jobs = []
    for j, v in plan_runs:
        target = drive / Path(j["run_dir"]).relative_to(tmp)
        if not (target / "events.json.gz").exists():
            jobs.append(({**j, "fingerprint": fp}, str(target), v))
    print(f"예측 제어 단독 모형(내다보는 길이 {horizon}초): 실행 {len(plan_runs)}개 중 새로 {len(jobs)}개 → {drive}", flush=True)
    for date in sorted({j["date"] for j, _, _ in jobs}):
        day_route_file(date, CACHE / "routes")
        if date in split["validation"]:
            platoon_route_file(date, CACHE / "routes")
    dqn_template_network(CACHE)
    with ProcessPoolExecutor(max_workers=args.workers or fc.auto_workers()) as pool:
        futures = [pool.submit(_single_job, j, t, v, horizon) for j, t, v in jobs]
        for n, f in enumerate(as_completed(futures), 1):
            print(f"  {n}/{len(futures)} {f.result()}", flush=True)
    fc.summary_all(drive)


def network12(args) -> None:
    import network_eval as ne
    import network_rl as nr
    import network_windows as nw
    horizon = _load_check(args.check)
    dates = json.loads((PROJECT / "config/focused_split.json").read_text())["validation"]
    fp = ne.evaluation_fingerprint()
    root = Path(args.out) / fp
    windows = nw.evaluation_windows(dates)
    nr.template_network_12()
    jobs = []
    for v in VARIANTS:
        for s in SEEDS:
            for j in ne.window_runs(root, windows, ["greedy"], [], s):
                j["run_dir"] = j["run_dir"].replace(f"/greedy_eval{s}", f"/{_pred_tag(v)}_eval{s}")
                jobs.append(({**j, "fingerprint": fp}, v))
    todo = [(j, v) for j, v in jobs if not (Path(j["run_dir"]) / "metrics.json").exists()]
    print(f"예측 제어 12개 지도(내다보는 길이 {horizon}초): 실행 {len(jobs)}개 중 새로 {len(todo)}개 → {root}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_window_job, j, v, horizon) for j, v in todo]
        for n, f in enumerate(as_completed(futures), 1):
            r = f.result()
            if n % 10 == 0 or n == len(futures):
                print(f"  {n}/{len(futures)} {r}", flush=True)


def _check_window(variant: str, horizon_s: int, workdir: str) -> dict:
    import network_eval as ne
    import network_rl as nr
    fp = ne.evaluation_fingerprint()
    job = ne.window_runs(Path(workdir) / variant / fp, [(CHECK_DATE, CHECK_HOUR)], ["greedy"], [], 42)[0]
    cls = install(variant, horizon_s)
    nr.template_network_12()
    r = ne.run_window({**job, "fingerprint": fp})
    s = cls.stats
    return {"map": "network12", "variant": variant, "horizon_s": horizon_s, "planned": s["planned"], "decisions": s["decisions"],
            "mean_seconds": round(s["seconds"] / max(1, s["planned"]), 4), "max_seconds": round(s["max_seconds"], 3),
            "mean_nodes": round(s["nodes"] / max(1, s["planned"])), "outside_allowed": s["outside_allowed"],
            "changed_by_rules": r["guard"]["changed_by_rules"], "delay_s": round(r["window_204820"]["mean_delay_s"], 2)}


def _check_single(horizon_s: int, workdir: str) -> dict:
    """단독 모형 코드 길을 확인(학습 날짜 첫 2시간, P1). 판단 시간은 12개 지도 저녁 조각으로 정한다."""
    import focused_conflicts as fc
    from focused_eval import CACHE, dqn_template_network, evaluation_fingerprint, planned_runs
    from focused_scenario import day_route_file
    day_route_file(CHECK_DATE, CACHE / "routes")
    dqn_template_network(CACHE)
    job = planned_runs(Path(workdir) / "single", [CHECK_DATE], ["greedy"], [], 42, "random")[0]
    cls = install("p1", horizon_s)
    fc.record_run({**job, "fingerprint": evaluation_fingerprint()}, until=7200)
    s = cls.stats
    return {"map": "single", "variant": "p1", "horizon_s": horizon_s, "until_s": 7200, "planned": s["planned"],
            "decisions": s["decisions"], "mean_seconds": round(s["seconds"] / max(1, s["planned"]), 4),
            "outside_allowed": s["outside_allowed"]}


def check(args) -> None:
    """평가 전 점검(Colab, SUMO): 학습 날짜 08-11 18시 12개 지도 조각에서 P0·P1·P2를 120초로 돌려 판단 시간을 잰다.
    하나라도 평균 0.3초를 넘으면 90초로 정한다(98.1, 시간으로만). 허용 밖 선택·규칙이 선택을 바꾼 횟수는 0이어야 한다."""
    workdir = tempfile.mkdtemp(prefix="pred_check_")
    with ProcessPoolExecutor(max_workers=3) as pool:
        runs = list(pool.map(_check_window, VARIANTS, [HORIZON_S] * 3, [workdir] * 3))
    horizon = HORIZON_S if all(r["mean_seconds"] <= TIME_LIMIT_S for r in runs) else FALLBACK_HORIZON_S
    if horizon != HORIZON_S:
        with ProcessPoolExecutor(max_workers=3) as pool:
            runs += list(pool.map(_check_window, VARIANTS, [horizon] * 3, [workdir] * 3))
    with ProcessPoolExecutor(max_workers=1) as pool:
        runs.append(pool.submit(_check_single, horizon, workdir).result())
    final = [r for r in runs if r["horizon_s"] == horizon]
    ok = all(r["planned"] > 0 and r["outside_allowed"] == 0 for r in final) and \
        all(sum(r["changed_by_rules"].values()) == 0 for r in final if "changed_by_rules" in r)
    report = {"code": code_fingerprint(), "horizon_s": horizon, "time_limit_s": TIME_LIMIT_S, "ok": ok,
              "rule": "98.1: 120초에서 판단 한 번 평균이 0.3초를 넘는 변형이 있으면 90초(시간으로만 정함)",
              "check_scenario": f"{CHECK_DATE} {CHECK_HOUR:02d}시(학습 날짜)", "runs": runs}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if not ok:
        raise SystemExit("점검 실패: 평가하지 않는다.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--out", required=True)
    s = sub.add_parser("single")
    s.add_argument("--check", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--h1-models", nargs="*", default=[], help="H1 seed 5개(검증 2일 무리 도착 DQN 평가가 없어서 함께 돌림)")
    s.add_argument("--tmp", default="/content/pred_tmp", help="기록을 잠시 둘 곳(분석 뒤 지움)")
    s.add_argument("--workers", type=int, default=0, help="0이면 CPU·메모리에 맞춰 자동")
    n = sub.add_parser("network12")
    n.add_argument("--check", required=True)
    n.add_argument("--out", required=True)
    n.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) - 1)))
    args = ap.parse_args()
    {"check": check, "single": single, "network12": network12}[args.cmd](args)


if __name__ == "__main__":
    main()
