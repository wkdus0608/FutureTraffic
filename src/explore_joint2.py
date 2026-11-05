"""[추가 탐색] 0단계 JOINT2 — 석천사거리와 동쪽 이웃(204822)을 함께 예측 제어해 '이웃까지 바꾸면 여지가 있나'를 잰다.

학습 없는 진단이다(지문 밖). 사전 등록: results/explore/joint2_20261031/사전등록.md, 근거: docs/12개교차로_제어_검토안_20261031.md 4절.
평가 코드(network_eval)와 예측 제어(explore_pred)는 고치지 않고 부른다. 실행할 때만 focused_eval.QueueFirst를 바꿔 끼우고
network_eval.window_metrics를 감싸 '도로망 지연'을 더한다(Max-Pressure·PRED와 같은 방식).

정책(12개 지도 검증 2일 46조각 × 평가 seed 5개):
- fixedc: 보정 고정신호(기존 그대로). p2: 기존 예측 제어 P2(explore_pred 그대로).
- e: 석천사거리만 바꾼다. P2와 같은 줄 모형에, 이웃 실제 시간표를 정확히 쓰고, 동쪽으로 내보낸 직진 차가 이웃에서
  기다릴 시간을 점수에 넣는다. 이웃은 실제 신호.
- j2: e와 같은 석천사거리 제어(이웃 시간표 대신 이웃의 지금 계획). 이웃도 예측 제어로 바꾼다:
  실제 신호와 같은 녹색 4개·같은 순서·황색 4초, 녹색 길이만 그 날 실제 길이의 5~95% 범위 안에서 5초마다 '유지/다음'.
  점수 = 이웃 이동별 줄 + 이웃이 서쪽으로 내보낸 직진 차가 석천사거리에서 기다릴 시간(석천사거리의 직전 계획 기준).
  한 판단 안에서 이웃이 먼저 계획·실행하고, 석천사거리가 이웃의 새 계획을 받아 계획한다.

사용(Colab):
  FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240 python src/explore_joint2.py smoke --out <작업 폴더>
  FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240 python src/explore_joint2.py run --out <Drive>/explore_joint2 --workers 8
  FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240 python src/explore_joint2.py report --out <Drive>/explore_joint2 \
      --reference <Drive>/explore_pred/network12
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
import sys
import time
import xml.etree.ElementTree as ET
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import explore_pred as ep  # noqa: E402
from explore_pred import SignalState, step_queue, _insert  # noqa: E402

NEIGHBOR = "204822"
HORIZON_S = 120                      # 기존 P2 점검과 같은 내다보는 길이
YELLOW_S = 4                         # 이웃 실제 신호의 황색
ZONE_M = 300.0                       # 이웃 감지 구간(정지선에서 거슬러, 다른 신호 교차로 출구에서 멈춤)
BOUND_Q = (0.05, 0.95)               # 녹색 길이 범위: 그 날 실제 길이의 분위
DEFAULT_HEADWAY_S = 2.0
EAST_TO_WEST = ("477706392_02", "477707013_01")      # 석천사거리 길주로 동→서 직진
WEST_TO_EAST = ("477707013_02", "477706392_01")      # 석천사거리 길주로 서→동 직진
POLICIES = ("fixedc", "p2", "e", "j2")
SEEDS = [42, 43, 44, 45, 46]
EVENING_HOURS = (16, 17, 18)         # 저녁 16~19시 조각(제어 시작 시각)
LONG_S = 300.0
T975_DF45 = 2.0141
TOL = 1e-9


def code_fingerprint() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


# ------------------------------------------------------------------ 계획(SUMO 없음)
class CyclicModel:
    """정해진 순서로 도는 녹색(이웃). explore_pred.plan과 같은 자리(actions·advance·_rates·step·terminal_s)를 채운다."""

    def __init__(self, rates: list[dict[int, float]], green_min: list[float], green_max: list[float], step: float,
                 yellow: float = YELLOW_S, terminal_s: float = ep.TERMINAL_S, startup_s: float = ep.STARTUP_LOST_S):
        if not (len(rates) == len(green_min) == len(green_max)) or any(a > b for a, b in zip(green_min, green_max)):
            raise ValueError("녹색 수·길이 범위가 맞지 않습니다.")
        self._rates, self.green_min, self.green_max = rates, green_min, green_max
        self.step, self.yellow, self.terminal_s, self.startup_s = step, yellow, terminal_s, startup_s

    def next_green(self, g: int) -> int:
        return (g + 1) % len(self._rates)

    def actions(self, st: SignalState, t: float) -> list[int]:
        tau = t - st.switch_time
        if tau < st.intergreen + self.green_min[st.phase]:
            return [st.phase]
        if tau >= st.intergreen + self.green_max[st.phase] - TOL:
            return [self.next_green(st.phase)]
        return [st.phase, self.next_green(st.phase)]

    def advance(self, st: SignalState, t: float, action: int) -> tuple[SignalState, float]:
        if action != st.phase:
            if action != self.next_green(st.phase):
                raise ValueError("이웃 녹색은 정해진 순서로만 바뀝니다.")
            st = SignalState(action, t, float(self.yellow), ())
        start = st.switch_time + st.intergreen + self.startup_s
        return st, max(0.0, t + self.step - max(t, start))


def plan_release(model, state: SignalState, now: float, q0: list[float], arrivals: list[list[float]],
                 first_actions: list[int], release_wait: list[dict[int, float]] | None = None) -> tuple[int, dict]:
    """explore_pred.plan과 같되, k단계에 이동 m에서 통과시킨 차 한 대마다 release_wait[k][m]초를 점수에 더한다
    (그 차가 다음 신호에서 기다릴 시간). 기다림이 0 이상이고 통과 수가 줄에 대해 늘기만 하므로 합치기 규칙은 그대로 맞다."""
    current, step = state.phase, model.step
    level = {state: [(0.0, tuple(q0), (False, current, ()))]}
    expanded = 0
    for k, arriving in enumerate(arrivals):
        t = now + k * step
        wait = release_wait[k] if release_wait else {}
        nxt: dict[SignalState, list] = {}
        for st, nodes in level.items():
            for a in (first_actions if k == 0 else model.actions(st, t)):
                st2, effective = model.advance(st, t, a)
                rate = model._rates[a] if effective > 0 else {}
                bucket = nxt.setdefault(st2, [])
                for cost, q, rank in nodes:
                    expanded += 1
                    q2 = step_queue(q, arriving, rate, effective)
                    released = sum((q[m] + arriving[m] - q2[m]) * w for m, w in wait.items())
                    seq = rank[2] + (a,)
                    _insert(bucket, cost + step * (sum(q) + sum(q2)) / 2 + released, q2,
                            (seq[0] != current, seq[0], seq))
        level = nxt
    total, rank = min(((cost + model.terminal_s * sum(q), rank) for nodes in level.values() for cost, q, rank in nodes),
                      key=lambda x: (round(x[0], 6), x[1][0], x[1][1]))
    return rank[2][0], {"cost": total, "nodes": expanded, "plan": list(rank[2])}


def wait_until(served_at, t_arr: float, now: float, horizon_s: float) -> float:
    """t_arr에 도착한 차가 다음 녹색까지 기다릴 초. 아는 범위 [now, now + horizon_s) 밖에 도착하면 0(모름),
    범위 끝까지 녹색이 없으면 (범위 끝 − 도착) + TERMINAL_S."""
    end = now + horizon_s
    if not (now <= t_arr < end):
        return 0.0
    t = t_arr
    while t < end:
        if served_at(t):
            return t - t_arr
        t += 1.0
    return end - t_arr + ep.TERMINAL_S


def green_bounds(phases: list[tuple[int, str]]) -> dict:
    """하루 실제 신호에서 녹색 4개(처음 나온 순서), 녹색마다 뒤 황색, 녹색 길이의 5~95% 범위."""
    greens, durations, yellow_after = [], {}, {}
    for i, (d, s) in enumerate(phases):
        if "y" in s:
            continue
        if s not in durations:
            greens.append(s)
            durations[s] = []
        durations[s].append(d)
        nxt = phases[i + 1][1] if i + 1 < len(phases) else None
        if nxt and "y" in nxt:
            yellow_after.setdefault(s, {}).setdefault(nxt, 0)
            yellow_after[s][nxt] += 1
    if len(greens) != 4 or set(yellow_after) != set(greens):
        raise ValueError(f"{NEIGHBOR} 녹색이 4개가 아니거나 황색이 없는 녹색이 있습니다: {len(greens)}")
    seq = [greens.index(s) for d, s in phases if "y" not in s]
    if sum(b != (a + 1) % 4 for a, b in zip(seq, seq[1:])) > 2:
        raise ValueError(f"{NEIGHBOR} 녹색 순서가 정해져 있지 않습니다.")

    def q(xs, p):
        xs = sorted(xs)
        return float(xs[int(p * (len(xs) - 1))])
    return {"greens": greens, "yellows": [max(yellow_after[s], key=yellow_after[s].get) for s in greens],
            "min": [q(durations[s], BOUND_Q[0]) for s in greens], "max": [q(durations[s], BOUND_Q[1]) for s in greens]}


# ------------------------------------------------------------------ 지도(한 번만)
_LAYOUT: dict | None = None


def neighbor_layout(net_file: Path | None = None) -> dict:
    """이웃 연결(들어오는 차로 → 나가는 도로), 감지 구간, 서쪽행 직진 연결, 석천사거리 ↔ 이웃 이동 시간."""
    global _LAYOUT
    if net_file is None:
        if _LAYOUT is None:
            import network_rl as nr
            _LAYOUT = neighbor_layout(nr.template_network_12())
        return _LAYOUT
    from focused_sumo import configure_sumo
    configure_sumo()
    import sumolib
    import explore_coord12 as ec
    net = sumolib.net.readNet(str(net_file))
    tls = net.getTLS(NEIGHBOR)
    links, direction = {}, {}
    for inl, out, idx in tls.getConnections():
        links[idx] = (inl.getID(), inl.getEdge().getID(), out.getEdge().getID())
        direction[idx] = next(c.getDirection() for c in inl.getOutgoing() if c.getToLane() == out)
    incoming = sorted({f for _, f, _ in links.values()})
    zone: dict[str, tuple[str, float]] = {}
    for a in incoming:
        frontier = [(a, 0.0)]
        while frontier:
            eid, off = frontier.pop()
            if eid in zone and zone[eid][1] <= off:
                continue
            zone[eid] = (a, off)
            e = net.getEdge(eid)
            if e.getFromNode().getType() == "traffic_light" or off + e.getLength() >= ZONE_M:
                continue
            frontier += [(p.getID(), off + e.getLength()) for p in e.getIncoming() if p.getFunction() != "internal"]
    travel = {}
    for eid, (a, off) in zone.items():
        e = net.getEdge(eid)
        travel[a] = max(travel.get(a, 0.0), (off + e.getLength()) / max(1.0, e.getSpeed()))
    east = ec.east_layout(net_file)
    to_204820 = sum(net.getEdge(c["edge"]).getLength() / net.getEdge(c["edge"]).getSpeed() for c in east["chain"])
    # 석천사거리 동쪽 출구 → 이웃 서쪽 진입로(직진으로 따라감)
    edge, path = net.getEdge(WEST_TO_EAST[1]), []
    while True:
        path.append(edge)
        if edge.getID() in incoming:
            break
        nxt = [to for to, conns in edge.getOutgoing().items() if any(c.getDirection() == "s" for c in conns)]
        if len(nxt) != 1 or len(path) > 30:
            raise ValueError(f"{WEST_TO_EAST[1]}에서 {NEIGHBOR}까지 직진 길을 하나로 찾지 못했습니다.")
        edge = nxt[0]
    eastbound = sorted(i for i, (_, f, _) in links.items() if f == path[-1].getID() and direction[i] == "s")
    groups = sorted({(lane, to) for lane, _, to in links.values()})
    return {"links": links, "direction": direction, "groups": groups,
            "group_of_link": {i: groups.index((lane, to)) for i, (lane, _, to) in links.items()},
            "zone": zone, "travel": travel, "westbound_straight": east["straight_links"],
            "eastbound_straight": eastbound, "travel_to_204820": to_204820,
            "travel_to_204822": sum(e.getLength() / e.getSpeed() for e in path),
            "nodes": sorted({inl.getEdge().getToNode().getID() for inl, _, _ in tls.getConnections()}),
            "edge_pairs": sorted({(f, t) for _, f, t in links.values()})}


def saturation(lane_edge: str, to_edge: str) -> float:
    cfg = json.loads((PROJECT / "config/network12_capacity.json").read_text())["saturation_per_lane_h"]
    v = cfg.get(f"{lane_edge}>{to_edge}")
    return float(v) / 3600.0 if v else 1.0 / DEFAULT_HEADWAY_S


def share(state: str, links: list[int]) -> float:
    return sum(state[i] in "Gg" for i in links) / len(links) if links else 0.0


# ------------------------------------------------------------------ 이웃 제어(SUMO에 붙음)
class NeighborControl:
    """이웃 신호를 넘겨받아 5초마다 '유지/다음'을 고른다. 석천사거리 제어가 판단마다 부른다."""

    def __init__(self, date: str, step: float, horizon_s: float, route_edges: dict[str, list[str]]):
        import network_signals as ns
        from traci import trafficlight as tl
        self.layout = neighbor_layout()
        self.bounds = green_bounds(ns.day_programs(date)[0][NEIGHBOR])
        greens = self.bounds["greens"]
        lay = self.layout
        rates = []
        for g in greens:
            r = {}
            for i, (lane, f, to) in lay["links"].items():
                if g[i] == "G":
                    r[lay["group_of_link"][i]] = saturation(f, to)
            rates.append(r)
        self.model = CyclicModel(rates, self.bounds["min"], self.bounds["max"], step)
        self.step, self.K, self.horizon_s = step, int(round(horizon_s / step)), horizon_s
        self.route_edges = route_edges
        self._logic = tl.Logic("joint2", 0, 0, [p for g, y, mx in zip(greens, self.bounds["yellows"], self.bounds["max"])
                                                 for p in (tl.Phase(mx, g), tl.Phase(YELLOW_S, y))])
        self.ready = False
        self._seen: set[str] = set()
        self._entries: deque = deque()
        self._first: float | None = None
        self._takeover_start: float | None = None      # 넘겨받은 녹색이 실제로 시작된 시각
        self._green_start = 0.0
        self.plan: tuple[float, list[int]] | None = None
        self.log = {"decisions": 0, "switches": 0, "forced": 0, "greens_done": [], "seconds": 0.0, "max_seconds": 0.0,
                    "queued_seen": 0, "bounds": self.bounds}

    def _takeover(self, conn, now: float) -> None:
        state, spent = conn.trafficlight.getRedYellowGreenState(NEIGHBOR), float(conn.trafficlight.getSpentDuration(NEIGHBOR))
        greens, yellows = self.bounds["greens"], self.bounds["yellows"]
        conn.trafficlight.setProgramLogic(NEIGHBOR, self._logic)
        conn.trafficlight.setProgram(NEIGHBOR, "joint2")
        if state in greens:
            g = greens.index(state)
            conn.trafficlight.setPhase(NEIGHBOR, 2 * g)
            conn.trafficlight.setPhaseDuration(NEIGHBOR, max(1.0, self.bounds["max"][g] - spent))
            self._takeover_start = now - spent
        else:
            g = min(range(4), key=lambda k: sum(a != b for a, b in zip(yellows[k], state)))
            conn.trafficlight.setPhase(NEIGHBOR, 2 * g + 1)
            conn.trafficlight.setPhaseDuration(NEIGHBOR, max(1.0, YELLOW_S - spent))
        self._phase_seen = conn.trafficlight.getPhase(NEIGHBOR)
        self._green_start = self._takeover_start if self._takeover_start is not None else now
        self.ready = True

    def _state(self, conn, now: float) -> SignalState:
        idx = conn.trafficlight.getPhase(NEIGHBOR)
        if idx != self._phase_seen:                          # 지난 판단 뒤에 SUMO가 바꿈(황색 끝·최대 녹색)
            if self._phase_seen % 2 == 0:                     # 녹색이 최대 길이로 끝남
                self.log["forced"] += 1
                self.log["greens_done"].append((self._phase_seen // 2, None, self._takeover_start is not None))
            self._takeover_start = None
            self._phase_seen = idx
        nxt = float(conn.trafficlight.getNextSwitch(NEIGHBOR))
        if idx % 2 == 0:
            g = idx // 2
            start = self._takeover_start if self._takeover_start is not None else nxt - self.bounds["max"][g]
            self._green_start = start
            return SignalState(g, start - YELLOW_S, float(YELLOW_S), ())
        start = nxt - YELLOW_S
        return SignalState(self.model.next_green(idx // 2), start, float(YELLOW_S), ())

    def _queues(self, conn, now: float) -> tuple[list[float], list[list[float]]]:
        lay, n, K, step = self.layout, len(self.layout["groups"]), self.K, self.step
        q0, arrivals = [0.0] * n, [[0.0] * n for _ in range(K)]
        fresh = self._first is None
        if fresh:
            self._first = now
        for e, (approach, _) in lay["zone"].items():
            for v in conn.edge.getLastStepVehicleIDs(e):
                hit = next((x for x in conn.vehicle.getNextTLS(v) if x[0] == NEIGHBOR), None)
                if hit is None:
                    continue
                m = lay["group_of_link"].get(int(hit[1]))
                if m is None or not any(m in r for r in self.model._rates):
                    continue
                if v not in self._seen:
                    self._seen.add(v)
                    if not fresh:
                        self._entries.append((now, approach, m, None))
                if conn.vehicle.getSpeed(v) < 0.1:
                    q0[m] += 1
                else:
                    k = int(max(0.0, float(hit[2])) / max(1.0, conn.vehicle.getAllowedSpeed(v)) // step)
                    if k < K:
                        arrivals[k][m] += 1
        pairs = {(f, t): lay["group_of_link"][i] for i, (_, f, t) in lay["links"].items()}
        for v in conn.simulation.getPendingVehicles():
            edges = self.route_edges.get(v)
            if not edges or edges[0] not in lay["zone"]:
                continue
            m = next((pairs[(a, b)] for a, b in zip(edges, edges[1:]) if (a, b) in pairs), None)
            if m is not None:
                q0[m] += 1
        while self._entries and self._entries[0][0] < now - ep.ENTRY_WINDOW_S:
            self._entries.popleft()
        for (approach, m), r in ep.entry_rates(self._entries, now, self._first).items():
            for k in range(math.ceil(lay["travel"][approach] / step), K):
                arrivals[k][m] += r * step
        self.log["queued_seen"] = max(self.log["queued_seen"], int(sum(q0)))
        return q0, arrivals

    def decide(self, conn, now: float, served_204820) -> None:
        """served_204820(t): 석천사거리의 직전 계획에서 t에 동→서 직진이 녹색인가(없으면 None → 하류 대기 0)."""
        started = time.perf_counter()
        if not self.ready:
            self._takeover(conn, now)
        st = self._state(conn, now)
        q0, arrivals = self._queues(conn, now)
        lay = self.layout
        wb = sorted({lay["group_of_link"][i] for i in lay["westbound_straight"]})
        release = None
        if served_204820 is not None:
            release = [{m: wait_until(served_204820, now + k * self.step + lay["travel_to_204820"], now, self.horizon_s)
                        for m in wb} for k in range(self.K)]
        first = self.model.actions(st, now)
        action, info = plan_release(self.model, st, now, q0, arrivals, first, release)
        if action != st.phase:
            idx = conn.trafficlight.getPhase(NEIGHBOR)
            if idx % 2 == 0:
                self.log["greens_done"].append((idx // 2, now - self._green_start, self._takeover_start is not None))
                conn.trafficlight.setPhase(NEIGHBOR, idx + 1)
                self._takeover_start = None
                self._phase_seen = idx + 1
                self.log["switches"] += 1
        self.plan = (now, info["plan"])
        took = time.perf_counter() - started
        self.log["decisions"] += 1
        self.log["seconds"] += took
        self.log["max_seconds"] = max(self.log["max_seconds"], took)

    def green_at(self, links: list[int]):
        """이웃의 지금 계획으로 본 't에 links가 녹색인가'(계획 범위 밖은 None)."""
        if self.plan is None:
            return None
        t0, greens = self.plan
        states = self.bounds["greens"]

        def f(t: float):
            j = int((t - t0) // self.step)
            return share(states[greens[j]], links) >= 0.5 if 0 <= j < len(greens) else None
        return f


def timetable_green_at(conn, links: list[int]):
    """실제 신호(지금 프로그램)를 그대로 따라간 't에 links가 녹색인가'(지금부터 앞으로)."""
    pid = conn.trafficlight.getProgram(NEIGHBOR)
    logic = next(lg for lg in conn.trafficlight.getAllProgramLogics(NEIGHBOR) if lg.programID == pid)
    phases = [(float(p.duration), share(p.state, links) >= 0.5) for p in logic.phases]
    idx = conn.trafficlight.getPhase(NEIGHBOR)
    now_switch = float(conn.trafficlight.getNextSwitch(NEIGHBOR))
    now = float(conn.simulation.getTime())

    def f(t: float):
        if t < now:
            return None
        k, end = idx, now_switch
        while t >= end:
            k += 1
            if k >= len(phases):
                return None
            end += phases[k][0]
        return phases[k][1]
    return f


# ------------------------------------------------------------------ 석천사거리 제어(e·j2)
def controller_class(policy: str, horizon_s: int = HORIZON_S):
    if policy not in ("e", "j2"):
        raise ValueError(policy)
    base = ep.controller_class("p2", horizon_s)

    class JointControl(base):
        stats: dict = {}
        route_file: Path | None = None
        date: str | None = None

        @staticmethod
        def movements(route_file: Path):
            JointControl.route_file = Path(route_file)
            return base.movements(route_file)

        def __init__(self, movement_of, greens):
            super().__init__(movement_of, greens)
            self.neighbor: NeighborControl | None = None
            self._last_plan: tuple[float, list[int]] | None = None
            type(self).stats.update({"policy": policy, "joint2_code": code_fingerprint()})

        def _wb_on(self, env):
            """석천사거리 동쪽 진입 예측에 쓸 '이웃 서쪽행 직진 녹색' 시간표(e: 실제, j2: 이웃 계획)."""
            import explore_coord12 as ec
            raw = getattr(env.sumo, "_conn", env.sumo)
            return (self.neighbor.green_at(self._east["straight_links"]) if policy == "j2"
                    else timetable_green_at(raw, self._east["straight_links"]))

        def _outside(self, arrivals, sumo, now: float, step: float) -> None:
            """explore_pred P2와 같되 이웃 켜짐·꺼짐을 시간표(아는 범위)로, 모르면 P2의 중앙값 외삽으로."""
            import explore_coord12 as ec
            K, east = len(arrivals), ec.APPROACH
            for (approach, i), r in ep.entry_rates(self._entries, now, self._first_time).items():
                if approach == east:
                    continue
                for k in range(math.ceil(self.geo["travel"][approach] / step), K):
                    arrivals[k][i] += r * step
            durations, on = self._runs
            cur, elapsed, med = ep.neighbor_history(durations, on, sumo.trafficlight.getPhase(ec.NEIGHBOR),
                                                    sumo.trafficlight.getSpentDuration(ec.NEIGHBOR))
            counts = {True: {}, False: {}}
            for t, a, i, o in self._entries:
                if a == east and t >= now - ep.NEIGHBOR_WINDOW_S:
                    counts[o][i] = counts[o].get(i, 0) + 1
            seconds = {s: step * sum(1 for t, o in self._neighbor_log if o == s) for s in (True, False)}
            rates = {s: {i: c / max(ep.MIN_RATE_WINDOW_S, seconds[s]) for i, c in counts[s].items()} for s in (True, False)}
            travel = self.geo["travel"][east]
            known = self._schedule
            for k in range(math.ceil(travel / step), K):
                t = k * step - travel
                s = known(now + t) if known is not None else None
                if s is None:
                    s = ep.predicted_on(cur, elapsed, med, t)
                for i, r in rates[s].items():
                    arrivals[k][i] += r * step

        def __call__(self, env, allowed: list[int]) -> int:
            from focused_data import TARGET_TLS
            stats = type(self).stats
            stats["decisions"] += 1
            if self._model is None:
                self._setup(env)
                if policy == "j2":
                    root = ET.parse(type(self).route_file).getroot()
                    routes = {r.get("id"): r.get("edges").split() for r in root.iter("route")}
                    edges = {v.get("id"): routes[v.get("route")] for v in root.iter("vehicle") if v.get("route") in routes}
                    self.neighbor = NeighborControl(type(self).date, float(env.delta_time), horizon_s, edges)
            raw = getattr(env.sumo, "_conn", env.sumo)
            now, step = float(env.sim_step), float(env.delta_time)
            started = time.perf_counter()
            if policy == "j2":
                self.neighbor.decide(raw, now, self._served_204820(now))
            self._observe(env)
            self._schedule = self._wb_on(env)
            ts = env.traffic_signals[TARGET_TLS]
            served = env.signal_constraints._served_now(ts, now)
            state = SignalState(ts.green_phase, now - ts.time_since_last_phase_change, float(ts.yellow_time),
                                tuple(served[c] for c in self._model.crosswalks))
            q0, arrivals = self.queues_and_arrivals(env)
            release = None
            iw = self.index.get(WEST_TO_EAST)
            lay = neighbor_layout()
            eb = (self.neighbor.green_at(lay["eastbound_straight"]) if policy == "j2"
                  else timetable_green_at(raw, lay["eastbound_straight"]))
            if iw is not None and eb is not None:
                served_eb = lambda t: bool(eb(t))               # noqa: E731  (계획 밖은 wait_until이 0으로 둠)
                release = [{iw: wait_until(served_eb, now + k * step + lay["travel_to_204822"], now, horizon_s)}
                           for k in range(self._steps)]
            action, info = plan_release(self._model, state, now, q0, arrivals, list(allowed), release)
            self._last_plan = (now, info["plan"])
            took = time.perf_counter() - started
            stats["planned"] += 1
            stats["seconds"] += took
            stats["max_seconds"] = max(stats["max_seconds"], took)
            stats["nodes"] += info["nodes"]
            stats["outside_allowed"] += action not in allowed
            if self.neighbor is not None:
                stats["neighbor"] = self.neighbor.log
            return action if len(allowed) > 1 else allowed[0]

        def _served_204820(self, now: float):
            """석천사거리의 직전 계획에서 t에 동→서 직진이 보호 녹색인가(계획 밖은 False로 보고 wait_until이 범위로 자름)."""
            if self._last_plan is None:
                return None
            t0, greens = self._last_plan
            step = self._model.step

            def f(t: float) -> bool:
                j = int((t - t0) // step)
                return 0 <= j < len(greens) and EAST_TO_WEST in self.served[greens[j]]
            return f

    JointControl.__name__ = f"JointControl_{policy}"
    return JointControl


def install(policy: str, date: str):
    import focused_eval
    cls = controller_class(policy)
    cls.date = date
    focused_eval.QueueFirst = cls
    return cls


# ------------------------------------------------------------------ 점수(도로망 지연)
def network_metrics(route_file: Path, tripinfo: Path, h0: float, h1: float) -> dict:
    """제어 60분 [h0, h1)에 출발 예정인 모든 차. 지연 = timeLoss + 출발 대기, 못 들어온 차는 h1 − 출발 예정(하한)."""
    lay = neighbor_layout()
    pairs = set(map(tuple, lay["edge_pairs"]))
    root = ET.parse(route_file).getroot()
    routes = {r.get("id"): r.get("edges").split() for r in root.iter("route")}
    plan = {}
    for v in root.iter("vehicle"):
        edges = routes[v.get("route")] if v.get("route") in routes else v.find("route").get("edges").split()
        plan[v.get("id")] = (edges, float(v.get("depart")))
    info = {}
    for t in ET.parse(tripinfo).getroot().iter("tripinfo"):
        info[t.get("id")] = (float(t.get("timeLoss", 0)) + float(t.get("departDelay", 0)), float(t.get("arrival", -1)) >= 0)
    rows = {"all": [], "via_204822": []}
    unfinished = not_inserted = 0
    for vid, (edges, dep) in plan.items():
        if not (h0 <= dep < h1):
            continue
        if vid in info:
            d, fin = info[vid]
        else:
            d, fin = h1 - dep, False
            not_inserted += 1
        unfinished += not fin
        rows["all"].append(d)
        if any((a, b) in pairs for a, b in zip(edges, edges[1:])):
            rows["via_204822"].append(d)

    def summary(xs):
        return {"vehicles": len(xs), "mean_delay_s": statistics.fmean(xs) if xs else 0.0,
                "long_delays": sum(x > LONG_S for x in xs), "total_delay_s": math.fsum(xs)}
    return {"network": {**summary(rows["all"]), "unfinished": unfinished, "not_inserted": not_inserted},
            "via_204822": summary(rows["via_204822"])}


def junction_collisions(path: Path, nodes: list[str], since: float) -> int:
    if not path.exists():
        return 0
    n = 0
    for c in ET.parse(path).getroot().iter("collision"):
        lane = c.get("lane", "")
        if float(c.get("time", 0)) >= since and any(lane.startswith(f":{node}_") for node in nodes):
            n += 1
    return n


def install_metrics() -> None:
    import network_eval as ne
    original = getattr(ne, "_joint2_original_window_metrics", ne.window_metrics)
    ne._joint2_original_window_metrics = original

    def wrapped(route_file, tripinfo, vehroute, local, near, h0, h1, end_state, run_dir):
        extra = network_metrics(route_file, tripinfo, h0, h1)
        extra["collisions_204822"] = junction_collisions(Path(run_dir) / "collisions.xml", neighbor_layout()["nodes"], h0)
        return {**original(route_file, tripinfo, vehroute, local, near, h0, h1, end_state, run_dir), **extra}
    ne.window_metrics = wrapped


# ------------------------------------------------------------------ 실행
def _job(job: dict, policy: str) -> str:
    import network_eval as ne
    install_metrics()
    cls = None
    if policy == "p2":
        cls = ep.install("p2", HORIZON_S)
    elif policy in ("e", "j2"):
        cls = install(policy, job["date"])
    ne.run_window(job)
    if cls is not None:
        ep._add_stats(Path(job["run_dir"]) / "metrics.json", cls.stats)
    for big in ("train_routes.rou.xml", "routes.rou.xml", "neighbors.add.xml", "target.add.xml"):   # 같은 설정으로 다시 만들 수 있는 큰 파일
        (Path(job["run_dir"]) / big).unlink(missing_ok=True)
    return f"{job['date']} {job['hour']:02d}시 {policy} seed {job['eval_seed']}"


def planned_jobs(out: Path, windows, policies, seeds) -> list[tuple[dict, str]]:
    import network_eval as ne
    fp = ne.evaluation_fingerprint()
    root = Path(out) / fp
    jobs = []
    for policy in policies:
        for s in seeds:
            for j in ne.window_runs(root, windows, ["fixedc" if policy == "fixedc" else "greedy"], [], s):
                if policy != "fixedc":
                    j["run_dir"] = j["run_dir"].replace(f"/greedy_eval{s}", f"/{policy}_eval{s}")
                jobs.append(({**j, "fingerprint": fp}, policy))
    return jobs


def run(args) -> None:
    import network_rl as nr
    import network_windows as nw
    dates = json.loads((PROJECT / "config/focused_split.json").read_text())["validation"]
    windows = nw.evaluation_windows(dates)
    nr.template_network_12()
    neighbor_layout()
    jobs = planned_jobs(Path(args.out), windows, args.policies, SEEDS)
    todo = [(j, p) for j, p in jobs if not (Path(j["run_dir"]) / "metrics.json").exists()]
    meta = Path(args.out) / "experiment.json"
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps({"code": code_fingerprint(), "pred_code": ep.code_fingerprint(), "windows": windows,
                                "policies": list(args.policies), "seeds": SEEDS}, ensure_ascii=False, indent=1))
    print(f"JOINT2: 실행 {len(jobs)}개 중 새로 {len(todo)}개(조각 {len(windows)}개) → {args.out}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_job, j, p) for j, p in todo]
        for n, f in enumerate(as_completed(futures), 1):
            r = f.result()
            if n % 10 == 0 or n == len(futures):
                print(f"  {n}/{len(futures)} {r}", flush=True)


def smoke(args) -> None:
    """학습 날짜 한 조각(08-11 18시)에서 e·j2를 돌려 이웃 제어 규칙과 점수 계산을 확인한다(검증 날짜는 쓰지 않음)."""
    import network_rl as nr
    nr.template_network_12()
    out = Path(args.out)
    report = {}
    for policy in args.policies:
        jobs = planned_jobs(out, [(ep.CHECK_DATE, ep.CHECK_HOUR)], [policy], [42])
        for j, p in jobs:
            if not (Path(j["run_dir"]) / "metrics.json").exists():
                _job(j, p)
            m = json.loads((Path(j["run_dir"]) / "metrics.json").read_text())
            report[policy] = {"network": m["network"], "via_204822": m["via_204822"], "window_204820": m["window_204820"]["mean_delay_s"],
                              "collisions_204822": m["collisions_204822"], "pred": m.get("pred", {}), "seconds": m["seconds"]}
    nb = report.get("j2", {}).get("pred", {}).get("neighbor")
    ok = True
    if nb:
        b = nb["bounds"]
        done = [(g, d) for g, d, takeover in nb["greens_done"] if d is not None and not takeover]
        bad = [(g, d) for g, d in done if not (b["min"][g] - 1 <= d <= b["max"][g] + 1)]
        ok = not bad and nb["decisions"] > 0 and nb["queued_seen"] > 0
        report["j2_check"] = {"greens_done": len(nb["greens_done"]), "forced": nb["forced"], "switches": nb["switches"],
                              "outside_bounds": bad, "queued_seen": nb["queued_seen"], "ok": ok}
    (out / "smoke.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: (v if k == "j2_check" else {kk: vv for kk, vv in v.items() if kk != "pred"}) for k, v in report.items()},
                     ensure_ascii=False, indent=1))
    if not ok:
        raise SystemExit("이웃 제어 점검 실패")


# ------------------------------------------------------------------ 보고
def _paired(a: dict, b: dict) -> dict:
    keys = sorted(set(a) & set(b))
    diff = [b[k] - a[k] for k in keys]                    # 양수 = a가 짧음
    mean = statistics.fmean(diff)
    half = T975_DF45 * statistics.stdev(diff) / math.sqrt(len(diff)) if len(diff) > 1 else math.inf
    return {"mean": mean, "t95": [mean - half, mean + half], "n": len(diff), "positive": sum(d > 0 for d in diff)}


def collect(root: Path, policy: str) -> dict:
    rows = {}
    for d in sorted((root / "windows").iterdir()):
        for s in SEEDS:
            f = d / f"{policy}_eval{s}" / "metrics.json"
            if f.exists():
                rows[(d.name, s)] = json.loads(f.read_text())
    return rows


def report(args) -> None:
    import network_eval as ne
    root = Path(args.out) / ne.evaluation_fingerprint()
    data = {p: collect(root, p) for p in POLICIES}
    windows = sorted({w for rows in data.values() for w, _ in rows})
    missing = {p: len(windows) * len(SEEDS) - len(rows) for p, rows in data.items()}
    out = {"code": code_fingerprint(), "missing": missing}
    # 재현 점검: 이번 p2 대 기존 P2
    ref = Path(args.reference) / ne.evaluation_fingerprint() / "windows"
    diffs = []
    for (w, s), m in data["p2"].items():
        f = ref / w / f"pred_p2_eval{s}" / "metrics.json"
        if f.exists():
            diffs.append(abs(json.loads(f.read_text())["window_204820"]["mean_delay_s"] - m["window_204820"]["mean_delay_s"]))
    out["reproduction_p2"] = {"compared": len(diffs), "max_abs_diff": max(diffs) if diffs else None}

    def per_window(policy, getter, hours=None):
        acc = {}
        for (w, s), m in data[policy].items():
            if hours is not None and int(w.split("_")[1]) not in hours:
                continue
            acc.setdefault(w, []).append(getter(m))
        return {w: statistics.fmean(v) for w, v in acc.items() if len(v) == len(SEEDS)}

    def movement(m, mv):
        r = m["window_204820"]["by_movement"].get(f"{mv[0]}->{mv[1]}")
        return (r["vehicles"] * r["mean_delay_s"], r["vehicles"]) if r else (0.0, 0)

    def weighted(policy, mv, hours):
        tot = n = 0.0
        for (w, s), m in data[policy].items():
            if int(w.split("_")[1]) in hours:
                a, b = movement(m, mv)
                tot, n = tot + a, n + b
        return tot / n if n else None

    metrics = {
        "network": lambda m: m["network"]["mean_delay_s"],
        "window_204820": lambda m: m["window_204820"]["mean_delay_s"],
        "via_204822": lambda m: m["via_204822"]["mean_delay_s"],
        "network_long": lambda m: m["network"]["long_delays"],
        "collisions_204822": lambda m: m.get("collisions_204822", 0),
        "collisions_204820": lambda m: m.get("collisions_in_204820", 0),
    }
    table = {p: {k: per_window(p, g) for k, g in metrics.items()} for p in POLICIES if data[p]}
    out["means"] = {p: {k: statistics.fmean(v.values()) if v else None for k, v in t.items()} for p, t in table.items()}
    out["evening_east_to_west"] = {p: weighted(p, EAST_TO_WEST, EVENING_HOURS) for p in table}
    out["evening_west_to_east"] = {p: weighted(p, WEST_TO_EAST, EVENING_HOURS) for p in table}
    comps = {}
    for a, b in (("j2", "p2"), ("e", "p2"), ("j2", "e"), ("j2", "fixedc"), ("p2", "fixedc"), ("e", "fixedc")):
        if a in table and b in table:
            comps[f"{a}_vs_{b}"] = {k: _paired(table[a][k], table[b][k]) for k in ("network", "window_204820", "via_204822")}
    out["comparisons"] = comps
    unstable = "20220928_08"
    out["without_unstable"] = {f"{a}_vs_{b}": _paired({w: v for w, v in table[a]["network"].items() if w != unstable},
                                                      {w: v for w, v in table[b]["network"].items() if w != unstable})
                               for a, b in (("j2", "p2"), ("e", "p2"), ("j2", "e")) if a in table and b in table}
    if "j2" in table and "p2" in table:
        c = comps["j2_vs_p2"]
        p2_net = out["means"]["p2"]["network"]
        coll_up = (out["means"]["j2"]["collisions_204822"] + out["means"]["j2"]["collisions_204820"]
                   > out["means"]["p2"]["collisions_204822"] + out["means"]["p2"]["collisions_204820"])
        if c["network"]["mean"] >= 0.02 * p2_net and c["network"]["t95"][0] > 0 and c["window_204820"]["mean"] >= -2.2 \
                and not coll_up:
            ew = (out["evening_east_to_west"]["p2"] or 0) - (out["evening_east_to_west"]["j2"] or 0)
            verdict = "여지 있음" + ("" if ew >= 3.0 else "(원인 미확인)")
        elif c["network"]["mean"] <= 0:
            verdict = "없음"
        else:
            verdict = "불분명"
        out["verdict"] = verdict
    (Path(args.out) / "report.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1)[:6000])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("smoke", "run", "report"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--workers", type=int, default=8)
        p.add_argument("--policies", nargs="+", default=list(POLICIES) if name != "smoke" else ["e", "j2"])
        p.add_argument("--reference", default=None)
    args = ap.parse_args()
    {"smoke": smoke, "run": run, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
