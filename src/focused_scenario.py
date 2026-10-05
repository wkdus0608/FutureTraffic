"""석천사거리(204820) 시나리오를 만든다: 차량 파일(수요·운전자)과 신호 프로그램.

차량(write_routes)
- 핵심 원칙: 204820에서 기록된 회전 1건 = 시뮬레이션 차량 1대.
- 지도는 AI Hub 원본을 그대로 쓰고, 차량은 204820 주변 도로에서만 다닌다.
- 각 이동은 "상류 직진 구간 → 진입 도로 → 진출 도로 → 하류 직진 구간" 경로로 만든다.
  상류·하류 구간은 다음 신호교차로 직전에서 끝나므로 다른 신호의 영향을 받지 않는다.
- 이동별 15분 교통량을 그 15분 동안 무작위 시각에 정지선에 도착하도록 출발 시각을 정한다.

신호(aihub_signal_program)
- AI Hub 신호 자료를 "한 현시 밀림"으로 보정한 실제 운영 신호를 하루 시간축으로 복원한다.
  자료를 그대로 읽으면 관측 교통량을 처리할 수 없는 시간대가 대부분이라 그대로는 실제 신호일 수
  없다(1분 통과량과 녹색 시간의 상관으로 보정을 확인, docs/가상도로_정상화_20261004.md).
- DQN은 신호 순서를 스스로 정하므로, 녹색 상태 4개만 담은 틀 지도를 따로 쓴다.
"""

from __future__ import annotations

import copy
import functools
import random
import statistics
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from focused_data import PROJECT, SIGNALS, TARGET_TLS, VTYPES, find_network, read_signal_phases, signal_xml_files

UPSTREAM_LIMIT_M = 300.0
# 정지선 도착 시각을 흩뿌리는 구간(초). 통과 기록의 1분·5분 변화는 대부분 실제 신호의 녹색
# 시점이 찍힌 것이다(같은 15분 안에서 5분 통과량과 녹색 시간의 상관 0.47~0.60).
# 그대로 도착 시각으로 쓰면 실제 고정신호에 유리하므로 15분 교통량을 15분 동안 무작위로 도착시킨다.
ARRIVAL_WINDOW_S = 900
DOWNSTREAM_LIMIT_M = 150.0
# 진입로가 상류에서 두 갈래로 나뉘는 곳. 기본 경로(_chain)는 차로가 많은 갈래 하나만 따라가므로,
# 다른 갈래로 들어오는 실제 차량까지 그 갈래로 몰면 현실에 없는 엇갈림(차로 변경)과 병목이 생긴다.
# 길주로 서쪽 접근로(477707013_02, 송내 쪽): 4차로 가지(477767666_02_01 → 0~3차로)로 오는 차량은 약 46%뿐이고(상류 송내 신호의
# 회전 기록, focused_data.BRANCH_INFLOW), 나머지는 2차로 가지(477744390_01 → 4·5차로)로 들어온다.
ENTRY_BRANCHES = {
    "477707013_02": {"merge_edge": "477715088_02_02", "main": "477767666_02_01", "alternative": "477744390_01"},
}
# 모든 정책이 같이 쓰는 운전자 설정(SUMO vType 속성). 값과 근거는 config/driver_model.json.
DRIVER_MODEL = PROJECT / "config/driver_model.json"


def driver_vtype_attributes() -> dict[str, str]:
    import json
    if not DRIVER_MODEL.exists():
        return {}
    return {k: str(v) for k, v in json.loads(DRIVER_MODEL.read_text())["vtype_attributes"].items()}


def driver_model_version() -> str:
    import json
    if not DRIVER_MODEL.exists():
        return "sumo_default"
    return json.loads(DRIVER_MODEL.read_text())["version"]


class FocusedNetwork:
    """원본 지도에서 204820 주변 경로와 신호 정보를 계산한다."""

    def __init__(self, network: Path | None = None):
        self.network = network or find_network()
        root = ET.parse(self.network).getroot()
        self.root = root
        self.edges = {e.get("id"): e for e in root.findall("edge") if not e.get("id").startswith(":")}
        self.length = {k: float(e.find("lane").get("length")) for k, e in self.edges.items()}
        self.speed = {k: float(e.find("lane").get("speed")) for k, e in self.edges.items()}
        self.tls_nodes = {t.get("id") for t in root.findall("tlLogic")}
        self.succ: dict[str, dict[str, str]] = defaultdict(dict)
        self.pred: dict[str, dict[str, str]] = defaultdict(dict)
        self.links: dict[int, tuple[str, str, str]] = {}
        for c in root.findall("connection"):
            f, t = c.get("from"), c.get("to")
            if f.startswith(":"):
                continue
            self.succ[f][t] = c.get("dir")
            self.pred[t][f] = c.get("dir")
            if c.get("tl") == TARGET_TLS:
                self.links[int(c.get("linkIndex"))] = (f, t, c.get("dir"))
        self.movements = {(f, t): d for f, t, d in self.links.values()}

    def lanes(self, edge: str) -> int:
        return len(self.edges[edge].findall("lane"))

    def _chain(self, edge: str, backward: bool, limit: float) -> list[str]:
        """다음 신호교차로 또는 차로가 크게 줄어드는 지점 전까지 직진 도로를 모은다.

        실제로는 차량이 여러 길로 흩어지거나 합류하므로, 좁은 도로 하나에 모든 차량을
        몰아넣으면 현실에 없는 병목이 생긴다. 그래서 차로 수가 시작 도로의 절반 미만으로
        줄어들면 그 앞에서 멈춘다.
        """
        chain: list[str] = []
        current, distance = edge, 0.0
        min_lanes = max(1, -(-self.lanes(edge) // 2))
        while distance < limit:
            neighbours = self.pred[current] if backward else self.succ[current]
            node = self.edges[current].get("from" if backward else "to")
            candidates = [x for x, d in neighbours.items() if d == "s" and x not in chain and x != edge]
            if not candidates or (node in self.tls_nodes and node != TARGET_TLS):
                break
            nxt = sorted(candidates, key=lambda x: (-self.lanes(x), x))[0]
            if self.lanes(nxt) < min_lanes:
                break
            chain.append(nxt)
            distance += self.length[nxt]
            current = nxt
        return chain[::-1] if backward else chain

    def route_for(self, origin: str, destination: str, branch: str = "main") -> list[str]:
        up = self._chain(origin, True, UPSTREAM_LIMIT_M)
        if branch != "main":
            spec = ENTRY_BRANCHES[origin]
            up = [spec[branch]] + up[up.index(spec["merge_edge"]):]
        down = self._chain(destination, False, DOWNSTREAM_LIMIT_M)
        route = up + [origin, destination] + down
        for a, b in zip(route, route[1:]):
            if b not in self.succ[a]:
                raise ValueError(f"경로가 끊어졌습니다: {a} -> {b}")
        return route

    def travel_time_to_stopline(self, route: list[str], origin: str) -> float:
        """출발점에서 정지선까지 자유 주행 시간(초)."""
        index = route.index(origin)
        return sum(self.length[e] / self.speed[e] for e in route[: index + 1])


def entry_branch_shares(rows: list[dict]) -> dict[str, float]:
    """진입로별 기본(main) 갈래 비율. 그 15분의 상류 기록 ÷ 그 진입로의 204820 기록(0~1로 자름)."""
    from focused_data import BRANCH_INFLOW
    if not rows:
        return {}
    inflow = _branch_inflow_cache(str(BRANCH_INFLOW))
    key = (rows[0]["date"], rows[0]["period"])
    shares = {}
    for approach in ENTRY_BRANCHES:
        volume = sum(r["count"] for r in rows if r["from_edge"] == approach)
        if volume:
            if key not in inflow:
                raise KeyError(f"{key}의 상류 갈래 기록이 없습니다. focused_data.extract_branch_inflow()를 먼저 실행하세요.")
            shares[approach] = min(1.0, inflow[key] / volume)
    return shares


@functools.lru_cache(maxsize=1)
def _branch_inflow_cache(path: str) -> dict:
    from focused_data import load_branch_inflow
    return load_branch_inflow(Path(path))


def write_routes(fn: FocusedNetwork, rows: list[dict], output: Path, seed: int,
                 vtypes_file: Path | None = None, arrival_window: int = ARRIVAL_WINDOW_S,
                 time_offset: float = 0.0, vtype_attrs: dict[str, str] | None = None,
                 split_entry_branches: bool = True) -> dict:
    """15분 시간대의 1분 기록을 차량으로 바꿔 route 파일을 쓴다.

    arrival_window: 정지선 도착 시각을 흩뿌리는 구간(초). 기본 900이면 그 15분 안에 무작위로
    도착시킨다(60이면 통과가 기록된 1분 안). 1분 통과 기록에는 실제 신호의 녹색 시점이 그대로
    찍혀 있어서(녹색 없는 분 2대, 녹색 40초 이상인 분 40대), 1분 단위로 도착시키면 차량이
    실제 신호의 녹색에 맞춰 도착해 고정신호에 유리해진다.
    time_offset: 출발 시각에 더할 초(하루를 이어 붙일 때 그 시간대의 시작 시각). 0초 앞으로는 당기지 않는다.
    vtype_attrs: 모든 차종에 덧붙일 운전자 설정. None이면 config/driver_model.json.
    split_entry_branches: ENTRY_BRANCHES 진입로의 차량을 상류 기록 비율대로 두 갈래에 나눈다.
        우회전은 오른쪽 4차로 가지, 좌회전·유턴은 왼쪽 2차로 가지에서 오고(차로 위치상 가정),
        직진으로 그 15분의 갈래 비율을 맞춘다. 차량 수·차종·도착 시각은 바뀌지 않는다.
    """
    rng = random.Random(seed)
    branch_rng = random.Random(seed * 7919 + 1)   # 도착 시각 난수열을 건드리지 않도록 따로 둔다.
    vtypes_file = vtypes_file or VTYPES
    attrs = driver_vtype_attributes() if vtype_attrs is None else vtype_attrs
    root = ET.Element("routes")
    for vtype in ET.parse(vtypes_file).getroot().findall("vType"):
        vtype = copy.deepcopy(vtype)
        for k, v in attrs.items():
            vtype.set(k, v)
        root.append(vtype)
    begin = min(r["begin_unix"] for r in rows) if rows else 0
    period_start = begin - (begin % 900)
    shares = entry_branch_shares(rows) if split_entry_branches else {}
    # 갈래별 직진(·좌회전) 배정 확률: 기본 갈래 몫 = 비율 × 진입량 = 우회전 + 직진 일부 (+ 좌회전 일부)
    p_main = {}
    for approach, share in shares.items():
        by_dir = defaultdict(int)
        for r in rows:
            if r["from_edge"] == approach:
                by_dir[fn.movements[(approach, r["to_edge"])]] += r["count"]
        want = share * sum(by_dir.values()) - by_dir["r"]
        p_main[(approach, "s")] = min(1.0, max(0.0, want / by_dir["s"])) if by_dir["s"] else 0.0
        rest = want - by_dir["s"]
        left = by_dir["l"] + by_dir["t"]
        p_main[(approach, "l")] = p_main[(approach, "t")] = min(1.0, max(0.0, rest / left)) if left else 0.0
        p_main[(approach, "r")] = 1.0
    drawn = []   # (도착 시각, 차종, 이동)
    for r in rows:
        movement = (r["from_edge"], r["to_edge"])
        minute_start = r["begin_unix"] - period_start
        window_start = (minute_start // arrival_window) * arrival_window
        for k in range(r["count"]):
            drawn.append((window_start + rng.uniform(0, arrival_window), r["vehicle_type"], movement))
    # 갈래 배정: (진입로, 회전 방향)마다 정확히 round(확률 × 대수)대를 기본 갈래로 무작위 선택한다.
    branch_of = ["main"] * len(drawn)
    groups = defaultdict(list)
    for i, (_, _, movement) in enumerate(drawn):
        if movement[0] in shares:
            groups[(movement[0], fn.movements[movement])].append(i)
    for key, idx in sorted(groups.items()):
        n_main = round(p_main[key] * len(idx))
        main = set(branch_rng.sample(idx, n_main))
        for i in idx:
            branch_of[i] = "main" if i in main else "alternative"
    routes = {}
    vehicles = []
    for (arrival, vtype, movement), branch in zip(drawn, branch_of):
        key = movement + (branch,)
        if key not in routes:
            routes[key] = fn.route_for(*key)
        lead = fn.travel_time_to_stopline(routes[key], movement[0])
        depart = max(0.0, arrival - lead + time_offset)
        vehicles.append((depart, vtype, key, arrival))
    vehicles.sort(key=lambda v: v[0])
    route_ids = {}
    for i, key in enumerate(sorted(routes)):
        rid = f"r{i}"
        route_ids[key] = rid
        ET.SubElement(root, "route", id=rid, edges=" ".join(routes[key]))
    for n, (depart, vtype, key, _) in enumerate(vehicles):
        ET.SubElement(root, "vehicle", id=f"{vtype[0]}{n}", type=vtype, route=route_ids[key],
                      depart=f"{depart:.2f}", departLane="best", departSpeed="max")
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return {"vehicles": len(vehicles), "movements": len({k[:2] for k in routes}), "routes": len(routes),
            "entry_branch_shares": shares}


# ---------------------------------------------------------------- 신호

def _green_cycle(phases: list[tuple[float, str]]) -> list[str]:
    greens = [s for d, s in phases if "y" not in s and d > 0]
    cycle: list[str] = []
    for g in greens:
        if g not in cycle:
            cycle.append(g)
    return cycle


def aihub_states(period_files: dict[str, Path] | None = None, date: str = "20220810") -> dict:
    """AI Hub 신호의 녹색 4개, 각 녹색의 황색, 순환 순서를 구한다."""
    files = period_files or signal_xml_files(date)
    phases = read_signal_phases(files[sorted(files)[0]])
    cycle = _green_cycle(phases)
    yellow_of = {}
    seq = [(d, s) for d, s in phases if d > 0]
    for (d1, s1), (d2, s2) in zip(seq, seq[1:]):
        if "y" not in s1 and "y" in s2:
            yellow_of[s1] = s2
    previous = {cycle[i]: cycle[i - 1] for i in range(len(cycle))}
    return {"greens": cycle, "yellow_of": yellow_of, "previous": previous}


# ---------------------------------------------------------------- AI Hub 신호 하루 시간축
# 15분 신호 파일의 기록 방식(19일 1,824개 파일 전수 확인, docs/가상도로_정상화_20261004.md)
# - 파일은 15분 칸의 시작 시각부터 기록한다. 첫 현시는 칸 시작에서 잘린 나머지다.
# - 칸이 끝날 때 진행 중이던 현시는 그 파일에 없다(파일 길이 564~902초, 중앙값 873초).
#   그 현시는 다음 파일의 첫 현시(잘린 나머지)로 이어지고, 칸 경계에서 마침 끝났으면 다음 파일은
#   길이 0인 그 현시를 적은 뒤 그다음 현시부터 시작한다.
# 따라서 빈 시간은 '마지막 현시의 다음 순서 현시'로 채워야 시각이 맞는다(경계 1,805개 중 1,766개가
# 이 규칙과 정확히 맞고, 32개는 2초 겹침, 4개는 딱 맞음, 3개만 순서가 어긋난다).
# 파일을 그냥 이어 붙이면(v3.2까지의 방식) 경계마다 진행 중이던 녹색이 잘리거나 통째로 빠진다.
SIGNAL_DROPOUT_RATIO = 1.6     # 같은 현시의 앞뒤 30분 중앙값보다 1.6배 이상이고
SIGNAL_DROPOUT_EXTRA_S = 40    # 40초 이상 긴 현시는 기록 누락으로 본다(아래 aihub_day_timeline 설명).


def _raw_order() -> tuple[list[str], dict[str, str], dict]:
    info = aihub_states(signal_xml_files("20220810"))  # 녹색 4개·순서는 19일 모두 같다(1,824개 파일 확인)
    order = [s for g in info["greens"] for s in (g, info["yellow_of"][g])]
    successor = {s: order[(i + 1) % len(order)] for i, s in enumerate(order)}
    return order, successor, info


def _corrected_state(state: str, info: dict) -> str:
    """'한 현시 밀림' 보정: 녹색은 앞 순서 녹색으로, 황색은 앞 순서 녹색의 황색으로."""
    if state in info["previous"]:
        return info["previous"][state]
    green = {y: g for g, y in info["yellow_of"].items()}[state]
    return info["yellow_of"][info["previous"][green]]


@functools.lru_cache(maxsize=32)
def aihub_day_timeline(date: str) -> tuple[tuple[tuple[float, float, str], ...], tuple[dict, ...]]:
    """하루 신호를 끊김 없는 시간축 [(시작초, 길이, 원본 상태)]로 복원하고, 손본 곳을 기록으로 돌려준다.

    1. 빈 시간 채우기: 위 설명의 규칙. 파일이 900초보다 길면(902초) 넘친 만큼 마지막 현시를 줄인다.
    2. 순서 어긋남: 녹색 뒤에 그 황색이 아닌 다른 상태가 오면(경계 3곳) 녹색 끝 3초를 황색으로 바꾼다.
    3. 기록 누락: 같은 현시가 앞뒤 30분 중앙값의 1.6배 이상이면서 40초 이상 길면, 실제로는 신호가
       계속 돌았는데 기록이 빠진 것으로 보고 앞뒤 30분의 중앙값 주기로 채운다. 근거: 20220810
       15:17의 325초 녹색 동안에도 1분 통과 기록에는 석천로 직진·좌회전 차량이 평소 주기대로
       지나갔다. 19일 중 15곳.
    """
    order, successor, _ = _raw_order()
    files = signal_xml_files(date)
    keys = sorted(files)
    segs: list[list] = []
    notes: list[dict] = []
    carry = 0.0   # 앞 파일이 900초를 넘겨 겹친 시간. 다음 파일 첫 녹색에서 뺀다(황색 길이는 고정이므로).
    for i, key in enumerate(keys):
        t0 = int(key[:2]) * 3600 + int(key[2:]) * 60
        t = float(t0) + carry
        for d, s in read_signal_phases(files[key]):
            if d <= 0:
                continue
            if s not in successor:
                raise ValueError(f"{date} {key}: 알 수 없는 신호 상태 {s}")
            if carry:
                if "y" not in s and d > carry:
                    d -= carry
                else:
                    t -= carry
                    segs[-1][1] -= carry
                carry = 0.0
            if segs and segs[-1][2] == s and abs(segs[-1][0] + segs[-1][1] - t) < 1e-6:
                segs[-1][1] += d
            else:
                segs.append([t, d, s])
            t += d
        gap = t0 + 900 - t
        if gap < 0:
            carry = -gap
            notes.append({"kind": "overlap_trimmed", "time": t0 + 900, "seconds": carry})
            if i + 1 == len(keys):
                segs[-1][1] -= carry
                carry = 0.0
        elif gap > 0:
            segs.append([t, gap, successor[segs[-1][2]]])
    # 2. 녹색 → (그 황색이 아닌 상태): 녹색 끝을 황색으로
    fixed: list[list] = []
    for seg, nxt in zip(segs, segs[1:] + [None]):
        fixed.append(seg)
        s = seg[2]
        if nxt is not None and "y" not in s and nxt[2] != successor[s]:
            yellow = min(3.0, seg[1] - 1)
            seg[1] -= yellow
            fixed.append([seg[0] + seg[1], yellow, successor[s]])
            notes.append({"kind": "yellow_inserted", "time": seg[0] + seg[1], "between": [s, nxt[2]]})
    segs = fixed
    # 3. 기록 누락으로 비정상적으로 긴 현시를 주변 주기로 채운다.
    def local_median(state: str, around: float) -> float | None:
        vals = [d for s0, d, s in segs if s == state and abs(s0 - around) < 1800 and d > 0]
        return statistics.median(vals) if vals else None

    medians = [local_median(s, s0) for s0, _, s in segs]
    out: list[tuple[float, float, str]] = []
    for (s0, d, s), med, nxt in zip(segs, medians, segs[1:] + [None]):
        if med is None or not (d >= SIGNAL_DROPOUT_RATIO * med and d >= med + SIGNAL_DROPOUT_EXTRA_S):
            out.append((s0, d, s))
            continue
        # s부터 다음 기록 상태의 바로 앞 상태까지, 주변 중앙값 길이로 주기를 반복해 d초를 채운다.
        end_state = order[order.index(nxt[2]) - 1] if nxt is not None else None
        typical = {x: (local_median(x, s0) or 3.0) for x in order}
        # 끝 상태로 끝나는 후보 중 주변 중앙값 길이의 합이 d에 가장 가까운 것을 고른다.
        seq, state, best = [], s, None
        while True:
            seq.append(state)
            length = sum(typical[x] for x in seq)
            if end_state is None or state == end_state:
                if best is None or abs(length - d) < abs(best[0] - d):
                    best = (length, list(seq))
                if length >= d:
                    break
            state = successor[state]
        seq = best[1]
        greens = [k for k, x in enumerate(seq) if "y" not in x]
        lengths = [typical[x] for x in seq]
        scale = (d - sum(l for k, l in enumerate(lengths) if k not in greens)) / sum(lengths[k] for k in greens)
        for k in greens:
            lengths[k] *= scale
        # 정수 초로 맞추고 반올림 오차는 마지막 녹색에서 맞춘다.
        lengths = [float(round(l)) for l in lengths]
        lengths[greens[-1]] += d - sum(lengths)
        t = s0
        for x, l in zip(seq, lengths):
            out.append((t, l, x))
            t += l
        notes.append({"kind": "dropout_filled", "time": s0, "seconds": d, "state_median_s": med,
                      "phases_filled": len(seq)})
    total = sum(d for _, d, _ in out)
    if abs(total - len(keys) * 900) > 1e-6:
        raise ValueError(f"{date}: 복원한 신호 길이 {total}초가 {len(keys) * 900}초와 다릅니다.")
    return tuple(out), tuple(notes)


def aihub_unreliable_periods(date: str) -> list[str]:
    """신호 기록을 고쳐 쓴(기록 누락 채움·황색 삽입) 15분 시간대. 보정·평가 지표에서 따로 표시한다."""
    _, notes = aihub_day_timeline(date)
    periods = set()
    for n in notes:
        if n["kind"] in ("dropout_filled", "yellow_inserted"):
            start = n["time"]
            end = start + n.get("seconds", 0)
            t = start - start % 900
            while t < max(end, start + 1):
                periods.add(f"{int(t // 3600):02d}{int(t % 3600 // 60):02d}")
                t += 900
    return sorted(periods)


def _next_date(date: str) -> str:
    import datetime as _dt
    return (_dt.date(int(date[:4]), int(date[4:6]), int(date[6:])) + _dt.timedelta(days=1)).strftime("%Y%m%d")


def aihub_signal_program(date: str, start_s: float, duration_s: float) -> list[tuple[int, str]]:
    """복원한 하루 시간축에서 [start_s, start_s + duration_s) 구간의 실제 운영 신호 프로그램을 만든다.

    - 상태는 '한 현시 밀림'으로 보정한다(_corrected_state).
    - 하루를 넘기면 다음 날 자료가 있으면 이어 쓰고, 없으면 그날 마지막 15분의 완결 주기를 순서대로 반복한다.
    - 유턴 황색: 원본 문자열은 좌회전 황색 동안 유턴(11·23번 연결)을 g로 두어 유턴 차량이 황색 없이
      적색을 받는다. 그런 연결을 황색으로 바꾼다(DQN이 쓰는 SUMO-RL의 황색 규칙과 같다).
    """
    order, successor, info = _raw_order()
    timeline, day, offset = [], date, 0.0
    while True:
        segs, _ = aihub_day_timeline(day)
        timeline += [(s0 + offset, d, s) for s0, d, s in segs]
        offset += 86400
        following = SIGNALS / _next_date(day)
        if offset >= start_s + duration_s or not following.is_dir() or len(list(following.glob("*s.xml"))) != 96:
            break
        day = _next_date(day)
    end = timeline[-1][0] + timeline[-1][1]
    if end < start_s + duration_s:
        last = [x for x in timeline if x[0] >= end - 900]
        cycle = [(d, s) for _, d, s in last[1:]][:len(order)]
        k = [s for _, s in cycle].index(successor[timeline[-1][2]])
        cycle = cycle[k:] + cycle[:k]
        t = end
        while t < start_s + duration_s:
            for d, s in cycle:
                timeline.append((t, d, s))
                t += d
    phases: list[tuple[int, str]] = []
    for s0, d, s in timeline:
        a, b = max(s0, start_s), min(s0 + d, start_s + duration_s)
        if b > a:
            phases.append((int(round(b - a)), _corrected_state(s, info)))
    # 유턴 황색(위 설명): 황색 현시에서 다음 현시에 적색이 되는 g 연결을 y로 바꾼다.
    return [(d, "".join("y" if a == "g" and b == "r" else a for a, b in zip(s, phases[i + 1][1]))
             if "y" in s and i + 1 < len(phases) else s) for i, (d, s) in enumerate(phases)]


def replay_phases(period: str, duration: int = 1800, date: str = "20220810") -> list[tuple[int, str]]:
    """시나리오 시작 시각(period, 예: '0800')부터 duration초 동안의 실제 운영 신호(복원한 시간축)."""
    if period not in signal_xml_files(date):
        raise KeyError(f"{date} {period} 신호 파일이 없습니다.")
    return aihub_signal_program(date, int(period[:2]) * 3600 + int(period[2:]) * 60, duration)


def write_tls_additional(phases: list[tuple[int, str]], output: Path, program_id: str) -> Path:
    root = ET.Element("additional")
    logic = ET.SubElement(root, "tlLogic", id=TARGET_TLS, type="static", programID=program_id, offset="0")
    for d, s in phases:
        ET.SubElement(logic, "phase", duration=str(max(1, d)), state=s)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output


def write_dqn_template_network(output: Path) -> Path:
    """204820 프로그램만 AI Hub 녹색 4개(+황색) 틀로 바꾼 지도 사본을 만든다."""
    info = aihub_states()
    text = find_network().read_text(encoding="utf-8")
    root = ET.fromstring(text)
    logic = root.find(f"tlLogic[@id='{TARGET_TLS}']")
    # 원본 지도의 다른 부분은 바이트 단위로 유지하기 위해 해당 tlLogic 블록만 문자열로 교체한다.
    start = text.index(f'<tlLogic id="{TARGET_TLS}"')
    end = text.index("</tlLogic>", start) + len("</tlLogic>")
    indent = "\n" + " " * 8
    phases = []
    for g in info["greens"]:
        phases.append(f'<phase duration="30" state="{g}"/>')
        phases.append(f'<phase duration="3" state="{info["yellow_of"][g]}"/>')
    attrs = " ".join(f'{k}="{v}"' for k, v in logic.attrib.items())
    block = f"<tlLogic {attrs}>" + "".join(indent + p for p in phases) + "\n    </tlLogic>"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text[:start] + block + text[end:], encoding="utf-8")
    return output


# DQN·감응식의 황색시간(초). AI Hub 실제 운영은 대부분 4초다(보정 뒤 석천로 직진 황색만 3초).
AIHUB_YELLOW_S = 4


def write_actuated_additional(greens: list[str], output: Path, program_id: str, min_durs: list[int],
                              max_durs: list[int], yellow: int) -> Path:
    """감응식(actuated) 신호 프로그램. 차량 감지에 반응해 녹색을 줄이거나 늘린다.

    현시 순서는 그대로 두고, 각 녹색은 min_durs(보행자·차량 최소)부터 max_durs까지 움직인다.
    SUMO가 진입 차로마다 감지기를 만든다. 단, 모든 현시에서 '양보 녹색(g)'인 연결만 가진
    차로(우회전 전용 차로)는 감지기를 끈다. 그러지 않으면 우회전 차량이 지나갈 때마다
    모든 현시가 연장되어 녹색이 최대까지 늘어난다.
    """
    lane_links = defaultdict(list)
    for c in ET.parse(find_network()).getroot().findall(f"connection[@tl='{TARGET_TLS}']"):
        lane_links[f"{c.get('from')}_{c.get('fromLane')}"].append(int(c.get("linkIndex")))
    always_minor = sorted(lane for lane, links in lane_links.items()
                          if all(g[i] == "g" for g in greens for i in links))
    root = ET.Element("additional")
    logic = ET.SubElement(root, "tlLogic", id=TARGET_TLS, type="actuated", programID=program_id, offset="0")
    for i, g in enumerate(greens):
        nxt = greens[(i + 1) % len(greens)]
        y = "".join("y" if a in "Gg" and b in "rs" else a for a, b in zip(g, nxt))
        ET.SubElement(logic, "phase", duration=str(min_durs[i]), minDur=str(min_durs[i]),
                      maxDur=str(max(min_durs[i], max_durs[i])), state=g)
        ET.SubElement(logic, "phase", duration=str(yellow), state=y)
    for lane in always_minor:
        ET.SubElement(logic, "param", key=lane, value="NO_DETECTOR")
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output
