"""12개 신호 교차로 지도의 하루 수요(차량 경로)를 만든다 — 지문 밖, SUMO 실행 없음.

AI Hub 기록은 교차로마다 따로 센 1분 회전 통과 횟수라서, 같은 차가 다음 교차로에서 어디로 갔는지 이어져 있지 않다.
그래서 회전 기록에 가장 잘 맞는 차량 경로 묶음을 추정한다(AI Hub 구축 가이드 60~61쪽의 routeSampler와 같은 생각).

1. 이동 서명: 신호 교차로 12곳의 연결을 '지도에서 지나는 도로 순서'로 적는다.
   - 기록 이동: 직접 연결은 [진입, 진출], 번호 대응 3개는 [진입, 477706911_02_02], 분리 우회전 6개는 분리 차로 도로 3개
     (오류 목록 2절). 지도에 길이 없는 201306 두 이동은 뺀다(생략, 대수는 장부에 남김).
   - 201293 유턴은 두 판: drop(지운 판) / straight(같은 진입로의 직진 477603993_02_02로 본 판).
   - 기록에 없는 지도 연결도 서명으로 둔다. 기록이 없으면 그 15분에 지나간 차가 0대라고 보고 맞춘다(가정).
2. 후보 경로(12개 지도 network_map.network_12 위, 빈 도로 주행 시간 기준, 길 중간 유턴 없음):
   - 출발·도착 도로(기본 chains): 신호 교차로 이동마다 진입 도로의 '진입 구간' 시작과 진출 도로의 '진출 구간' 끝.
     구간 규칙은 석천사거리 단독 모형(focused_scenario.FocusedNetwork._chain)과 같다: 직진 연결만 따라가며 다른 신호
     교차로, 차로가 절반 미만으로 줄어드는 곳, 진입 300m·진출 150m에서 멈춘다(석천사거리 14개 이동에서 같음을 검사).
     단독 모형처럼 석천사거리 서쪽 진입로의 다른 갈래(477744390_01)도 출발 도로로 둔다. 이어지는 교차로를 지나는 차는
     한 대로 이어진다. boundary는 지도 끝 → 지도 끝(처음 방식): 08-11 하루 점검에서 지도 가장자리 옆길과 신호 없는 교차로에
     차가 몰려 순간이동을 끄면 하루 안에 바둑판 전체가 멈췄고(세 번), 순간이동 300초를 켜도 123개 교차로에서 3,424번
     옮기며 낮 동안 회복하지 못했다.
   - 모든 출발·도착 쌍의 최단 경로
   - 이동마다 그 이동을 지나는 경로: 같은 쌍의 최단보다 30% 이내로 더 걸리는 것과, 적어도 가장 짧은 10개
3. 15분마다 선형계획(네 차종을 한꺼번에): Σ 가중치 × (모자람 + 3 × 넘침) + Σ 경로 대수 × (0.01 + 0.001 × 경로 주행 분) 을
   최소로. 회전 기록은 차종마다 맞추고, 아래 상한은 네 차종의 합에 건다.
   - 차 한 대마다 0.01: 같은 통과를 더 적은 차(여러 교차로를 지나는 긴 경로)로 설명한다(routeSampler의 차 수 줄이기와 같음).
     없으면 '석천사거리를 지나 옆길로 나가는 차 + 옆길에서 들어와 204822로 가는 차'처럼 짧은 경로로 쪼개, 08-11 아침
     1차로 옆길(477703841_01)에 시간당 1,963대가 몰렸다.
   - 지도 끝 도로마다 15분에 드나드는 차(네 차종 합)는 차로 수 × 시간당 600대 이하(가정).
     1차로 지도 끝만 시간당 200대로 더 묶어 보았으나(08-11 학습 날짜), 석천사거리를 맞추려고 cluster에 기록에 없는 통과가
     하루 11,157대 생겨 쓰지 않는다. 지도 끝은 지도를 자른 자리라 1차로라고 실제로 작은 길이라는 보장이 없다.
   - 이웃 신호 11곳의 직진·좌회전·유턴은 15분마다 그 이동의 녹색·황색 초 × 차로 수 × 처리 능력(SUMO에서 잰 직진 시간당
     1,880대·좌회전 1,340대의 90%) 이하. 실제 신호(network_signals)로는 보낼 수 없는 몫까지 넣으면 줄이 짧은 진입로를 채우고 바둑판 전체가
     막힌다(08-11 하루 점검). 석천사거리는 묶지 않는다(기록대로 맞추고, 단독 모형에서 처리됨을 확인). 줄인 대수는 장부에.
   가중치는 석천사거리 10, 가볍게 맞추는 곳(보완 지점 201300·201329·201330과 자료끼리 맞지 않는 201306) 0.25, 나머지 1.
   넘침(기록보다 많이 만든 통과)을 모자람보다 3배 무겁게 본다. 그러지 않으면 이웃끼리 안 맞는 기록을 메우려고 기록에 없는
   흐름(예: 08-11 204920 유턴 기록 240대 → 계획 2,475대, 차 한 대로 cluster의 두 직진을 채우는 왕복 경로)을 만들어 낸다.
4. 정수로: 경로 대수의 소수 부분만큼의 확률로 한 대를 더한다(날짜·시간대·차종마다 고정 난수).
5. 출발 시각: 석천사거리를 지나는 차는 석천사거리 정지선에, 아닌 차는 경로의 첫 신호 교차로 정지선에 그 15분 안의
   무작위 시각에 닿도록 빈 도로 주행 시간만큼 당긴다(단독 모형과 같은 원칙). 0초보다 앞으로는 당기지 않는다.
   - 출발 시각 보정(depart_fix, network_day가 만든다): 석천사거리를 지나는 차는 단독 모형 구간 입구(진입 구간 시작)까지
     오는 동안 이웃 신호·출발 대기로 늦는다(08-11 배경 80%: 동쪽 직진 평균 60초, 서쪽 직진 중 cluster를 지난 차 193초).
     단독 모형에서는 차가 바로 그 입구에서 출발하므로 이 늦음이 0이다. 그래서 보정 없이 한 번 돌린 하루 점검에서 경로·
     시간대별로 '입구에 계획보다 늦게 온 초'를 재고, 그만큼 출발을 더 당긴다(차·경로·도착 시각은 그대로).
6. 장부: 계획된 통과(차량 경로 기준, 지나는 시각은 무시)를 기록과 비교한다(15분·1시간, 교차로별, GEH).

사용(시험 날짜는 기본으로 막힘):
  .venv/bin/python src/network_demand.py build --date 20220811 [--uturn drop|straight] [--origins chains|boundary]
결과: results/network_demand/<날짜>_<유턴 판>[_boundary]/routes.rou.xml, ledger.json, audit.md
"""
from __future__ import annotations

import argparse
import copy
import heapq
import json
import math
import random
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sumolib
from scipy.optimize import linprog
from scipy.sparse import block_diag, csr_matrix, hstack, identity, vstack

import focused_data as fd
from focused_scenario import driver_vtype_attributes, route_seed
from network_map import network_12

OUT = fd.PROJECT / "results/network_demand"
SPLIT = fd.PROJECT / "config/focused_split.json"
DAY0_UNIX = 1660057200                         # 2022-08-10 00:00 (KST)
PERIOD_S = 900
TARGET_TLS = "204820"
# 가볍게 맞추는 곳: AI Hub 설명서상 측정 지점이 아닌 보완 지점 3곳 + 자료끼리 맞지 않는 201306
# (201306: 신호와 통과 시각이 어느 읽기로도 맞지 않고(오류 목록 4절), 이웃과 차 수도 맞지 않음 — 201329 → 201306 −49%,
#  201306 → 201293 +203%, 사이에 옆길 없음. 그래서 위아래 이웃 기록을 더 믿는다.)
LOW_TRUST_TLS = {"201300", "201329", "201330", "201306"}
WEIGHT = {"target": 10.0, "low_trust": 0.25, "other": 1.0}
DETOUR = 0.3
MIN_ROUTES_PER_MOVEMENT = 10
CHAIN_UP_M, CHAIN_DOWN_M = 300.0, 150.0        # 진입·진출 구간 한도(단독 모형 focused_scenario의 UPSTREAM/DOWNSTREAM_LIMIT_M과 같음)
EXTRA_ENTRIES = ("477744390_01",)              # 단독 모형 ENTRY_BRANCHES: 석천사거리 서쪽 진입로의 다른 갈래(2차로)
LINK_LIMIT_M = 1500.0                          # 이웃 신호 사이를 직진으로 잇는 길로 볼 최대 거리
LENGTH_COST_PER_MIN = 0.001
VEHICLE_COST = 0.01                            # 차 한 대마다(긴 경로 한 대 > 짧은 경로 여러 대)
BOUNDARY_CAP_PER_LANE_H = 600                  # 지도 끝 도로마다 드나드는 차의 상한(차로당 시간당)
# 이웃 신호 이동의 상한 = 녹색·황색 1시간 동안 차로당 SUMO에서 실제로 지나는 차 × 0.9(포화도 0.9, 줄이 끝없이 자라지 않게).
# 처리 능력은 단독 모형에서 잰 값(config/driver_model.json, sigma 0): 직진 약 1,880대, 석천로 좌회전 약 1,340대.
# 처음 넣었던 직진 1,800·회전 1,500은 SUMO 좌회전보다 커서 cluster 북쪽 좌회전 줄이 13m 진입로를 넘쳐 지도 끝까지 찼다(08-11).
SATURATION_H = {"s": 1880 * 0.9, "l": 1340 * 0.9, "t": 1340 * 0.9}
UTILIZATION = 0.9
# 이동별 보정(config/network12_capacity.json): 일반값보다 SUMO가 실제로 덜 내보내는 이동만, 학습 날짜 하루 점검에서 잰 값
CAPACITY_FILE = fd.PROJECT / "config/network12_capacity.json"
# 출발 시각 보정: 경로·시간대 평균은 차가 적으면 흔들리므로 그 경로의 하루 평균 쪽으로 당긴다(차 5대 몫).
# 보정은 0~900초(15분 칸 하나)로 자른다.
DEPART_FIX_SHRINK = 5
DEPART_FIX_MAX_S = 900.0


def route_key(edges) -> str:
    """경로(도로 순서)의 짧은 이름. 경로 번호는 후보 목록 순서라 보정표는 도로 순서로 찾는다."""
    import hashlib
    return hashlib.sha1(" ".join(edges).encode()).hexdigest()[:16]


def depart_shift(fix: dict, key: str, hour: int) -> float:
    """출발 시각 보정표(network_day.measure_lag)에서 경로·도착 시의 당길 초."""
    day = fix["by_route"].get(key)
    if day is None:
        return 0.0
    n, m = fix["by_route_hour"].get(f"{key}|{hour}", (0, 0.0))
    value = (n * m + DEPART_FIX_SHRINK * day[1]) / (n + DEPART_FIX_SHRINK)
    return float(min(DEPART_FIX_MAX_S, max(0.0, value)))


def capacity_overrides() -> dict[str, float]:
    if not CAPACITY_FILE.exists():
        return {}
    return {k: float(v) for k, v in json.loads(CAPACITY_FILE.read_text())["saturation_per_lane_h"].items()}
OVER_FACTOR = 3.0                              # 넘침 1대 = 모자람 3대
CODES = ("p", "b", "t", "m")                   # 승용·버스·트럭·이륜(파일 이름 끝 글자)
VTYPE_OF = {"p": "passenger", "b": "bus", "t": "truck", "m": "motorcycle"}
# 오류 목록(docs/12개교차로_오류목록_20261015.md) 2절: 기록 이동 → 지도에서 지나는 도로 순서와 신호
SPECIAL = {
    ("477706883_01", "477706911_02"): (("477706883_01", "477706911_02_02"), "204820", "번호 대응"),
    ("477707013_02", "477706911_02"): (("477707013_02", "477706911_02_02"), "204820", "번호 대응"),
    ("477706392_02", "477706911_02"): (("477706392_02", "477706911_02_02"), "204820", "번호 대응"),
    ("477765074_01", "477757554_01"): (("477714244_02_01", "477712245_01", "477707232_01_01"), "cluster_204796_205202", "분리 우회전"),
    ("477765078_01", "477753313_01"): (("477754785_0202_01", "E1", "477753312_0101_01"), "cluster_204796_205202", "분리 우회전"),
    ("477605419_02", "477643151_01"): (("477643154_02020202_01", "477643146_01", "477649538_0101_01"), "201330", "분리 우회전"),
    ("477643151_02", "477605685_01"): (("477649538_0101_02", "477643145_01", "477643149_02_02"), "201330", "분리 우회전"),
    ("477605685_02", "477657424_01"): (("477643149_02_01", "477643147_01", "-E4"), "201330", "분리 우회전"),
    ("477657424_02", "477605419_01"): (("E4", "477643148_01", "477643154_02020202_02"), "201330", "분리 우회전"),
}
OMITTED = {("477610076_02", "477649580_01_01"): "201306", ("477651439_02", "477649578_02_02"): "201306"}
UTURN_201293 = ("477670332_01", "477670332_02")
UTURN_AS_STRAIGHT = ("477670332_01", "477603993_02_02")


@dataclass(frozen=True)
class Movement:
    key: tuple[str, str]           # 기록의 (진입, 진출). 기록에 없는 지도 연결은 지도 도로 그대로
    signature: tuple[str, ...]     # 지도에서 지나는 도로 순서
    tls: str
    kind: str                      # 직접 연결 / 번호 대응 / 분리 우회전 / 유턴을 직진으로 / 기록 없는 연결

    @property
    def weight(self) -> float:
        if self.tls == TARGET_TLS:
            return WEIGHT["target"]
        return WEIGHT["low_trust"] if self.tls in LOW_TRUST_TLS else WEIGHT["other"]


class Graph:
    """지도의 보통 도로와 연결. 네 차종이 모두 같은 연결을 쓸 수 있는지 확인한다(이 지도는 같음)."""

    def __init__(self, network: Path):
        net = sumolib.net.readNet(str(network), withInternal=False)
        self.edges = {e.getID(): e for e in net.getEdges() if e.getFunction() != "internal"}
        self.cost = {i: e.getLength() / max(e.getSpeed(), 1.0) for i, e in self.edges.items()}
        per_class = {}
        for code, vc in VTYPE_OF.items():
            succ = defaultdict(set)
            for i, e in self.edges.items():
                if not e.allows(vc):
                    continue
                for t, conns in e.getOutgoing().items():
                    if t.getID() in self.edges and t.allows(vc) and any(
                            c.getFromLane().allows(vc) and c.getToLane().allows(vc) for c in conns):
                        succ[i].add(t.getID())
            per_class[code] = {k: frozenset(v) for k, v in succ.items()}
        if any(per_class[c] != per_class["p"] for c in CODES):
            raise ValueError("차종마다 지나갈 수 있는 연결이 다릅니다. 후보 경로를 차종별로 만들어야 합니다.")
        self.succ = {k: sorted(v) for k, v in per_class["p"].items()}
        self.pred = defaultdict(list)
        for a, ts in self.succ.items():
            for b in ts:
                self.pred[b].append(a)
        self.dir = {(i, t.getID()): conns[0].getDirection() for i, e in self.edges.items()
                    for t, conns in e.getOutgoing().items() if t.getID() in self.edges}
        self.tls_nodes = {n.getID() for n in net.getNodes() if n.getType().startswith("traffic_light")}
        self.tls_links = defaultdict(set)              # 신호 → {(진입, 진출)}
        for i, e in self.edges.items():
            for t, conns in e.getOutgoing().items():
                for c in conns:
                    if c.getTLSID():
                        self.tls_links[c.getTLSID()].add((i, t.getID()))
        self.entries = sorted(e for e in self.edges if not [p for p in self.pred[e] if not self.reverse(p, e)])
        self.exits = sorted(e for e in self.edges if not [s for s in self.succ.get(e, ()) if not self.reverse(e, s)])

    def lanes(self, e: str) -> int:
        return self.edges[e].getLaneNumber()

    def chain(self, edge: str, backward: bool, limit: float) -> list[str]:
        """단독 모형 FocusedNetwork._chain과 같은 규칙: 직진 연결만 따라가고, 신호 교차로·차로가 시작 도로의 절반 미만으로
        줄어드는 곳·거리 한도에서 멈춘다. 거꾸로면 시작 쪽부터 순서대로 돌려준다."""
        out, current, distance = [], edge, 0.0
        min_lanes = max(1, -(-self.lanes(edge) // 2))
        while distance < limit:
            nbr = self.pred.get(current, []) if backward else self.succ.get(current, [])
            node = (self.edges[current].getFromNode() if backward else self.edges[current].getToNode()).getID()
            cands = [x for x in nbr if self.dir.get((x, current) if backward else (current, x)) == "s"
                     and x not in out and x != edge]
            if not cands or node in self.tls_nodes:
                break
            nxt = sorted(cands, key=lambda x: (-self.lanes(x), x))[0]
            if self.lanes(nxt) < min_lanes:
                break
            out.append(nxt)
            distance += self.edges[nxt].getLength()
            current = nxt
        return out[::-1] if backward else out

    def reverse(self, a: str, b: str) -> bool:
        ea, eb = self.edges[a], self.edges[b]
        return ea.getFromNode() == eb.getToNode() and ea.getToNode() == eb.getFromNode()

    def dijkstra(self, start: str, forward: bool, allowed: set[str] | None = None) -> tuple[dict[str, float], dict[str, str]]:
        """빈 도로 주행 시간 최단 경로(길 중간 유턴 없음). dist는 양 끝 도로를 포함한 합. allowed가 있으면 그 도로만 다닌다."""
        dist, link, pq = {start: self.cost[start]}, {}, [(self.cost[start], start)]
        nbr = self.succ if forward else self.pred
        while pq:
            d, e = heapq.heappop(pq)
            if d > dist[e]:
                continue
            for t in nbr.get(e, ()):
                if (self.reverse(e, t) if forward else self.reverse(t, e)) or (allowed is not None and t not in allowed):
                    continue
                nd = d + self.cost[t]
                if nd < dist.get(t, math.inf):
                    dist[t], link[t] = nd, e
                    heapq.heappush(pq, (nd, t))
        return dist, link


def movements(graph: Graph, recorded: set[tuple[str, str]], uturn: str) -> tuple[list[Movement], dict]:
    """서명 목록과, 뺀 기록 이동 {(진입, 진출): 이유}."""
    out, dropped, seen = [], {}, set()
    tls_of = {link: tls for tls, links in graph.tls_links.items() for link in links}
    for key in sorted(recorded):
        if key in OMITTED:
            dropped[key] = "지도에 길 없음(201306, 생략)"
            continue
        if key == UTURN_201293:
            if uturn == "drop":
                dropped[key] = "201293 유턴(지운 판)"
                continue
            out.append(Movement(key, UTURN_AS_STRAIGHT, "201293", "유턴을 직진으로"))
            seen.add(UTURN_AS_STRAIGHT)
            continue
        if key in SPECIAL:
            sig, tls, kind = SPECIAL[key]
        elif key in tls_of:
            sig, tls, kind = key, tls_of[key], "직접 연결"
        else:
            raise ValueError(f"기록 이동 {key}를 지도에 놓을 수 없습니다(오류 목록에 없는 새 이동).")
        if any(b not in graph.succ.get(a, ()) for a, b in zip(sig, sig[1:])):
            raise ValueError(f"{key}의 서명 {sig}이 지도에서 이어지지 않습니다.")
        out.append(Movement(key, sig, tls, kind))
        if len(sig) == 2:
            seen.add(sig)
    for tls, links in sorted(graph.tls_links.items()):
        for link in sorted(links):
            if link not in seen and link not in recorded:
                out.append(Movement(link, link, tls, "기록 없는 연결"))
    return out, dropped


def chain_ends(graph: Graph, mvs: list[Movement]) -> tuple[list[str], list[str]]:
    """'구간' 방식의 출발·도착 도로: 신호 교차로 이동마다 진입 도로의 진입 구간 시작, 진출 도로의 진출 구간 끝.
    진출 구간은 다음 신호 교차로의 진입 도로(끝이 신호 교차로인 도로) 앞에서 끝낸다. 그 도로에서 도착하는 차는 다음 신호의
    적색 앞에 줄을 섰다가 정지선에서 사라져, 실제로 없는 줄을 만든다(08-11: 석천사거리 남쪽 출구 → 201292 북쪽 진입로에
    하루 약 4,900대, 그 줄이 석천사거리 남쪽 출구까지 차올라 한낮 교착의 시작이 됨)."""
    starts, ends = set(EXTRA_ENTRIES) & set(graph.edges), set()
    for m in mvs:
        up = graph.chain(m.signature[0], True, CHAIN_UP_M)
        down = graph.chain(m.signature[-1], False, CHAIN_DOWN_M)
        while down and graph.edges[down[-1]].getToNode().getID() in graph.tls_nodes:
            down.pop()
        starts.add(up[0] if up else m.signature[0])
        ends.add(down[-1] if down else m.signature[-1])
    return sorted(starts), sorted(ends)


def corridor_edges(graph: Graph, mvs: list[Movement]) -> set[str]:
    """'큰길 축' 도로: 이동 서명 + 진입 구간(300m) + 진출 구간(150m) + 진출 도로에서 직진으로 다음 신호 교차로까지 이어지는
    길(1.5km 안) + 석천사거리 서쪽 진입로의 다른 갈래에서 진입로까지의 길."""
    out, approaches = set(), {m.signature[0] for m in mvs}
    for m in mvs:
        out |= set(m.signature)
        out |= set(graph.chain(m.signature[0], True, CHAIN_UP_M))
        out |= set(graph.chain(m.signature[-1], False, CHAIN_DOWN_M))
        link = graph.chain(m.signature[-1], False, LINK_LIMIT_M)
        end = link[-1] if link else m.signature[-1]
        nxt = [x for x in graph.succ.get(end, ()) if x in approaches]
        if graph.edges[end].getToNode().getID() in graph.tls_nodes or nxt:
            out |= set(link) | set(nxt)
    for e in EXTRA_ENTRIES:
        if e in graph.edges:
            dist, prev = graph.dijkstra(e, True)
            targets = [a for a in approaches if a in dist and graph.edges[a].getToNode().getID() == TARGET_TLS]
            if targets:
                t = min(targets, key=lambda a: dist[a])
                p = [t]
                while p[-1] != e:
                    p.append(prev[p[-1]])
                out |= set(p)
    return out


def candidate_routes(graph: Graph, mvs: list[Movement], origins: str = "chains", roads: str = "corridor") -> list[tuple[str, ...]]:
    entries, exits = chain_ends(graph, mvs) if origins == "chains" else (graph.entries, graph.exits)
    allowed = corridor_edges(graph, mvs) if roads == "corridor" else None
    if allowed is not None:
        entries, exits = [a for a in entries if a in allowed], [b for b in exits if b in allowed]
    fwd = {a: graph.dijkstra(a, True, allowed) for a in entries}
    bwd = {b: graph.dijkstra(b, False, allowed) for b in exits}

    def path_from(a, e):
        p = [e]
        while p[-1] != a:
            p.append(fwd[a][1][p[-1]])
        return p[::-1]

    def path_to(e, b):
        p = [e]
        while p[-1] != b:
            p.append(bwd[b][1][p[-1]])
        return p

    routes = set()
    for a in entries:
        for b in exits:
            if b != a and b in fwd[a][0]:
                routes.add(tuple(path_from(a, b)))
    for m in mvs:
        s = m.signature
        mid = sum(graph.cost[e] for e in s[1:-1])
        options = []
        for a in entries:
            da = fwd[a][0].get(s[0])
            if da is None:
                continue
            for b in exits:
                db = bwd[b][0].get(s[-1])
                if db is not None:
                    options.append((da + mid + db, fwd[a][0].get(b), a, b))
        options.sort()
        chosen = [o for o in options if o[1] is not None and o[0] <= (1 + DETOUR) * o[1]]
        chosen += options[:MIN_ROUTES_PER_MOVEMENT]
        for _, _, a, b in chosen:
            p = tuple(path_from(a, s[0]) + list(s[1:-1]) + path_to(s[-1], b))
            if len(set(p)) == len(p):
                routes.add(p)
    return sorted(routes)


def incidence(routes: list[tuple[str, ...]], mvs: list[Movement]) -> csr_matrix:
    """(이동 × 경로) 행렬: 경로가 그 이동을 몇 번 지나는가."""
    by_first = defaultdict(list)
    for i, m in enumerate(mvs):
        by_first[m.signature[0]].append(i)
    rows, cols = [], []
    for j, r in enumerate(routes):
        for k, e in enumerate(r):
            for i in by_first.get(e, ()):
                s = mvs[i].signature
                if r[k:k + len(s)] == s:
                    rows.append(i)
                    cols.append(j)
    return csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(mvs), len(routes)))


def anchor_times(graph: Graph, routes: list[tuple[str, ...]], mvs: list[Movement]) -> tuple[np.ndarray, np.ndarray]:
    """경로마다 (빈 도로 전체 주행 시간, 출발 → 기준 정지선 시간). 기준은 석천사거리 이동, 없으면 첫 신호 교차로 이동."""
    first = defaultdict(list)
    for m in mvs:
        first[m.signature[0]].append(m)
    total, lead = np.zeros(len(routes)), np.full(len(routes), np.nan)
    for j, r in enumerate(routes):
        t, first_any, at_target = 0.0, None, None
        for k, e in enumerate(r):
            t += graph.cost[e]
            for m in first.get(e, ()):
                if r[k:k + len(m.signature)] == m.signature:
                    if first_any is None:
                        first_any = t
                    if m.tls == TARGET_TLS and at_target is None:
                        at_target = t
        total[j] = t
        lead[j] = at_target if at_target is not None else (first_any if first_any is not None else np.nan)
    return total, lead


def read_counts(dates: set[str]) -> dict[str, dict[tuple[str, str], Counter]]:
    """{날짜: {(15분 이름, 차종 글자): Counter{(진입, 진출): 대수}}}"""
    out = defaultdict(lambda: defaultdict(Counter))
    for name, data in fd.iter_turn_files():
        date = re.search(r"/(\d{8})/", name).group(1)
        if date not in dates:
            continue
        code = Path(name).stem[-1]
        if code not in VTYPE_OF:
            raise ValueError(f"모르는 차종 글자: {name}")
        for interval in ET.fromstring(data).iter("interval"):
            sec = (int(float(interval.get("begin"))) - DAY0_UNIX) % 86400
            period = f"{sec // 3600:02d}{sec % 3600 // PERIOD_S * 15:02d}"
            for rel in interval.iter("edgeRelation"):
                out[date][(period, code)][(rel.get("from"), rel.get("to"))] += int(float(rel.get("count")))
    return out


def boundary_limits(graph: Graph, routes: list[tuple[str, ...]]) -> tuple[csr_matrix, np.ndarray]:
    """(지도 끝 도로 × 경로) 행렬과 15분 상한: 그 도로에서 출발하거나 그 도로에서 끝나는 경로 대수의 합 ≤ 차로 수 × 600 / 4."""
    ends = sorted({r[0] for r in routes} | {r[-1] for r in routes})
    row = {(e, kind): i for i, (e, kind) in enumerate((e, k) for e in ends for k in ("start", "end"))}
    rows, cols = [], []
    for j, r in enumerate(routes):
        rows += [row[(r[0], "start")], row[(r[-1], "end")]]
        cols += [j, j]
    cap = np.array([graph.edges[e].getLaneNumber() * BOUNDARY_CAP_PER_LANE_H * PERIOD_S / 3600 for e, _ in row])
    return csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(row), len(routes))), cap


def capped_movements(mvs: list[Movement], graph: Graph) -> dict[int, tuple[list[tuple[str, int, str]], str]]:
    """처리 능력 상한을 거는 이동: 이웃 신호의 직진·좌회전·유턴(서명이 진입·진출 두 도로인 직접 연결).
    돌려줌: {이동 번호: ([(신호, 칸, 방향)], 'straight' 또는 'turn')}."""
    lane_links = defaultdict(list)                      # (진입, 진출) → [신호 칸 번호]
    for i, e in graph.edges.items():
        for t, conns in e.getOutgoing().items():
            for c in conns:
                if c.getTLSID():
                    lane_links[(i, t.getID())].append((c.getTLSID(), c.getTLLinkIndex(), c.getDirection()))
    out = {}
    for j, m in enumerate(mvs):
        if m.tls == TARGET_TLS or len(m.signature) != 2:
            continue
        links = lane_links.get(m.signature, [])
        dirs = {d for _, _, d in links}
        if not links or dirs & {"r", "R"}:
            continue
        out[j] = (links, "straight" if dirs <= {"s"} else "turn")
    return out


def green_lane_seconds(date: str, mvs: list[Movement], graph: Graph) -> np.ndarray:
    """(96, 이동) 15분마다 그 이동의 신호 칸(차로마다 하나)이 녹색·황색인 초의 합. 상한을 거는 이동만, 나머지 0."""
    import network_signals as ns
    programs, _ = ns.day_programs(date)
    usable = {}
    for tls, phases in programs.items():
        if tls == TARGET_TLS:
            continue
        per_second = np.zeros((len(phases[0][1]), 86400), dtype=bool)
        t = 0
        for d, st in phases:
            if t >= 86400:
                break
            for k, ch in enumerate(st):
                if ch in "Ggy":
                    per_second[k, t:min(86400, t + d)] = True
            t += d
        usable[tls] = per_second.reshape(len(phases[0][1]), 96, 900).sum(2)    # (칸, 96) 녹색·황색 초
    out = np.zeros((96, len(mvs)))
    for j, (links, _) in capped_movements(mvs, graph).items():
        out[:, j] = sum(usable[tls][k] for tls, k, _ in links if tls in usable)
    return out


def movement_capacity(date: str, mvs: list[Movement], graph: Graph) -> np.ndarray:
    """(96, 이동) 15분 상한. 이웃 신호의 직진·좌회전·유턴만, 나머지는 무한."""
    cap = np.full((96, len(mvs)), np.inf)
    seconds = green_lane_seconds(date, mvs, graph)
    override = capacity_overrides()
    for j, (_, kind) in capped_movements(mvs, graph).items():
        m = mvs[j]
        key = f"{m.key[0]}>{m.key[1]}"
        sat = (UTILIZATION * override[key] if key in override else SATURATION_H["s" if kind == "straight" else "l"]) / 3600
        cap[:, j] = seconds[:, j] * sat
    return cap


def fit_interval(A: csr_matrix, Y: np.ndarray, w: np.ndarray, minutes: np.ndarray,
                 limits: tuple[csr_matrix, np.ndarray], cap: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """15분 한 칸의 선형계획. 차종 k개를 한꺼번에 푼다. Y: (k, 이동) 기록, 돌려줌: (k, 경로) 대수.
    최소화 Σ차종 Σ이동 w(모자람 + 3 × 넘침) + Σ차종 Σ경로 대수 × (0.01 + 0.001 × 주행 분)
    조건: 차종마다 A x + 모자람 − 넘침 = y / 지도 끝마다 드나드는 대수(차종 합) ≤ 상한 / 이웃 신호 이동(차종 합) ≤ 처리 능력."""
    k = Y.shape[0]
    X = np.zeros((k, A.shape[1]))
    useful = np.flatnonzero(np.asarray(A[Y.sum(0) > 0].sum(0)).ravel() > 0)   # 기록이 있는 이동을 하나라도 지나는 경로
    if useful.size == 0:
        return X, 0.0
    sub = A.tocsc()[:, useful]
    n_m, n_r = sub.shape
    B = limits[0].tocsc()[:, useful]
    b_ub = limits[1]
    if cap is not None and np.isfinite(cap).any():
        capped = np.flatnonzero(np.isfinite(cap))
        B = vstack([B, sub[capped]], format="csc")
        b_ub = np.concatenate([b_ub, cap[capped]])
    # 변수 순서: 차종별 경로 대수 [x_1 … x_k], 차종별 모자람 [u_1 … u_k], 차종별 넘침 [o_1 … o_k]
    A_eq = hstack([block_diag([sub] * k), identity(k * n_m), -identity(k * n_m)], format="csr")
    A_ub = hstack([hstack([B] * k), csr_matrix((B.shape[0], 2 * k * n_m))], format="csr")
    route_cost = VEHICLE_COST + LENGTH_COST_PER_MIN * minutes[useful]
    c = np.concatenate([np.tile(route_cost, k), np.tile(w, k), np.tile(OVER_FACTOR * w, k)])
    res = linprog(c, A_eq=A_eq, b_eq=Y.ravel(), A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method="highs")
    if res.status != 0:
        raise RuntimeError(f"선형계획 실패: {res.message}")
    X[:, useful] = np.clip(res.x[:k * n_r].reshape(k, n_r), 0, None)
    return X, float(res.fun)


def fit_period(A: csr_matrix, y: np.ndarray, w: np.ndarray, minutes: np.ndarray,
               limits: tuple[csr_matrix, np.ndarray], cap: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """차종 하나짜리 fit_interval."""
    X, f = fit_interval(A, y[None, :], w, minutes, limits, cap)
    return X[0], f


def build_day(date: str, uturn: str, output: Path, counts: dict | None = None, origins: str = "chains",
              background: float = 1.0, roads: str = "corridor", depart_fix: dict | None = None) -> dict:
    """background: 석천사거리를 지나지 않는 차(배경 교통)만 이 비율로 줄인다(선형계획 뒤, 정수로 바꾸기 전).
    석천사거리를 지나는 차는 그대로라 석천사거리 수요는 단독 모형과 같다. 12개 지도가 한낮 교통량을 SUMO에서 다 실어
    나르지 못해(docs/12개지도_수요_20261015.md 4절) 하루가 끝까지 도는 비율을 학습 날짜로 찾는다(2026-10-15 사용자 결정).
    depart_fix: 출발 시각 보정표(network_day.measure_lag). 주면 석천사거리를 지나는 차의 출발을 그만큼 더 당긴다.
    선형계획·난수가 같아 차·경로·도착 시각은 보정 없는 수요와 똑같고 출발 시각만 다르다."""
    t0 = time.time()
    network = network_12()
    graph = Graph(network)
    counts = counts if counts is not None else read_counts({date})[date]
    recorded = {k for c in counts.values() for k in c}
    mvs, dropped = movements(graph, recorded, uturn)
    routes = candidate_routes(graph, mvs, origins, roads)
    A = incidence(routes, mvs)
    total_s, lead_s = anchor_times(graph, routes, mvs)
    w = np.array([m.weight for m in mvs])
    index = {m.key: i for i, m in enumerate(mvs)}
    through_target = np.asarray(A[[i for i, m in enumerate(mvs) if m.tls == TARGET_TLS]].sum(0)).ravel() > 0
    limits = boundary_limits(graph, routes)
    capacity = movement_capacity(date, mvs, graph)
    Y = {}                                             # (차종, 15분 번호) → 이동별 기록
    dropped_n = Counter()
    for code in CODES:
        for p in range(96):
            period = f"{p // 4:02d}{p % 4 * 15:02d}"
            y = np.zeros(len(mvs))
            for key, n in counts.get((period, code), {}).items():
                if key in dropped:
                    dropped_n[(key, code)] += n
                else:
                    y[index[key]] += n
            Y[(code, p)] = y
    total_y = {p: sum(Y[(c, p)] for c in CODES) for p in range(96)}
    capped_off = Counter()                             # 상한 때문에 줄어든 기록(이동별)
    print(f"이동 {len(mvs)}개(기록 {sum(m.kind != '기록 없는 연결' for m in mvs)}), 후보 경로 {len(routes):,}개, "
          f"출발·도착 방식 {origins} ({time.time() - t0:.0f}초)", flush=True)
    vehicles, ledger, shifts = [], {"planned": {}, "observed": {}}, []
    for p in range(96):
        period = f"{p // 4:02d}{p % 4 * 15:02d}"
        Yp = np.stack([Y[(code, p)] for code in CODES])
        for i in np.flatnonzero(total_y[p] > capacity[p]):
            capped_off[mvs[i].key] += total_y[p][i] - capacity[p][i]
        X, _ = fit_interval(A, Yp, w, total_s / 60.0, limits, capacity[p])
        if background < 1.0:
            X[:, ~through_target] *= background
        for code_i, code in enumerate(CODES):
            x, y = X[code_i], Yp[code_i]
            rng = random.Random(route_seed(date, period) * 10 + code_i)
            n_route = np.floor(x).astype(int)
            frac = x - n_route
            for j in np.flatnonzero(frac > 1e-9):
                if rng.random() < frac[j]:
                    n_route[j] += 1
            start = p * PERIOD_S
            for j in np.flatnonzero(n_route):
                key = route_key(routes[j]) if depart_fix is not None and through_target[j] else None
                for _ in range(n_route[j]):
                    arrival = start + rng.uniform(0, PERIOD_S)
                    shift = 0.0
                    if key:
                        shift = depart_shift(depart_fix, key, int(arrival // 3600))
                        shifts.append(shift)
                    vehicles.append((max(0.0, arrival - lead_s[j] - shift), code, int(j), arrival))
            planned = A @ n_route
            ledger["planned"][f"{period}{code}"] = {mvs[i].key[0] + ">" + mvs[i].key[1]: int(v)
                                                    for i, v in enumerate(planned) if v}
            ledger["observed"][f"{period}{code}"] = {mvs[i].key[0] + ">" + mvs[i].key[1]: int(v)
                                                     for i, v in enumerate(y) if v}
        if p % 16 == 15:
            print(f"  {period} 끝 ({time.time() - t0:.0f}초)", flush=True)
    write_route_file(output / "routes.rou.xml", routes, vehicles)
    meta = {
        "date": date, "uturn_201293": uturn, "origins": origins, "roads": roads, "background": background,
        "network": network.name,
        "method": {"weight": WEIGHT, "low_trust_tls": sorted(LOW_TRUST_TLS), "over_factor": OVER_FACTOR, "detour": DETOUR, "min_routes_per_movement": MIN_ROUTES_PER_MOVEMENT,
                   "length_cost_per_min": LENGTH_COST_PER_MIN, "vehicle_cost": VEHICLE_COST,
                   "boundary_cap_per_lane_h": BOUNDARY_CAP_PER_LANE_H, "saturation_h": SATURATION_H,
                   "capacity_overrides": capacity_overrides(), "utilization": UTILIZATION,
                   "anchor": "석천사거리 정지선, 없으면 첫 신호 교차로 정지선",
                   "arrival_window_s": PERIOD_S},
        "movements": [{"key": list(m.key), "signature": list(m.signature), "tls": m.tls, "kind": m.kind} for m in mvs],
        "dropped": {f"{k[0]}>{k[1]}": {"why": why, "vehicles": {c: dropped_n[(k, c)] for c in CODES if dropped_n[(k, c)]}}
                    for k, why in dropped.items()},
        "capped_off": {f"{k[0]}>{k[1]}": round(v, 1) for k, v in sorted(capped_off.items(), key=lambda kv: -kv[1]) if v >= 1},
        "candidates": len(routes), "routes_used": len({v[2] for v in vehicles}), "vehicles": len(vehicles),
        "vehicles_by_type": dict(Counter(VTYPE_OF[v[1]] for v in vehicles)),
        "mean_route_min": float(np.mean([total_s[v[2]] for v in vehicles]) / 60) if vehicles else 0.0,
        "share_through_204820": float(np.mean([through_target[v[2]] for v in vehicles])) if vehicles else 0.0,
        "depart_fix": None if depart_fix is None else {
            **depart_fix["meta"], "shrink_vehicles": DEPART_FIX_SHRINK, "max_s": DEPART_FIX_MAX_S,
            "shifted_vehicles": int(sum(s > 0 for s in shifts)), "mean_shift_s": float(np.mean(shifts)) if shifts else 0.0},
        "seconds": round(time.time() - t0, 1),
    }
    ledger["meta"] = meta
    (output / "ledger.json").write_text(json.dumps(ledger, ensure_ascii=False))
    return ledger


def write_route_file(path: Path, routes: list[tuple[str, ...]], vehicles: list[tuple]) -> None:
    root = ET.Element("routes")
    attrs = driver_vtype_attributes()
    for vtype in ET.parse(fd.VTYPES).getroot().findall("vType"):
        vtype = copy.deepcopy(vtype)
        for k, v in attrs.items():
            vtype.set(k, v)
        root.append(vtype)
    used = sorted({v[2] for v in vehicles})
    for j in used:
        ET.SubElement(root, "route", id=f"r{j}", edges=" ".join(routes[j]))
    serial = Counter()
    for depart, code, j, arrival in sorted(vehicles):
        period = f"{int(arrival // 3600):02d}{int(arrival % 3600 // PERIOD_S) * 15:02d}"
        serial[(period, code)] += 1
        ET.SubElement(root, "vehicle", id=f"{period}{code}{serial[(period, code)]}", type=VTYPE_OF[code], route=f"r{j}",
                      depart=f"{depart:.2f}", departLane="best", departSpeed="max")
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def audit(ledger: dict) -> str:
    """계획된 통과(경로 기준) vs 기록: 교차로별 하루 대수, 오차율(15분×차종×이동, 15분×이동, 1시간×이동), 1시간 GEH."""
    meta = ledger["meta"]
    tls_of = {f"{m['key'][0]}>{m['key'][1]}": m["tls"] for m in meta["movements"]}
    grid = {}
    for name in ("planned", "observed"):
        g = defaultdict(lambda: np.zeros((96, len(CODES))))
        for pc, row in ledger[name].items():
            p = int(pc[:2]) * 4 + int(pc[2:4]) // 15
            c = CODES.index(pc[4])
            for mv, n in row.items():
                g[mv][p, c] += n
        grid[name] = g
    keys = sorted(set(grid["planned"]) | set(grid["observed"]))
    zero = np.zeros((96, len(CODES)))
    lines = [f"# 수요 장부 {meta['date']} (201293 유턴: {meta['uturn_201293']}, 배경 교통 {meta.get('background', 1.0):.0%})", "",
             f"- 후보 경로 {meta['candidates']:,}개 중 {meta['routes_used']:,}개 사용, 차량 {meta['vehicles']:,}대 "
             f"({', '.join(f'{k} {v:,}' for k, v in meta['vehicles_by_type'].items())}), 평균 경로 {meta['mean_route_min']:.1f}분(빈 도로), "
             f"석천사거리를 지나는 차 {meta['share_through_204820']:.0%}, 만드는 데 {meta['seconds']:.0f}초",
             "- 뺀 기록: " + ("; ".join(f"{k} {v['why']} {sum(v['vehicles'].values()):,}대" for k, v in meta["dropped"].items()) or "없음"),
             f"- 이웃 신호 처리 능력을 넘어 줄인 기록: 합 {sum(meta.get('capped_off', {}).values()):,.0f}대 — "
             + (", ".join(f"{k} {v:,.0f}" for k, v in list(meta.get("capped_off", {}).items())[:6]) or "없음"),
             "- 계획된 통과는 차량 경로가 지나는 이동을 그 15분에 센 값이다(실제로 지나는 시각과 신호 지연은 하루 실행으로 따로 본다).", "",
             "| 교차로 | 하루 기록 | 하루 계획 | 오차율: 15분×차종×이동 | 15분×이동 | 1시간×이동 | 1시간 모자람 | 1시간 넘침 | 1시간 GEH 5 미만 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    per_tls = defaultdict(list)
    for k in keys:
        per_tls[tls_of.get(k, "?")].append(k)
    for tls in sorted(per_tls, key=lambda t: (t != TARGET_TLS, t)):
        P = np.stack([grid["planned"].get(k, zero) for k in per_tls[tls]])
        O = np.stack([grid["observed"].get(k, zero) for k in per_tls[tls]])
        obs = O.sum()
        e1 = np.abs(P - O).sum() / max(obs, 1)
        e2 = np.abs(P.sum(2) - O.sum(2)).sum() / max(obs, 1)
        Ph, Oh = P.sum(2).reshape(len(P), 24, 4).sum(2), O.sum(2).reshape(len(O), 24, 4).sum(2)
        e3 = np.abs(Ph - Oh).sum() / max(obs, 1)
        busy = (Ph + Oh) > 0
        geh = np.sqrt(2 * (Ph - Oh) ** 2 / np.maximum(Ph + Oh, 1e-9))
        short, over = np.clip(Oh - Ph, 0, None).sum(), np.clip(Ph - Oh, 0, None).sum()
        lines.append(f"| {tls} | {obs:,.0f} | {P.sum():,.0f} | {e1:.1%} | {e2:.1%} | {e3:.1%} | {short:,.0f} | {over:,.0f} | "
                     f"{(geh[busy] < 5).mean():.1%} |")
    worst = []
    for k in keys:
        Ph = grid["planned"].get(k, zero).sum(1).reshape(24, 4).sum(1)
        Oh = grid["observed"].get(k, zero).sum(1).reshape(24, 4).sum(1)
        worst.append((float(np.abs(Ph - Oh).sum()), k, Oh.sum(), Ph.sum()))
    worst.sort(reverse=True)
    lines += ["", "## 1시간 오차 합이 가장 큰 이동 15개", "", "| 교차로 | 이동 | 하루 기록 | 하루 계획 | 1시간 오차 합 |", "|---|---|---:|---:|---:|"]
    for err, k, o, p in worst[:15]:
        lines.append(f"| {tls_of.get(k, '?')} | {k} | {o:,.0f} | {p:,.0f} | {err:,.0f} |")
    return "\n".join(lines) + "\n"


def demand_dir(date: str, uturn: str, origins: str = "chains", background: float = 1.0, roads: str = "corridor",
               depart_fix: bool = False) -> Path:
    return OUT / (f"{date}_{uturn}" + ("" if origins == "chains" else f"_{origins}") + ("" if roads == "corridor" else f"_{roads}")
                  + ("" if background >= 1.0 else f"_bg{round(background * 100)}") + ("_dep" if depart_fix else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--date", required=True)
    b.add_argument("--uturn", choices=("drop", "straight"), default="drop")
    b.add_argument("--background", type=float, default=1.0, help="석천사거리를 지나지 않는 차의 비율(0~1, 기본 1)")
    b.add_argument("--roads", choices=("corridor", "all"), default="corridor",
                   help="corridor: 큰길 축(구간·신호 사이 직진 길)만 다님(기본), all: 지도의 모든 도로")
    b.add_argument("--origins", choices=("chains", "boundary"), default="chains",
                   help="chains: 12곳 진입·진출 구간에서 출발·도착(기본), boundary: 지도 끝에서 지도 끝까지")
    b.add_argument("--allow-test-date", action="store_true", help="시험 날짜도 만든다(방법을 고르는 데 쓰면 안 됨)")
    args = ap.parse_args()
    test = set(json.loads(SPLIT.read_text())["test"])
    if args.date in test and not args.allow_test_date:
        raise SystemExit(f"{args.date}는 시험 날짜입니다. 수요 방법을 고르는 동안에는 만들지 않습니다.")
    output = demand_dir(args.date, args.uturn, args.origins, args.background, args.roads)
    ledger = build_day(args.date, args.uturn, output, origins=args.origins, background=args.background, roads=args.roads)
    report = audit(ledger)
    (output / "audit.md").write_text(report)
    print(report)
    print(f"저장: {output}")


if __name__ == "__main__":
    main()
