"""석천사거리 녹색 사이 전환 12개의 정리 시간 점검표(지도 계산만, SUMO를 돌리지 않음).

녹색 i가 끝나고 녹색 j가 켜질 때, i에서 출발해 교차로 안에 남은 차(빠져나가는 차)와 j에서 막 출발한 차
(들어오는 차)의 경로가 겹치는 곳이 있으면, 들어오는 차가 그 구역에 닿기 전에 빠져나가는 차가 다 지나가야 한다.
교통공학의 녹색 사이 시간(intergreen) 계산과 같은 틀이다.

    필요한 녹색 사이 시간 = 넘는 시각 + 빠져나가는 시간 − 들어오는 시간

- 넘는 시각(t_over): 황색이 켜진 뒤에도 정지선을 넘어가는 마지막 시각. SUMO에서 차는 황색에 멈출 수 있으면
  멈추고(감속 b), 못 멈추면 지나간다. 그래서 진입 도로 제한속도 v에서 v/(2b)초까지 넘어갈 수 있다.
  보수적 기준으로 '황색 끝까지 넘어갈 수 있다'(t_over = 황색 4초)도 같이 계산한다.
- 빠져나가는 시간: 정지선에서 겹치는 구역의 먼 끝까지 거리 + 차 길이를, 교차로 안 경로의 제한속도로 나눈 값.
  버스 12m·트럭 7.1m·승용차 5m·이륜차 2.2m를 모두 보고 가장 긴 값을 쓴다.
- 들어오는 시간: 정지선 1m 앞에 서 있던 차가 녹색에 출발해(가속 a, 제한속도까지) 겹치는 구역의 가까운 끝에
  닿는 시간. 가장 빨리 닿는 차종을 쓴다. 보수적 기준으로 '멈추지 않고 제한속도로 들어온다'도 같이 계산한다.
- 겹치는 구역: 두 경로의 중심선이 (두 차 폭의 합) ÷ 2보다 가까운 곳. 경로는 정지선(진입 도로 끝)에서 교차로 안
  차로(좌회전·유턴의 대기 위치 앞뒤 조각 포함)를 지나 나가는 도로 시작까지다.
- 지금 녹색 사이 시간은 황색 4초뿐이다(전적색 0초). 필요한 값이 4초보다 길면 그 차이만큼 전적색이 필요하다.

차종 값은 SUMO 차종 기본값이다. 이 계산은 근사다(차를 폭이 있는 선으로 보고, 교차로 안 속도를 제한속도로 둠).
결과는 SUMO 실행(충돌 재현)으로 따로 검증한다.

사용: .venv/bin/python src/focused_intergreen.py
결과: results/focused_204820/intergreen/intergreen_table.json, intergreen_table.md
"""
from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from focused_data import PROJECT, TARGET_TLS, find_network
from focused_scenario import AIHUB_YELLOW_S, signal_greens

OUT = PROJECT / "results/focused_204820/intergreen"
# SUMO 차종 기본값(길이·폭 m, 가속·감속 m/s²)
VEHICLES = {
    "passenger": {"length": 5.0, "width": 1.8, "accel": 2.6, "decel": 4.5},
    "bus": {"length": 12.0, "width": 2.5, "accel": 1.2, "decel": 4.0},
    "truck": {"length": 7.1, "width": 2.4, "accel": 1.3, "decel": 4.0},
    "motorcycle": {"length": 2.2, "width": 0.9, "accel": 6.0, "decel": 10.0},
}
STOP_OFFSET_M = 1.0     # 녹색을 기다리는 맨 앞 차는 정지선 약 1m 앞에 선다(09-28 재현 기록: 1.0m)
STEP_M = 0.2
GREEN_NAMES = {0: "석천로 좌회전", 1: "길주로 직진", 2: "길주로 좌회전", 3: "석천로 직진"}
REAL_ORDER = [3, 0, 1, 2]          # 실제 운영 순서(석천로 직진 → 석천로 좌회전 → 길주로 직진 → 길주로 좌회전)


def parse_shape(text: str) -> list[tuple[float, float]]:
    return [tuple(map(float, p.split(","))) for p in text.split()]


def polyline_length(points) -> float:
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


def resample(points, step: float = STEP_M) -> list[tuple[float, float, float]]:
    """(x, y, 시작점에서의 거리) 목록."""
    out, total = [(points[0][0], points[0][1], 0.0)], 0.0
    for a, b in zip(points, points[1:]):
        seg = math.dist(a, b)
        n = max(1, int(seg / step))
        for k in range(1, n + 1):
            f = k / n
            out.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, total + seg * f))
        total += seg
    return out


def point_segment_distance(p, a, b) -> float:
    ax, ay, bx, by = a[0], a[1], b[0], b[1]
    dx, dy = bx - ax, by - ay
    if dx == dy == 0:
        return math.dist(p, a)
    t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy)))
    return math.dist(p, (ax + t * dx, ay + t * dy))


def distance_to_polyline(p, points) -> float:
    return min(point_segment_distance(p, a, b) for a, b in zip(points, points[1:]))


def link_paths(network: Path) -> dict[int, dict]:
    """신호 연결 번호 → 교차로 안 경로(정지선에서 나가는 도로 시작까지)."""
    root = ET.parse(network).getroot()
    lanes = {lane.get("id"): lane for edge in root.iter("edge") for lane in edge.iter("lane")}
    internal = {}       # (내부 도로, 차로) → 다음 내부 차로(via)
    for c in root.iter("connection"):
        if c.get("from", "").startswith(":") and c.get("via"):
            internal[(c.get("from"), c.get("fromLane"))] = c.get("via")
    paths = {}
    for c in root.iter("connection"):
        if c.get("tl") != TARGET_TLS:
            continue
        chain, via = [], c.get("via")
        while via:
            chain.append(via)
            edge, lane = via.rsplit("_", 1)
            via = internal.get((edge, lane))
        shape, offsets = [], []
        for lane_id in chain:
            pts = parse_shape(lanes[lane_id].get("shape"))
            # SUMO의 차로 위치(pos)는 차로 길이 속성 기준이라, 모양 길이로 바꾸는 비율을 같이 둔다
            offsets.append((lane_id, polyline_length(shape), polyline_length(pts) / float(lanes[lane_id].get("length"))))
            shape += pts if not shape else pts[1:]
        from_lane, to_lane = f"{c.get('from')}_{c.get('fromLane')}", f"{c.get('to')}_{c.get('toLane')}"
        paths[int(c.get("linkIndex"))] = {
            "from": c.get("from"), "to": c.get("to"), "dir": c.get("dir"), "internal_lanes": chain, "shape": shape,
            "length": polyline_length(shape), "speed": min(float(lanes[l].get("speed")) for l in chain),
            "approach_speed": float(lanes[from_lane].get("speed")), "from_lane": from_lane, "to_lane": to_lane,
            "from_lane_length": float(lanes[from_lane].get("length")), "lane_offsets": offsets,
        }
    return paths


def conflict(path_e: dict, path_s: dict, half_gap: float) -> tuple[float, float, float, float] | None:
    """겹치는 구역의 (e 경로 가까운 끝, e 경로 먼 끝, s 경로 가까운 끝, s 경로 먼 끝)까지 거리(정지선부터). 안 겹치면 None."""
    on_e = [d for x, y, d in resample(path_e["shape"]) if distance_to_polyline((x, y), path_s["shape"]) < half_gap]
    on_s = [d for x, y, d in resample(path_s["shape"]) if distance_to_polyline((x, y), path_e["shape"]) < half_gap]
    if not on_e or not on_s:
        return None
    return min(on_e), max(on_e), min(on_s), max(on_s)


def standing_start_time(distance: float, accel: float, vmax: float) -> float:
    """서 있다가 가속 a로 출발해(최고 vmax) distance를 가는 시간."""
    d_acc = vmax * vmax / (2 * accel)
    if distance <= d_acc:
        return math.sqrt(2 * distance / accel)
    return vmax / accel + (distance - d_acc) / vmax


def pair_requirement(path_e: dict, path_s: dict) -> dict | None:
    """끝나는 연결 e와 시작하는 연결 s 한 쌍의 필요한 녹색 사이 시간(기준 두 가지). 안 겹치면 None."""
    rows = []
    for ce, ve in VEHICLES.items():          # 빠져나가는 차종
        for cs, vs in VEHICLES.items():      # 들어오는 차종
            hit = conflict(path_e, path_s, (ve["width"] + vs["width"]) / 2)
            if hit is None:
                continue
            _, s_clear, s_enter, _ = hit
            clear_s = (s_clear + ve["length"]) / path_e["speed"]
            over_sumo = path_e["approach_speed"] / (2 * ve["decel"])
            enter_stand = standing_start_time(s_enter + STOP_OFFSET_M, vs["accel"], path_s["speed"])
            enter_fly = s_enter / path_s["speed"]
            rows.append({"clearing_vehicle": ce, "entering_vehicle": cs, "clear_far_edge_m": round(s_clear, 1),
                         "enter_near_edge_m": round(s_enter, 1), "clear_time_s": round(clear_s, 2),
                         "overrun_sumo_s": round(over_sumo, 2), "enter_standing_s": round(enter_stand, 2),
                         "enter_flying_s": round(enter_fly, 2),
                         # 기준 1(SUMO 동작): 못 멈춘 차만 황색 초반에 넘어감 + 서 있던 차가 출발
                         "need_sumo_s": round(over_sumo + clear_s - enter_stand, 2),
                         # 기준 2(보수적): 황색 끝까지 넘어갈 수 있음 + 멈추지 않고 들어옴
                         "need_conservative_s": round(AIHUB_YELLOW_S + clear_s - enter_fly, 2)})
    if not rows:
        return None
    return {"sumo": max(rows, key=lambda r: r["need_sumo_s"]),
            "conservative": max(rows, key=lambda r: r["need_conservative_s"])}


def transitions(greens: list[str]) -> list[tuple[int, int]]:
    return [(i, j) for i in range(len(greens)) for j in range(len(greens)) if i != j]


def is_real_order(i: int, j: int) -> bool:
    return REAL_ORDER[(REAL_ORDER.index(i) + 1) % len(REAL_ORDER)] == j


def build_table() -> dict:
    greens = signal_greens()
    paths = link_paths(find_network())
    rows = []
    for i, j in transitions(greens):
        ending = [k for k in paths if greens[i][k] in "Gg" and greens[j][k] == "r"]
        starting = [k for k in paths if greens[j][k] in "Gg" and greens[i][k] == "r"]
        pairs = []
        for e in ending:
            for s in starting:
                req = pair_requirement(paths[e], paths[s])
                if req:
                    pairs.append({"ending_link": e, "starting_link": s, **req})
        worst_sumo = max(pairs, key=lambda p: p["sumo"]["need_sumo_s"], default=None)
        worst_cons = max(pairs, key=lambda p: p["conservative"]["need_conservative_s"], default=None)
        need_sumo = worst_sumo["sumo"]["need_sumo_s"] if worst_sumo else 0.0
        need_cons = worst_cons["conservative"]["need_conservative_s"] if worst_cons else 0.0
        rows.append({"from_green": i, "to_green": j, "from_name": GREEN_NAMES[i], "to_name": GREEN_NAMES[j],
                     "real_order": is_real_order(i, j), "ending_links": ending, "starting_links": starting,
                     "conflicting_pairs": len(pairs),
                     "need_sumo_s": need_sumo, "all_red_sumo_s": max(0, math.ceil(need_sumo - AIHUB_YELLOW_S)),
                     "worst_pair_sumo": worst_sumo,
                     "need_conservative_s": need_cons,
                     "all_red_conservative_s": max(0, math.ceil(need_cons - AIHUB_YELLOW_S)),
                     "worst_pair_conservative": worst_cons, "pairs": pairs})
    return {"tls": TARGET_TLS, "greens": greens, "yellow_s": AIHUB_YELLOW_S, "vehicles": VEHICLES,
            "links": {k: {kk: v for kk, v in p.items() if kk != "shape"} | {"length": round(p["length"], 1)}
                      for k, p in sorted(paths.items())},
            "transitions": rows}


def write_markdown(table: dict, path: Path) -> None:
    lines = ["# 석천사거리 녹색 사이 전환 12개 점검표", "",
             f"지금 녹색 사이 시간: 황색 {table['yellow_s']}초, 전적색 0초. "
             "'필요'가 이보다 길면 그만큼 전적색이 필요하다(올림).", "",
             "| 전환 | 실제 순서 | 겹치는 연결 쌍 | 필요(SUMO 동작) | 전적색 | 필요(보수적) | 전적색 | 가장 빠듯한 쌍(SUMO 동작) |",
             "|---|:---:|---:|---:|---:|---:|---:|---|"]
    for r in table["transitions"]:
        w = r["worst_pair_sumo"]
        detail = "—" if not w else (f"연결 {w['ending_link']}→{w['starting_link']}: {w['sumo']['clearing_vehicle']} 빠져나감 "
                                    f"{w['sumo']['clear_time_s']}초, {w['sumo']['entering_vehicle']} 들어옴 {w['sumo']['enter_standing_s']}초")
        lines.append(f"| {r['from_green']} {r['from_name']} → {r['to_green']} {r['to_name']} | {'○' if r['real_order'] else ''} | "
                     f"{r['conflicting_pairs']} | {r['need_sumo_s']:.1f}초 | {r['all_red_sumo_s']}초 | "
                     f"{r['need_conservative_s']:.1f}초 | {r['all_red_conservative_s']}초 | {detail} |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table = build_table()
    (OUT / "intergreen_table.json").write_text(json.dumps(table, ensure_ascii=False, indent=1))
    write_markdown(table, OUT / "intergreen_table.md")
    print((OUT / "intergreen_table.md").read_text())


if __name__ == "__main__":
    main()
