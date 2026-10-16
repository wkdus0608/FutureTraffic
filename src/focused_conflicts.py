"""녹색 전환마다 '통과 후 간격'(PET)을 SUMO 실행에서 직접 잰다(전적색 길이를 정하는 근거).

녹색 i가 끝나고 녹색 j가 켜질 때, 서로 겹치는 경로 쌍(focused_intergreen의 겹치는 구역)마다
    PET = (j에서 출발한 첫 차의 앞끝이 겹치는 구역에 닿은 시각) − (i에서 출발한 마지막 차의 뒤끝이 그 구역을 빠져나간 시각)
을 잰다. 0보다 작으면 두 차가 같은 때 같은 자리에 있었다는 뜻(충돌 또는 아주 가까움)이다. 교차로 안전 분석에서
흔히 1초 미만을 위험한 상충으로 본다. 전환 종류(12개)별로 모아 가장 작은 PET를 본다.

record  평가(focused_eval.run_day)를 그대로 다시 돌리며, 1초마다 석천사거리 교차로 안·정지선 앞 25m·나가는 도로 30m에
        있는 차의 차로·위치·속도를 기록한다. 값을 묻기만 하므로 차의 움직임은 평가와 같다(결과 metrics.json으로 확인).
        결과를 낸 곳과 같은 Colab에서 돌린다. 날짜는 학습·검증 날짜만 쓴다(시험 날짜로 규칙을 정하지 않음).
analyze 기록에서 전환을 찾아 PET를 계산하고, 전환 종류별 요약과 필요한 전적색(PET 기준)을 낸다. 맥에서도 돌린다.

잠근 환경 파일·평가 코드를 고치지 않는다(두 지문 그대로).

measure 위 둘을 한 번에(Colab). 기록은 임시 폴더에 두고 실행마다 바로 분석해, 분석 결과(events.json.gz)와 평가 결과만
        Drive에 남긴다. 분석 결과가 있는 실행은 건너뛴다(끊기면 다시 실행). 마지막에 전체 요약표를 쓴다.

사용(Colab):
  python src/focused_conflicts.py measure --dates <학습·검증 15일> --policies fixed actuated greedy dqn \
      --models "$DRIVE/models_v5/seed_42" "$DRIVE/models_v5/seed_44" --arrivals random platoon --out "$DRIVE/conflict_check"
사용(맥): .venv/bin/python src/focused_conflicts.py analyze --root <Drive>/conflict_check
시험(맥, 짧게): ... record --dates 20220928 --policies fixed --out <임시 폴더> --workers 1 --until 3600
"""
from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import json
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from focused_data import TARGET_TLS, find_network
from focused_intergreen import (AIHUB_YELLOW_S, GREEN_NAMES, VEHICLES, conflict, is_real_order, link_paths,
                                signal_greens)

UPSTREAM_M, DOWNSTREAM_M = 25.0, 30.0
PET_THRESHOLDS = (1.0, 1.5)


class StopEarly(Exception):
    """--until 시각에 기록을 멈춘다(맥에서 짧게 시험할 때)."""


# ------------------------------------------------------------------ 기록(Colab)
def watched_lanes(paths: dict) -> dict[str, tuple[str, float]]:
    """기록할 차로 → (종류, 차로 길이). in = 정지선 앞 진입 차로, internal = 교차로 안, out = 나가는 차로."""
    out = {}
    for p in paths.values():
        out[p["from_lane"]] = ("in", p["from_lane_length"])
        out[p["to_lane"]] = ("out", 0.0)
        for lane_id, _, _ in p["lane_offsets"]:
            out[lane_id] = ("internal", 0.0)
    return out


def watch_radius(paths: dict) -> tuple[tuple[float, float], float]:
    """교차로 중심과, 기록할 곳(교차로 안·정지선 앞 25m·나가는 도로 30m)을 모두 덮는 반지름(+5m 여유)."""
    import xml.etree.ElementTree as ET
    from focused_intergreen import parse_shape, resample
    root = ET.parse(find_network()).getroot()
    j = root.find(f"junction[@id='{TARGET_TLS}']")
    center = (float(j.get("x")), float(j.get("y")))
    lanes = {lane.get("id"): lane for edge in root.iter("edge") for lane in edge.iter("lane")}
    far = 0.0
    for lane_id, (kind, length) in watched_lanes(paths).items():
        pts = resample(parse_shape(lanes[lane_id].get("shape")), 1.0)
        scale = float(lanes[lane_id].get("length")) / max(pts[-1][2], 1e-9)
        for x, y, d in pts:
            pos = d * scale
            if (kind == "in" and pos < length - UPSTREAM_M - 1) or (kind == "out" and pos > DOWNSTREAM_M + 1):
                continue
            far = max(far, math.dist(center, (x, y)))
    return center, far + 5.0


def record_run(job: dict, until: int | None = None) -> dict:
    import traci.constants as tc

    import focused_eval
    import focused_sumo

    paths = link_paths(find_network())
    watch = watched_lanes(paths)
    _, radius = watch_radius(paths)
    run_dir = Path(job["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    traj = gzip.open(run_dir / "trajectory.csv.gz", "wt", newline="")
    writer = csv.writer(traj)
    writer.writerow(["time", "vehicle", "lane", "pos", "speed"])
    vehicles: dict[str, list] = {}
    signal: list[tuple[int, str]] = []
    original = focused_sumo.make_environment

    def recording_environment(**kwargs):
        env = original(**kwargs)
        step = env._sumo_step
        state = {"subscribed": False}

        def recorded_step():
            step()
            sumo = env.sumo
            if not state["subscribed"]:
                # 교차로 중심 반지름 안의 차를 차로·위치·속도와 함께 1초에 한 번에 받는다(차마다 따로 묻지 않음)
                sumo.junction.subscribeContext(TARGET_TLS, tc.CMD_GET_VEHICLE_VARIABLE, radius,
                                               [tc.VAR_LANE_ID, tc.VAR_LANEPOSITION, tc.VAR_SPEED])
                state["subscribed"] = True
                return
            t = int(env.sim_step)
            signal.append((t, sumo.trafficlight.getRedYellowGreenState(TARGET_TLS)))
            for v, var in sorted((sumo.junction.getContextSubscriptionResults(TARGET_TLS) or {}).items()):
                lane = var[tc.VAR_LANE_ID]
                if lane not in watch:
                    continue
                kind, length = watch[lane]
                pos = var[tc.VAR_LANEPOSITION]
                if (kind == "in" and pos < length - UPSTREAM_M) or (kind == "out" and pos > DOWNSTREAM_M):
                    continue
                if v not in vehicles:
                    vehicles[v] = [sumo.vehicle.getTypeID(v), sumo.vehicle.getLength(v), sumo.vehicle.getWidth(v)]
                writer.writerow([t, v, lane, round(pos, 2), round(var[tc.VAR_SPEED], 2)])
            if until is not None and t >= until:
                raise StopEarly

        env._sumo_step = recorded_step
        return env

    focused_sumo.make_environment = recording_environment     # run_day는 실행할 때 이 이름을 가져다 쓴다
    result = None
    try:
        result = focused_eval.run_day(job)
    except StopEarly:
        pass
    finally:
        focused_sumo.make_environment = original
        traj.close()
    with gzip.open(run_dir / "signal.csv.gz", "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "tls_state"])
        w.writerows(signal)
    (run_dir / "vehicles.json").write_text(json.dumps(vehicles))     # 마지막에 써서 '끝난 기록'의 표시로 쓴다
    return {"run": run_dir.name, "date": job["date"], "until": until,
            "mean_delay_s": result["all"]["mean_delay_s"] if result else None,
            "collisions_unexplained": len(result["collisions_unexplained"]) if result else None}


def record(args) -> None:
    from focused_eval import CACHE, dqn_template_network, evaluation_fingerprint, planned_runs
    from focused_scenario import day_route_file

    fp = evaluation_fingerprint()
    out = Path(args.out) / fp
    jobs = []
    for arrivals in args.arrivals:
        jobs += [{**j, "fingerprint": fp} for j in planned_runs(out, args.dates, args.policies, args.models, 42, arrivals)]
    for date in args.dates:            # 차량 파일은 먼저 순서대로 만든다(동시 생성 충돌 방지)
        day_route_file(date, CACHE / "routes")
        if "platoon" in args.arrivals:
            from focused_platoon import platoon_route_file
            platoon_route_file(date, CACHE / "routes")
    dqn_template_network(CACHE)
    todo = [j for j in jobs if not (Path(j["run_dir"]) / "vehicles.json").exists()]
    print(f"평가 지문 {fp}: 실행 {len(jobs)}개 중 새로 {len(todo)}개 → {out}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(record_run, j, args.until) for j in todo]):
            r = f.result()
            if r["until"] is None:
                print(f"  {r['date']} {r['run']}: 차량당 지연 {r['mean_delay_s']:.2f}초, "
                      f"확인 안 된 충돌 {r['collisions_unexplained']}건", flush=True)
            else:
                print(f"  {r['date']} {r['run']}: {r['until']}초에서 멈춤(시험용)", flush=True)


# ------------------------------------------------------------------ 분석(맥)
def vehicle_class(type_id: str) -> str:
    return next((c for c in VEHICLES if type_id.startswith(c)), "passenger")


class Geometry:
    def __init__(self):
        self.greens = signal_greens()
        self.paths = link_paths(find_network())
        self.lane_link = {lane: (k, off, scale) for k, p in self.paths.items() for lane, off, scale in p["lane_offsets"]}
        self._zones = {}

    def zone(self, e: int, s: int, ce: str, cs: str):
        """겹치는 구역(e 가까운 끝, e 먼 끝, s 가까운 끝, s 먼 끝).

        같은 나가는 차로로 합류하는 쌍은 두 경로가 만나는 점(교차로 안 경로의 끝) 하나로 본다. 합류한 뒤에는 앞뒤로
        줄을 서므로, 구역으로 재면 안전하게 따라가는 차도 겹친 것처럼 나온다. 만나는 점에서 앞차 뒤끝이 지나간 뒤
        뒤차 앞끝이 닿기까지의 시간이 PET의 원래 정의다.
        """
        key = (e, s, ce, cs)
        if key not in self._zones:
            pe, ps = self.paths[e], self.paths[s]
            if pe["to_lane"] == ps["to_lane"]:
                self._zones[key] = (pe["length"], pe["length"], ps["length"], ps["length"])
            else:
                self._zones[key] = conflict(pe, ps, (VEHICLES[ce]["width"] + VEHICLES[cs]["width"]) / 2)
        return self._zones[key]

    def changing_links(self, i: int, j: int) -> tuple[list[int], list[int]]:
        g = self.greens
        ending = [k for k in self.paths if g[i][k] in "Gg" and g[j][k] == "r"]
        starting = [k for k in self.paths if g[j][k] in "Gg" and g[i][k] == "r"]
        return ending, starting


class Track:
    """차 한 대의 경로 위 앞끝 위치(정지선 = 0, 교차로 안 경로를 따라 + 방향)."""

    def __init__(self, link: int, cls: str, length: float):
        self.link, self.cls, self.length = link, cls, length
        self.t: list[float] = []
        self.d: list[float] = []

    def cross(self, threshold: float) -> float | None:
        """앞끝이 threshold를 처음 넘은 시각(1초 사이는 직선으로 이음)."""
        for k, d in enumerate(self.d):
            if d >= threshold:
                if k == 0 or self.t[k] - self.t[k - 1] > 1.5:
                    return self.t[k]
                d0 = self.d[k - 1]
                return self.t[k - 1] + (threshold - d0) / (d - d0) * (self.t[k] - self.t[k - 1])
        return None

    def speed_at(self, threshold: float) -> float | None:
        for k, d in enumerate(self.d):
            if d >= threshold and k > 0:
                return (d - self.d[k - 1]) / max(1e-9, self.t[k] - self.t[k - 1])
        return None


def load_tracks(run_dir: Path, geo: Geometry, cache_dir: Path | None = None) -> list[Track]:
    import pickle
    cache = cache_dir / f"{run_dir.parent.name}_{run_dir.name}.pkl.gz" if cache_dir else None
    if cache and cache.exists():
        with gzip.open(cache, "rb") as f:
            return pickle.load(f)
    tracks = _read_tracks(run_dir, geo)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(cache, "wb") as f:
            pickle.dump(tracks, f)
    return tracks


def _read_tracks(run_dir: Path, geo: Geometry) -> list[Track]:
    info = json.loads((run_dir / "vehicles.json").read_text())
    rows = defaultdict(list)
    with gzip.open(run_dir / "trajectory.csv.gz", "rt") as f:
        for r in csv.DictReader(f):
            rows[r["vehicle"]].append((int(r["time"]), r["lane"], float(r["pos"])))
    tracks = []
    for v, rs in rows.items():
        link = next((geo.lane_link[lane][0] for _, lane, _ in rs if lane in geo.lane_link), None)
        if link is None:
            continue
        p = geo.paths[link]
        tr = Track(link, vehicle_class(info[v][0]), info[v][1])
        for t, lane, pos in sorted(rs):
            if lane in geo.lane_link:
                _, off, scale = geo.lane_link[lane]
                d = off + pos * scale
            elif lane == p["from_lane"]:
                d = pos - p["from_lane_length"]
            elif lane == p["to_lane"]:
                d = p["length"] + pos
            else:
                continue
            tr.t.append(t)
            tr.d.append(d)
        if tr.t:
            tracks.append(tr)
    return tracks


def find_transitions(run_dir: Path, geo: Geometry) -> list[dict]:
    """매초 신호 기록에서 녹색 i → (황색) → 녹색 j 전환을 찾는다."""
    index = {g: k for k, g in enumerate(geo.greens)}
    events, prev_green, yellow_start, last = [], None, None, None
    with gzip.open(run_dir / "signal.csv.gz", "rt") as f:
        for r in csv.DictReader(f):
            t, s = int(r["time"]), r["tls_state"]
            if s == last:
                continue
            last = s
            if s in index:
                g = index[s]
                if prev_green is not None and g != prev_green:
                    events.append({"from": prev_green, "to": g, "yellow_start": yellow_start, "green_start": t})
                prev_green, yellow_start = g, None
            elif "y" in s and yellow_start is None:
                yellow_start = t
    return events


def occupancy(tr: Track, near: float, far: float) -> tuple[float | None, float | None]:
    """겹치는 구역에 앞끝이 들어간 시각과 뒤끝이 빠져나간 시각. 기록이 끝날 때까지 못 빠져나가면 마지막 기록 + 1초."""
    t_in = tr.cross(near)
    if t_in is None:
        return None, None
    t_out = tr.cross(far + tr.length)
    return t_in, (t_out if t_out is not None else tr.t[-1] + 1.0)


def event_pet(ev: dict, geo: Geometry, by_link: dict[int, list[Track]], line_times: dict[int, list[float]]) -> dict:
    """전환 하나에서 겹치는 쌍 가운데 가장 작은 PET. 어느 차가 먼저 지나갔는지에 따라 따로 낸다.

    끝나는 녹색 차가 먼저(clearing_first): 녹색 사이 시간이 모자라면 생기는 상충이다. 전적색 규칙은 이것으로 정한다.
    시작하는 녹색 차가 먼저(entering_first): 끝나는 녹색 차가 교차로 안에서 기다리다(유턴 대기, 앞이 막힘 등)
    나중에 지나간 경우다. 전적색으로 풀리지 않으므로 따로 센다.
    """
    ending, starting = geo.changing_links(ev["from"], ev["to"])
    T = ev["green_start"]
    lo = (ev["yellow_start"] or T) - 30
    best_by_order = {True: None, False: None}       # True: 끝나는 녹색 차가 먼저 지나감, False: 시작하는 녹색 차가 먼저
    for e in ending:
        ts = line_times[e]
        clearing = [by_link[e][k] for k in range(bisect.bisect_left(ts, lo), bisect.bisect_right(ts, T))]
        if not clearing:
            continue
        for s in starting:
            ts2 = line_times[s]
            entering = [by_link[s][k] for k in range(bisect.bisect_left(ts2, T - 1), bisect.bisect_right(ts2, T + 30))]
            for c in clearing:
                for n in entering:
                    z = geo.zone(e, s, c.cls, n.cls)
                    if z is None:
                        continue
                    near_e, far_e, near_s, far_s = z
                    c_in, c_out = occupancy(c, near_e, far_e)
                    n_in, n_out = occupancy(n, near_s, far_s)
                    if c_in is None or n_in is None:          # 한 차가 구역에 닿지 않음(기록 안에서)
                        continue
                    clearing_first = c_in <= n_in
                    # 먼저 들어간 차가 다 빠져나간 뒤 다음 차가 들어오기까지의 간격. 0보다 작으면 같은 때 구역에 있었다
                    pet = (n_in - c_out) if clearing_first else (c_in - n_out)
                    best = best_by_order[clearing_first]
                    if best is None or pet < best["pet"]:
                        best_by_order[clearing_first] = {"pet": round(pet, 2), "clearing_first": clearing_first, "ending_link": e,
                                "starting_link": s, "clearing_class": c.cls, "entering_class": n.cls,
                                "clear_line_time": round(c.cross(0.0), 1),
                                "clear_speed_at_line": round(c.speed_at(0.0) or float("nan"), 2),
                                "clearing_in_out": [round(c_in, 1), round(c_out, 1)],
                                "entering_in_out": [round(n_in, 1), round(n_out, 1)]}
    return {"clearing_first": best_by_order[True], "entering_first": best_by_order[False]}


def analyze_run(run_dir: Path, geo: Geometry, cache_dir: Path | None = None) -> list[dict]:
    tracks = load_tracks(run_dir, geo, cache_dir)
    by_link = defaultdict(list)
    for tr in tracks:
        if tr.cross(0.0) is not None:
            by_link[tr.link].append(tr)
    line_times = {}
    for k in geo.paths:
        by_link[k].sort(key=lambda tr: tr.cross(0.0))
        line_times[k] = [tr.cross(0.0) for tr in by_link[k]]
    out = []
    for ev in find_transitions(run_dir, geo):
        best = event_pet(ev, geo, by_link, line_times)
        out.append({**ev, "run": run_dir.parent.name + "/" + run_dir.name, "pet": best})
    return out


def summarize(events: list[dict]) -> list[dict]:
    rows = []
    geo_order = [(i, j) for i in range(4) for j in range(4) if i != j]
    for i, j in geo_order:
        evs = [e for e in events if e["from"] == i and e["to"] == j]
        main = [e for e in evs if e["pet"]["clearing_first"]]
        pets = sorted(e["pet"]["clearing_first"]["pet"] for e in main)
        worst = min(main, key=lambda e: e["pet"]["clearing_first"]["pet"], default=None)
        rev = sorted(e["pet"]["entering_first"]["pet"] for e in evs if e["pet"]["entering_first"])
        worst_rev = min((e for e in evs if e["pet"]["entering_first"]), key=lambda e: e["pet"]["entering_first"]["pet"],
                        default=None)
        row = {"from_green": i, "to_green": j, "from_name": GREEN_NAMES[i], "to_name": GREEN_NAMES[j],
               "real_order": is_real_order(i, j), "transitions": len(evs), "with_conflict": len(pets),
               "pet_min": pets[0] if pets else None,
               "pet_p1": pets[max(0, math.ceil(0.01 * len(pets)) - 1)] if pets else None,
               "pet_lt_0": sum(p < 0 for p in pets), "pet_lt_1": sum(p < 1.0 for p in pets),
               "worst": worst, "entering_first_count": len(rev), "entering_first_lt_0": sum(p < 0 for p in rev),
               "entering_first_min": rev[0] if rev else None, "worst_entering_first": worst_rev}
        for th in PET_THRESHOLDS:
            row[f"all_red_for_pet_{th}"] = max(0, math.ceil(th - pets[0])) if pets else 0
        rows.append(row)
    return rows


POLICY_GROUPS = ("fixed", "fixedc", "actuated", "greedy", "dqn")


def policy_of(run_label: str) -> str:
    return run_label.split("/")[1].split("_")[0]


def write_summary(events: list[dict], out_dir: Path, runs: list[str]) -> list[dict]:
    """전환 종류별 요약표(전체, 정책별)를 md·json으로 쓴다."""
    table = summarize(events)
    days = sorted({r.split("/")[0] for r in runs})
    (out_dir / "pet_summary.json").write_text(json.dumps(
        {"runs": runs, "days": days, "yellow_s": AIHUB_YELLOW_S, "summary": table}, ensure_ascii=False, indent=1))
    fmt = lambda x: "—" if x is None else f"{x:.1f}"
    lines = [f"날짜 {len(days)}일({', '.join(days)}), 실행 {len(runs)}개. 끝나는 녹색의 차가 먼저 지나간 경우(전적색으로 풀 수 있는 "
             "상충). PET = 다음 차 앞끝이 구역(합류는 만나는 점)에 닿은 시각 − 앞차 뒤끝이 빠져나간 시각.", "",
             "| 전환 | 실제 순서 | 전환 수 | 겹침 있음 | 가장 작은 PET | 1% 값 | PET<0 | PET<1초 | 전적색(PET≥1초) | 전적색(PET≥1.5초) | 가장 빠듯한 경우 | 반대 순서 PET<0 |",
             "|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
    for r in table:
        w = r["worst"]
        q = w["pet"]["clearing_first"] if w else None
        detail = "—" if not w else (f"{w['run']} {w['green_start']}초: 연결 {q['ending_link']}→{q['starting_link']} "
                                    f"{q['clearing_class']}(정지선 {q['clear_speed_at_line']}m/s)→{q['entering_class']}")
        lines.append(f"| {r['from_green']} {r['from_name']} → {r['to_green']} {r['to_name']} | {'○' if r['real_order'] else ''} | "
                     f"{r['transitions']} | {r['with_conflict']} | {fmt(r['pet_min'])} | {fmt(r['pet_p1'])} | {r['pet_lt_0']} | "
                     f"{r['pet_lt_1']} | {r['all_red_for_pet_1.0']}초 | {r['all_red_for_pet_1.5']}초 | {detail} | "
                     f"{r['entering_first_lt_0']}/{r['entering_first_count']} |")
    lines += ["", "정책별: PET<0 / PET<1초 / 전환 수", "",
              "| 전환 | 실제 순서 | " + " | ".join(POLICY_GROUPS) + " |", "|---|:---:|" + "---:|" * len(POLICY_GROUPS)]
    for i, j in [(i, j) for i in range(4) for j in range(4) if i != j]:
        cells = []
        for pol in POLICY_GROUPS:
            evs = [e for e in events if e["from"] == i and e["to"] == j and policy_of(e["run"]) == pol]
            pets = [e["pet"]["clearing_first"]["pet"] for e in evs if e["pet"]["clearing_first"]]
            cells.append(f"{sum(x < 0 for x in pets)} / {sum(x < 1.0 for x in pets)} / {len(evs)}")
        lines.append(f"| {i} {GREEN_NAMES[i]} → {j} {GREEN_NAMES[j]} | {'○' if is_real_order(i, j) else ''} | " + " | ".join(cells) + " |")
    (out_dir / "pet_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return table


def save_events(events: list[dict], path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as f:
        json.dump(events, f, ensure_ascii=False)
    tmp.replace(path)                     # 끊겨도 반쯤 쓴 파일이 '끝난 분석'으로 보이지 않게


def analyze_existing(run_dir: str) -> str:
    """기록(trajectory)이 이미 있는 실행을 분석해 events.json.gz를 같은 폴더에 남긴다."""
    run_dir = Path(run_dir)
    save_events(analyze_run(run_dir, Geometry()), run_dir / "events.json.gz")
    return f"{run_dir.parent.name}/{run_dir.name}"


def measure_job(job: dict, drive_run_dir: str, until: int | None = None) -> dict:
    """임시 폴더에 기록 → 바로 분석 → 분석 결과와 평가 결과만 Drive에 남기고 기록은 지운다."""
    import shutil
    local = Path(job["run_dir"])
    r = record_run(job, until)
    events = analyze_run(local, Geometry())
    out = Path(drive_run_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("metrics.json", "collisions.xml", "statistics.xml"):
        if (local / name).exists():
            shutil.copy(local / name, out / name)
    save_events(events, out / "events.json.gz")
    shutil.rmtree(local, ignore_errors=True)
    return r


def auto_workers() -> int:
    import os
    n = os.cpu_count() or 1
    try:
        import psutil                      # 실행 하나에 메모리 약 1.5GB를 잡는다
        n = min(n, max(1, int(psutil.virtual_memory().total / 1.5e9)))
    except ImportError:
        pass
    return max(1, n)


def measure(args) -> None:
    """학습·검증 날짜 전체를 한 번에: 기록이 남은 예전 실행은 분석만, 없는 실행은 기록→분석, 마지막에 요약."""
    from focused_eval import CACHE, dqn_template_network, evaluation_fingerprint, planned_runs
    from focused_scenario import day_route_file

    fp = evaluation_fingerprint()
    drive = Path(args.out) / fp
    tmp = Path(args.tmp) / fp
    workers = args.workers or auto_workers()
    old = [p.parent for p in drive.glob("*/*/vehicles.json") if not (p.parent / "events.json.gz").exists()]
    jobs = []
    for arrivals in args.arrivals:
        for j in planned_runs(tmp, args.dates, args.policies, args.models, 42, arrivals):
            target = drive / Path(j["run_dir"]).relative_to(tmp)
            if not (target / "events.json.gz").exists() and not (target / "vehicles.json").exists():   # 예전 기록은 분석만
                jobs.append(({**j, "fingerprint": fp}, str(target)))
    print(f"평가 지문 {fp}: 예전 기록 분석 {len(old)}개, 새 실행 {len(jobs)}개, 동시에 {workers}개 → {drive}", flush=True)
    for date in sorted({j["date"] for j, _ in jobs}):      # 차량 파일은 먼저 순서대로 만든다(동시 생성 충돌 방지)
        day_route_file(date, CACHE / "routes")
        if "platoon" in args.arrivals:
            from focused_platoon import platoon_route_file
            platoon_route_file(date, CACHE / "routes")
    dqn_template_network(CACHE)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(analyze_existing, str(r)) for r in old]
        futures += [pool.submit(measure_job, j, target, args.until) for j, target in jobs]
        for n, f in enumerate(as_completed(futures), 1):
            r = f.result()
            label = r if isinstance(r, str) else f"{r['date']}/{r['run']} 차량당 지연 {r['mean_delay_s'] or 0:.2f}초"
            print(f"  {n}/{len(futures)} {label}", flush=True)
    summary_all(drive)


def summary_all(drive: Path) -> None:
    events, runs = [], []
    for path in sorted(drive.glob("*/*/events.json.gz")):
        with gzip.open(path, "rt") as f:
            events += json.load(f)
        runs.append(f"{path.parent.parent.name}/{path.parent.name}")
    write_summary(events, drive, runs)


def analyze(args) -> None:
    """맥에서: 기록이 있는 실행을 모두 분석해 요약한다(기록을 읽은 결과는 tracks_cache에 저장)."""
    geo = Geometry()
    root = Path(args.root)
    out_dir = Path(args.out or args.root)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = sorted(p.parent for p in root.glob("*/*/*/vehicles.json"))
    events = []
    for r in runs:
        evs = analyze_run(r, geo, out_dir / "tracks_cache")
        events += evs
        print(f"{r.parent.name}/{r.name}: 전환 {len(evs)}번", flush=True)
    save_events(events, out_dir / "pet_events.json.gz")
    write_summary(events, out_dir, [f"{r.parent.name}/{r.name}" for r in runs])


def check_dates(dates: list[str]) -> None:
    split = json.loads((Path(__file__).resolve().parents[1] / "config/focused_split.json").read_text())
    if set(dates) & set(split["test"]):
        raise SystemExit("시험 날짜로 전적색 규칙을 정하지 않습니다. 학습·검증 날짜만 쓰세요.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("record", "measure"):
        r = sub.add_parser(name)
        r.add_argument("--dates", nargs="+", required=True)
        r.add_argument("--policies", nargs="+", default=["fixed", "actuated", "greedy", "dqn"],
                       choices=["fixed", "fixedc", "actuated", "greedy", "dqn"])
        r.add_argument("--models", nargs="*", default=[])
        r.add_argument("--arrivals", nargs="+", default=["random"])
        r.add_argument("--out", required=True)
        r.add_argument("--workers", type=int, default=0 if name == "measure" else 4, help="0이면 CPU·메모리에 맞춰 자동")
        r.add_argument("--until", type=int, default=None, help="이 시각(초)에 멈춘다(맥에서 짧게 시험할 때만)")
        if name == "measure":
            r.add_argument("--tmp", default="/content/conflict_tmp", help="기록을 잠시 둘 곳(분석 뒤 지움)")
    a = sub.add_parser("analyze")
    a.add_argument("--root", required=True, help="record의 --out 폴더")
    a.add_argument("--out", default=None, help="요약을 쓸 폴더(없으면 --root)")
    args = p.parse_args()
    if args.cmd in ("record", "measure"):
        check_dates(args.dates)
        record(args) if args.cmd == "record" else measure(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
