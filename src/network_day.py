"""12개 지도 하루 점검(고정신호) — 실제 신호 12곳 + 하루 수요로 하루를 끊지 않고 돌린다. 지문 밖.

입력: 12개 지도(network_map), 하루 수요(network_demand), 12곳 실제 신호(network_signals, 석천사거리는 단독 모형과 같은 신호).
SUMO 설정은 단독 모형 하루 점검(focused_day)과 같다: 순간이동 끔, 교차로 안 충돌까지 검사(경고만, 차는 그대로), seed 42.

확인하는 것:
1. 교착: 달리는 차와 지도에 못 들어간 차가 시간에 따라 계속 늘어나는지, 하루가 끝날 때 남은 차
2. 충돌: 곳·때
3. 통과량: 실제로 지나간 대수(차가 진입 도로를 떠난 시각 기준) vs 기록, 15분·1시간, 교차로별(특히 석천사거리)
4. 막힌 도로: 15분 동안 점유 60% 넘고 속도 1m/s 미만

사용: .venv/bin/python src/network_day.py run --date 20220811 [--uturn drop|straight]
      .venv/bin/python src/network_day.py report --date 20220811 [--uturn ...]   (이미 돌린 결과만 다시 정리)
결과: results/network_day/<날짜>_<유턴 판>/report.md, result.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import focused_data as fd
import network_demand as nd
import network_signals as ns
from focused_sumo import COLLISION_OPTIONS, configure_sumo
from network_map import network_12

OUT = fd.PROJECT / "results/network_day"
DAY_S, TAIL_S = 86400, 1800
CODES = nd.CODES


GIVE_UP_WAITING = 15000                # 못 들어간 차가 이만큼 넘으면 교착으로 보고 멈춘다(무너진 판은 아주 느려짐)


def run_dir(date: str, uturn: str, teleport: int, background: float = 1.0, depart_fix: bool = False) -> Path:
    return OUT / (f"{date}_{uturn}" + ("" if background >= 1.0 else f"_bg{round(background * 100)}")
                  + ("_dep" if depart_fix else "") + (f"_tp{teleport}" if teleport > 0 else ""))


def measure_lag(date: str, uturn: str, background: float) -> dict:
    """보정 없는 하루 점검에서, 석천사거리를 지나는 차가 단독 모형 구간 입구(그 경로에서 처음 만나는 단독 모형 도로)에
    계획보다 몇 초 늦게 왔는지 잰다. 계획 = 계획 출발 + 빈 도로 주행 시간(network_demand.anchor_times와 같은 값),
    실제 = 그 입구 도로에 들어간 시각(앞 도로를 떠난 시각, 입구에서 출발한 차는 실제 출발 시각).
    돌려줌: 경로·도착 시별 (차 수, 평균 초)와 경로별 하루 (차 수, 평균 초). 도착 시는 차 이름의 15분 칸(기록 시각)이다."""
    from network_map import single_model_edges
    out = run_dir(date, uturn, -1, background)
    demand = nd.demand_dir(date, uturn, background=background)
    run_info = json.loads((out / "run.json").read_text())
    if run_info.get("gave_up"):
        raise RuntimeError(f"{out}: 하루를 끝까지 돌지 못해(교착) 출발 시각 보정을 잴 수 없습니다.")
    meta = json.loads((demand / "ledger.json").read_text())["meta"]
    target_sigs = [tuple(m["signature"]) for m in meta["movements"] if m["tls"] == nd.TARGET_TLS]
    graph = nd.Graph(network_12())
    near = single_model_edges()
    root = ET.parse(demand / "routes.rou.xml").getroot()
    route_edges = {r.get("id"): r.get("edges").split() for r in root.iter("route")}
    planned = {v.get("id"): (float(v.get("depart")), v.get("route")) for v in root.iter("vehicle")}
    entry = {}                                         # 경로 → (입구 위치, 출발 → 입구 빈 도로 시간) 또는 None
    for rid, edges in route_edges.items():
        hit = any(tuple(edges[k:k + len(s)]) == s for s in target_sigs for k in range(len(edges)) if edges[k] == s[0])
        c0 = next((k for k, e in enumerate(edges) if e in near), None) if hit else None
        entry[rid] = None if c0 is None else (c0, sum(graph.cost[e] for e in edges[:c0]))
    rows = defaultdict(list)
    for _, v in ET.iterparse(out / "vehroute.xml"):
        if v.tag != "vehicle":
            continue
        vid = v.get("id")
        info = entry.get(planned.get(vid, (0, None))[1])
        r = v.find("route")
        if info is None or r is None or not r.get("exitTimes"):
            v.clear()
            continue
        c0, lead = info
        exits = [float(x) for x in r.get("exitTimes").split()]
        if c0 - 1 < len(exits):
            actual = exits[c0 - 1] if c0 > 0 else float(v.get("depart"))
            hour = int(vid[:2])
            rows[(nd.route_key(route_edges[planned[vid][1]]), hour)].append(actual - planned[vid][0] - lead)
        v.clear()
    by_route_hour = {f"{k}|{h}": (len(x), float(np.mean(x))) for (k, h), x in rows.items()}
    per_route = defaultdict(list)
    for (k, _), x in rows.items():
        per_route[k] += x
    by_route = {k: (len(x), float(np.mean(x))) for k, x in per_route.items()}
    lags = np.concatenate([np.array(x) for x in rows.values()]) if rows else np.zeros(1)
    return {"by_route_hour": by_route_hour, "by_route": by_route,
            "meta": {"source_run": out.name, "vehicles_measured": int(len(lags)), "routes": len(by_route),
                     "lag_mean_s": float(lags.mean()), "lag_median_s": float(np.median(lags)),
                     "lag_p90_s": float(np.percentile(lags, 90)), "lag_over_60s_share": float((lags > 60).mean())}}


def demand_for(date: str, uturn: str, teleport: int, background: float, depart_fix: bool) -> Path:
    """이 실행이 쓰는 수요 폴더. depart_fix면 보정 없는 하루 점검(없으면 먼저 돌림)에서 늦음을 재 보정 수요를 만든다."""
    fixed = nd.demand_dir(date, uturn, background=background, depart_fix=True)
    if depart_fix and (fixed / "routes.rou.xml").exists():
        return fixed                                   # 이미 있음(맥에서 만든 묶음을 Colab에서 풀었을 때도)
    demand = nd.demand_dir(date, uturn, background=background)
    if not (demand / "routes.rou.xml").exists():
        ledger = nd.build_day(date, uturn, demand, background=background)
        (demand / "audit.md").write_text(nd.audit(ledger))
    if not depart_fix:
        return demand
    if teleport > 0:
        raise SystemExit("출발 시각 보정은 순간이동을 끈 하루 점검으로만 잽니다.")
    first = run_dir(date, uturn, -1, background)
    if not (first / "run.json").exists():
        run(date, uturn, -1, background)
    if not (first / "result.json").exists():
        report(date, uturn, -1, background)
    fix = measure_lag(date, uturn, background)
    fixed.mkdir(parents=True, exist_ok=True)
    (fixed / "depart_fix.json").write_text(json.dumps(fix, ensure_ascii=False))
    ledger = nd.build_day(date, uturn, fixed, background=background, depart_fix=fix)
    (fixed / "audit.md").write_text(nd.audit(ledger))
    return fixed


def run(date: str, uturn: str, teleport: int = -1, background: float = 1.0, depart_fix: bool = False) -> Path:
    """teleport: SUMO --time-to-teleport(초). 기본 -1(끔, 단독 모형과 같음). 양수면 그만큼 꼼짝 못 한 차를 옮긴다(시험용).
    background: 석천사거리를 지나지 않는 차의 비율(network_demand.build_day).
    depart_fix: 출발 시각 보정 수요로 돈다(demand_for)."""
    demand = demand_for(date, uturn, teleport, background, depart_fix)
    out = run_dir(date, uturn, teleport, background, depart_fix)
    out.mkdir(parents=True, exist_ok=True)
    programs, _ = ns.day_programs(date)
    for tls, ph in programs.items():       # SUMO가 마지막 → 첫 현시 되돌이도 검사하므로 끝이 녹색이면 황색을 붙인다
        if "y" not in ph[-1][1]:
            last, first = ph[-1][1], ph[0][1]
            ph.append((4, "".join("y" if a in "Gg" and b == "r" else a for a, b in zip(last, first))))
    tls_add = ns.write_additional(programs, out / "tls.add.xml")
    meandata = out / "meandata.add.xml"
    meandata.write_text(f'<additional><edgeData id="ed" period="900" file="{out / "edgedata.xml"}" excludeEmpty="true"/></additional>')
    configure_sumo()
    import sumolib
    cmd = [sumolib.checkBinary("sumo"), "-n", str(network_12()), "-r", str(demand / "routes.rou.xml"),
           "-a", f"{tls_add},{meandata}", "--seed", "42", "--end", str(DAY_S + TAIL_S), "--no-step-log",
           "--time-to-teleport", str(teleport), *COLLISION_OPTIONS.split(), "--collision-output", str(out / "collisions.xml"),
           "--tripinfo-output", str(out / "tripinfo.xml"), "--tripinfo-output.write-unfinished", "true",
           "--vehroute-output", str(out / "vehroute.xml"), "--vehroute-output.exit-times", "true",
           "--vehroute-output.write-unfinished", "true", "--summary-output", str(out / "summary.xml"),
           "--statistic-output", str(out / "stats.xml"), "--duration-log.statistics", "true"]
    t0 = time.time()
    gave_up = None
    with (out / "sumo.log").open("w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            time.sleep(15)
            step = _last_step(out / "summary.xml")
            if step and step["waiting"] > GIVE_UP_WAITING:
                gave_up = step
                proc.terminate()
                proc.wait()
                break
    if gave_up is None and proc.returncode != 0:
        raise RuntimeError(f"SUMO 실패(종료 코드 {proc.returncode}): {out / 'sumo.log'}")
    (out / "run.json").write_text(json.dumps({"seconds": round(time.time() - t0), "command": cmd, "gave_up": gave_up},
                                             ensure_ascii=False))
    return out


def _last_step(summary: Path) -> dict | None:
    """쓰는 중인 요약 파일의 마지막 step."""
    import re
    if not summary.exists():
        return None
    with summary.open("rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 4000))
        tail = f.read().decode(errors="ignore")
    steps = re.findall(r'<step time="([0-9.]+)"[^>]*?inserted="(\d+)"[^>]*?running="(\d+)"[^>]*?waiting="(\d+)"', tail)
    if not steps:
        return None
    t, ins, run_, wait = steps[-1]
    return {"time": float(t), "inserted": int(ins), "running": int(run_), "waiting": int(wait)}


def geh_hourly(simulated: float, observed: float) -> float:
    """15분 대수를 시간당으로 바꿔 계산한 GEH(단독 모형 focused_day.geh_hourly와 같음)."""
    s, o = 4 * simulated, 4 * observed
    return (2 * (s - o) ** 2 / (s + o)) ** 0.5 if s + o else 0.0


def passages(out: Path, movements: list[dict]) -> dict[str, np.ndarray]:
    """실제 통과: {'진입>진출': (96, 차종) 대수}. 서명의 첫 도로를 떠난 시각(교차로에 들어간 시각)으로 15분 칸을 정한다."""
    first = defaultdict(list)
    for m in movements:
        first[m["signature"][0]].append((f"{m['key'][0]}>{m['key'][1]}", tuple(m["signature"])))
    grid = defaultdict(lambda: np.zeros((96, len(CODES))))
    code_of = {v: k for k, v in nd.VTYPE_OF.items()}
    for _, v in ET.iterparse(out / "vehroute.xml"):
        if v.tag != "vehicle":
            continue
        r = v.find("route")
        if r is None or not r.get("exitTimes"):
            v.clear()
            continue
        edges, exits = r.get("edges").split(), [float(x) for x in r.get("exitTimes").split()]
        c = CODES.index(code_of.get(v.get("type"), "p"))
        for k, e in enumerate(edges[:len(exits)]):
            for key, sig in first.get(e, ()):
                if tuple(edges[k:k + len(sig)]) == sig and k + len(sig) - 1 < len(exits):
                    t = exits[k]
                    if 0 <= t < DAY_S:
                        grid[key][int(t // 900), c] += 1
        v.clear()
    return grid


def stopline_arrivals(out: Path, movements: list[dict]) -> dict[str, np.ndarray]:
    """석천사거리 이동마다 '단독 모형 구간 입구에 들어온 시각 + 입구 → 정지선 빈 도로 시간'(15분 칸, 차종 합).
    단독 모형에서는 이 시각이 곧 기록 시각이다(차가 입구에서 출발). 그래서 '기록 − 실제 통과'로 잰 밀린 차를
    '기록 − 입구 기준 도착'(상류에서 늦게 온 몫)과 '입구 기준 도착 − 실제 통과'(석천사거리 자체 줄)로 나눌 수 있다."""
    from network_map import single_model_edges
    near = single_model_edges()
    graph = nd.Graph(network_12())
    first = defaultdict(list)
    for m in movements:
        first[m["signature"][0]].append((f"{m['key'][0]}>{m['key'][1]}", tuple(m["signature"])))
    grid = defaultdict(lambda: np.zeros(96))
    for _, v in ET.iterparse(out / "vehroute.xml"):
        if v.tag != "vehicle":
            continue
        r = v.find("route")
        if r is None or not r.get("exitTimes"):
            v.clear()
            continue
        edges, exits = r.get("edges").split(), [float(x) for x in r.get("exitTimes").split()]
        for k, e in enumerate(edges[:len(exits)]):
            for key, sig in first.get(e, ()):
                if tuple(edges[k:k + len(sig)]) != sig or e not in near:
                    continue
                c0 = k
                while c0 > 0 and edges[c0 - 1] in near:
                    c0 -= 1
                t_in = exits[c0 - 1] if c0 > 0 else float(v.get("depart"))
                t = t_in + sum(graph.cost[x] for x in edges[c0:k + 1])
                if 0 <= t < DAY_S:
                    grid[key][int(t // 900)] += 1
        v.clear()
    return grid


def teleports(out: Path) -> list[tuple[float, str, str]]:
    """SUMO 기록의 순간이동: (시각, 차로, 이유)."""
    import re
    pat = re.compile(r"Teleporting vehicle '[^']*'; ([^,]*), lane='([^']*)', time=([0-9]+(?:\.[0-9]+)?)")
    rows = []
    with (out / "sumo.log").open(errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if m:
                rows.append((float(m.group(3)), m.group(2), m.group(1)))
    return rows


def report(date: str, uturn: str, teleport: int = -1, background: float = 1.0, depart_fix: bool = False) -> str:
    out = run_dir(date, uturn, teleport, background, depart_fix)
    run_info = json.loads((out / "run.json").read_text())
    fix_note = ", 출발 시각 보정" if depart_fix else ""
    if run_info.get("gave_up"):
        g = run_info["gave_up"]
        text = (f"# 12개 지도 하루 점검 {date} (유턴 {uturn}, 배경 교통 {background:.0%}{fix_note}) — 교착으로 중단\n\n"
                f"- {g['time'] / 3600:.2f}시에 못 들어간 차 {g['waiting']:,}대(> {GIVE_UP_WAITING:,})로 멈춤. 들어간 차 {g['inserted']:,}, "
                f"달리는 차 {g['running']:,}\n- 합격: 아니오(하루를 끝까지 돌지 못함)\n")
        (out / "report.md").write_text(text)
        (out / "result.json").write_text(json.dumps({"gave_up": g, "accepted": False}, ensure_ascii=False, indent=1))
        return text
    ledger = json.loads((nd.demand_dir(date, uturn, background=background, depart_fix=depart_fix) / "ledger.json").read_text())
    meta = ledger["meta"]
    tls_of = {f"{m['key'][0]}>{m['key'][1]}": m["tls"] for m in meta["movements"]}
    obs = defaultdict(lambda: np.zeros((96, len(CODES))))
    for pc, row in ledger["observed"].items():
        p, c = int(pc[:2]) * 4 + int(pc[2:4]) // 15, CODES.index(pc[4])
        for k, n in row.items():
            obs[k][p, c] += n
    sim = passages(out, meta["movements"])
    # 시간에 따른 차 수
    steps = [(float(s.get("time")), int(s.get("running")), int(s.get("waiting")), int(s.get("halting")))
             for s in ET.parse(out / "summary.xml").getroot().iter("step")]
    hourly = {h: max((x for x in steps if h * 3600 <= x[0] < (h + 1) * 3600), key=lambda x: x[2], default=None)
              for h in range(25)}
    stats = ET.parse(out / "stats.xml").getroot()
    veh = dict(stats.find("vehicles").attrib)
    trip = dict(stats.find("vehicleTripStatistics").attrib)
    safety = dict(stats.find("safety").attrib)
    cols = list(ET.parse(out / "collisions.xml").getroot())
    where = Counter(c.get("lane").lstrip(":").rsplit("_", 2)[0] if c.get("lane").startswith(":") else "도로 " + c.get("lane")
                    for c in cols)
    when = Counter(int(float(c.get("time")) // 3600) for c in cols)
    # 막힌 도로(15분 칸): 점유 60% 넘고 속도 1m/s 미만. 굳음(마지막 칸 판정)은 점유 70%·속도 0.3m/s
    jam, jam_last = Counter(), defaultdict(set)
    for iv in ET.parse(out / "edgedata.xml").getroot().iter("interval"):
        for e in iv.iter("edge"):
            occ, sp = float(e.get("occupancy", 0)), float(e.get("speed", 99))
            if occ > 60 and sp < 1.0:
                jam[e.get("id")] += 1
            if occ > 70 and sp < 0.3 and not e.get("id").startswith(":"):
                jam_last[e.get("id")].add(float(iv.get("begin")))
    zero = np.zeros((96, len(CODES)))
    lines = [f"# 12개 지도 하루 점검 {date} (201293 유턴: {uturn}, 배경 교통 {background:.0%}{fix_note}, 고정신호)", "",
             f"- 실행 {json.loads((out / 'run.json').read_text())['seconds']}초, 수요 {meta['vehicles']:,}대"
             + (f", 출발을 더 당긴 차 {meta['depart_fix']['shifted_vehicles']:,}대(평균 {meta['depart_fix']['mean_shift_s']:.0f}초, "
                f"1차 실행 {meta['depart_fix']['source_run']}의 입구 늦음 평균 {meta['depart_fix']['lag_mean_s']:.0f}초)"
                if meta.get("depart_fix") else ""),
             f"- 차: 불러옴 {veh.get('loaded')}, 들어감 {veh.get('inserted')}, 끝날 때 달리는 {veh.get('running')}, "
             f"못 들어감 {veh.get('waiting')}",
             f"- 차 한 대 평균: 걸린 시간 {float(trip['duration']):.0f}초, 손실 시간 {float(trip['timeLoss']):.0f}초, "
             f"출발 지연 {float(trip['departDelay']):.0f}초, 경로 {float(trip['routeLength']):.0f}m",
             f"- 안전: 충돌 {safety.get('collisions')}, 급정지 {safety.get('emergencyStops')}, 비상 제동 {safety.get('emergencyBraking')}",
             "", "## 시간대별 가장 붐빈 순간(그 시간 안에서 못 들어간 차가 가장 많은 때)", "",
             "| 시 | 달리는 차 | 못 들어간 차 | 멈춘 차 |", "|---:|---:|---:|---:|"]
    for h, x in hourly.items():
        if x:
            lines.append(f"| {h} | {x[1]:,} | {x[2]:,} | {x[3]:,} |")
    lines += ["", "## 통과량: 실제 통과 vs 기록", "",
              "| 교차로 | 하루 기록 | 하루 계획 | 하루 실제 | 15분×이동 오차율 | 1시간×이동 오차율 | 1시간 GEH 5 미만 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    planned_day = defaultdict(float)
    for pc, row in ledger["planned"].items():
        for k, n in row.items():
            planned_day[tls_of.get(k, "?")] += n
    per_tls = defaultdict(list)
    for k in set(obs) | set(sim):
        per_tls[tls_of.get(k, "?")].append(k)
    result_tls = {}
    for tls in sorted(per_tls, key=lambda t: (t != nd.TARGET_TLS, t)):
        O = np.stack([obs.get(k, zero).sum(1) for k in per_tls[tls]])
        S = np.stack([sim.get(k, zero).sum(1) for k in per_tls[tls]])
        e15 = np.abs(S - O).sum() / max(O.sum(), 1)
        Oh, Sh = O.reshape(len(O), 24, 4).sum(2), S.reshape(len(S), 24, 4).sum(2)
        e60 = np.abs(Sh - Oh).sum() / max(O.sum(), 1)
        busy = (Sh + Oh) > 0
        geh = np.sqrt(2 * (Sh - Oh) ** 2 / np.maximum(Sh + Oh, 1e-9))
        share = float((geh[busy] < 5).mean()) if busy.any() else 1.0
        result_tls[tls] = {"observed": float(O.sum()), "planned": planned_day[tls], "simulated": float(S.sum()),
                           "err_15min": float(e15), "err_60min": float(e60), "geh_lt5": share}
        lines.append(f"| {tls} | {O.sum():,.0f} | {planned_day[tls]:,.0f} | {S.sum():,.0f} | {e15:.1%} | {e60:.1%} | {share:.1%} |")
    # 합격 판정(단독 모형 focused_day.ACCEPTANCE와 같은 틀, 순간이동 끔)
    target = per_tls.get(nd.TARGET_TLS, [])
    arrived = stopline_arrivals(out, [m for m in meta["movements"] if m["tls"] == nd.TARGET_TLS])
    cells, backlog, own, supply = [], {}, {}, {}
    for k in target:
        O, S = obs.get(k, zero).sum(1), sim.get(k, zero).sum(1)
        E = arrived.get(k, np.zeros(96))
        cells += [geh_hourly(S[p], O[p]) for p in range(96) if O[p] or S[p]]
        backlog[k] = float(np.max(np.cumsum(O - S))) if len(O) else 0.0
        own[k] = float(np.max(np.cumsum(E - S)))           # 석천사거리 자체 줄(입구 기준 도착 − 통과)
        supply[k] = float(np.max(np.cumsum(O - E)))        # 상류에서 늦게 온 몫(기록 − 입구 기준 도착)
    geh_share = float(np.mean([g < 5 for g in cells])) if cells else 1.0
    last_iv = max((float(iv.get("begin")) for iv in ET.parse(out / "edgedata.xml").getroot().iter("interval")), default=0)
    jam_end = sorted(e for e, ts in jam_last.items() if last_iv in ts)
    never = int(veh.get("waiting", 0))
    inside = where.get(nd.TARGET_TLS, 0)               # 석천사거리 교차로 안(내부 차로) 충돌
    accept = {"never_inserted_0": never == 0, "no_jam_at_end": not jam_end,
              "geh15_204820_ge_95": geh_share >= 0.95, "backlog_204820_le_150": max(backlog.values(), default=0) <= 150,
              "no_collision_in_204820": inside == 0}
    lines += ["", "## 합격 판정(순간이동 끔, 단독 모형과 같은 틀)", "",
              f"- 끝날 때 못 들어간 차: {never:,}대 → {'통과' if accept['never_inserted_0'] else '불합격'}",
              f"- 마지막 15분에 굳어 있는 도로: {len(jam_end)}개 → {'통과' if accept['no_jam_at_end'] else '불합격'}"
              + (f" ({', '.join(jam_end[:6])})" if jam_end else ""),
              f"- 석천사거리 이동 15분 GEH 5 미만: {geh_share:.1%}(칸 {len(cells)}개) → {'통과' if accept['geh15_204820_ge_95'] else '불합격'}",
              f"- 석천사거리 이동별 밀린 차 누적 최대: {max(backlog.values(), default=0):,.0f}대 → "
              f"{'통과' if accept['backlog_204820_le_150'] else '불합격'}",
              f"  - 나눠 보기(참고, 판정에 안 씀): 석천사거리 자체 줄(단독 모형 구간 입구 기준 도착 − 통과) 최대 "
              f"{max(own.values(), default=0):,.0f}대, 상류에서 늦게 온 몫(기록 − 입구 기준 도착) 최대 {max(supply.values(), default=0):,.0f}대",
              f"- 석천사거리 교차로 안 충돌: {inside}건 → {'통과' if accept['no_collision_in_204820'] else '불합격'}"
              " (단독 모형은 15일 모두 0건. 다른 곳 충돌은 아래 목록에 따로 적는다)",
              f"- **합격: {'예' if all(accept.values()) else '아니오'}**"]
    lines += ["", "## 충돌", "", f"- 곳: {', '.join(f'{k} {v}' for k, v in where.most_common(10)) or '없음'}",
              f"- 시: {', '.join(f'{k}시 {v}' for k, v in sorted(when.items())) or '없음'}",
              "", "## 막힌 도로(15분 칸 수, 점유 60% 넘고 속도 1m/s 미만)", ""]
    lines += [f"- {e}: {n}칸" for e, n in jam.most_common(12)] or ["- 없음"]
    tps = teleports(out)
    if tps:
        from network_map import single_model_edges
        near = single_model_edges()
        place = Counter((l.lstrip(":").rsplit("_", 2)[0] if l.startswith(":") else l.rsplit("_", 1)[0]) for _, l, _ in tps)
        lines += ["", f"## 순간이동 {len(tps):,}번(꼼짝 못 한 지 {teleport}초)", "",
                  f"- 이유: {dict(Counter(r for _, _, r in tps))}",
                  f"- 석천사거리 단독 모형이 쓰는 도로에서: {sum(1 for _, l, _ in tps if l.rsplit('_', 1)[0] in near):,}번",
                  f"- 시: {', '.join(f'{h}시 {n}' for h, n in sorted(Counter(int(t // 3600) for t, _, _ in tps).items()))}",
                  f"- 곳(많은 순): {', '.join(f'{k} {v}' for k, v in place.most_common(12))}"]
    text = "\n".join(lines) + "\n"
    (out / "report.md").write_text(text)
    (out / "result.json").write_text(json.dumps({"accepted": all(accept.values()), "acceptance": accept,
                                                  "geh15_204820": geh_share, "backlog_204820": backlog,
                                                  "backlog_204820_own": own, "backlog_204820_supply": supply,
                                                  "jam_at_end": jam_end, "vehicles": veh, "trip": trip, "safety": safety,
                                                  "collisions_where": dict(where), "collisions_hour": dict(when),
                                                  "jam_15min": dict(jam.most_common(50)), "tls": result_tls,
                                                  "hourly_peak": {h: x for h, x in hourly.items() if x}},
                                                 ensure_ascii=False, indent=1))
    return text


def _run_one(args: tuple) -> dict:
    date, uturn, teleport, background, depart_fix = args
    if depart_fix:                                     # 1차 실행이 교착이면 보정할 수 없다 → 1차 결과를 그대로 낸다
        first = run_dir(date, uturn, teleport, background)
        if not (first / "run.json").exists():
            run(date, uturn, teleport, background)
        if json.loads((first / "run.json").read_text()).get("gave_up"):
            report(date, uturn, teleport, background)
            res = json.loads((first / "result.json").read_text())
            res["seconds"] = json.loads((first / "run.json").read_text())["seconds"]
            res["note"] = "1차 실행(보정 없음)이 교착이라 출발 시각 보정을 하지 못함"
            return res
    out = run_dir(date, uturn, teleport, background, depart_fix)
    if not (out / "run.json").exists():
        run(date, uturn, teleport, background, depart_fix)
    report(date, uturn, teleport, background, depart_fix)
    res = json.loads((out / "result.json").read_text())
    res["seconds"] = json.loads((out / "run.json").read_text())["seconds"]
    if depart_fix:                                     # 보정 전(1차 실행) 결과도 같이 남긴다(같은 판정 틀로 다시 정리)
        report(date, uturn, teleport, background)
        first = json.loads((run_dir(date, uturn, teleport, background) / "result.json").read_text())
        res["before_fix"] = {"accepted": first.get("accepted"), "gave_up": first.get("gave_up"),
                             "backlog_204820_max": max(first.get("backlog_204820", {}).values(), default=None),
                             "geh15_204820": first.get("geh15_204820")}
    return res


def check(dates: list[str], uturn: str, teleport: int, background: float, workers: int, depart_fix: bool = False) -> str:
    """여러 날짜 하루 점검(최종 시험). 날짜 하나를 한 작업이 맡아 수요 만들기부터 끝까지 한다(같은 날짜를 동시에 만들지 않음).
    depart_fix면 날짜마다 보정 없는 1차 실행 → 늦음 재기 → 보정 수요 → 2차 실행(판정)을 한다."""
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers) as ex:
        results = dict(zip(dates, ex.map(_run_one, [(d, uturn, teleport, background, depart_fix) for d in dates])))
    split = json.loads(nd.SPLIT.read_text())
    kind = {d: "학습" for d in split["train"]} | {d: "검증" for d in split["validation"]}
    lines = [f"# 12개 지도 여러 날짜 하루 점검(배경 교통 {background:.0%}, 유턴 {uturn}, "
             f"{'출발 시각 보정, ' if depart_fix else ''}순간이동 {'끔' if teleport <= 0 else f'{teleport}초'})", "",
             "| 날짜 | 묶음 | 끝까지 | 못 들어간 차 | 굳은 도로(끝) | 석천사거리 15분 GEH 5 미만 | 석천사거리 밀린 차 최대"
             + (" (보정 전)" if depart_fix else "") + " | 그중 석천사거리 자체 줄 | 충돌(석천사거리 안) | 차 한 대 손실 시간 | 합격 | 실행 |",
             "|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|"]
    passed = 0
    for date, r in results.items():
        if r.get("gave_up"):
            g = r["gave_up"]
            lines.append(f"| {date} | {kind.get(date, '?')} | ✕({g['time'] / 3600:.1f}시 교착) | {g['waiting']:,} | — | — | — | — | — | — | ✕ | {r['seconds']}초 |")
            continue
        acc = r["accepted"]
        passed += acc
        before = r.get("before_fix", {}).get("backlog_204820_max")
        lines.append(f"| {date} | {kind.get(date, '?')} | ○ | {int(r['vehicles'].get('waiting', 0)):,} | {len(r['jam_at_end'])} | "
                     f"{r['geh15_204820']:.1%} | {max(r['backlog_204820'].values(), default=0):,.0f}"
                     + (f" ({before:,.0f})" if before is not None else "")
                     + f" | {max(r.get('backlog_204820_own', {}).values(), default=0):,.0f} | {r['safety'].get('collisions')} "
                     f"({r['collisions_where'].get(nd.TARGET_TLS, 0)}) | {float(r['trip']['timeLoss']):.0f}초 | "
                     f"{'○' if acc else '✕'} | {r['seconds']}초 |")
    fails = Counter(k for r in results.values() if not r.get("gave_up") for k, ok in r["acceptance"].items() if not ok)
    lines += ["", f"합격 {passed}/{len(dates)}일. 기준(단독 모형과 같은 틀): 끝날 때 못 들어간 차 0, 마지막 15분에 굳은 도로 0, "
              "석천사거리 이동 15분 GEH 5 미만 95% 이상, 석천사거리 이동별 밀린 차 누적 최대 150대 이하, 석천사거리 교차로 안 충돌 0.",
              f"- 떨어진 기준(날짜 수): {dict(fails) or '없음'}"]
    where = Counter()
    for r in results.values():
        where.update(r.get("collisions_where", {}))
    lines.append(f"- 충돌 곳(모든 날짜 합): {', '.join(f'{k} {v}' for k, v in where.most_common(12)) or '없음'}")
    text = "\n".join(lines) + "\n"
    tag = f"bg{round(background * 100)}_{uturn}" + ("_dep" if depart_fix else "") + (f"_tp{teleport}" if teleport > 0 else "")
    path = OUT / f"check_{tag}_{time.strftime('%Y%m%d_%H%M')}.md"
    path.write_text(text)
    (path.with_suffix(".json")).write_text(json.dumps(results, ensure_ascii=False, indent=1))
    return text + f"\n저장: {path}\n"


def saturated_greens(out: Path, date: str, mvs: list, capped: dict, graph) -> dict[str, list[tuple[int, float]]]:
    """이동마다 '줄이 다 풀리지 않은 녹색'의 (통과 대수, 녹색·황색 차로·초). 그 녹색이 끝날 때까지 진입 도로에 들어와
    있던 차(빈 도로라면 녹색 안에 지날 수 있었던 차) 가운데 다음 녹색에야 지나간 차가 있으면 '줄이 남은 녹색'이다.
    그런 녹색에서 지나간 대수 ÷ 녹색·황색 차로·초가 그 이동이 SUMO에서 실제로 내보내는 속도다.
    다만 출구 도로가 막혔던 녹색(그 15분 출구 평균 속도가 제한 속도의 절반 미만)은 뺀다. 그때 느린 것은 그 이동의 처리
    능력이 아니라 앞 교차로에서 거꾸로 찬 줄이기 때문이다(특히 석천사거리로 들어가는 이동은 석천사거리 신호의 결과가
    수요에 들어가면 안 된다)."""
    import network_signals as ns
    programs, _ = ns.day_programs(date)
    exit_edges = {mvs[j].signature[1] for j in capped}
    slow = set()                                       # (도로, 15분 칸): 출구가 막힌 칸
    for iv in ET.parse(out / "edgedata.xml").getroot().iter("interval"):
        p = int(float(iv.get("begin")) // 900)
        for e in iv.iter("edge"):
            eid = e.get("id")
            if eid in exit_edges and float(e.get("speed", 99)) < 0.5 * graph.edges[eid].getSpeed():
                slow.add((eid, p))
    per_second = {tls: [st for d, st in ph for _ in range(d)][:DAY_S] for tls, ph in programs.items() if tls != nd.TARGET_TLS}
    by_first = defaultdict(list)
    for j in capped:
        by_first[mvs[j].signature[0]].append(j)
    times = defaultdict(list)                          # 이동 → [(진입 도로에 들어온 시각, 교차로에 들어간 시각)]
    for _, v in ET.iterparse(out / "vehroute.xml"):
        if v.tag != "vehicle":
            continue
        r = v.find("route")
        if r is None or not r.get("exitTimes"):
            v.clear()
            continue
        edges, exits = r.get("edges").split(), [float(x) for x in r.get("exitTimes").split()]
        for k, e in enumerate(edges[:len(exits) - 1]):
            for j in by_first.get(e, ()):
                if edges[k + 1] == mvs[j].signature[1]:
                    t_in = exits[k - 1] if k > 0 else float(v.get("depart"))
                    times[j].append((t_in, exits[k]))
        v.clear()
    result = {}
    for j, (links, _) in capped.items():
        tls = links[0][0]
        if tls not in per_second or not times[j]:
            continue
        states = per_second[tls]
        green = np.array([any(st[k] in "Gg" for _, k, _ in links) for st in states])
        yellow = np.array([any(st[k] == "y" for _, k, _ in links) for st in states])
        t_in, t_pass = np.array(sorted(times[j], key=lambda x: x[1])).T
        free = graph.cost[mvs[j].signature[0]]
        rows, t = [], 0
        while t < len(green):
            if not green[t]:
                t += 1
                continue
            g0 = t
            while t < len(green) and green[t]:
                t += 1
            g1 = t
            while t < len(green) and yellow[t]:
                t += 1
            end = t                                         # 녹색 + 뒤따르는 황색
            lane_s = (end - g0) * len(links)
            passed = int(((t_pass >= g0) & (t_pass < end)).sum())
            left = ((t_in < g1 - free - 2) & (t_pass >= end + 1)).any()
            if left and end - g0 >= 10 and (mvs[j].signature[1], g0 // 900) not in slow:
                rows.append((passed, float(lane_s)))
        result[f"{mvs[j].key[0]}>{mvs[j].key[1]}"] = rows
    return result


def calibrate(dates: list[str], uturn: str, background: float, min_greens: int = 30) -> str:
    """이웃 신호 이동마다 SUMO가 줄이 있을 때 실제로 내보내는 속도를 보정 없는 하루 점검(1차 실행)에서 잰다
    (saturated_greens: 줄이 다 풀리지 않은 녹색만). 속도 = 그런 녹색에서 지나간 대수 합 ÷ 녹색·황색 차로·초 합 × 3600
    (차로·녹색 1시간당). 그런 녹색이 날짜를 합쳐 min_greens번 이상이고 속도가 일반값(직진 1,880·회전 1,340)보다 5% 넘게
    작으면 후보로 적는다. 기존 보정값은 새로 잰 값이 더 작을 때만 바꾼다(기존 보정으로 상한이 걸려 줄이 안 생기면 잴 녹색이
    적다). 학습 날짜만 쓴다(main에서 막음)."""
    graph = nd.Graph(network_12())
    generic = {"straight": 1880.0, "turn": 1340.0}
    totals, info = defaultdict(lambda: [0, 0.0, 0]), {}
    for date in dates:
        out = run_dir(date, uturn, -1, background)
        if json.loads((out / "run.json").read_text()).get("gave_up"):
            print(f"{date}: 교착이라 뺌", flush=True)
            continue
        counts = nd.read_counts({date})[date]
        mvs, _ = nd.movements(graph, {k for c in counts.values() for k in c}, uturn)
        capped = nd.capped_movements(mvs, graph)
        for key, rows in saturated_greens(out, date, mvs, capped, graph).items():
            t = totals[key]
            t[0] += sum(n for n, _ in rows)
            t[1] += sum(ls for _, ls in rows)
            t[2] += len(rows)
        for j, (links, kind) in capped.items():
            info[f"{mvs[j].key[0]}>{mvs[j].key[1]}"] = {"tls": mvs[j].tls, "kind": kind, "lanes": len(links)}
        print(f"{date}: 끝", flush=True)
    current = nd.capacity_overrides()
    rows, proposal = [], dict(current)
    for key, (passed, lane_s, n) in totals.items():
        gen = generic[info[key]["kind"]]
        rate = passed / lane_s * 3600 if lane_s else None
        row = {"movement": key, **info[key], "generic": gen, "saturated_greens": n, "rate": rate, "current": current.get(key)}
        if n >= min_greens and rate < 0.95 * gen and (key not in current or rate < current[key]):
            proposal[key] = round(rate)
            row["proposed"] = round(rate)
        rows.append(row)
    rows.sort(key=lambda x: -x["saturated_greens"])
    lines = [f"# 이웃 신호 처리 능력 보정 후보(배경 {background:.0%}, 유턴 {uturn}, 날짜 {len(dates)}일: {', '.join(dates)})", "",
             "줄이 다 풀리지 않은 녹색(그 녹색 안에 지날 수 있었던 차가 다음 녹색에야 지나감)에서 지나간 대수 ÷ 녹색·황색 차로·초. "
             "차로·녹색 1시간당 대수.", "",
             "| 이동 | 신호 | 종류 | 차로 | 일반값 | 줄 남은 녹색 | 잰 속도 | 지금 보정 | 후보 |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for x in rows:
        if x["saturated_greens"] == 0 and not x["current"]:
            continue
        rate = "" if x["rate"] is None else f"{x['rate']:.0f}"
        lines.append(f"| {x['movement']} | {x['tls']} | {x['kind']} | {x['lanes']} | {x['generic']:.0f} | "
                     f"{x['saturated_greens']} | {rate} | {x['current'] or ''} | {x.get('proposed', '')} |")
    lines += ["", f"후보 {sum('proposed' in x for x in rows)}개. 적용하면 config/network12_capacity.json의 saturation_per_lane_h가 된다."]
    text = "\n".join(lines) + "\n"
    path = OUT / f"capacity_proposal_bg{round(background * 100)}_{uturn}_{time.strftime('%Y%m%d_%H%M')}.md"
    path.write_text(text)
    path.with_suffix(".json").write_text(json.dumps({"dates": dates, "uturn": uturn, "background": background,
                                                     "min_greens": min_greens, "rows": rows, "proposal": proposal},
                                                    ensure_ascii=False, indent=1))
    return text + f"\n저장: {path}\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("run", "report", "check", "calibrate"))
    ap.add_argument("--date", help="run·report: 날짜")
    ap.add_argument("--dates", help="check: 쉼표로 나눈 날짜들, 또는 train·validation·train+validation")
    ap.add_argument("--workers", type=int, default=4, help="check: 동시에 돌릴 하루 실행 수")
    ap.add_argument("--uturn", choices=("drop", "straight"), default="drop")
    ap.add_argument("--allow-test-date", action="store_true")
    ap.add_argument("--teleport", type=int, default=-1, help="SUMO --time-to-teleport(초), 기본 -1(끔)")
    ap.add_argument("--background", type=float, default=1.0, help="석천사거리를 지나지 않는 차의 비율(0~1, 기본 1)")
    ap.add_argument("--depart-fix", action="store_true",
                    help="출발 시각 보정: 보정 없는 하루 점검에서 잰 '석천사거리 구간 입구 늦음'만큼 출발을 당긴 수요로 돈다")
    args = ap.parse_args()
    split = json.loads(nd.SPLIT.read_text())
    if args.cmd == "calibrate":                       # 처리 능력은 학습 날짜로만 정한다
        dates = split["train"] if args.dates in (None, "train") else args.dates.split(",")
        if set(dates) - set(split["train"]):
            raise SystemExit(f"처리 능력 보정은 학습 날짜로만 잽니다: {sorted(set(dates) - set(split['train']))}")
        print(calibrate(sorted(dates), args.uturn, args.background))
        return
    if args.cmd == "check":
        names = {"train": split["train"], "validation": split["validation"],
                 "train+validation": sorted(split["train"] + split["validation"])}
        dates = names.get(args.dates) or args.dates.split(",")
        if set(dates) & set(split["test"]) and not args.allow_test_date:
            raise SystemExit(f"시험 날짜 {sorted(set(dates) & set(split['test']))}는 쓰지 않습니다.")
        print(check(sorted(dates), args.uturn, args.teleport, args.background, args.workers, args.depart_fix))
        return
    if not args.date:
        raise SystemExit("--date가 필요합니다.")
    if args.date in set(split["test"]) and not args.allow_test_date:
        raise SystemExit(f"{args.date}는 시험 날짜입니다.")
    if args.cmd == "run":
        run(args.date, args.uturn, args.teleport, args.background, args.depart_fix)
    print(report(args.date, args.uturn, args.teleport, args.background, args.depart_fix))


if __name__ == "__main__":
    main()
