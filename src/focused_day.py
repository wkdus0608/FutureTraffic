"""석천사거리(204820) 하루 연속 실행과 가상 도로 점검.

15분마다 빈 도로에서 다시 시작하면 앞 시간대의 대기열·교착이 가려진다(2026-10-04 설계 재분석).
여기서는 하루 96개 시간대를 끊지 않고 이어서 돌린다.

- 차량: 하루 차량 파일(focused_scenario.day_route_file, 환경 지문별로 따로 둔다).
- 신호: 실제 운영 신호를 하루 시간축으로 복원한 것(focused_scenario.aihub_signal_program).
- 점검(check): DQN 없이 실제 운영 고정신호로만 돌려, 관측 교통량을 실제 신호가 처리하는지 본다.
  교차로 안 충돌까지 검사하고, 평가와 같은 규칙으로 판정한다(focused_eval.judge_collisions).

사용:
  .venv/bin/python src/focused_day.py check --dates 20220811 20220928 --workers 2
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from focused_data import COUNTS, PROJECT, environment_fingerprint, find_network
from focused_scenario import (FocusedNetwork, aihub_signal_program, aihub_unreliable_periods, day_route_file,
                              driver_model_version, driver_vtype_attributes, period_of_time, write_tls_additional)

PERIOD_S = 900
DAY_S = 96 * PERIOD_S
TAIL_S = 1800            # 하루가 끝난 뒤 남은 차량이 빠져나갈 시간
OUT = PROJECT / "results/focused_204820/day"
LONG_WAIT_S = 300        # '오래 기다린 차' 기준: 진입 전 대기 + 도로에서 멈춰 있던 시간 합이 300초 초과
# 환경 점검 통과 기준(실제 운영 고정신호, 순간이동 끔).
# - 실제로는 관측량이 그 15분 안에 교차로를 지나갔으므로, 시뮬레이션이 같은 신호로 따라가지 못해
#   밀린 대수(이동별 관측 누적 - 시뮬레이션 누적)가 계속 쌓이면 처리 능력이 모자란 것이다.
#   허용 폭 150대 = 가장 붐비는 이동(길주로 직진, 15분 최대 약 740대)의 한 주기(약 200초) 도착량.
# - GEH는 시간당 교통량으로 계산하는 지표다(15분 대수 × 4). 교통 모형 보정에서 흔히 쓰는 기준은
#   'GEH<5가 85% 이상'이고, 여기서는 15분·이동별로 더 엄격하게 95%를 쓴다.
# - 충돌: 어디서 나든(석천사거리 안·밖) 모두 센다. 원인을 확인해 등록한 지도 오류와 모양이 같은 겹침만 따로 세고
#   (config/collision_exceptions.json: 서쪽 접근로 정지선 약 69m 상류 분기점), 나머지 '확인 안 된 충돌'은 0이어야 한다.
ACCEPTANCE = {
    "never_inserted": 0,                 # 끝까지 도로에 못 들어온 차량(교착이면 생긴다)
    "collisions_unexplained": 0,
    "max_backlog_vehicles": 150,
    "movement_15min_GEH_lt5_share": 0.95,
}


def geh_hourly(simulated: int, observed: int) -> float:
    """15분 대수를 시간당으로 바꿔 계산한 GEH."""
    s, o = 4 * simulated, 4 * observed
    return (2 * (s - o) ** 2 / (s + o)) ** 0.5 if s + o else 0.0


def analyse(run: Path, date: str, route_file: Path, period_of: dict, phases: list[tuple[int, str]]) -> dict:
    """하루 실행 결과: 시간대별 지연, 이동별 통과량 대 관측, 녹색 1초당 통과 분포, 안전 통계."""
    fn = FocusedNetwork()
    end = DAY_S + TAIL_S
    loss, planned = defaultdict(float), Counter(period_of.values())
    routes_root = ET.parse(route_file).getroot()
    route_edges = {r.get("id"): r.get("edges").split() for r in routes_root.iter("route")}
    movement_of = {}
    for v in routes_root.iter("vehicle"):
        e = route_edges[v.get("route")]
        mv = next(((a, b) for a, b in zip(e, e[1:]) if (a, b) in fn.movements), None)
        movement_of[v.get("id")] = f"{mv[0]}->{mv[1]} ({fn.movements[mv]})" if mv else "?"
    by_movement = defaultdict(lambda: {"vehicles": 0, "delay_s": 0.0, "waiting_over_300s": 0})
    seen, long_wait, rows = set(), 0, []
    for t in ET.parse(run / "tripinfo.xml").getroot().iter("tripinfo"):
        vid = t.get("id")
        seen.add(vid)
        depart_delay, waiting = float(t.get("departDelay", 0)), float(t.get("waitingTime", 0))
        delay = float(t.get("timeLoss", 0)) + depart_delay
        loss[period_of[vid]] += delay
        over = waiting + depart_delay > LONG_WAIT_S        # 진입 전 대기까지 넣은 누적 대기
        long_wait += over
        m = by_movement[movement_of[vid]]
        m["vehicles"] += 1
        m["delay_s"] += delay
        m["waiting_over_300s"] += over
        rows.append((vid, period_of[vid], movement_of[vid], t.get("depart"), t.get("departDelay"),
                     t.get("waitingTime"), t.get("timeLoss"), t.get("arrival")))
    never = 0
    for v in routes_root.iter("vehicle"):
        if v.get("id") not in seen:
            never += 1
            lower = max(0.0, end - float(v.get("depart")))
            loss[period_of[v.get("id")]] += lower
            long_wait += lower > LONG_WAIT_S
            m = by_movement[movement_of[v.get("id")]]
            m["vehicles"] += 1
            m["delay_s"] += lower
            m["waiting_over_300s"] += lower > LONG_WAIT_S
    # 차량별 기록(차량 ID, 시간대, 이동, 출발, 진입 지연, 멈춘 시간 합, 지연, 도착)을 작게 남긴다.
    with gzip.open(run / "vehicles.csv.gz", "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["vehicle", "period", "movement", "depart", "depart_delay", "waiting_time", "time_loss", "arrival"])
        w.writerows(rows)
    cross, cross_minute = defaultdict(Counter), defaultdict(Counter)
    for v in ET.parse(run / "vehroute.xml").getroot().iter("vehicle"):
        r = v.find("route")
        if r is None or r.get("exitTimes") is None:
            continue
        edges, exits = r.get("edges").split(), r.get("exitTimes").split()
        for i, (a, b) in enumerate(zip(edges, edges[1:])):
            if (a, b) in fn.movements and i < len(exits) and 0 <= float(exits[i]) < DAY_S:
                cross[period_of_time(float(exits[i]))][(a, b)] += 1
                cross_minute[int(float(exits[i]) // 60)][(a, b)] += 1
    observed, obs_minute = defaultdict(Counter), defaultdict(Counter)
    with COUNTS.open() as handle:
        for r in csv.DictReader(handle):
            if r["date"] == date:
                observed[r["period"]][(r["from_edge"], r["to_edge"])] += int(r["count"])
                obs_minute[(int(r["begin_unix"]) - 1660057200) % 86400 // 60][(r["from_edge"], r["to_edge"])] += int(r["count"])
    periods = sorted(planned)
    flows, cum, max_gap = [], Counter(), Counter()
    for p in periods:
        for mv in fn.movements:
            o, s = observed[p][mv], cross[p][mv]
            cum[mv] += o - s
            max_gap[mv] = max(max_gap[mv], cum[mv])
            if o or s:
                flows.append({"period": p, "from": mv[0], "to": mv[1], "dir": fn.movements[mv], "observed": o, "simulated": s})
    with (run / "movement_15min.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flows[0]))
        w.writeheader()
        w.writerows(flows)
    geh = [geh_hourly(r["simulated"], r["observed"]) for r in flows]
    per_period = [{"period": p, "vehicles": planned[p], "delay_per_vehicle_s": round(loss[p] / planned[p], 2)} for p in periods]
    with (run / "per_period.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_period[0]))
        w.writeheader()
        w.writerows(per_period)
    # 녹색(+황색) 30초 이상인 분의 '녹색 1초당 차로당 통과'(관측 자료와 같은 방법)
    per_sec = [s for d, s in phases for _ in range(d)]
    lanes_of, links_of = defaultdict(set), defaultdict(list)
    for c in fn.root.findall("connection"):
        if c.get("tl") == "204820":
            lanes_of[(c.get("from"), c.get("to"))].add(int(c.get("fromLane")))
    for i, (a, b, _) in fn.links.items():
        links_of[(a, b)].append(i)

    def discharge(source) -> dict:
        out = {}
        for mv, d in fn.movements.items():
            if d not in "sl":
                continue
            vals = []
            for minute in range(DAY_S // 60):
                gy = sum(1 for s in per_sec[minute * 60:minute * 60 + 60] if any(s[i] in "Gy" for i in links_of[mv]))
                if gy >= 30:
                    vals.append(source[minute][mv] / gy / len(lanes_of[mv]) * 3600)
            vals.sort()
            if vals:   # 녹색+황색이 30초 넘는 분이 하루에 한 번도 없으면 뺀다(예: 추석 연휴 좌회전)
                out[f"{mv[0]}->{mv[1]} ({d})"] = {q: round(vals[min(len(vals) - 1, int(q * len(vals)))])
                                                  for q in (0.5, 0.9, 0.95, 0.99)}
        return out

    stats = ET.parse(run / "stats.xml").getroot()
    warn = Counter()
    for line in (run / "sumo.log").read_text(errors="replace").splitlines():
        if "Missing yellow" in line:
            warn["missing_yellow"] += 1
        elif "Teleporting vehicle" in line:
            m = re.search(r"\((\w+)\)", line)
            warn[f"teleport_{m.group(1) if m else 'collision'}"] += 1
        elif "emergency braking" in line:
            warn["emergency_braking"] += 1
        elif "collision" in line.lower():
            warn["collision"] += 1
    unreliable = set(aihub_unreliable_periods(date))
    reliable = [r for r in per_period if r["period"] not in unreliable]
    return {
        "date": date,
        "day_delay_per_vehicle_s": round(sum(loss.values()) / sum(planned.values()), 2),
        "day_delay_reliable_periods_s": round(sum(r["delay_per_vehicle_s"] * r["vehicles"] for r in reliable)
                                              / sum(r["vehicles"] for r in reliable), 2),
        "unreliable_signal_periods": sorted(unreliable),
        "worst_period": max(per_period, key=lambda r: r["delay_per_vehicle_s"]),
        "never_inserted": never,
        "vehicles_waiting_over_300s": long_wait,
        "teleports": int(stats.find("teleports").get("total")),
        "collisions": int(stats.find("safety").get("collisions")),
        "emergency_braking": int(stats.find("safety").get("emergencyBraking")),
        "warnings": dict(warn),
        "movement_15min_GEH_lt5_share": round(sum(g < 5 for g in geh) / len(geh), 3),
        "max_backlog_by_movement": {f"{a}->{b} ({fn.movements[(a, b)]})": v for (a, b), v in max_gap.most_common()},
        "by_movement": {k: {"vehicles": m["vehicles"], "mean_delay_s": round(m["delay_s"] / m["vehicles"], 1),
                            "waiting_over_300s_pct": round(100 * m["waiting_over_300s"] / m["vehicles"], 2)}
                        for k, m in sorted(by_movement.items())},
        "discharge_simulated": discharge(cross_minute),
        "discharge_observed": discharge(obs_minute),
    }


def check_day(date: str, out: Path = OUT, seed: int = 42) -> dict:
    """실제 운영 고정신호로 하루를 돌려 환경 점검 지표와 통과 여부를 낸다."""
    route_file, period_of = day_route_file(date, out / "routes")
    run = out / "check" / date
    run.mkdir(parents=True, exist_ok=True)
    phases = aihub_signal_program(date, 0, DAY_S + TAIL_S)
    # SUMO는 고정 프로그램의 마지막 현시 → 첫 현시 되돌이도 검사한다. 실행 시간 안에서는 일어나지 않지만
    # 경고가 남지 않도록 끝이 녹색이면 그 녹색의 황색(다음=첫 현시 기준)을 붙인다.
    if "y" not in phases[-1][1]:
        last, first = phases[-1][1], phases[0][1]
        phases.append((4, "".join("y" if a in "Gg" and b == "r" else a for a, b in zip(last, first))))
    add = write_tls_additional(phases, run / "tls.add.xml", "aihub_rebuilt")
    from focused_sumo import COLLISION_OPTIONS, configure_sumo
    configure_sumo()
    import sumolib
    cmd = [sumolib.checkBinary("sumo"), "-n", str(find_network()), "-r", str(route_file), "-a", str(add),
           "--seed", str(seed), "--end", str(DAY_S + TAIL_S), "--no-step-log", "--time-to-teleport", "-1",
           *COLLISION_OPTIONS.split(), "--collision-output", str(run / "collisions.xml"),
           "--tripinfo-output", str(run / "tripinfo.xml"), "--tripinfo-output.write-unfinished", "true",
           "--vehroute-output", str(run / "vehroute.xml"), "--vehroute-output.exit-times", "true",
           "--vehroute-output.write-unfinished", "true", "--statistic-output", str(run / "stats.xml"),
           "--duration-log.statistics", "true"]
    with (run / "sumo.log").open("w") as log:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=log)
    result = analyse(run, date, route_file, period_of, phases)
    from focused_eval import evaluation_fingerprint, judge_collisions
    result.update(judge_collisions(run / "collisions.xml"))
    result["environment_fingerprint"] = environment_fingerprint()
    result["evaluation_fingerprint"] = evaluation_fingerprint()     # 충돌 판정 규칙이 들어 있는 지문
    result["driver_model"] = {"version": driver_model_version(), "vtype_attributes": driver_vtype_attributes()}
    backlog = max(result["max_backlog_by_movement"].values())
    result["acceptance"] = {
        "never_inserted": result["never_inserted"] <= ACCEPTANCE["never_inserted"],
        "collisions_unexplained": len(result["collisions_unexplained"]) <= ACCEPTANCE["collisions_unexplained"],
        "max_backlog_vehicles": backlog <= ACCEPTANCE["max_backlog_vehicles"],
        "movement_15min_GEH_lt5_share": result["movement_15min_GEH_lt5_share"] >= ACCEPTANCE["movement_15min_GEH_lt5_share"],
    }
    result["passed"] = all(result["acceptance"].values())
    (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    for name in ("tripinfo.xml", "vehroute.xml"):
        (run / name).unlink(missing_ok=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check")
    c.add_argument("--dates", nargs="+", required=True)
    c.add_argument("--workers", type=int, default=2)
    c.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    out = Path(args.out)
    for date in args.dates:             # 차량 파일은 먼저 순서대로 만든다(동시 생성 충돌 방지).
        day_route_file(date, out / "routes")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(check_day, args.dates, [out] * len(args.dates)))
    for r in results:
        backlog = max(r["max_backlog_by_movement"].values())
        print(f"{r['date']} {'통과' if r['passed'] else '불통과'}  하루 차량당 지연 {r['day_delay_per_vehicle_s']}초 "
              f"(최악 {r['worst_period']['period']} {r['worst_period']['delay_per_vehicle_s']}초)  "
              f"미진입 {r['never_inserted']}  확인 안 된 충돌 {len(r['collisions_unexplained'])}(지도 오류 겹침 "
              f"{sum(r['collisions_known_map_defect'].values())})  최대 밀림 {backlog}대  "
              f"GEH<5 {r['movement_15min_GEH_lt5_share']}  300초 초과 {r['vehicles_waiting_over_300s']}대")


if __name__ == "__main__":
    sys.exit(main())
