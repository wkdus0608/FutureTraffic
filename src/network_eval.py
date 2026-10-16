"""12개 지도 평가: 석천사거리 정책을 12개 지도 하루 연속으로 비교한다 — 지문 밖.

단독 모형 평가(focused_eval)와 같은 원칙이다. 같은 날짜의 같은 하루 차량(12개 지도 최종 환경 수요, network_rl.final_environment)·
운전자·평가 seed를 모든 정책이 쓰고, 하루를 끊지 않고 돌린다(+30분 배출). 이웃 11곳은 모든 정책에서 실제 신호 그대로다.
다른 것은 석천사거리 신호뿐이다.
- 정책: fixed = 실제 운영 고정신호(원자료), fixedc = 전환 규칙의 전적색 보정(규칙 none이면 돌리지 않음),
  greedy = 단순 제어(focused_eval.QueueFirst, 같은 운영 규칙), dqn = 12개 지도에서 학습한 모델(network_rl.train).
  감응식은 뺀다(10-15 사용자 결정).
- '못 들어온 차'·'출발 대기'·'누적 대기'는 학습과 같이 석천사거리 앞에서 기다리는 것만 센다(network_rl.LocalView: 구간에서 출발하려다
  못 들어온 차 + 구간 바로 위 큰길에 멈춘 석천사거리행 차). 모든 정책에 같게 건다.
- 지표
  - 석천사거리 구간 지연(주 지표): 석천사거리를 지나는 차가 단독 모형 도로(진입 구간 입구 → 진출 구간 끝) 안에서 늦어진
    시간 = 구간 통과 시간 − 빈 도로 주행 시간(일반 도로 길이 ÷ 제한 속도) + 석천사거리 구간에서 출발한 차의 출발 대기.
    단독 모형의 차량당 지연과 같은 범위다(교차로 안 차로의 빈 도로 시간을 빼지 않아 몇 초 크게 나온다, 정책끼리는 같음).
    300초 넘게 늦은 차도 이 값으로 센다.
  - 석천사거리를 지나는 차의 전체 통행 지연, 모든 차의 전체 통행 지연(SUMO timeLoss + 출발 대기, 이웃 영향을 본다)
  - 충돌(석천사거리 교차로 안 / 다른 곳), 급제동, 보행(단독 모형과 같은 계산), 운영 규칙이 바꾼 판단
- 결과: <out>/<평가 지문>/<날짜>/<정책>/metrics.json. 평가 지문 = 잠근 환경 파일 + 12개 지도 코드·설정 + 이 파일 + 전환 규칙.
  지문이 같고 metrics.json이 있으면 다시 돌리지 않는다. 날짜 × 정책(DQN은 모델마다) 가운데 하나라도 빠지면 보고하지 않는다.

- 조각 평가(--windows, 10-16 사용자 결정으로 기본): 하루 연속 대신 90분 조각(준비 30분 실제 신호 + 정책 60분, 학습과 같은 틀)
  마다 정책을 비교한다. 조각은 고정신호로 무너지지 않은 것만(network_windows.evaluation_windows). 주 지표는 제어 60분 안에
  석천사거리 진입 구간에 들어온(또는 들어가려고 기다리기 시작한) 차의 석천사거리 구간 지연이다(window_metrics).

사용:
  .venv/bin/python src/network_eval.py run --windows --dates 20220928 20220813 --policies fixed fixedc greedy --workers 4
  .venv/bin/python src/network_eval.py report --windows --dates 20220928 20220813 --models <seed 폴더들>
  (하루 연속 평가는 --windows 없이)
  .venv/bin/python src/network_eval.py run --dates 20220928 20220813 --policies fixed fixedc greedy --workers 2
  .venv/bin/python src/network_eval.py run --dates 20220928 --policies dqn --models <Drive>/models_12map/pet1/seed_42
  .venv/bin/python src/network_eval.py report --dates 20220928 20220813 --models <seed 폴더들>
  (전환 규칙은 학습과 같이 FOCUSED_RULESET으로 고른다. 시험 날짜는 --allow-test-date 없이는 막힘)
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import statistics
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import focused_data as fd
import network_demand as nd
import network_rl as nr
import network_signals as ns
from focused_data import ENV_FILES, TARGET_TLS, environment_fingerprint, fingerprint, variant, variant_signature
from network_map import network_12, single_model_edges

OUT = fd.PROJECT / "results/network_eval"
DAY_S, TAIL_S = 86400, 1800
LONG_DELAY_S = 300
POLICIES = ("fixed", "fixedc", "greedy", "dqn")
LABELS = {"fixed": "고정신호(원자료 그대로)", "fixedc": "고정신호(전적색 보정)", "greedy": "단순 제어(학습 없음)", "dqn": "DQN"}


def evaluation_fingerprint() -> str:
    return fingerprint(ENV_FILES + nr.CODE_FILES + ["src/network_eval.py"])


def _closing_yellow(phases: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """끝이 녹색이면 첫 현시로 되돌아가는 황색을 붙인다(SUMO 되돌이 검사, network_day.run과 같음)."""
    if "y" not in phases[-1][1]:
        last, first = phases[-1][1], phases[0][1]
        phases = phases + [(4, "".join("y" if a in "Gg" and b == "r" else a for a, b in zip(last, first)))]
    return phases


def local_vehicles(route_file: Path, near: set[str]) -> set[str]:
    """하루 수요에서 석천사거리 구간에서 출발하고 석천사거리를 지나는 차(LocalView가 세는 차)."""
    from focused_rl import MOVEMENT_LINKS
    root = ET.parse(route_file).getroot()
    edges_of = {r.get("id"): r.get("edges").split() for r in root.iter("route")}
    ok = {rid for rid, e in edges_of.items() if e[0] in near and any(m in MOVEMENT_LINKS for m in zip(e, e[1:]))}
    return {v.get("id") for v in root.iter("vehicle") if v.get("route") in ok}


# ------------------------------------------------------------------ 하루 실행
def run_day(job: dict) -> dict:
    from focused_eval import QueueFirst, fixed_day_program, model_identity
    from focused_rl import (ALL_ZONE_EDGES, VEHICLE_MAX_WAIT_S, SignalConstraints, ZoneObservation, hold_aware_dqn_class,
                            pedestrian_service, zone_queue_reward)
    from focused_scenario import AIHUB_YELLOW_S
    from focused_sumo import make_environment, parse_statistics, sumo_output_options

    date, policy, eval_seed = job["date"], job["policy"], job["eval_seed"]
    run_dir = Path(job["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    final = nr.final_environment()
    route_file = nd.demand_dir(date, final["uturn"], background=final["background"],
                               depart_fix=final["depart_fix"]) / "routes.rou.xml"
    if not route_file.exists():
        raise SystemExit(f"{route_file}가 없습니다. network_rl.py prepare(또는 Colab 15번 칸 묶음)를 먼저 하세요.")
    programs, _ = ns.day_programs(date)
    neighbors = {t: _closing_yellow(ph) for t, ph in programs.items() if t != TARGET_TLS}
    command, tripinfo, stats = sumo_output_options(run_dir)
    vehroute = (run_dir / "vehroute.xml").resolve()
    command += (f" --vehroute-output {vehroute} --vehroute-output.exit-times true"
                " --vehroute-output.write-unfinished true")
    chooses = policy in ("dqn", "greedy")
    if chooses:
        add = ns.write_additional(neighbors, run_dir / "neighbors.add.xml")
    else:
        add = ns.write_additional({**neighbors, TARGET_TLS: fixed_day_program(date, corrected=policy == "fixedc")},
                                  run_dir / "tls.add.xml")
    command += f" --additional-files {add.resolve()}"
    green = {"min_green": 15, "max_green": 120, "yellow_time": AIHUB_YELLOW_S}
    model = trained = None
    if policy == "dqn":
        trained = json.loads((Path(job["model"]) / "training_config.json").read_text())
        want = {"map": "network12", "environment_fingerprint": environment_fingerprint(),
                "network12_fingerprint": nr._code_fingerprint()}
        got = {k: trained.get(k) for k in want}
        if got != want:
            raise SystemExit(f"{job['model']}: 모델의 학습 환경 {got}이 지금 {want}와 다릅니다. 지금 환경으로 학습한 모델만 평가합니다.")
        if model_identity(job["model"])["model_sha256"] != job["model_sha256"]:
            raise SystemExit(f"{job['model']}: 평가를 준비한 뒤 모델 파일이 바뀌었습니다. 다시 실행하세요.")
        green = {k: trained["environment"][k] for k in ("min_green", "max_green", "yellow_time")}
        model = hold_aware_dqn_class().load(Path(job["model"]) / f"dqn_{TARGET_TLS}")
    near = single_model_edges()
    keep = local_vehicles(route_file, near)
    env = make_environment(seed=eval_seed, fixed_ts=not chooses, additional_sumo_cmd=command, record_per_second_queue=True,
                           net_file=nr.template_network_12() if chooses else network_12(), route_file=route_file,
                           simulation_duration_sec=DAY_S + TAIL_S, observation_class=ZoneObservation,
                           reward_fn=zone_queue_reward, extra_queue_edges=ALL_ZONE_EDGES, **green)
    nr.use_local_view(env, keep, nr.approach_chain_edges(), nr.approach_feeder_edges())
    guard = controller = None
    if policy == "dqn":
        guard = SignalConstraints(max_wait=trained["signal_constraints"]["vehicle_max_cumulative_wait_s"]).bind(env)
    elif policy == "greedy":
        guard = SignalConstraints(max_wait=VEHICLE_MAX_WAIT_S).bind(env)
        controller = QueueFirst(QueueFirst.movements(route_file),
                                [p.state for p in env.traffic_signals[TARGET_TLS].green_phases])
    actions = Counter()
    started = time.time()
    try:
        obs, _ = env.reset(seed=eval_seed)
        done = False
        while not done:
            action = None
            if model is not None:
                action = guard(env, int(model.predict(obs, deterministic=True)[0]))
            elif controller is not None:
                action = guard(env, controller(env, guard.allowed(env)))
            if action is not None:
                actions[action] += 1
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    finally:
        env.close()
    records = env.per_second_queue_records
    with gzip.open(run_dir / "per_second.csv.gz", "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "zone_queue", "pending_204820", "tls_state"])
        for r in records:
            w.writerow([int(r["simulation_time_sec"]), r["zone_queue_vehicles"], r["pending_vehicles"], r["tls_state"]])
    result = {"date": date, "policy": policy, "eval_seed": eval_seed, "train_seed": job.get("train_seed"),
              "model": job.get("model"), "model_sha256": job.get("model_sha256"), "model_timesteps": job.get("model_timesteps"),
              "evaluation_fingerprint": job["fingerprint"], "variant": variant_signature(),
              "environment_fingerprint": environment_fingerprint(), "network12_fingerprint": nr._code_fingerprint(),
              "final_environment": final, "seconds": round(time.time() - started, 1), "action_counts": dict(actions)}
    result.update(vehicle_metrics(route_file, tripinfo, vehroute, keep, near, run_dir))
    result.update(parse_statistics(stats))
    result.update(collision_metrics(run_dir / "collisions.xml"))
    result["pedestrian"] = pedestrian_service([r["tls_state"] for r in records])
    result["max_pending_204820"] = max((r["pending_vehicles"] or 0 for r in records), default=0)
    result["mean_zone_queue"] = statistics.fmean(r["zone_queue_vehicles"] for r in records) if records else 0.0
    if guard is not None:
        result["guard"] = {"decisions": guard.decisions, "rule_decided": dict(guard.rule_decided),
                           "changed_by_rules": {"ped": guard.ped_interventions, "vehicle": guard.vehicle_interventions,
                                                "max_green": guard.max_green_switches}}   # DQN은 0이어야 한다
    tmp = run_dir / "metrics.json.tmp"          # 끊겨도 반쯤 쓴 결과가 '끝난 평가'로 보이지 않게 이름을 마지막에 바꾼다
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    tmp.replace(run_dir / "metrics.json")
    for big in (tripinfo, vehroute):            # 큰 기록은 지운다(결과는 metrics.json·vehicles.csv.gz에)
        Path(big).unlink(missing_ok=True)
    return result


def vehicle_metrics(route_file: Path, tripinfo: Path, vehroute: Path, local: set[str], near: set[str], run_dir: Path) -> dict:
    """차량별 지연. 석천사거리를 지나는 차는 단독 모형 도로 안의 지연(주 지표)과 전체 통행 지연, 모든 차는 전체 통행 지연."""
    from focused_rl import MOVEMENT_LINKS
    graph = nd.Graph(network_12())
    root = ET.parse(route_file).getroot()
    edges_of = {r.get("id"): r.get("edges").split() for r in root.iter("route")}
    plan = {v.get("id"): edges_of[v.get("route")] for v in root.iter("vehicle")}
    planned_depart = {v.get("id"): float(v.get("depart")) for v in root.iter("vehicle")}
    movement = {vid: next((f"{a}->{b}" for a, b in zip(e, e[1:]) if (a, b) in MOVEMENT_LINKS), None) for vid, e in plan.items()}
    trip = {}
    for t in ET.parse(tripinfo).getroot().iter("tripinfo"):
        trip[t.get("id")] = (float(t.get("timeLoss", 0)) + float(t.get("departDelay", 0)), float(t.get("arrival", -1)) >= 0,
                             float(t.get("departDelay", 0)), float(t.get("depart", 0)))
    segment = {}
    for _, v in ET.iterparse(vehroute):
        if v.tag != "vehicle":
            continue
        vid = v.get("id")
        r = v.find("route")
        if movement.get(vid) is None or r is None or not r.get("exitTimes"):
            v.clear()
            continue
        edges, exits = r.get("edges").split(), [float(x) for x in r.get("exitTimes").split()]
        k = next(i for i, (a, b) in enumerate(zip(edges, edges[1:])) if (a, b) in MOVEMENT_LINKS)
        c0, c1 = k, k + 1
        while c0 > 0 and edges[c0 - 1] in near:
            c0 -= 1
        while c1 + 1 < len(edges) and edges[c1 + 1] in near:
            c1 += 1
        if c1 < len(exits):                             # 구간을 다 지난 차만(못 지난 차는 아래에서 하한)
            start = exits[c0 - 1] if c0 > 0 else float(v.get("depart"))
            delay = exits[c1] - start - sum(graph.cost[e] for e in edges[c0:c1 + 1])
            segment[vid] = max(0.0, delay) + (trip.get(vid, (0, 0, 0.0, 0))[2] if vid in local else 0.0)
        v.clear()
    end = DAY_S + TAIL_S
    rows = []
    for vid, edges in plan.items():
        t = trip.get(vid)
        through = movement[vid] is not None
        whole = t[0] if t else 0.0
        if t is None:                                   # 끝까지 도로에 못 들어온 차: (끝 − 예정 출발)을 하한으로
            whole = max(0.0, end - planned_depart[vid])
        seg = segment.get(vid)
        if through and seg is None:                     # 석천사거리 구간을 다 지나지 못한 차: 전체 지연을 하한으로
            seg = whole
        rows.append({"vehicle": vid, "movement": movement[vid] or "", "segment_delay": seg, "trip_delay": whole,
                     "arrived": bool(t and t[1]), "local": vid in local})
    with gzip.open(run_dir / "vehicles.csv.gz", "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    th = [r for r in rows if r["movement"]]
    by_mv = defaultdict(list)
    for r in th:
        by_mv[r["movement"]].append(r)
    return {
        "through_204820": {"vehicles": len(th), "mean_segment_delay_s": statistics.fmean(r["segment_delay"] for r in th),
                           "long_segment_delays": sum(r["segment_delay"] > LONG_DELAY_S for r in th),
                           "mean_trip_delay_s": statistics.fmean(r["trip_delay"] for r in th),
                           "not_arrived": sum(not r["arrived"] for r in th),
                           "by_movement": {k: {"vehicles": len(v), "mean_segment_delay_s":
                                               round(statistics.fmean(r["segment_delay"] for r in v), 1)}
                                           for k, v in sorted(by_mv.items())}},
        "all_vehicles": {"vehicles": len(rows), "mean_trip_delay_s": statistics.fmean(r["trip_delay"] for r in rows),
                         "not_arrived": sum(not r["arrived"] for r in rows),
                         "never_inserted": sum(1 for vid in plan if vid not in trip)},
    }


def collision_metrics(path: Path, since: float = 0.0) -> dict:
    from focused_sumo import parse_collisions
    events = [e for e in parse_collisions(path) if float(e.get("time", 0)) >= since]
    where = Counter((e.get("lane", "").lstrip(":").rsplit("_", 2)[0] if e.get("lane", "").startswith(":") else "도로 " + e.get("lane", ""))
                    for e in events)
    inside = [e for e in events if e.get("lane", "").startswith(f":{TARGET_TLS}_")]
    return {"collisions_total": len(events), "collisions_in_204820": len(inside), "collisions_in_204820_events": inside,
            "collisions_where": dict(where.most_common())}


# ------------------------------------------------------------------ 90분 조각 평가(10-16 사용자 결정)
def window_runs(out: Path, windows: list[tuple[str, int]], policies: list[str], models: list[str], eval_seed: int) -> list[dict]:
    """조각 평가 실행 목록: 조각 × 정책(DQN은 모델마다). 결과 폴더 = windows/<날짜>_<시>/<정책>_eval<seed>."""
    from focused_eval import model_identity
    identities = [model_identity(m) for m in models] if "dqn" in policies else []
    seeds = [i["train_seed"] for i in identities]
    if len(seeds) != len(set(seeds)):
        raise SystemExit(f"학습 seed가 같은 모델이 둘 이상입니다({sorted(seeds)}).")
    runs = []
    for date, hour in windows:
        for policy in policies:
            for ident in (identities if policy == "dqn" else [{}]):
                tag = f"dqn_seed{ident['train_seed']}_{ident['model_sha256'][:10]}" if ident else policy
                runs.append({"date": date, "hour": hour, "policy": policy, "eval_seed": eval_seed, **ident,
                             "run_dir": str(out / "windows" / f"{date}_{hour:02d}" / f"{tag}_eval{eval_seed}")})
    return runs


def run_window(job: dict) -> dict:
    """조각 하나: 준비 30분은 모든 정책이 실제 신호(원자료)로 같고, 넘겨받은 뒤 60분을 정책이 맡는다(학습과 같은 틀).
    고정신호는 그 60분에도 실제 신호(원자료, 또는 전적색 보정), 단순 제어·DQN은 학습 환경(WarmStart12Env)으로 돌린다."""
    from focused_eval import QueueFirst, model_identity
    from focused_rl import (ALL_ZONE_EDGES, VEHICLE_MAX_WAIT_S, SignalConstraints, ZoneObservation, hold_aware_dqn_class,
                            pedestrian_service, zone_queue_reward)
    from focused_scenario import AIHUB_YELLOW_S, aihub_signal_program, apply_all_red, real_signal_states
    from focused_sumo import make_environment, parse_statistics, sumo_output_options
    date, hour, policy, eval_seed = job["date"], job["hour"], job["policy"], job["eval_seed"]
    run_dir = Path(job["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    final = nr.final_environment()
    demand = nd.demand_dir(date, final["uturn"], background=final["background"], depart_fix=final["depart_fix"])
    origin = hour * 3600 - nr.WARMUP_S
    command, tripinfo, stats = sumo_output_options(run_dir)
    vehroute = (run_dir / "vehroute.xml").resolve()
    command += (f" --vehroute-output {vehroute} --vehroute-output.exit-times true"
                " --vehroute-output.write-unfinished true")
    near = single_model_edges()
    guard = model = None
    started = time.time()
    if policy in ("dqn", "greedy"):
        green = {"min_green": 15, "max_green": 120, "yellow": AIHUB_YELLOW_S}
        if policy == "dqn":
            trained = json.loads((Path(job["model"]) / "training_config.json").read_text())
            want = {"map": "network12", "environment_fingerprint": environment_fingerprint(),
                    "network12_fingerprint": nr._code_fingerprint()}
            got = {k: trained.get(k) for k in want}
            if got != want:
                raise SystemExit(f"{job['model']}: 모델의 학습 환경 {got}이 지금 {want}와 다릅니다.")
            if model_identity(job["model"])["model_sha256"] != job["model_sha256"]:
                raise SystemExit(f"{job['model']}: 평가를 준비한 뒤 모델 파일이 바뀌었습니다.")
            green = {"min_green": trained["environment"]["min_green"], "max_green": trained["environment"]["max_green"],
                     "yellow": trained["environment"]["yellow_time"]}
            guard = SignalConstraints(max_wait=trained["signal_constraints"]["vehicle_max_cumulative_wait_s"])
            model = hold_aware_dqn_class().load(Path(job["model"]) / f"dqn_{TARGET_TLS}")
        else:
            guard = SignalConstraints(max_wait=VEHICLE_MAX_WAIT_S)
        env12 = nr.WarmStart12Env([(date, f"{hour:02d}00")], eval_seed, run_dir, constraints=guard, uturn=final["uturn"],
                                  background=final["background"], depart_fix=final["depart_fix"], extra_sumo_cmd=command,
                                  record_per_second_queue=True, extra_queue_edges=ALL_ZONE_EDGES, **green)
        env, route_file, local = env12.current, env12.route, env12.local
        controller = (QueueFirst(QueueFirst.movements(route_file), [p.state for p in env.traffic_signals[TARGET_TLS].green_phases])
                      if policy == "greedy" else None)
        actions = Counter()
        try:
            obs, _ = env12.reset()
            handover, done = env12.handover_s, False
            while not done:
                action = (int(model.predict(obs, deterministic=True)[0]) if model is not None
                          else controller(env, guard.allowed(env)))
                obs, _, terminated, truncated, info = env12.step(action)
                actions[info["executed_action"]] += 1
                done = terminated or truncated
            end_state = _end_state(env)
        finally:
            env12.close()
    else:
        route_file = run_dir / "routes.rou.xml"
        _, local = nr.window_routes(demand / "routes.rou.xml", origin, nr.WARMUP_S + nr.CONTROL_S, route_file, near)
        programs, _ = ns.day_programs(date)
        neighbors = nr.neighbor_window(programs, origin, nr.WARMUP_S + nr.CONTROL_S + 900, run_dir / "neighbors.add.xml")
        states = real_signal_states(date, origin, nr.WARMUP_S + 600)
        handover = next(i for i in range(nr.WARMUP_S, len(states)) if "y" not in states[i] and states[i] != states[i - 1])
        rest = aihub_signal_program(date, origin + handover, nr.CONTROL_S + 900)
        if policy == "fixedc":
            rest, _ = apply_all_red(rest)
        target = ns.write_additional({TARGET_TLS: ns.compress(states[:handover]) + rest}, run_dir / "target.add.xml")
        env = make_environment(seed=eval_seed, fixed_ts=True, net_file=network_12(), route_file=route_file,
                               simulation_duration_sec=handover + nr.CONTROL_S, observation_class=ZoneObservation,
                               reward_fn=zone_queue_reward, extra_queue_edges=ALL_ZONE_EDGES, record_per_second_queue=True,
                               additional_sumo_cmd=f"-a {neighbors.resolve()},{target.resolve()} {command}",
                               min_green=15, max_green=120, yellow_time=AIHUB_YELLOW_S)
        nr.use_local_view(env, set(local), nr.approach_chain_edges(), nr.approach_feeder_edges())
        actions = Counter()
        try:
            env.reset(seed=eval_seed)
            done = False
            while not done:
                _, _, terminated, truncated, _ = env.step(None)
                done = terminated or truncated
            end_state = _end_state(env)
        finally:
            env.close()
    records = [r for r in env.per_second_queue_records if r["simulation_time_sec"] >= handover]
    result = {"date": date, "hour": hour, "policy": policy, "eval_seed": eval_seed, "train_seed": job.get("train_seed"),
              "model": job.get("model"), "model_sha256": job.get("model_sha256"), "model_timesteps": job.get("model_timesteps"),
              "evaluation_fingerprint": job["fingerprint"], "variant": variant_signature(),
              "environment_fingerprint": environment_fingerprint(), "network12_fingerprint": nr._code_fingerprint(),
              "final_environment": final, "handover_s": handover, "seconds": round(time.time() - started, 1),
              "action_counts": dict(actions)}
    result.update(window_metrics(route_file, tripinfo, vehroute, set(local), near, handover, handover + nr.CONTROL_S,
                                 end_state, run_dir))
    result.update(parse_statistics(stats))
    cols = collision_metrics(run_dir / "collisions.xml", since=handover)
    result.update(cols)
    result["pedestrian"] = pedestrian_service([r["tls_state"] for r in records])
    if guard is not None:
        result["guard"] = {"decisions": guard.decisions, "rule_decided": dict(guard.rule_decided),
                           "changed_by_rules": {"ped": guard.ped_interventions, "vehicle": guard.vehicle_interventions,
                                                "max_green": guard.max_green_switches}}
    tmp = run_dir / "metrics.json.tmp"
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    tmp.replace(run_dir / "metrics.json")
    for big in (tripinfo, vehroute):
        Path(big).unlink(missing_ok=True)
    return result


def _end_state(env) -> dict:
    """조각이 끝날 때(닫기 전) LocalView가 본 상태: 구간에서 못 들어온 차, 구간 위 큰길에서 기다리는 차, 기다린 초."""
    view = env.sumo
    return {"pending": list(view.simulation.getPendingVehicles()), "queued": dict(view._queued),
            "feeder_wait": dict(view._feeder_wait)}


def window_metrics(route_file: Path, tripinfo: Path, vehroute: Path, local: set[str], near: set[str], h0: float, h1: float,
                   end_state: dict, run_dir: Path) -> dict:
    """석천사거리 구간 지연(조각): 제어 60분 [h0, h1) 안에 석천사거리 진입 구간에 들어온(또는 들어가려고 기다리기 시작한)
    석천사거리행 차만. 지연 = 구간 통과 시간 − 빈 도로 시간 + 구간 위 큰길에서 기다린 초 + (구간에서 출발한 차의) 출발 대기.
    조각이 끝날 때 아직 못 지난 차·못 들어온 차는 (끝 − 들어온 때 − 빈 도로 시간)·(끝 − 기다리기 시작한 때)를 하한으로."""
    from focused_rl import MOVEMENT_LINKS
    graph = nd.Graph(network_12())
    root = ET.parse(route_file).getroot()
    edges_of = {r.get("id"): r.get("edges").split() for r in root.iter("route")}
    plan = {v.get("id"): (edges_of[v.get("route")], float(v.get("depart"))) for v in root.iter("vehicle")}
    movement = {vid: next((f"{a}->{b}" for a, b in zip(e, e[1:]) if (a, b) in MOVEMENT_LINKS), None) for vid, (e, _) in plan.items()}
    depart_delay = {t.get("id"): float(t.get("departDelay", 0)) for t in ET.parse(tripinfo).getroot().iter("tripinfo")}
    entered = {}
    for _, v in ET.iterparse(vehroute):
        if v.tag != "vehicle":
            continue
        vid, r = v.get("id"), v.find("route")
        if movement.get(vid) is None or r is None or not r.get("exitTimes"):
            v.clear()
            continue
        edges, exits = r.get("edges").split(), [float(x) for x in r.get("exitTimes").split()]
        k = next(i for i, (a, b) in enumerate(zip(edges, edges[1:])) if (a, b) in MOVEMENT_LINKS)
        c0, c1 = k, k + 1
        while c0 > 0 and edges[c0 - 1] in near:
            c0 -= 1
        while c1 + 1 < len(edges) and edges[c1 + 1] in near:
            c1 += 1
        if c0 == 0 or c0 - 1 < len(exits):
            entry = exits[c0 - 1] if c0 > 0 else float(v.get("depart"))
            entered[vid] = (entry, exits[c1] if c1 < len(exits) else None, sum(graph.cost[e] for e in edges[c0:c1 + 1]))
        v.clear()
    queued, feeder_wait, pending = end_state["queued"], end_state["feeder_wait"], set(end_state["pending"])
    rows = []
    for vid, mv in movement.items():
        if mv is None:
            continue
        if vid in entered:
            entry, exit_t, ff = entered[vid]
            start = entry - feeder_wait.get(vid, 0.0) - (depart_delay.get(vid, 0.0) if vid in local else 0.0)
            if not (h0 <= start < h1):
                continue
            delay = (exit_t if exit_t is not None else h1) - start - ff
            rows.append({"vehicle": vid, "movement": mv, "delay": max(0.0, delay), "finished": exit_t is not None})
        elif vid in queued and h0 <= queued[vid] < h1:
            rows.append({"vehicle": vid, "movement": mv, "delay": h1 - queued[vid], "finished": False})
        elif vid in pending and h0 <= plan[vid][1] < h1:
            rows.append({"vehicle": vid, "movement": mv, "delay": h1 - plan[vid][1], "finished": False})
    with gzip.open(run_dir / "vehicles.csv.gz", "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["vehicle", "movement", "delay", "finished"])
        w.writeheader()
        w.writerows(rows)
    by_mv = defaultdict(list)
    for r in rows:
        by_mv[r["movement"]].append(r["delay"])
    return {"window_204820": {"vehicles": len(rows), "mean_delay_s": statistics.fmean(r["delay"] for r in rows) if rows else 0.0,
                              "long_delays": sum(r["delay"] > LONG_DELAY_S for r in rows),
                              "unfinished": sum(not r["finished"] for r in rows),
                              "pending_at_end": len(pending), "queued_upstream_at_end": len(queued),
                              "by_movement": {k: {"vehicles": len(v), "mean_delay_s": round(statistics.fmean(v), 1)}
                                              for k, v in sorted(by_mv.items())}}}


# ------------------------------------------------------------------ 실행과 보고
def planned(out: Path, dates: list[str], policies: list[str], models: list[str], eval_seed: int) -> list[dict]:
    from focused_eval import planned_runs
    return planned_runs(out, dates, policies, models, eval_seed, "random")


def run(args) -> None:
    if "fixedc" in args.policies and not variant()["all_red"]:
        args.policies = [p for p in args.policies if p != "fixedc"]
        print("전환 규칙이 없어(none) 전적색 보정 고정신호는 원자료 고정신호와 같습니다. 돌리지 않습니다.", flush=True)
    fp = evaluation_fingerprint()
    out = Path(args.out) / fp
    if args.windows:
        import network_windows as nw
        jobs = [{**j, "fingerprint": fp}
                for j in window_runs(out, nw.evaluation_windows(args.dates), args.policies, args.models, args.eval_seed)]
    else:
        jobs = [{**j, "fingerprint": fp} for j in planned(out, args.dates, args.policies, args.models, args.eval_seed)]
    nr.template_network_12()
    todo = [j for j in jobs if not (Path(j["run_dir"]) / "metrics.json").exists()]
    print(f"평가 지문 {fp}({variant()['rule_set']}{', 조각' if args.windows else ', 하루 연속'}): 실행 {len(jobs)}개 중 "
          f"새로 {len(todo)}개 → {out}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_window if args.windows else run_day, j) for j in todo]
        for n, future in enumerate(as_completed(futures), 1):
            r = future.result()
            if args.windows:
                w = r["window_204820"]
                print(f"  {n}/{len(todo)} {r['date']} {r['hour']:02d}시 {r['policy']}"
                      f"{'' if r['train_seed'] is None else ' seed ' + str(r['train_seed'])}: 석천사거리 구간 지연 "
                      f"{w['mean_delay_s']:.1f}초(차 {w['vehicles']}), 300초 초과 {w['long_delays']}대, "
                      f"석천사거리 안 충돌 {r['collisions_in_204820']}건, {r['seconds']:.0f}초", flush=True)
                continue
            t = r["through_204820"]
            print(f"  {n}/{len(todo)} {r['date']} {r['policy']}{'' if r['train_seed'] is None else ' seed ' + str(r['train_seed'])}: "
                  f"석천사거리 구간 지연 {t['mean_segment_delay_s']:.1f}초, 300초 초과 {t['long_segment_delays']}대, "
                  f"모든 차 통행 지연 {r['all_vehicles']['mean_trip_delay_s']:.1f}초, 석천사거리 안 충돌 {r['collisions_in_204820']}건, "
                  f"{r['seconds']:.0f}초", flush=True)


def report_windows(args, out: Path) -> None:
    """조각 평가 보고: 정책마다 모든 조각의 차를 합친 석천사거리 구간 지연(DQN은 seed별 평균 ± 표준편차)과 시간대별 표."""
    from focused_eval import _load
    import network_windows as nw
    wins = nw.evaluation_windows(args.dates)
    base = ["fixed"] + (["fixedc"] if variant()["all_red"] else []) + ["greedy"]
    rows = _load(window_runs(out, wins, base + (["dqn"] if args.models else []), args.models, args.eval_seed))

    def values(rs: list[dict]) -> dict:
        n = sum(r["window_204820"]["vehicles"] for r in rs)
        return {"delay": sum(r["window_204820"]["mean_delay_s"] * r["window_204820"]["vehicles"] for r in rs) / max(n, 1),
                "vehicles": n, "long": sum(r["window_204820"]["long_delays"] for r in rs),
                "unfinished": sum(r["window_204820"]["unfinished"] for r in rs),
                "col_in": sum(r["collisions_in_204820"] for r in rs), "col_all": sum(r["collisions_total"] for r in rs),
                "ped_over": sum(c["waits_over_limit"] for r in rs for c in r["pedestrian"].values()),
                "rule_changed": sum(sum(r.get("guard", {}).get("changed_by_rules", {}).values()) for r in rs)}

    def by_policy(policy: str) -> tuple[dict, str, list[dict]]:
        rs = [r for r in rows if r["policy"] == policy]
        if policy != "dqn":
            return values(rs), "", rs
        seeds = defaultdict(list)
        for r in rs:
            seeds[r["train_seed"]].append(r)
        per = [values(x) for x in seeds.values()]
        v = {k: statistics.fmean(p[k] for p in per) for k in per[0]}
        return v, (f" ± {statistics.stdev(p['delay'] for p in per):.1f}" if len(per) > 1 else ""), rs

    skipped = {d: sorted(set(nw.HOURS) - {h for dd, h in wins if dd == d}) for d in args.dates}
    lines = [f"# 12개 지도 조각 평가 — 날짜 {', '.join(args.dates)} (평가 지문 {out.name}, {variant()['rule_set']}, 평가 seed {args.eval_seed})", "",
             "90분 조각(준비 30분은 모든 정책이 실제 신호, 넘겨받은 뒤 60분을 정책이 맡음). 모든 정책이 같은 조각·차량·운전자·평가 seed·"
             "이웃 신호를 쓴다. 석천사거리 구간 지연 = 제어 60분 안에 석천사거리 진입 구간에 들어온(또는 들어가려고 기다리기 시작한) "
             "석천사거리행 차가 단독 모형 도로 안에서 늦어진 시간(+ 구간 위 큰길 대기·구간 출발 대기). DQN은 학습 seed별 평균 ± 표준편차.", "",
             f"- 조각 {len(wins)}개. 고정신호에서 무너져 뺀 조각: " + (", ".join(f"{d} {', '.join(f'{h:02d}시' for h in hs)}"
                                                                  for d, hs in skipped.items() if hs) or "없음"), "",
             "| 정책 | 석천사거리 구간 지연(초) | 차 | 300초 넘게 늦은 차 | 조각 끝까지 못 지난 차 | 석천사거리 안 충돌 | 충돌 전체 | 보행 한도 초과 | 운영 규칙이 바꾼 판단 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    table = {}
    for policy in POLICIES:
        if not any(r["policy"] == policy for r in rows):
            continue
        v, sd, rs = by_policy(policy)
        table[policy] = rs
        lines.append(f"| {LABELS[policy]} | {v['delay']:.1f}{sd} | {v['vehicles']:,.0f} | {v['long']:.0f} | {v['unfinished']:.0f} | "
                     f"{v['col_in']:.0f} | {v['col_all']:.0f} | {v['ped_over']:.0f} | {v['rule_changed']:.0f} |")
    lines += ["", "## 시간대별 석천사거리 구간 지연(초, 날짜 합침, DQN은 seed 평균)", "",
              "| 제어 시작 | " + " | ".join(LABELS[p] for p in table) + " |", "|---|" + "---:|" * len(table)]
    for h in nw.HOURS:
        cells = []
        for policy, rs in table.items():
            hr = [r for r in rs if r["hour"] == h]
            if not hr:
                cells.append("—")
                continue
            n = sum(r["window_204820"]["vehicles"] for r in hr)
            cells.append(f"{sum(r['window_204820']['mean_delay_s'] * r['window_204820']['vehicles'] for r in hr) / max(n, 1):.1f}")
        lines.append(f"| {h:02d}시 | " + " | ".join(cells) + " |")
    target = out / ("report_windows_" + "_".join(args.dates) + ".md")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"저장: {target}")


def report(args) -> None:
    from focused_eval import _load
    out = Path(args.out) / (args.fingerprint or evaluation_fingerprint())
    if args.windows:
        report_windows(args, out)
        return
    base = ["fixed"] + (["fixedc"] if variant()["all_red"] else []) + ["greedy"]
    rows = _load(planned(out, args.dates, base + (["dqn"] if args.models else []), args.models, args.eval_seed))

    def values(rs: list[dict]) -> dict:
        n = sum(r["through_204820"]["vehicles"] for r in rs)
        m = sum(r["all_vehicles"]["vehicles"] for r in rs)
        return {"segment": sum(r["through_204820"]["mean_segment_delay_s"] * r["through_204820"]["vehicles"] for r in rs) / n,
                "long": sum(r["through_204820"]["long_segment_delays"] for r in rs),
                "trip_204820": sum(r["through_204820"]["mean_trip_delay_s"] * r["through_204820"]["vehicles"] for r in rs) / n,
                "trip_all": sum(r["all_vehicles"]["mean_trip_delay_s"] * r["all_vehicles"]["vehicles"] for r in rs) / m,
                "not_arrived": sum(r["all_vehicles"]["not_arrived"] for r in rs),
                "col_in": sum(r["collisions_in_204820"] for r in rs), "col_all": sum(r["collisions_total"] for r in rs),
                "braking": sum(r["emergency_braking_count"] for r in rs),
                "ped_over": sum(c["waits_over_limit"] for r in rs for c in r["pedestrian"].values())}
    lines = [f"# 12개 지도 평가 — 날짜 {', '.join(args.dates)} (평가 지문 {out.name}, {variant()['rule_set']}, 평가 seed {args.eval_seed})", "",
             "하루를 끊지 않고 돌린 결과. 모든 정책이 같은 차량·운전자·평가 seed·이웃 신호를 쓴다. DQN은 학습 seed별 평균 ± 표준편차.",
             "석천사거리 구간 지연 = 석천사거리를 지나는 차가 단독 모형 도로 안에서 늦어진 시간(+ 그 구간에서 출발한 차의 출발 대기).", "",
             "| 정책 | 석천사거리 구간 지연(초) | 300초 넘게 늦은 차 | 석천사거리 지나는 차 통행 지연 | 모든 차 통행 지연 | 도착 못 한 차 | "
             "석천사거리 안 충돌 | 충돌 전체 | 급제동 | 보행 한도 초과 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for policy in POLICIES:
        rs = [r for r in rows if r["policy"] == policy]
        if not rs:
            continue
        if policy != "dqn":
            v, sd = values(rs), ""
        else:
            seeds = defaultdict(list)
            for r in rs:
                seeds[r["train_seed"]].append(r)
            per = [values(x) for x in seeds.values()]
            v = {k: statistics.fmean(p[k] for p in per) for k in per[0]}
            sd = f" ± {statistics.stdev(p['segment'] for p in per):.1f}" if len(per) > 1 else ""
        lines.append(f"| {LABELS[policy]} | {v['segment']:.1f}{sd} | {v['long']:.0f} | {v['trip_204820']:.1f} | {v['trip_all']:.1f} | "
                     f"{v['not_arrived']:.0f} | {v['col_in']:.0f} | {v['col_all']:.0f} | {v['braking']:.0f} | {v['ped_over']:.0f} |")
    target = out / ("report_" + "_".join(args.dates) + ".md")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"저장: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--dates", nargs="+", required=True)
    r.add_argument("--policies", nargs="+", default=list(POLICIES), choices=POLICIES)
    r.add_argument("--models", nargs="*", default=[])
    r.add_argument("--eval-seed", type=int, default=42)
    r.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    r.add_argument("--out", default=str(OUT))
    r.add_argument("--allow-test-date", action="store_true")
    r.add_argument("--windows", action="store_true", help="90분 조각 평가(막히지 않은 조각만, config/network12_windows.json)")
    p = sub.add_parser("report")
    p.add_argument("--dates", nargs="+", required=True)
    p.add_argument("--models", nargs="*", default=[])
    p.add_argument("--eval-seed", type=int, default=42)
    p.add_argument("--fingerprint")
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--allow-test-date", action="store_true")
    p.add_argument("--windows", action="store_true", help="90분 조각 평가 보고")
    args = parser.parse_args()
    test = set(json.loads(nd.SPLIT.read_text())["test"])
    if set(args.dates) & test and not args.allow_test_date:
        parser.error(f"시험 날짜 {sorted(set(args.dates) & test)}는 설정을 모두 정한 뒤 --allow-test-date로만 돌린다")
    if args.command == "run" and "dqn" in args.policies and not args.models:
        parser.error("dqn 정책에는 --models가 필요합니다")
    {"run": run, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
