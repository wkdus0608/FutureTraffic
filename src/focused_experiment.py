"""석천사거리(204820) 실험: DQN 학습·정책 평가·보고.

비교하는 정책 셋 (모두 같은 차량 파일·같은 운전자 설정·같은 평가 seed)
- fixed: 고정신호 = AI Hub 실제 운영 신호("한 현시 밀림" 보정, 하루 시간축 복원)
- actuated: 감응식 = 실제 신호의 녹색 4개를 차량 감지에 따라 늘이고 줄임(참고 기준)
- dqn: DQN = 실제 신호의 녹색 4개 중 하나를 5초마다 고름(운영 제약 적용)

주의(2026-10-05): docs/석천사거리_전체점검_20261005.md 5.2절의 결함을 고치기 전에는 본 학습·평가에
쓰지 않는다. 특히 지금 평가는 30분 단위(빈 도로에서 시작)이고, 결과 폴더가 있으면 다시 계산하지 않는다.

사용 예
-------
.venv/bin/python src/focused_experiment.py train --seed 42 --timesteps 100000
.venv/bin/python src/focused_experiment.py evaluate --date 20220928 --workers 6
.venv/bin/python src/focused_experiment.py report --dates 20220928
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import gymnasium as gym

from focused_data import COUNTS, PROJECT, TARGET_TLS, find_network, load_counts, signal_xml_files
from focused_scenario import (AIHUB_YELLOW_S, ARRIVAL_WINDOW_S, ENTRY_BRANCHES, FocusedNetwork, aihub_states,
                              driver_vtype_attributes, replay_phases, write_actuated_additional,
                              write_dqn_template_network, write_routes, write_tls_additional)

OUT = PROJECT / "results/focused_204820"            # 기본 출력 위치(--out으로 바꿀 수 있다)
TEMPLATE_NET = OUT / "net_aihub_template.net.xml"   # DQN용 틀 지도(없으면 만든다)
SPLIT_FILE = PROJECT / "config/focused_split.json"
DURATION = 1800
# v4_env (2026-10-04): 가상 도로 정상화 — 실제 신호를 하루 시간축으로 복원(파일 경계·기록 누락·유턴 황색),
# 길주로 서쪽 접근로(477707013_02) 차량을 상류 기록대로 두 갈래에 나눔, 운전자 설정(config/driver_model.json)을 모든 정책에 같게.
# v3.2 이전 모델은 다른 환경에서 학습했으므로 평가에서 거부한다.
EXPERIMENT_VERSION = "focused_204820_v4_env"


def dqn_network() -> Path:
    """DQN용 지도: 원본 지도에서 204820 신호 프로그램만 실제 신호의 녹색 4개 틀로 바꾼 사본."""
    if not TEMPLATE_NET.exists():
        write_dqn_template_network(TEMPLATE_NET)
    return TEMPLATE_NET


def route_seed(date: str, period: str) -> int:
    return int(date) * 10_000 + int(period)


def signal_greens() -> list[str]:
    """실제 운영 신호의 녹색 현시 4개(DQN의 행동 4개)."""
    return aihub_states()["greens"]


def actuated_plan(period: str, date: str) -> tuple[list[str], list[int], list[int]]:
    """감응식 기준선의 현시·최소·최대 녹색.

    최소 = DQN과 같은 운영 제약(보행자 횡단 46/33초, 그 밖 15초).
    최대 = 그 시간대 실제 운영 녹색(중앙값) × 1.25. 실제 계획을 감응식으로 돌린 형태다.
    """
    from focused_rl import phase_rules
    greens = signal_greens()
    mins, _ = phase_rules(greens)
    durations = defaultdict(list)
    for d, s in replay_phases(period, date=date)[1:]:
        durations[s].append(d)
    maxes = [max(m, round(1.25 * statistics.median(durations[g]))) if durations[g] else m
             for m, g in zip(mins, greens)]
    return greens, mins, maxes


# ------------------------------------------------------------------ 학습

class RandomScenarioEnv(gym.Env):
    """에피소드마다 학습 날짜의 15분 시간대 하나를 무작위로 골라 새 SUMO를 연다."""

    def __init__(self, scenarios: list[tuple[str, str]], counts, seed: int, workdir: Path,
                 min_green: int = 15, max_green: int = 120, yellow: int = AIHUB_YELLOW_S, constraints=None):
        super().__init__()
        from focused_sumo import make_environment
        self.make_environment = make_environment
        self.scenarios = scenarios
        self.counts = counts
        self.net = dqn_network()
        self.rng = random.Random(seed)
        self.seed_base = seed
        self.episode = 0
        self.workdir = workdir
        self.fn = FocusedNetwork()
        self.min_green, self.max_green, self.yellow = min_green, max_green, yellow
        self.constraints = constraints
        self.history: list[tuple[str, str]] = []
        self.current = self._open()
        if self.constraints is not None:
            self.constraints.bind(self.current)
        self.observation_space = self.current.observation_space
        self.action_space = self.current.action_space
        self.started = False

    def _write_next_routes(self) -> Path:
        date, period = self.rng.choice(self.scenarios)
        self.history.append((date, period))
        route = self.workdir / "train_routes.rou.xml"
        write_routes(self.fn, self.counts[(date, period)], route, seed=route_seed(date, period))
        return route

    def _open(self):
        route = self._write_next_routes()
        from focused_rl import ZoneObservation, zone_queue_reward
        return self.make_environment(seed=self.seed_base + self.episode, net_file=self.net,
                                     route_file=route, simulation_duration_sec=DURATION,
                                     observation_class=ZoneObservation, reward_fn=zone_queue_reward,
                                     min_green=self.min_green, max_green=self.max_green, yellow_time=self.yellow)

    def reset(self, *, seed=None, options=None):
        if self.started:
            # 환경을 새로 만들지 않고 같은 경로 파일에 다음 시나리오를 쓴 뒤 다시 시작한다.
            # SUMO-RL은 reset 때 경로 파일을 새로 읽으므로 결과는 같고 SUMO 시작 횟수만 줄어든다.
            self.episode += 1
            self._write_next_routes()
        self.started = True
        if self.constraints is not None:
            self.constraints.reset()
        return self.current.reset(seed=self.seed_base + self.episode, options=options)

    def step(self, action):
        if self.constraints is not None:
            action = self.constraints(self.current, action)
        obs, reward, terminated, truncated, info = self.current.step(action)
        # 학습기(HoldAwareDQN)가 재생 버퍼에 저장할 '실제로 실행된 행동' = 이번 5초 동안 켜진 현시.
        info = dict(info or {})
        info["executed_action"] = int(self.current.traffic_signals[TARGET_TLS].green_phase)
        return obs, reward, terminated, truncated, info

    def close(self):
        self.current.close()


def train(args) -> None:
    import torch
    torch.set_num_threads(1)  # 여러 학습을 동시에 돌리므로 프로세스당 1개 코어만 쓴다.
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor

    from focused_rl import CROSSWALK_MIN_GREEN, CROSSWALKS, SignalConstraints, hold_aware_dqn_class
    DQN = hold_aware_dqn_class()

    counts = load_counts(COUNTS)
    # 학습 날짜는 항상 분할 파일의 train만 쓴다(시험·검증 날짜가 섞이지 않게).
    split = json.loads(Path(args.split_file).read_text())
    train_dates = set(split["train"])
    scenarios = sorted(k for k, rows in counts.items() if k[0] in train_dates and rows)
    out = Path(args.out) / "dqn" / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    constraints = SignalConstraints(max_wait=args.max_wait)
    workdir = Path(tempfile.mkdtemp(prefix=f"focused_dqn_{args.seed}_"))
    env = RandomScenarioEnv(scenarios, counts, args.seed, workdir, min_green=args.min_green,
                            max_green=args.max_green, yellow=args.yellow, constraints=constraints)
    monitored = Monitor(env, filename=str(out / "monitor.csv"))
    model = DQN(policy="MlpPolicy", env=monitored, learning_rate=1e-3, buffer_size=100_000,
                learning_starts=1_000, batch_size=64, train_freq=4, target_update_interval=5_000,
                exploration_fraction=0.2, exploration_initial_eps=1.0, exploration_final_eps=0.05,
                seed=args.seed, verbose=0)
    started = time.time()
    callback = (CheckpointCallback(save_freq=args.checkpoint_freq, save_path=str(out / "checkpoints"),
                                   name_prefix=f"dqn_{TARGET_TLS}") if args.checkpoint_freq else None)
    model.learn(total_timesteps=args.timesteps, progress_bar=False, callback=callback)
    model.save(out / f"dqn_{TARGET_TLS}")
    monitored.close()
    config = {
        "experiment_version": EXPERIMENT_VERSION, "seed": args.seed,
        "timesteps": args.timesteps, "split_file": str(args.split_file), "train_dates": sorted(train_dates),
        "training_scenarios_available": len(scenarios), "episodes": len(env.history),
        "scenarios_used": env.history, "network": str(dqn_network()),
        "arrival_window_s": ARRIVAL_WINDOW_S,
        "driver_model_vtype_attributes": driver_vtype_attributes(),
        "entry_branches": ENTRY_BRANCHES,
        "fixed_signal_program": "focused_scenario.aihub_signal_program (하루 시간축 복원)",
        "seconds": round(time.time() - started, 1),
        "dqn": {"learning_rate": 1e-3, "buffer_size": 100_000, "learning_starts": 1_000,
                "batch_size": 64, "train_freq": 4, "target_update_interval": 5_000,
                "exploration_fraction": 0.2, "exploration_final_eps": 0.05,
                "algorithm": "HoldAwareDQN: replay stores executed action; next value uses current phase "
                             "while minimum green is running"},
        "signal_constraints": {
            "max_wait_s": args.max_wait, "phase_min_green_s": constraints.phase_min,
            "crosswalks": {n: {"min_green_s": CROSSWALK_MIN_GREEN[n], "max_wait_s": c["max_wait"],
                               "phases": constraints.serving.get(n, [])} for n, c in CROSSWALKS.items()},
            "decisions": constraints.decisions, "ped_interventions": constraints.ped_interventions,
            "vehicle_interventions": constraints.vehicle_interventions},
        "environment": {"delta_time": 5, "yellow_time": args.yellow, "min_green": args.min_green, "max_green": args.max_green,
                        "observation": "phase, min-green flag, green elapsed, incoming lane density/queue, "
                                       "per-approach full-upstream zone occupancy/halting, pending entries, pedestrian waits",
                        "reward": "-0.1 x (halting vehicles in full-upstream zones + vehicles waiting to enter)",
                        "time_to_teleport": -1, "simulation_seconds": DURATION},
    }
    (out / "training_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in config.items() if k != "scenarios_used"}, ensure_ascii=False))


# ------------------------------------------------------------------ 평가

# 이름: 종류. fixed 종류는 SUMO가 신호 프로그램대로 운영하고, dqn은 5초마다 녹색 현시를 고른다(운영 제약 적용).
POLICIES = {"fixed": "fixed", "actuated": "fixed", "dqn": "dqn"}


def run_job(job: dict) -> dict:
    import xml.etree.ElementTree as ET
    from focused_rl import ALL_ZONE_EDGES, ZoneObservation, pedestrian_service, zone_queue_reward
    from focused_sumo import make_environment, parse_statistics, parse_tripinfo, sumo_output_options

    policy, date, period, eval_seed = job["policy"], job["date"], job["period"], job["eval_seed"]
    kind = POLICIES[policy]
    fixed = kind == "fixed"
    run_dir = Path(job["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    command, tripinfo, stats = sumo_output_options(run_dir)
    vehroute = (run_dir / "vehroute.xml").resolve()
    command += (f" --vehroute-output {vehroute} --vehroute-output.exit-times true"
                " --vehroute-output.write-unfinished true")
    if policy == "actuated":
        greens, mins, maxes = actuated_plan(period, date)
        add = write_actuated_additional(greens, run_dir / "tls.add.xml", "actuated", mins, maxes, AIHUB_YELLOW_S)
        command += f" --additional-files {add}"
    elif policy == "fixed":
        add = write_tls_additional(replay_phases(period, date=date), run_dir / "tls.add.xml", "aihub_corrected")
        command += f" --additional-files {add}"
    green = {"min_green": 15, "max_green": 120}
    guard = None
    if kind == "dqn":
        trained = json.loads((Path(job["model"]).parent / "training_config.json").read_text())
        if trained.get("experiment_version") != EXPERIMENT_VERSION:
            raise SystemExit(f"{job['model']}: {trained.get('experiment_version')} 모델은 지금 코드"
                             f"({EXPERIMENT_VERSION})의 관측과 맞지 않습니다. 다시 학습하세요.")
        green = {k: trained["environment"][k] for k in green}
        green["yellow_time"] = trained["environment"]["yellow_time"]
        from focused_rl import SignalConstraints
        guard = SignalConstraints(max_wait=trained["signal_constraints"]["max_wait_s"])
    env = make_environment(seed=eval_seed, fixed_ts=fixed, additional_sumo_cmd=command,
                           record_per_second_queue=True, net_file=find_network() if fixed else dqn_network(),
                           route_file=Path(job["route_file"]), simulation_duration_sec=DURATION,
                           observation_class=ZoneObservation, reward_fn=zone_queue_reward,
                           extra_queue_edges=ALL_ZONE_EDGES, **green)
    if guard is not None:
        guard.bind(env)
    model = None
    if kind == "dqn":
        from stable_baselines3 import DQN
        model = DQN.load(job["model"])
    actions = Counter()
    try:
        obs, _ = env.reset(seed=eval_seed)
        done = False
        while not done:
            action = None
            if model is not None:
                action = guard(env, int(model.predict(obs, deterministic=True)[0]))
                actions[action] += 1
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
    finally:
        env.close()
    queues = [r["zone_queue_vehicles"] for r in env.per_second_queue_records]
    # 매초 신호 상태와 대기열을 남겨 녹색 길이·전환 빈도·대기열 변화를 사후 분석한다.
    with (run_dir / "per_second.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "zone_queue", "stopline_queue", "pending", "tls_state"])
        for r in env.per_second_queue_records:
            writer.writerow([r["simulation_time_sec"], r["zone_queue_vehicles"], r["local_queue_vehicles"],
                             r.get("pending_vehicles"), r["tls_state"]])
    # 교차로 건넘 = 차량 경로에서 204820 진입 도로를 빠져나와 진출 도로로 넘어간 시각.
    movements = FocusedNetwork().movements
    crossings = []
    for vehicle in ET.parse(vehroute).getroot().iter("vehicle"):
        route = vehicle.find("route")
        if route is None or route.get("exitTimes") is None:
            continue
        edges, exits = route.get("edges").split(), route.get("exitTimes").split()
        for i, (a, b) in enumerate(zip(edges, edges[1:])):
            # 끝나기 전에 진입 도로를 떠나지 못한 차량은 exitTime이 -1이다(건너지 않음).
            if (a, b) in movements and i < len(exits) and float(exits[i]) >= 0:
                crossings.append((float(exits[i]), vehicle.get("id"), a, b))
    crossings.sort()
    with (run_dir / "crossings.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "vehicle", "from_edge", "to_edge"])
        writer.writerows(crossings)
    result = {
        "policy": policy, "date": date, "period": period, "eval_seed": eval_seed,
        "train_seed": job.get("train_seed"),
        "vehicles_in_route_file": job["vehicles"],
        "local_crossings": len(crossings),

        "local_queue_vehicle_seconds": float(sum(queues)),
        "local_mean_queue": statistics.fmean(queues) if queues else 0.0,
        "local_max_queue": max(queues, default=0),
        "local_wait_per_crossing_sec": (sum(queues) / len(crossings)) if crossings else None,
        "dqn_action_counts": dict(actions),
        "min_green": green["min_green"], "max_green": green["max_green"],
        "guard_interventions": guard.interventions if guard else 0,
        "guard_ped_interventions": guard.ped_interventions if guard else 0,
        "guard_vehicle_interventions": guard.vehicle_interventions if guard else 0,
        "guard_decisions": guard.decisions if guard else 0,
        "max_pending_vehicles": max((r.get("pending_vehicles") or 0 for r in env.per_second_queue_records), default=0),
        "pedestrian": pedestrian_service([r["tls_state"] for r in env.per_second_queue_records]),
    }
    result.update(parse_tripinfo(tripinfo))
    result.update(parse_statistics(stats))
    trips = ET.parse(tripinfo).getroot().findall("tripinfo")
    result["vehicles_waiting_over_300s"] = sum(float(t.get("waitingTime", 0)) > 300 for t in trips)
    # SUMO는 끝까지 도로에 들어오지 못한 차량을 tripinfo에 쓰지 않는다. 그 차량은 적어도
    # (종료 시각 - 예정 출발 시각)만큼 기다렸으므로 그 값을 지연에 더한다(하한).
    recorded = {t.get("id") for t in trips}
    never = [float(v.get("depart")) for v in ET.parse(job["route_file"]).getroot().iter("vehicle")
             if v.get("id") not in recorded]
    result["never_inserted"] = len(never)
    result["never_inserted_delay_lower_bound_sec"] = sum(max(0.0, DURATION - d) for d in never)
    result["vehicles_waiting_over_300s"] += sum(DURATION - d > 300 for d in never)
    result["total_time_loss_sec"] = (sum(float(t.get("timeLoss", 0)) + float(t.get("departDelay", 0)) for t in trips)
                                     + result["never_inserted_delay_lower_bound_sec"])
    result["not_arrived"] = result["loaded_vehicles"] - result["completed_tripinfo_vehicles"]
    # 차량별 핵심 값(이동·지연)만 작은 CSV로 남긴다. 방향별 차별 여부 분석에 쓴다.
    movement_of = {}
    for vehicle in ET.parse(vehroute).getroot().iter("vehicle"):
        route = vehicle.find("route")
        if route is not None:
            edges = route.get("edges").split()
            for a, b in zip(edges, edges[1:]):
                if (a, b) in movements:
                    movement_of[vehicle.get("id")] = (a, b)
    with (run_dir / "vehicles.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["vehicle", "type", "from_edge", "to_edge", "depart", "depart_delay", "arrival",
                         "waiting_time", "time_loss"])
        for t in trips:
            a, b = movement_of.get(t.get("id"), ("", ""))
            writer.writerow([t.get("id"), t.get("vType"), a, b, t.get("depart"), t.get("departDelay"),
                             t.get("arrival"), t.get("waitingTime"), t.get("timeLoss")])
    (run_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    if not job.get("keep_raw"):
        for name in ("tripinfo.xml", "vehroute.xml", "summary.xml"):
            (run_dir / name).unlink(missing_ok=True)
    return result


def evaluate(args) -> None:
    counts = load_counts(COUNTS)
    fn = FocusedNetwork()
    periods = sorted(p for (d, p) in counts if d == args.date)
    if args.periods:
        periods = [p for p in periods if p in args.periods]
    available_signals = signal_xml_files(args.date)
    out = Path(args.out) / f"eval_{args.date}"
    jobs = []
    for period in periods:
        route = out / "routes" / f"{period}.rou.xml"
        rows = counts[(args.date, period)]
        # 이미 만든 차량 파일은 다시 쓰지 않는다(실행 중인 평가가 읽는 파일을 덮어쓰지 않기 위해).
        if route.exists():
            info = {"vehicles": sum(r["count"] for r in rows)}
        else:
            tmp = route.with_suffix(".tmp.xml")
            info = write_routes(fn, rows, tmp, seed=route_seed(args.date, period))
            tmp.replace(route)
        for policy in args.policies:
            if POLICIES[policy] == "fixed" and period not in available_signals:
                continue
            for train_seed in (args.train_seeds if policy == "dqn" else [None]):
                for eval_seed in args.eval_seeds:
                    tag = f"{policy}_train{train_seed}_eval{eval_seed}" if policy == "dqn" else f"{policy}_eval{eval_seed}"
                    job = {"policy": policy, "date": args.date, "period": period, "eval_seed": eval_seed,
                           "train_seed": train_seed, "route_file": str(route), "vehicles": info["vehicles"],
                           "run_dir": str(out / "runs" / period / tag), "keep_raw": args.keep_raw}
                    if policy == "dqn":
                        job["model"] = str(Path(args.out) / "dqn" / f"seed_{train_seed}" / f"dqn_{TARGET_TLS}.zip")
                    jobs.append(job)
    todo = [j for j in jobs if not (Path(j["run_dir"]) / "metrics.json").exists()]
    print(f"평가 실행 {len(jobs)}개 중 새로 실행 {len(todo)}개 (시간대 {len(periods)}, 정책 {len(args.policies)})")
    jobs = todo
    results = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_job, job) for job in jobs]
        for n, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if n % 50 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} 완료 ({time.time() - started:.0f}초)", flush=True)
    # 이전 실행 결과까지 모두 모아 runs.csv를 다시 쓴다.
    results = [json.loads(p.read_text()) for p in sorted((out / "runs").glob("*/*/metrics.json"))]
    results.sort(key=lambda r: (r["period"], r["policy"], str(r.get("train_seed")), r["eval_seed"]))
    keys = sorted({k for r in results for k in r})
    with (out / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"저장: {out / 'runs.csv'}")


# ------------------------------------------------------------------ 보고

BANDS = [("심야 00~06시", 0, 6), ("오전 첨두 07~09시", 7, 9), ("낮 10~15시", 10, 15),
         ("오후 첨두 16~19시", 16, 19), ("저녁 20~23시", 20, 23)]
LABELS = {
    "fixed": "고정신호 — 실제 운영(보정)",
    "actuated": "감응식 — 실제 신호 현시(참고)",
    "dqn": "DQN — 실제 신호 현시",
}
ORDER = ["fixed", "actuated", "dqn"]


def _num(r, k):
    v = r.get(k)
    return float(v) if v not in (None, "", "None") else float("nan")


def _ped(r: dict) -> dict:
    """실행 하나의 보행 지표(횡단보도 둘을 합침)."""
    ped = r.get("pedestrian") or {}
    return {
        "greens": sum(c["greens"] for c in ped.values()),
        "short_greens": sum(c["short_greens"] for c in ped.values()),
        "waits_over_limit": sum(c["waits_over_limit"] for c in ped.values()),
        "max_wait_s": max((c["max_wait_s"] for c in ped.values()), default=0.0),
    }


def _day_metrics(rows: list[dict]) -> dict:
    loaded = sum(_num(r, "loaded_vehicles") for r in rows)
    veh = sum(_num(r, "completed_tripinfo_vehicles") for r in rows)
    peds = [_ped(r) for r in rows]
    greens = sum(p["greens"] for p in peds)
    return {
        "runs": len(rows),
        "complete_runs": sum(1 for r in rows if int(_num(r, "not_arrived")) == 0),
        "not_arrived": int(sum(_num(r, "not_arrived") for r in rows)),
        "waiting_over_300s": int(sum(_num(r, "vehicles_waiting_over_300s") for r in rows)),
        # 차량당 평균 지연 = (감속·정지로 늦어진 시간 + 진입 대기 + 끝까지 못 들어온 차량의 하한) ÷ 투입 차량 수
        "mean_delay_s": sum(_num(r, "total_time_loss_sec") for r in rows) / loaded,
        "mean_wait_s": sum(_num(r, "global_mean_waiting_time_sec") * _num(r, "completed_tripinfo_vehicles") for r in rows) / veh,
        "mean_queue": statistics.fmean(_num(r, "local_mean_queue") for r in rows),
        "ped_short_pct": 100 * sum(p["short_greens"] for p in peds) / greens if greens else float("nan"),
        "ped_over_limit": sum(p["waits_over_limit"] for p in peds),
        "ped_max_wait_s": max(p["max_wait_s"] for p in peds),
        "band_delay_s": {
            name: (lambda sel: sum(_num(r, "total_time_loss_sec") for r in sel)
                   / max(1.0, sum(_num(r, "loaded_vehicles") for r in sel)))(
                [r for r in rows if lo <= int(r["period"][:2]) <= hi])
            for name, lo, hi in BANDS
        },
    }


SEED_MEAN = ("mean_delay_s", "mean_wait_s", "mean_queue", "ped_short_pct")
SEED_AVG_COUNT = ("not_arrived", "waiting_over_300s", "ped_over_limit")


def summarize(rows: list[dict]) -> dict:
    """정책별 요약. DQN은 학습 seed별로 계산한 뒤 평균 ± 표준편차를 낸다.
    개수 지표는 하루 1회 실행인 다른 정책과 같은 기준이 되도록 seed당 평균이다."""
    groups = defaultdict(list)
    for r in rows:
        groups[r["policy"]].append(r)
    summary = {}
    for policy, rs in groups.items():
        if POLICIES[policy] != "dqn":
            summary[policy] = _day_metrics(rs)
            continue
        by_seed = defaultdict(list)
        for r in rs:
            by_seed[r["train_seed"]].append(r)
        seeds = {seed: _day_metrics(v) for seed, v in sorted(by_seed.items())}
        m = {"by_train_seed": seeds, "train_seeds": len(seeds)}
        for key in SEED_MEAN:
            vals = [x[key] for x in seeds.values()]
            m[key] = statistics.fmean(vals)
            m[key + "_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        for key in SEED_AVG_COUNT:
            m[key] = round(statistics.fmean(x[key] for x in seeds.values()), 1)
        m["runs"] = sum(x["runs"] for x in seeds.values())
        m["complete_runs"] = sum(x["complete_runs"] for x in seeds.values())
        m["ped_max_wait_s"] = max(x["ped_max_wait_s"] for x in seeds.values())
        m["band_delay_s"] = {b: statistics.fmean(x["band_delay_s"][b] for x in seeds.values()) for b, _, _ in BANDS}
        decisions = sum(_num(r, "guard_decisions") for r in rs)
        if decisions:
            m["override_pct"] = {k: 100 * sum(_num(r, f"guard_{k}_interventions") for r in rs) / decisions
                                 for k in ("ped", "vehicle")}
        summary[policy] = m
    return summary


def validate_demand(out: Path, policy: str = "fixed") -> dict:
    """수요 검증을 세 가지로 나눠 본다.

    1. 입력 일치: 차량 파일의 이동별 15분 교통량이 기록과 같은가(구성상 같아야 한다).
    2. 15분 안 통과: 그 15분 동안 교차로를 건넌 비율. 나머지는 대기열에 있다가 15분 뒤에 건넜다.
    3. 최종 통과: 30분 실행이 끝날 때까지 건넌 비율.
    1분·5분 단위 통과 시각은 일부러 맞추지 않는다(focused_scenario.ARRIVAL_WINDOW_S 설명 참고).
    """
    import xml.etree.ElementTree as ET
    counts = load_counts(COUNTS)
    date = out.name.split("_")[-1]
    exact = n_mov = recorded = in_window = crossed = 0
    for run in sorted((out / "runs").glob(f"*/{policy}_eval*/crossings.csv")):
        period = run.parent.parent.name
        observed = Counter()
        for r in counts[(date, period)]:
            observed[(r["from_edge"], r["to_edge"])] += r["count"]
        root = ET.parse(out / "routes" / f"{period}.rou.xml").getroot()
        movement_of_route = {}
        fn_moves = _movements()
        for route in root.findall("route"):
            edges = route.get("edges").split()
            movement_of_route[route.get("id")] = next((a, b) for a, b in zip(edges, edges[1:]) if (a, b) in fn_moves)
        generated = Counter(movement_of_route[v.get("route")] for v in root.findall("vehicle"))
        for key in set(observed) | set(generated):
            exact += int(observed[key] == generated[key])
            n_mov += 1
        recorded += sum(observed.values())
        times = [float(r["time"]) for r in csv.DictReader(run.open())]
        crossed += len(times)
        in_window += sum(t < 900 for t in times)
    return {"policy": policy, "recorded_vehicles": recorded,
            "movement_15min_input_match": f"{exact}/{n_mov}",
            "crossed_within_15min_pct": round(100 * in_window / recorded, 1) if recorded else None,
            "crossed_by_end_pct": round(100 * crossed / recorded, 2) if recorded else None}


_MOVES = None


def _movements():
    global _MOVES
    if _MOVES is None:
        _MOVES = FocusedNetwork().movements
    return _MOVES


PAIRS = [("dqn", "fixed"), ("dqn", "actuated")]


def paired(rows: list[dict]) -> list[str]:
    """같은 날짜·시간대끼리 짝지어 비교한다. DQN은 학습 seed 평균, 지연은 차량당 평균 지연."""
    delay = defaultdict(lambda: defaultdict(list))
    for r in rows:
        delay[r["policy"]][(r["date"], r["period"])].append(_num(r, "total_time_loss_sec") / _num(r, "loaded_vehicles"))
    lines = []
    for a, b in PAIRS:
        keys = sorted(set(delay[a]) & set(delay[b]))
        if not keys:
            continue
        diff = [statistics.fmean(delay[a][k]) - statistics.fmean(delay[b][k]) for k in keys]
        better = sum(d < 0 for d in diff)
        seeds = {r["train_seed"] for r in rows if r["policy"] == a}
        # 학습 seed마다 따로 이겼는지도 본다(seed 하나만 좋아서 평균이 좋아 보이는 경우 확인).
        per_seed = []
        for seed in sorted(seeds, key=str):
            d = defaultdict(list)
            for r in rows:
                if r["policy"] == a and r["train_seed"] == seed:
                    d[(r["date"], r["period"])].append(_num(r, "total_time_loss_sec") / _num(r, "loaded_vehicles"))
            ks = sorted(set(d) & set(delay[b]))
            per_seed.append(sum(statistics.fmean(d[k]) < statistics.fmean(delay[b][k]) for k in ks) / len(ks) * 100)
        lines.append(f"| {LABELS[a]} 대 {LABELS[b]} | {len(keys)} | {better / len(keys) * 100:.1f}% | "
                     f"{statistics.median(diff):+.1f} | {min(diff):+.1f} ~ {max(diff):+.1f} | "
                     + ", ".join(f"{x:.0f}%" for x in per_seed) + " |")
    if lines:
        lines = ["| 비교 | 시간대 수 | DQN이 더 나은 시간대 | 지연 차이 중앙값(초) | 지연 차이 범위(초) | 학습 seed별 더 나은 비율 |",
                 "|---|---:|---:|---:|---:|---|"] + lines
    return lines


def _fmt(m: dict, key: str, digits: int = 1) -> str:
    v = m.get(key)
    if v is None or (isinstance(v, float) and v != v):
        return "-"
    if key + "_std" in m:
        return f"{v:.{digits}f} ± {m[key + '_std']:.{digits}f}"
    return f"{v:.{digits}f}" if isinstance(v, float) else str(v)


def _table(summary: dict) -> list[str]:
    lines = ["| 정책 | 실행(전원 도착) | 차량당 평균 지연(초) | 차량당 평균 대기(초) | 평균 대기행렬(대) | 300초 이상 대기(대) | 미도착(대) | 보행 녹색 최소 미달 | 보행 대기 한도 초과(회) | 보행 최대 대기(초) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for policy in [p for p in ORDER if p in summary]:
        m = summary[policy]
        lines.append(f"| {LABELS[policy]} | {m['runs']}({m['complete_runs']}) | {_fmt(m, 'mean_delay_s')} | {_fmt(m, 'mean_wait_s')} | "
                     f"{_fmt(m, 'mean_queue')} | {_fmt(m, 'waiting_over_300s')} | {_fmt(m, 'not_arrived')} | {_fmt(m, 'ped_short_pct')}% | "
                     f"{_fmt(m, 'ped_over_limit')} | {m['ped_max_wait_s']:.0f} |")
    return lines


def report(args) -> None:
    sections = []
    all_rows = []
    out_json = {}
    for date in args.dates:
        out = Path(args.out) / f"eval_{date}"
        rows = [json.loads(p.read_text()) for p in sorted((out / "runs").glob("*/*/metrics.json"))]
        if not rows:
            continue
        all_rows += rows
        summary = summarize(rows)
        validation = validate_demand(out)
        out_json[date] = {"summary": summary, "validation": validation}
        sections += [f"## 날짜 {date}", ""] + _table(summary) + [
            "", f"수요 검증(고정신호 — 실제 운영): 이동별 15분 교통량 입력 일치 {validation['movement_15min_input_match']}, "
                f"15분 안 통과 {validation['crossed_within_15min_pct']}%, 실행 끝까지 통과 {validation['crossed_by_end_pct']}%", ""]
    combined = summarize(all_rows)
    out_json["combined"] = combined
    lines = [f"# 석천사거리(204820) 실험 결과 — 날짜 {', '.join(args.dates)}", "",
             "모든 정책이 같은 차량 파일(15분 교통량을 15분 동안 무작위 도착)을 쓴다. DQN은 학습 seed별 결과의 평균 ± 표준편차이고,",
             "개수 지표는 seed당 평균이다. DQN에는 운영 제약(보행 최소 녹색·보행 최대 대기·최대 녹색·차량 최대 대기)이 걸린다.",
             "보행 지표는 실제 보행자를 움직인 것이 아니라, 보행자가 평행 직진 신호와 같이 건넌다고 가정하고 신호만으로 잰 값이다.", "",
             "## 날짜 전체", ""] + _table(combined) + [""]
    over = [f"{LABELS[p]}: 보행 {combined[p]['override_pct']['ped']:.2f}% · 차량 {combined[p]['override_pct']['vehicle']:.2f}%"
            for p in ORDER if p in combined and "override_pct" in combined[p]]
    if over:
        lines += ["운영 제약이 정책의 선택을 바꾼 비율 — " + "; ".join(over), ""]
    pair_lines = paired(all_rows)
    if pair_lines:
        lines += ["## 같은 시간대끼리 짝지은 비교 (날짜 전체)", ""] + pair_lines + [""]
    lines += ["## 시간대별 차량당 평균 지연(초, 날짜 전체)", "", "| 정책 | " + " | ".join(b[0] for b in BANDS) + " |",
              "|---|" + "---:|" * len(BANDS)]
    for policy in [p for p in ORDER if p in combined]:
        lines.append(f"| {LABELS[policy]} | " + " | ".join(f"{combined[policy]['band_delay_s'][b[0]]:.1f}" for b in BANDS) + " |")
    lines += [""] + sections
    target = Path(args.out) / ("report.md" if len(args.dates) > 1 else f"eval_{args.dates[0]}/report.md")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    target.with_suffix(".json").write_text(json.dumps(out_json, ensure_ascii=False, indent=2))
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--timesteps", type=int, default=100_000)
    t.add_argument("--split-file", default=str(SPLIT_FILE), help="train/validation/test 날짜 JSON. train 날짜만 학습한다")
    t.add_argument("--min-green", type=int, default=15, help="보행자 횡단이 없는 현시의 최소 녹색(초)")
    t.add_argument("--max-green", type=int, default=120)
    t.add_argument("--checkpoint-freq", type=int, default=0)
    t.add_argument("--max-wait", type=float, default=120, help="차량 최대 대기 보장(초). 0이면 끔")
    t.add_argument("--yellow", type=int, default=AIHUB_YELLOW_S, help="DQN 황색시간(초). 실제 운영 신호는 대부분 4초")
    t.add_argument("--out", default=str(OUT))
    e = sub.add_parser("evaluate")
    e.add_argument("--date", required=True)
    e.add_argument("--policies", nargs="+", default=list(POLICIES), choices=list(POLICIES))
    e.add_argument("--periods", nargs="*")
    e.add_argument("--eval-seeds", nargs="+", type=int, default=[42])
    e.add_argument("--train-seeds", nargs="+", type=int, default=[42, 43, 44])
    e.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    e.add_argument("--out", default=str(OUT))
    e.add_argument("--keep-raw", action="store_true", help="tripinfo·vehroute 원본 XML을 지우지 않는다")
    r = sub.add_parser("report")
    r.add_argument("--dates", nargs="+", required=True)
    r.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    {"train": train, "evaluate": evaluate, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
