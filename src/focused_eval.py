"""석천사거리(204820) 정책 평가: 하루를 끊지 않고 고정신호·감응식·DQN을 비교한다.

- 같은 날짜의 같은 하루 차량 파일(00:00~24:00)과 같은 운전자 설정, 같은 평가 seed를 모든 정책이 쓴다.
- 하루를 끊지 않고 돌린다(+30분 배출). 앞 시간대의 대기열이 다음 시간대로 그대로 넘어간다.
- 교차로 안 충돌까지 검사한다(focused_sumo.COLLISION_OPTIONS). 충돌은 어디서 나든 모두 세고, 원인을 확인해
  등록한 지도 오류(config/collision_exceptions.json)와 모양이 같은 것만 따로 센다. 나머지는 '확인 안 된 충돌'이고,
  그런 실행은 원인을 확인하기 전에는 결과로 쓰지 않는다(judge_collisions).
- 정책
  fixed    = 그날 실제 운영 신호(보정·복원)
  actuated = 실제 신호의 순서·녹색 4개로 도는 SUMO 감응식. 녹색마다 최소(보행 46·33초, 그 밖 15초)~
             최대(그날 15분 시간대별 실제 녹색 중앙값 × 1.25 중 가장 큰 값을 보행 최대 대기 한도 안으로 줄인 값).
             우회전 전용 차로 감지기는 끔
  greedy   = 단순 제어(학습 없음). DQN과 같은 운영 규칙 안에서, 기다리는 차가 가장 많은 녹색을 켠다(QueueFirst)
  dqn      = 학습한 모델(환경 지문이 같은 모델만). 운영 규칙(focused_rl.SignalConstraints) 적용
- 도착 방식(--arrivals): random = 본 결과(15분 안 무작위 도착), platoon = 민감도(동쪽 차량이 이웃 신호 204822의
  녹색에 맞춰 무리로 옴, focused_platoon.py). 학습한 모델·잠근 환경 파일은 그대로다.
- 결과는 '평가 지문'(환경 파일 + 이 파일 + 충돌 예외 목록 + 무리 도착 파일) 이름의 폴더에 쓴다. 지문이 같을 때만 이미 있는 결과를 다시 쓴다.
  DQN 결과 폴더 이름에는 모델 파일 지문(SHA-256 앞 10자리)을 넣는다. 같은 폴더의 모델을 더 학습해 바꾸거나 다른 학습의
  같은 seed 모델을 평가해도 예전 점수를 다시 쓰지 않는다.
- 보고는 전체 값과, 신호 기록을 손본 '불확실' 시간대를 뺀 값을 같이 낸다. 날짜 × 정책(DQN은 모델마다) 가운데 하나라도
  빠지면 비교표를 만들지 않는다(빠진 날짜가 있으면 정책마다 평균 낸 날짜가 달라져 비교가 틀어진다).

사용:
  .venv/bin/python src/focused_eval.py run --dates 20220928 20220813 --policies fixed actuated greedy --workers 2
  .venv/bin/python src/focused_eval.py run --dates 20220928 --policies dqn --models results/focused_204820/models/seed_42
  .venv/bin/python src/focused_eval.py report --dates 20220928 20220813 --models results/focused_204820/models/seed_42
  (동쪽 무리 도착 민감도는 run·report에 --arrivals platoon을 붙인다)
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import statistics
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from focused_data import (ENV_FILES, PROJECT, TARGET_TLS, environment_fingerprint, find_network, fingerprint,
                          variant, variant_signature)
from focused_scenario import (AIHUB_YELLOW_S, FocusedNetwork, apply_all_red, aihub_day_timeline, aihub_signal_program,
                              aihub_unreliable_periods, day_route_file, dqn_template_network, period_of_time,
                              signal_greens, write_actuated_additional, write_tls_additional, _corrected_state,
                              aihub_states)

CACHE = PROJECT / "results/focused_204820/cache"
OUT = PROJECT / "results/focused_204820/eval"
DAY_S, TAIL_S = 86400, 1800
LONG_WAIT_S = 300
POLICIES = ("fixed", "fixedc", "actuated", "greedy", "dqn")
LABELS = {"fixed": "고정신호(원자료 그대로)", "fixedc": "고정신호(전적색 보정)", "actuated": "감응식(참고)",
          "greedy": "단순 제어(학습 없음)", "dqn": "DQN"}
ARRIVALS = {"random": "무작위 도착(본 결과)", "platoon": "동쪽 무리 도착(민감도)"}
BANDS = [("심야 00~06시", 0, 6), ("오전 첨두 07~09시", 7, 9), ("낮 10~15시", 10, 15),
         ("오후 첨두 16~19시", 16, 19), ("저녁 20~23시", 20, 23)]


COLLISION_EXCEPTIONS = PROJECT / "config/collision_exceptions.json"


def evaluation_fingerprint() -> str:
    from focused_platoon import PLATOON_FILES
    return fingerprint(ENV_FILES + ["src/focused_eval.py", "config/collision_exceptions.json"] + PLATOON_FILES)


# ------------------------------------------------------------------ 충돌 판정
def _point_on_lane(point: tuple[float, float], shape: list) -> tuple[float, float]:
    """점에서 차로 중심선(선분 하나)까지의 거리와, 중심선 위 가장 가까운 지점에서 차로 끝까지의 거리."""
    (ax, ay), (bx, by) = shape
    dx, dy = bx - ax, by - ay
    t = max(0.0, min(1.0, ((point[0] - ax) * dx + (point[1] - ay) * dy) / (dx * dx + dy * dy)))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(point[0] - cx, point[1] - cy), math.hypot(bx - cx, by - cy)


def _is_exception(event: dict, m: dict) -> bool:
    """충돌 하나가 등록된 지도 오류와 모양이 같은지. 값이 빠져 있으면 같지 않은 것으로 본다."""
    if event.get("type") != m["type"] or event.get("lane") != m["collider_lane"]:
        return False
    if not float(event.get("victimSpeed", "inf")) <= m["victim_speed_max"]:
        return False
    back = tuple(float(v) for v in event.get("victimBack", "nan,nan").split(","))
    offset, to_end = _point_on_lane(back, m["victim_back_lane_shape"])
    return offset <= m["victim_back_offset_max_m"] and to_end <= m["victim_back_to_lane_end_max_m"]


def judge_collisions(path: Path) -> dict:
    """충돌 기록을 판정한다. 등록된 지도 오류와 모양이 같은 충돌만 따로 세고, 나머지는 어디서 났든 '확인 안 된 충돌'이다."""
    from focused_sumo import parse_collisions
    exceptions = json.loads(COLLISION_EXCEPTIONS.read_text())["exceptions"]
    events = parse_collisions(path)
    known, unexplained = Counter(), []
    for e in events:
        rule = next((x["id"] for x in exceptions if _is_exception(e, x["match"])), None)
        if rule is None:
            unexplained.append(e)
        else:
            known[rule] += 1
    return {"collisions_total": len(events),
            "collisions_at_204820": sum(e.get("lane", "").startswith(f":{TARGET_TLS}_") for e in events),
            "collisions_known_map_defect": dict(known),
            "collisions_unexplained": unexplained}


# ------------------------------------------------------------------ 신호 프로그램
def fixed_day_program(date: str, corrected: bool = False) -> list[tuple[int, str]]:
    """그날 실제 운영 신호(하루 + 배출). 끝이 녹색이면 첫 현시로 되돌아가는 황색을 붙인다(경고 방지).

    corrected=True이면 전환 규칙(config/transition_rules.json)의 전적색을 황색 뒤에 넣는다(끝나는 녹색에서 뺀다).
    """
    phases = aihub_signal_program(date, 0, DAY_S + TAIL_S)
    if "y" not in phases[-1][1]:
        last, first = phases[-1][1], phases[0][1]
        phases.append((AIHUB_YELLOW_S, "".join("y" if a in "Gg" and b == "r" else a for a, b in zip(last, first))))
    if corrected:
        phases, _ = apply_all_red(phases)
    return phases


def actuated_day_plan(date: str) -> tuple[list[str], list[int], list[int]]:
    """감응식 하루 계획: 현시·최소·최대 녹색.

    최대 = 15분 시간대별 실제 녹색 중앙값 × 1.25 중 가장 큰 값. 다만 DQN과 같은 보행자 최대 대기(155·120초)를
    지키도록, 한 횡단보도가 다시 열릴 때까지의 최악 시간(다른 세 현시의 최대 녹색 + 황색 4번)이 한도를 넘으면
    그 세 현시의 '최소 위 여유분'을 같은 비율로 줄인다(최소 녹색은 그대로). 감응식에만 보행 특혜를 주지 않기 위해서다.
    """
    from focused_rl import CROSSWALKS, phase_rules
    greens = signal_greens()
    mins, serving = phase_rules(greens)
    info = aihub_states()
    segs, _ = aihub_day_timeline(date)
    unreliable = set(aihub_unreliable_periods(date))
    by_period = defaultdict(lambda: defaultdict(list))
    for s0, d, s in segs:
        state, period = _corrected_state(s, info), period_of_time(s0)
        if "y" not in state and period not in unreliable:
            by_period[period][state].append(d)
    maxes = []
    for m, g in zip(mins, greens):
        medians = [statistics.median(v[g]) for v in by_period.values() if v[g]]
        maxes.append(max(m, round(1.25 * max(medians))) if medians else m)
    maxes = [float(x) for x in maxes]
    for _ in range(50):                      # 두 횡단보도 조건을 모두 만족할 때까지 번갈아 줄인다
        changed = False
        for name, phases in serving.items():
            others = [i for i in range(len(greens)) if i not in phases]
            room = CROSSWALKS[name]["max_wait"] - AIHUB_YELLOW_S * len(greens) - sum(maxes[i] for i in others)
            if room < 0:
                slack = sum(maxes[i] - mins[i] for i in others)
                keep = max(0.0, (slack + room) / slack) if slack > 0 else 0.0
                for i in others:
                    maxes[i] = mins[i] + (maxes[i] - mins[i]) * keep
                changed = True
        if not changed:
            break
    return greens, mins, [int(x) for x in maxes]   # 내림: 한도를 넘지 않게


# ------------------------------------------------------------------ 단순 제어(학습 없음)
class QueueFirst:
    """5초마다, 규칙상 그대로 켜질 수 있는 녹색 가운데 그 녹색이 통과시키는 이동에서 기다리는 차가 가장 많은 녹색을 켠다.

    기다리는 차 = 감지 구간에서 멈춘 차(속도 0.1m/s 미만, SUMO의 '정지' 기준) + 도로에 못 들어온 차. DQN 점수와 같은
    범위의 같은 차다. 녹색이 이동을 통과시킨다 = 그 이동의 신호 연결 중 하나라도 보호 녹색(G). 같은 수면 지금 녹색을
    유지하고, 그다음은 번호가 작은 녹색. 학습도, 조정할 숫자도 없다. 운영 규칙은 DQN과 같은 SignalConstraints를 쓴다.
    차가 어느 이동인지는 차의 경로로 안다(DQN은 같은 범위를 진입로·차로별로 본다).
    """

    def __init__(self, movement_of: dict[str, tuple[str, str] | None], greens: list[str]):
        from focused_rl import ALL_ZONE_EDGES, MOVEMENT_LINKS
        self.movement_of = movement_of
        self.served = [{m for m, links in MOVEMENT_LINKS.items() if any(g[l] == "G" for l in links)} for g in greens]
        self.edges = ALL_ZONE_EDGES

    @staticmethod
    def movements(route_file: Path) -> dict[str, tuple[str, str] | None]:
        from focused_rl import MOVEMENT_LINKS
        root = ET.parse(route_file).getroot()
        edges_of = {r.get("id"): r.get("edges").split() for r in root.iter("route")}
        return {v.get("id"): next(((a, b) for a, b in zip(e, e[1:]) if (a, b) in MOVEMENT_LINKS), None)
                for v in root.iter("vehicle") for e in [edges_of[v.get("route")]]}

    def waiting(self, sumo) -> Counter:
        n = Counter()
        for edge in self.edges:
            for v in sumo.edge.getLastStepVehicleIDs(edge):
                if sumo.vehicle.getSpeed(v) < 0.1:
                    n[self.movement_of.get(v)] += 1
        for v in sumo.simulation.getPendingVehicles():
            n[self.movement_of.get(v)] += 1
        return n

    def __call__(self, env, allowed: list[int]) -> int:
        if len(allowed) == 1:
            return allowed[0]
        current = env.traffic_signals[TARGET_TLS].green_phase
        n = self.waiting(env.sumo)
        score = {p: sum(n[m] for m in self.served[p]) for p in allowed}
        return max(allowed, key=lambda p: (score[p], p == current, -p))


# ------------------------------------------------------------------ 실행 계획
def model_identity(model_dir) -> dict:
    """DQN 모델의 신원: 학습 seed, 학습량, 모델 파일 SHA-256."""
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "training_config.json").read_text())
    digest = hashlib.sha256((model_dir / f"dqn_{TARGET_TLS}.zip").read_bytes()).hexdigest()
    return {"model": str(model_dir), "train_seed": config["seed"], "model_timesteps": config["timesteps"],
            "model_sha256": digest}


def planned_runs(out: Path, dates: list[str], policies: list[str], models: list[str], eval_seed: int,
                 arrivals: str = "random") -> list[dict]:
    """비교에 들어갈 실행 목록: 날짜 × 정책(DQN은 모델마다). 결과 폴더 = 날짜/정책(DQN은 seed와 모델 지문)_평가 seed
    (무리 도착이면 끝에 _platoon)."""
    if "dqn" in policies and not models:
        raise SystemExit("DQN 평가·보고에는 --models(학습 결과 폴더들)가 필요합니다.")
    identities = [model_identity(m) for m in models] if "dqn" in policies else []
    seeds = [i["train_seed"] for i in identities]
    if len(seeds) != len(set(seeds)):
        raise SystemExit(f"학습 seed가 같은 모델이 둘 이상입니다({sorted(seeds)}). 한 비교에는 seed마다 모델 하나만 넣으세요.")
    runs = []
    for date in dates:
        for policy in policies:
            for ident in (identities if policy == "dqn" else [{}]):
                tag = f"dqn_seed{ident['train_seed']}_{ident['model_sha256'][:10]}" if ident else policy
                runs.append({"date": date, "policy": policy, "eval_seed": eval_seed, "arrivals": arrivals, **ident,
                             "run_dir": str(out / date / f"{tag}_eval{eval_seed}{'' if arrivals == 'random' else '_' + arrivals}")})
    return runs


# ------------------------------------------------------------------ 하루 실행
def run_day(job: dict) -> dict:
    from focused_rl import (ALL_ZONE_EDGES, VEHICLE_MAX_WAIT_S, SignalConstraints, ZoneObservation, pedestrian_service,
                            zone_queue_reward)
    from focused_sumo import make_environment, parse_statistics, sumo_output_options

    date, policy, eval_seed = job["date"], job["policy"], job["eval_seed"]
    run_dir = Path(job["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    arrivals = job.get("arrivals", "random")
    if arrivals == "platoon":
        from focused_platoon import platoon_route_file
        route_file, period_of = platoon_route_file(date, CACHE / "routes")
    else:
        route_file, period_of = day_route_file(date, CACHE / "routes")
    command, tripinfo, stats = sumo_output_options(run_dir)
    if policy in ("fixed", "fixedc"):
        add = write_tls_additional(fixed_day_program(date, corrected=policy == "fixedc"), run_dir / "tls.add.xml",
                                   "aihub_corrected")
        command += f" --additional-files {add}"
    elif policy == "actuated":
        greens, mins, maxes = actuated_day_plan(date)
        add = write_actuated_additional(greens, run_dir / "tls.add.xml", "actuated", mins, maxes, AIHUB_YELLOW_S)
        command += f" --additional-files {add}"
    green = {"min_green": 15, "max_green": 120, "yellow_time": AIHUB_YELLOW_S}
    guard = model = controller = None
    if policy == "dqn":
        trained = json.loads((Path(job["model"]) / "training_config.json").read_text())
        if trained.get("environment_fingerprint") != environment_fingerprint():
            raise SystemExit(f"{job['model']}: 모델의 환경 지문 {trained.get('environment_fingerprint')}이 지금 환경 "
                             f"{environment_fingerprint()}과 다릅니다. 지금 환경으로 다시 학습하세요.")
        green = {k: trained["environment"][k] for k in ("min_green", "max_green")}
        green["yellow_time"] = trained["environment"]["yellow_time"]
        if model_identity(job["model"])["model_sha256"] != job["model_sha256"]:
            raise SystemExit(f"{job['model']}: 평가를 준비한 뒤 모델 파일이 바뀌었습니다. 다시 실행하세요.")
        from focused_rl import hold_aware_dqn_class      # 그대로 실행될 행동 안에서만 고르는 선택(predict)
        model = hold_aware_dqn_class().load(Path(job["model"]) / f"dqn_{TARGET_TLS}")
    chooses = policy in ("dqn", "greedy")        # 5초마다 녹색을 고르는 정책(DQN용 틀 지도 + 운영 규칙)
    env = make_environment(seed=eval_seed, fixed_ts=not chooses, additional_sumo_cmd=command,
                           record_per_second_queue=True,
                           net_file=dqn_template_network(CACHE) if chooses else find_network(),
                           route_file=route_file, simulation_duration_sec=DAY_S + TAIL_S,
                           observation_class=ZoneObservation, reward_fn=zone_queue_reward,
                           extra_queue_edges=ALL_ZONE_EDGES, **green)
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
        w.writerow(["time", "zone_queue", "pending", "tls_state"])
        for r in records:
            w.writerow([int(r["simulation_time_sec"]), r["zone_queue_vehicles"], r["pending_vehicles"], r["tls_state"]])
    result = {"date": date, "policy": policy, "arrivals": arrivals, "eval_seed": eval_seed, "train_seed": job.get("train_seed"),
              "model": job.get("model"), "model_sha256": job.get("model_sha256"),
              "model_timesteps": job.get("model_timesteps"), "evaluation_fingerprint": job["fingerprint"],
              "variant": variant_signature(),
              "environment_fingerprint": environment_fingerprint(), "seconds": round(time.time() - started, 1),
              "action_counts": dict(actions)}
    result.update(vehicle_metrics(route_file, period_of, tripinfo, date, run_dir))
    result.update(parse_statistics(stats))
    result.update(judge_collisions(run_dir / "collisions.xml"))
    result["pedestrian"] = pedestrian_service([r["tls_state"] for r in records])
    result["max_pending_vehicles"] = max((r["pending_vehicles"] or 0 for r in records), default=0)
    result["mean_zone_queue"] = statistics.fmean(r["zone_queue_vehicles"] for r in records) if records else 0.0
    if guard is not None:
        result["guard"] = {"decisions": guard.decisions, "rule_decided": dict(guard.rule_decided),
                           "changed_by_rules": {"ped": guard.ped_interventions, "vehicle": guard.vehicle_interventions,
                                                "max_green": guard.max_green_switches}}   # 0이어야 한다
    if policy == "actuated":
        result["actuated_plan"] = dict(zip(("greens", "min_green_s", "max_green_s"), actuated_day_plan(date)))
    tmp = run_dir / "metrics.json.tmp"      # 끊겨도 반쯤 쓴 결과가 '끝난 평가'로 보이지 않게 이름을 마지막에 바꾼다
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    tmp.replace(run_dir / "metrics.json")
    (run_dir / "tripinfo.xml").unlink(missing_ok=True)
    return result


def vehicle_metrics(route_file: Path, period_of: dict, tripinfo: Path, date: str, run_dir: Path) -> dict:
    """차량별 지연·누적 대기를 모아 전체 값과 '불확실 시간대 제외' 값을 낸다.

    지연 = 속도가 줄어 잃은 시간 + 진입 전 대기. 끝까지 도로에 못 들어온 차는 (끝 시각 − 예정 출발)을 하한으로 넣는다.
    누적 대기 = 진입 전 대기 + 도로에서 멈춰 있던 시간 합. 300초를 넘으면 '오래 기다린 차'.
    """
    fn = FocusedNetwork()
    root = ET.parse(route_file).getroot()
    edges_of = {r.get("id"): r.get("edges").split() for r in root.iter("route")}
    movement_of, depart_of = {}, {}
    for v in root.iter("vehicle"):
        e = edges_of[v.get("route")]
        mv = next(((a, b) for a, b in zip(e, e[1:]) if (a, b) in fn.movements), None)
        movement_of[v.get("id")] = f"{mv[0]}->{mv[1]} ({fn.movements[mv]})"
        depart_of[v.get("id")] = float(v.get("depart"))
    rows, seen = [], set()
    for t in ET.parse(tripinfo).getroot().iter("tripinfo"):
        vid = t.get("id")
        seen.add(vid)
        dd, wt, tl = float(t.get("departDelay", 0)), float(t.get("waitingTime", 0)), float(t.get("timeLoss", 0))
        rows.append({"vehicle": vid, "period": period_of[vid], "movement": movement_of[vid], "delay": tl + dd,
                     "wait": wt + dd, "arrived": float(t.get("arrival", -1)) >= 0})
    for vid, depart in depart_of.items():
        if vid not in seen:
            lower = max(0.0, DAY_S + TAIL_S - depart)
            rows.append({"vehicle": vid, "period": period_of[vid], "movement": movement_of[vid], "delay": lower,
                         "wait": lower, "arrived": False})
    with gzip.open(run_dir / "vehicles.csv.gz", "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    unreliable = set(aihub_unreliable_periods(date))

    def summary(sel: list[dict]) -> dict:
        by_mv = defaultdict(list)
        for r in sel:
            by_mv[r["movement"]].append(r)
        return {
            "vehicles": len(sel),
            "mean_delay_s": statistics.fmean(r["delay"] for r in sel),
            "long_waits": sum(r["wait"] > LONG_WAIT_S for r in sel),
            "not_arrived": sum(not r["arrived"] for r in sel),
            "by_movement": {k: {"vehicles": len(v), "mean_delay_s": round(statistics.fmean(r["delay"] for r in v), 1),
                                "long_wait_pct": round(100 * sum(r["wait"] > LONG_WAIT_S for r in v) / len(v), 2)}
                            for k, v in sorted(by_mv.items())},
            "by_band_delay_s": {name: (statistics.fmean(r["delay"] for r in band) if band else None)
                                for name, lo, hi in BANDS
                                for band in [[r for r in sel if lo <= int(r["period"][:2]) <= hi]]},
            "by_period_delay_s": {p: statistics.fmean(r["delay"] for r in v)
                                  for p, v in sorted(_group(sel, "period").items())},
        }

    return {"never_inserted": len(depart_of) - len(seen), "unreliable_periods": sorted(unreliable),
            "all": summary(rows), "reliable": summary([r for r in rows if r["period"] not in unreliable])}


def _group(rows: list[dict], key: str) -> dict:
    out = defaultdict(list)
    for r in rows:
        out[r[key]].append(r)
    return out


def run(args) -> None:
    if "fixedc" in args.policies and not variant()["all_red"]:
        args.policies = [p for p in args.policies if p != "fixedc"]     # 규칙이 없으면 원자료 고정신호와 같다
        print("전환 규칙이 없어(none) 전적색 보정 고정신호는 원자료 고정신호와 같습니다. 돌리지 않습니다.", flush=True)
    fp = evaluation_fingerprint()
    out = Path(args.out) / fp
    jobs = [{**j, "fingerprint": fp}
            for j in planned_runs(out, args.dates, args.policies, args.models, args.eval_seed, args.arrivals)]
    for date in args.dates:             # 하루 차량 파일은 먼저 순서대로 만든다(동시 생성 충돌 방지).
        day_route_file(date, CACHE / "routes")
        if args.arrivals == "platoon":
            from focused_platoon import platoon_route_file
            platoon_route_file(date, CACHE / "routes")
    dqn_template_network(CACHE)
    todo = [j for j in jobs if not (Path(j["run_dir"]) / "metrics.json").exists()]
    print(f"평가 지문 {fp}, {ARRIVALS[args.arrivals]}: 실행 {len(jobs)}개 중 새로 {len(todo)}개 → {out}")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_day, j) for j in todo]
        for n, future in enumerate(as_completed(futures), 1):
            r = future.result()
            print(f"  {n}/{len(todo)} {r['date']} {r['policy']}{'' if r['train_seed'] is None else ' seed ' + str(r['train_seed'])}"
                  f"{'' if r['arrivals'] == 'random' else ' (' + r['arrivals'] + ')'}: "
                  f"차량당 지연 {r['all']['mean_delay_s']:.1f}초, 300초 초과 {r['all']['long_waits']}대, "
                  f"확인 안 된 충돌 {len(r['collisions_unexplained'])}건(지도 오류 겹침 {sum(r['collisions_known_map_defect'].values())}건), "
                  f"{r['seconds']:.0f}초", flush=True)


# ------------------------------------------------------------------ 보고
def _load(runs: list[dict]) -> list[dict]:
    """계획한 실행의 결과를 모두 읽는다. 하나라도 빠졌거나 다른 모델의 결과면 멈춘다."""
    missing = [r for r in runs if not (Path(r["run_dir"]) / "metrics.json").exists()]
    if missing:
        listing = "\n".join(f"  - {r['date']} {LABELS[r['policy']]}"
                            + (f" 학습 seed {r['train_seed']}(모델 {r['model_sha256'][:10]})" if r["policy"] == "dqn" else "")
                            for r in missing)
        raise SystemExit(f"빠진 평가가 {len(missing)}개 있어 비교표를 만들지 않습니다. 빠진 날짜가 있으면 정책마다 평균 낸 "
                         f"날짜가 달라져 비교가 틀어집니다. 모두 평가한 뒤 다시 보고하세요.\n{listing}")
    rows = [json.loads((Path(r["run_dir"]) / "metrics.json").read_text()) for r in runs]
    for r, row in zip(runs, rows):
        if row.get("model_sha256") != r.get("model_sha256"):
            raise SystemExit(f"{r['run_dir']}: 결과의 모델 지문이 지금 모델 파일과 다릅니다.")
    return rows


def _summarize(rows: list[dict], part: str) -> dict:
    """정책별 요약. DQN은 학습 seed별로 계산한 뒤 평균 ± 표준편차."""
    def day_values(rs: list[dict]) -> dict:
        n = sum(r[part]["vehicles"] for r in rs)
        return {"mean_delay_s": sum(r[part]["mean_delay_s"] * r[part]["vehicles"] for r in rs) / n,
                "long_waits": sum(r[part]["long_waits"] for r in rs), "vehicles": n,
                "not_arrived": sum(r[part]["not_arrived"] for r in rs),
                "collisions_unexplained": sum(len(r["collisions_unexplained"]) for r in rs),
                "collisions_known": sum(sum(r["collisions_known_map_defect"].values()) for r in rs),
                "emergency_braking": sum(r["emergency_braking_count"] for r in rs),
                "ped_over_limit": sum(c["waits_over_limit"] for r in rs for c in r["pedestrian"].values()),
                "ped_short": sum(c["short_greens"] for r in rs for c in r["pedestrian"].values())}
    out = {}
    for policy in POLICIES:
        rs = [r for r in rows if r["policy"] == policy]
        if not rs:
            continue
        if policy != "dqn":
            out[policy] = day_values(rs)
            continue
        seeds = defaultdict(list)
        for r in rs:
            seeds[r["train_seed"]].append(r)
        per_seed = {s: day_values(v) for s, v in sorted(seeds.items())}
        vals = [v["mean_delay_s"] for v in per_seed.values()]
        out[policy] = {**{k: statistics.fmean(v[k] for v in per_seed.values()) for k in per_seed[next(iter(per_seed))]},
                       "mean_delay_s_std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                       "train_seeds": len(per_seed), "by_train_seed": per_seed}
    return out


def report(args) -> None:
    out = Path(args.out) / (args.fingerprint or evaluation_fingerprint())
    base = ["fixed"] + (["fixedc"] if variant()["all_red"] else []) + ["actuated", "greedy"]
    runs = planned_runs(out, args.dates, base + (["dqn"] if args.models else []), args.models,
                        args.eval_seed, args.arrivals)
    rows = _load(runs)
    lines = [f"# 석천사거리 평가 — 날짜 {', '.join(args.dates)} (평가 지문 {out.name}, 평가 seed {args.eval_seed})", "",
             f"도착 방식: {ARRIVALS[args.arrivals]}", "",
             "하루를 끊지 않고 돌린 결과. 모든 정책이 같은 차량·운전자·평가 seed를 쓰고, 모든 정책이 모든 날짜를 평가했다. "
             "DQN은 학습 seed별 평균 ± 표준편차.", ""]
    models = {(r["train_seed"], r["model_timesteps"], r["model_sha256"]) for r in runs if r["policy"] == "dqn"}
    if models:
        lines += ["DQN 모델: " + ", ".join(f"학습 seed {s}({t:,}단계, 모델 지문 {h[:10]})" for s, t, h in sorted(models)), ""]
    bad = [r for r in rows if r["collisions_unexplained"]]
    lines += ["## 충돌 판정", ""]
    if bad:
        lines += ["**확인 안 된 충돌이 있다. 원인을 확인하기 전에는 아래 표로 결론을 내지 않는다.**", ""]
        for r in bad:
            who = LABELS[r["policy"]] + ("" if r["train_seed"] is None else f" 학습 seed {r['train_seed']}")
            lines += [f"- {r['date']} {who}: {e['time']}초 {e['type']} {e['lane']} 가해 {e['collider']}({e['colliderType']}, "
                      f"{e['colliderSpeed']}m/s) 피해 {e['victim']}({e['victimType']}, {e['victimSpeed']}m/s)"
                      for e in r["collisions_unexplained"]]
    else:
        lines.append("확인 안 된 충돌 없음. 등록된 지도 오류 겹침(config/collision_exceptions.json)만 표에 따로 센다.")
    lines += ["", "충돌·급제동·보행 지표는 두 표 모두 하루 전체 값이다.", ""]
    num = lambda x: f"{x:.0f}" if float(x).is_integer() else f"{x:.1f}"
    for part, title in (("all", "전체"), ("reliable", "신호 기록 불확실 시간대 제외")):
        s = _summarize(rows, part)
        lines += [f"## {title}", "", "| 정책 | 차량당 평균 지연(초) | 300초 넘게 기다린 차(대, 날짜 합) | 도착 못 한 차 | 확인 안 된 충돌 | "
                  "지도 오류 겹침(예외) | 급제동 | 보행 대기 한도 초과 | 보행 녹색 최소 미달 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for policy, m in s.items():
            delay = f"{m['mean_delay_s']:.1f}" + (f" ± {m['mean_delay_s_std']:.1f}" if "mean_delay_s_std" in m else "")
            lines.append(f"| {LABELS[policy]} | {delay} | " + " | ".join(num(m[k]) for k in (
                "long_waits", "not_arrived", "collisions_unexplained", "collisions_known", "emergency_braking",
                "ped_over_limit", "ped_short")) + " |")
        lines.append("")
    target = out / ("report_" + "_".join(args.dates) + ("" if args.arrivals == "random" else "_" + args.arrivals) + ".md")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"저장: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--dates", nargs="+", required=True)
    r.add_argument("--policies", nargs="+", default=list(POLICIES), choices=POLICIES)
    r.add_argument("--models", nargs="*", default=[], help="학습 결과 폴더들(seed_42 등). dqn 정책에 필요")
    r.add_argument("--eval-seed", type=int, default=42)
    r.add_argument("--arrivals", choices=tuple(ARRIVALS), default="random", help="random = 본 결과, platoon = 동쪽 무리 도착(민감도)")
    r.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    r.add_argument("--out", default=str(OUT))
    p = sub.add_parser("report")
    p.add_argument("--dates", nargs="+", required=True)
    p.add_argument("--models", nargs="*", default=[], help="비교에 넣을 DQN 학습 결과 폴더들. 생략하면 고정신호·감응식만")
    p.add_argument("--eval-seed", type=int, default=42)
    p.add_argument("--arrivals", choices=tuple(ARRIVALS), default="random")
    p.add_argument("--fingerprint", help="생략하면 지금 코드의 평가 지문")
    p.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    if args.command == "run" and "dqn" in args.policies and not args.models:
        parser.error("dqn 정책에는 --models가 필요합니다")
    {"run": run, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
