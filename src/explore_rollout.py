"""[추가 탐색] 0단계 — SUMO 안에서 후보 녹색을 직접 굴려 보면 P0보다 좋은 선택이 있나(진행 기록 119절). 지문 밖, 학습 없음.

복원: 새 SUMO 프로세스가 저장 상태(`--save-state.rng`, 자릿수 20)를 읽고 시작한다(같은 SUMO 안에서 되돌리면 어긋남, 119절).
파이썬 쪽 기록(신호 객체의 숫자 값, 운영 규칙 SignalConstraints의 기록 전부, 신호 상태 문자열)도 함께 저장·복원한다.
표본 미래(119.2): 저장 상태 안의 '출발 시각이 t0 뒤인 실제 차'를 지우고, 최근 900초에 출발한 차의 경로·차종을 다시 뽑아 넣는다.
비용(119.4): 탐색 동안 쌓인 지연(모든 차의 시간 손실 증가 + 진입 대기 1초당 1초) + 끝에 남은 정체 × 75초.

명령(Colab 37번 칸):
  FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240 python src/explore_rollout.py check --out <Drive>/explore_rollout/check.json
  FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240 python src/explore_rollout.py sample --work /content/rollout --out <Drive>/explore_rollout
  FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240 python src/explore_rollout.py compare --out <Drive>/explore_rollout --work /content/rollout --workers 8
  python src/explore_rollout.py summary --out <Drive>/explore_rollout
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import pickle
import random
import shutil
import statistics
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

P0_VARIANT, P0_HORIZON_S, P0_CODE = "p0", 120, "9440fe1d3869"
# --route-steps 0: 차량 파일을 처음에 한꺼번에 읽는다. 기본값(200초씩)이면 저장 상태에 미리 읽힌 미래 차의 마지막 한 대가
# 복원 뒤 파일을 다시 읽는 경계에서 사라졌다(119절 복원 검사, 08-11 21:30). 한꺼번에 읽으면 복원은 파일을 다시 읽지 않는다.
STATE_OPTS = "--save-state.rng true --save-state.precision 20 --route-steps 0"
SUMO_SEED = 42
H_MAIN, H_SENS = 300, 600
TERMINAL_S = 75.0                    # P0의 끝값과 같음
HISTORY_S = 900
SELECT_SEEDS = tuple(range(1001, 1009))
CONFIRM_SEEDS = tuple(range(2001, 2009))
EPISODES = [(d, p) for d in ("20220811", "20220823", "20221030") for p in ("0730", "1230", "1730", "2130")]
CHECK_EPISODES = [("20220811", "0730"), ("20220811", "2130")]
CHECK_OFFSETS_S = (300, 1500)
CHECK_TRACE_S = 300
LEFTS = {"left_east": ("477706392_02", "477706883_02"), "left_south": ("477706883_01", "477707013_01")}
LEFT_Q, CONGESTED_Q, LIGHT_Q, LEFT_MIN = 0.90, 0.80, 0.40, 3
CLASS_N, PICK_SEED = 16, 20261023
CLASSES = ("left", "congested", "light")
SAMPLE_PREFIX = "smp"


def plan_record() -> dict:
    return {"p0": [P0_VARIANT, P0_HORIZON_S, P0_CODE], "state_opts": STATE_OPTS, "sumo_seed": SUMO_SEED,
            "horizons_s": [H_MAIN, H_SENS], "terminal_s": TERMINAL_S, "history_s": HISTORY_S,
            "select_seeds": list(SELECT_SEEDS), "confirm_seeds": list(CONFIRM_SEEDS), "episodes": EPISODES,
            "classes": {"left_q": LEFT_Q, "left_min": LEFT_MIN, "congested_q": CONGESTED_Q, "light_q": LIGHT_Q,
                        "per_class": CLASS_N, "pick_seed": PICK_SEED}}


def code_fingerprint() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


# ------------------------------------------------------------------ 순수 계산(SUMO 없음)
def movement_of_edges(edges: list[str]) -> tuple[str, str] | None:
    """경로에서 석천사거리를 지나는 이동(단순 제어 QueueFirst.movements와 같은 규칙)."""
    from focused_rl import MOVEMENT_LINKS
    return next(((a, b) for a, b in zip(edges, edges[1:]) if (a, b) in MOVEMENT_LINKS), None)


def read_routes(route_file: Path) -> tuple[dict, dict]:
    """차량 파일 → ({차량: (출발 시각, 경로 도로 목록, 차종)}, {경로 ID: 도로 목록})."""
    root = ET.parse(route_file).getroot()
    routes = {r.get("id"): r.get("edges").split() for r in root.iter("route")}
    vehicles = {v.get("id"): (float(v.get("depart")), routes[v.get("route")], v.get("type")) for v in root.iter("vehicle")}
    return vehicles, routes


def history(vehicles: dict, t0: float) -> list[tuple[tuple[str, ...], str]]:
    """119.2: 최근 HISTORY_S초 동안 출발한 차의 (경로, 차종). t0 뒤에 출발하는 차는 들어가지 않는다."""
    return [(tuple(edges), vt) for vid, (dep, edges, vt) in sorted(vehicles.items(), key=lambda kv: (kv[1][0], kv[0]))
            if t0 - HISTORY_S <= dep < t0]


def sample_future(hist: list, t0: float, seed: int, horizon: int = H_SENS) -> list[tuple[str, tuple[str, ...], str, float]]:
    """표본 미래 시나리오 하나: 대수 ~ 포아송(최근 대수 × horizon ÷ HISTORY_S), 시각은 (t0, t0 + horizon]에 고르게,
    경로·차종은 최근 차 가운데 하나를 고르게 뽑아 그대로. [(차량 ID, 경로, 차종, 출발 시각)]"""
    if not hist:
        return []
    g = np.random.default_rng(seed)
    n = int(g.poisson(len(hist) * horizon / HISTORY_S))
    times = np.sort(t0 + horizon * (1.0 - g.random(n)))
    picks = g.integers(0, len(hist), n)
    return [(f"{SAMPLE_PREFIX}{seed}_{k}", hist[i][0], hist[i][1], max(t0 + 0.01, round(float(t), 2)))
            for k, (t, i) in enumerate(zip(times, picks))]


def future_ids(state_file: Path, vehicles: dict, t0: float) -> list[str]:
    """저장 상태 안의 차 가운데 출발 시각이 t0 뒤인 실제 차(엿봄이 되므로 지운다)."""
    root = ET.parse(state_file).getroot()
    return sorted(c.get("id") for c in root if c.tag == "vehicle" and vehicles.get(c.get("id"), (-1,))[0] > t0)


def types_only_file(route_file: Path) -> Path:
    """차종만 있는 차량 파일(표본 미래 실행용: 실제 미래 차가 파일에서 새로 읽히지 않게)."""
    out = route_file.with_name(route_file.stem.replace(".rou", "") + ".types.rou.xml")
    if not out.exists():
        root = ET.Element("routes")
        root.extend(ET.parse(route_file).getroot().findall("vType"))
        ET.ElementTree(root).write(out, encoding="utf-8", xml_declaration=True)
    return out


PRIMITIVE = (int, float, bool, str, type(None), np.integer, np.floating)


def neutralize_idle_program(xml: Path, tls: str) -> int:
    """저장 상태에서 쓰이지 않는(active="0") 석천사거리 신호 프로그램의 현시 번호를 0으로 바꾼다. 몇 개 바꿨는지 돌려준다.
    SUMO-RL은 시작할 때 기본 프로그램을 녹색+황색 현시로 바꿔 쓰고 실제 신호는 'online' 프로그램으로 낸다. 그래서 저장 상태의
    기본 프로그램 현시 번호(예: 10)가 원래 지도의 기본 프로그램에는 없어, 새 SUMO가 읽기를 거부했다(119절). 쓰이지 않는 프로그램이라
    움직임에는 영향이 없다(복원 검사로 확인)."""
    import re
    text = xml.read_text()
    pattern = re.compile(rf'(<tlLogic id="{re.escape(tls)}" programID="[^"]*" phase=")(\d+)("[^>]*active="0")')
    new, n = pattern.subn(r"\g<1>0\g<3>", text)
    if n:
        xml.write_text(new)
    return n


def signal_values(ts) -> dict:
    """신호 객체의 숫자·문자 값 전부(현시, 황색 중인지, 경과, 다음 판단 시각, 이번 전환 길이 등)."""
    return {k: copy.deepcopy(v) for k, v in ts.__dict__.items() if isinstance(v, PRIMITIVE)}


def quantile(xs: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(xs, dtype=float), q)) if xs else math.nan


def classify(points: list[dict]) -> dict:
    """119.3 분류 기준(전체 선택 시점으로 정함)과 시점별 분류(좌회전 정체 → 혼잡 → 비혼잡 순, 나머지는 None)."""
    cut = {"W80": quantile([p["W"] for p in points], CONGESTED_Q), "W40": quantile([p["W"] for p in points], LIGHT_Q)}
    for name in LEFTS:
        cut[f"{name}90"] = quantile([p[name] for p in points], LEFT_Q)
    for p in points:
        left = any(not p[f"{n}_served"] and p[n] >= max(cut[f"{n}90"], LEFT_MIN) for n in LEFTS)
        p["class"] = "left" if left else "congested" if p["W"] >= cut["W80"] else "light" if p["W"] <= cut["W40"] else None
    return cut


def pick(points: list[dict], n: int = CLASS_N, seed: int = PICK_SEED) -> list[dict]:
    """분류마다 n개: 섞은 뒤 에피소드를 돌아가며 하나씩(모자라면 있는 만큼)."""
    rng = random.Random(seed)
    chosen = []
    for cls in CLASSES:
        pool = [p for p in points if p.get("class") == cls]
        rng.shuffle(pool)
        by_ep: dict = {}
        for p in pool:
            by_ep.setdefault(p["episode"], []).append(p)
        order = sorted(by_ep)
        rng.shuffle(order)
        got = []
        while len(got) < n and any(by_ep[e] for e in order):
            for e in order:
                if by_ep[e] and len(got) < n:
                    got.append(by_ep[e].pop(0))
        chosen += got
    return chosen


T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.16, 2.145, 2.131,
        2.12, 2.11, 2.101, 2.093, 2.086, 2.08, 2.074, 2.069, 2.064, 2.06, 2.056, 2.052, 2.048, 2.045, 2.042]


def t975(df: int) -> float:
    """t 분포 97.5% 값(자유도 30까지 표, 그 위는 1.96 + 2.4/df, 오차 0.003 미만)."""
    return T975[df - 1] if df <= 30 else 1.96 + 2.4 / df


def t_interval(xs: list[float]) -> tuple[float, float, float]:
    """평균과 95% t 구간."""
    m = statistics.fmean(xs)
    if len(xs) < 2:
        return m, math.nan, math.nan
    half = t975(len(xs) - 1) * statistics.stdev(xs) / math.sqrt(len(xs))
    return m, m - half, m + half


def cost_check(runs: dict, cands: list[int]) -> dict:
    """121절: 정확한 지연(delay)과 매초 증가분 셈(delay_increment)의 차이. 실행마다, 그리고 같은 시나리오에서 두 후보의
    차이(후보 비교에 실제로 쓰는 값)마다. runs = {(후보, seed): {표시 시점: 값}}."""
    out = {}
    for h in (H_MAIN, H_SENS):
        keys = [k for k in runs if h in runs[k]]
        diff = [runs[k][h]["delay"] - runs[k][h]["delay_increment"] for k in keys]
        rel = [abs(d) / max(1.0, runs[k][h]["delay"]) for k, d in zip(keys, diff)]
        pairs = []
        if len(cands) > 1:
            for seed in sorted({sd for _, sd in keys}):
                a, b = runs.get((cands[0], seed), {}).get(h), runs.get((cands[1], seed), {}).get(h)
                if a and b:
                    exact, inc = a["delay"] - b["delay"], a["delay_increment"] - b["delay_increment"]
                    pairs.append({"seed": seed, "exact_diff": round(exact, 2), "increment_diff": round(inc, 2),
                                  "error": round(inc - exact, 2)})
        out[str(h)] = {"exact_minus_increment": [round(x, 2) for x in diff], "max_rel": round(max(rel), 4) if rel else None,
                       "pairs": pairs}
    return out


def _paired(costs: dict, a: int, b: int, key: str) -> list[float]:
    """확인용 시나리오마다 (a의 비용 − b의 비용). 같은 시나리오끼리 짝지어 뺀다."""
    return [costs[a][s][key] - costs[b][s][key] for s in CONFIRM_SEEDS]


def _parts(costs: dict, p0: int, best: int, key: str) -> dict:
    """확인 이득을 실제 지연 몫과 끝에 남은 정체 벌점(× 75초) 몫으로 나눈다(121절). 값이 없으면 넣지 않는다."""
    out = {}
    for part in ("delay", "terminal"):
        k = f"{key}_{part}"
        if best == p0:
            out[f"{part}_gain"] = 0.0
        elif all(k in costs[c][s] for c in (p0, best) for s in CONFIRM_SEEDS):
            out[f"{part}_gain"] = round(statistics.fmean(_paired(costs, p0, best, k)), 2)
    return out


def judge_state(costs: dict, p0_action: int, key: str) -> dict:
    """119.4: costs = {후보: {seed: {key: 비용}}}. 고르는 용 평균 최소 후보(같으면 P0) → 확인용에서 P0 − 고른 후보(짝)."""
    sel = {c: statistics.fmean(costs[c][s][key] for s in SELECT_SEEDS) for c in costs}
    best_val = min(sel.values())
    best = p0_action if sel[p0_action] <= best_val + 1e-9 else min(c for c, v in sel.items() if v <= best_val + 1e-9)
    out = {"best": best, "differs": best != p0_action, "select_gain": round(sel[p0_action] - sel[best], 2),
           "select_means": {str(c): round(v, 1) for c, v in sel.items()}}
    out.update(evaluate_choice(costs, p0_action, best, key))
    return out


def evaluate_choice(costs: dict, p0: int, best: int, key: str) -> dict:
    """고른 후보 하나를 확인용 시나리오에서 잰다(key 비용). 121절: 300초에서 고른 후보를 600초 비용으로 다시 잴 때도 쓴다."""
    if best == p0:
        m = lo = hi = sd = 0.0
    else:
        diffs = _paired(costs, p0, best, key)
        m, lo, hi = t_interval(diffs)
        sd = statistics.stdev(diffs)
    return {"confirm_gain": round(m, 2), "confirm_ci": [round(lo, 2), round(hi, 2)], "confirm_sd": round(sd, 2),
            "worse": best != p0 and m < 0, "significantly_worse": best != p0 and hi < 0,
            "significantly_better": best != p0 and lo > 0, **_parts(costs, p0, best, key)}


def read_class(rows: list[dict]) -> dict:
    gains = [r["confirm_gain"] for r in rows]
    m, lo, hi = t_interval(gains) if gains else (math.nan, math.nan, math.nan)
    mean = lambda k: round(statistics.fmean(r[k] for r in rows), 2) if rows and all(k in r for r in rows) else None
    return {"states": len(rows), "differs": sum(r.get("differs", r["confirm_gain"] != 0) for r in rows), "mean_confirm_gain": round(m, 2),
            "ci95": [round(lo, 2), round(hi, 2)], "mean_select_gain": mean("select_gain"),
            "mean_delay_gain": mean("delay_gain"), "mean_terminal_gain": mean("terminal_gain"),
            "significantly_better": sum(r["significantly_better"] for r in rows), "worse": sum(r["worse"] for r in rows),
            "significantly_worse": sum(r["significantly_worse"] for r in rows)}


def by_group(states: list[dict], field) -> dict:
    """121절: 날짜·에피소드별로 같은 표(상태 48개는 3일·12개 에피소드에서 나와 서로 독립이 아니다)."""
    groups: dict = {}
    for s_ in states:
        groups.setdefault(field(s_), []).append(s_)
    return {g: {"main": read_class([x["main"] for x in rows]), "main_at_600": read_class([x["main_at_600"] for x in rows])}
            for g, rows in sorted(groups.items())}


def reading(classes: dict) -> str:
    """119.5 읽는 법: 혼잡이나 좌회전 정체에서 확인 이득 95% 구간 > 0이고 유의하게 나빠진 상태 ≤ 10%면 '이득 있음'."""
    hit = [c for c in ("congested", "left") if classes.get(c, {}).get("states")
           and classes[c]["ci95"][0] > 0 and classes[c]["significantly_worse"] <= 0.10 * classes[c]["states"]]
    return f"이득 있음({', '.join(hit)})" if hit else "이득이 보이지 않음"


# ------------------------------------------------------------------ SUMO
def _env(route_file: Path, extra: str, duration: int):
    import focused_train as ft
    from focused_rl import ZoneObservation, zone_queue_reward
    from focused_scenario import dqn_template_network
    from focused_sumo import make_environment
    for p in (route_file, *[x for x in extra.split() if x.endswith(".xml")]):
        if " " in str(p):
            raise SystemExit(f"SUMO에 넘기는 경로에 공백이 있으면 안 된다: {p}")
    return make_environment(seed=SUMO_SEED, net_file=dqn_template_network(ft.CACHE), route_file=route_file,
                            simulation_duration_sec=duration, observation_class=ZoneObservation, reward_fn=zone_queue_reward,
                            min_green=15, max_green=120, yellow_time=ft.AIHUB_YELLOW_S, additional_sumo_cmd=f"{STATE_OPTS} {extra}".strip())


def _p0():
    import explore_pred as ep
    if ep.code_fingerprint() != P0_CODE:
        raise SystemExit(f"예측 제어 코드 {ep.code_fingerprint()}가 100절의 {P0_CODE}와 다르다.")
    return ep.controller_class(P0_VARIANT, P0_HORIZON_S)


def waiting_counts(env, movement_of: dict) -> Counter:
    from explore_choice import waiting_by_movement
    return waiting_by_movement(env.sumo, movement_of)


def served(ts, movement: tuple[str, str]) -> bool:
    from focused_rl import MOVEMENT_LINKS
    state = ts.green_phases[ts.green_phase].state
    return not ts.is_yellow and any(state[l] == "G" for l in MOVEMENT_LINKS[movement])


class Episode:
    """학습 환경과 같은 에피소드(실제 신호 30분 → 넘겨받은 뒤 60분)를 상태 저장 옵션을 켠 SUMO로 열고 P0가 신호를 고른다."""

    def __init__(self, date: str, period: str, workdir: Path):
        import focused_train as ft
        from focused_data import COUNTS, TARGET_TLS, load_counts
        from focused_rl import SignalConstraints
        workdir.mkdir(parents=True, exist_ok=True)
        self.date, self.period = date, period
        self.route = workdir / f"{date}_{period}.rou.xml"
        self.start = ft.warm_start(date, period, load_counts(COUNTS), self.route)
        self.env = _env(self.route, "", self.start["handover_s"] + ft.CONTROL_S)
        self.env.warmup_states, self.env.warmup_handover = self.start["warmup_states"], self.start["handover_state"]
        self.cons = SignalConstraints().bind(self.env)
        self.cons.reset(self.start["last_served"])
        self.env.reset(seed=SUMO_SEED)
        self.ts = self.env.traffic_signals[TARGET_TLS]
        self.vehicles, _ = read_routes(self.route)
        P0 = _p0()
        self.movement_of = P0.movements(self.route)
        self.ctl = P0(self.movement_of, [p.state for p in self.ts.green_phases])
        self.handover = float(self.env.sim_step)
        self.done = False

    def decide(self) -> tuple[list[int], int]:
        allowed = self.cons.allowed(self.env)
        return allowed, int(self.ctl(self.env, allowed))

    def apply(self, action: int) -> int:
        a = self.cons(self.env, action)
        _, _, terminated, truncated, _ = self.env.step(a)
        self.done = terminated or truncated
        return a

    def snapshot(self, stem: Path, allowed: list[int], p0_action: int) -> dict:
        """판단 직전(운영 규칙을 씌우기 전) 상태를 저장한다. stem.xml(SUMO) + stem.pkl(파이썬 기록)."""
        from focused_data import TARGET_TLS
        stem.parent.mkdir(parents=True, exist_ok=True)
        t0 = float(self.env.sim_step)
        xml = stem.with_suffix(".xml")
        self.env.sumo.simulation.saveState(str(xml))
        patched = neutralize_idle_program(xml, TARGET_TLS)
        snap = {"date": self.date, "period": self.period, "t0": t0, "handover": self.handover, "route": self.route.name,
                "idle_program_patched": patched, "allowed": list(allowed), "p0_action": int(p0_action), "signal": signal_values(self.ts),
                "constraints": copy.deepcopy(self.cons.__dict__), "env_vehicles": copy.deepcopy(self.env.vehicles),
                "tls": self.env.sumo.trafficlight.getRedYellowGreenState(TARGET_TLS),
                "future_ids": future_ids(xml, self.vehicles, t0), "code": code_fingerprint()}
        stem.with_suffix(".pkl").write_bytes(pickle.dumps(snap))
        return snap

    def close(self) -> None:
        self.env.close()


def restore(stem: Path, route_dir: Path, future: str, seed: int | None = None, horizon: int = H_SENS,
            tripinfo: Path | None = None):
    """새 SUMO가 저장 상태를 읽고 시작한다. 차량 파일은 차종만 있는 것을 쓴다(저장 상태에 모든 차가 들어 있다, --route-steps 0).
    future='real': 저장 상태의 실제 미래 차를 그대로 둔다(복원 검사). 'sampled': 출발 시각이 t0 뒤인 실제 차를 모두 지우고
    표본 미래 시나리오 seed를 넣는다. tripinfo를 주면 SUMO가 도착한 차마다 최종 시간 손실을 그 파일에 적는다(정확한 지연 셈).
    돌려주는 값: (환경, 운영 규칙, 신호, P0, 기록, 넣은 차 목록, 차량 파일 내용)."""
    from focused_data import TARGET_TLS
    from focused_rl import SignalConstraints
    snap = pickle.loads(stem.with_suffix(".pkl").read_bytes())
    t0 = snap["t0"]
    vehicles, _ = read_routes(route_dir / snap["route"])
    extra = f"--load-state {stem.with_suffix('.xml')} --begin {int(t0)}"
    if tripinfo is not None:
        extra += f" --tripinfo-output {tripinfo}"
    env = _env(types_only_file(route_dir / snap["route"]), extra, int(t0) + horizon + 10)
    cons = SignalConstraints().bind(env)
    start = env._start_simulation

    def start_at_t0():                      # SUMO-RL은 시작 시각을 '-b t' 한 덩어리로 넘기므로 --begin을 쓰고,
        env.begin_time = 0                  # 신호 객체의 첫 판단 시각만 t0로 둔다
        start()
        env.begin_time = int(t0)
    env._start_simulation = start_at_t0
    env.reset(seed=SUMO_SEED)
    ts = env.traffic_signals[TARGET_TLS]
    for k, v in snap["signal"].items():
        setattr(ts, k, copy.deepcopy(v))
    cons.__dict__.update(copy.deepcopy(snap["constraints"]))
    env.vehicles = copy.deepcopy(snap["env_vehicles"])
    env.sumo.trafficlight.setRedYellowGreenState(TARGET_TLS, snap["tls"])
    movement_of = {vid: movement_of_edges(edges) for vid, (_, edges, _) in vehicles.items()}
    injected = []
    if future == "sampled":
        for vid in snap["future_ids"]:                 # 아직 출발하지 않은 실제 미래 차(SUMO 시간 진행 전에 지움)
            env.sumo.vehicle.remove(vid)
        injected = sample_future(history(vehicles, t0), t0, seed, horizon)
        made: dict = {}
        for vid, edges, vt, dep in injected:
            if edges not in made:
                made[edges] = f"{SAMPLE_PREFIX}_r{len(made)}"
                env.sumo.route.add(made[edges], list(edges))
            env.sumo.vehicle.add(vid, made[edges], typeID=vt, depart=f"{dep:.2f}", departLane="best", departSpeed="max")
            movement_of[vid] = movement_of_edges(list(edges))
    P0 = _p0()
    ctl = P0(movement_of, [p.state for p in ts.green_phases])
    return env, cons, ts, ctl, snap, injected, vehicles


class DelayMeter:
    """표시 시점(t0 + h)마다 그때까지 쌓인 지연(평가 지표와 같은 정의: 시간 손실 + 진입 대기)과 남은 정체를 적는다.

    정확한 셈(delay, 121절): finish()에서 SUMO 차량별 기록(tripinfo)을 읽어 마친다.
      - 시간 손실: 표시 시점에 도로 위인 차는 (그때 값 − t0 값), 그 전에 도착한 차는 (도착 때 최종 값 − t0 값).
        t0에 도로 위가 아니던 차의 t0 값은 0이다.
      - 진입 대기: t0 뒤 표시 시점까지 들어온 차는 (실제 진입 − max(원래 출발, t0)), 아직 못 들어온 차는
        (표시 시점 − max(원래 출발, t0)). 원래 출발은 차량 파일(실제 차)·표본 미래(넣은 차)의 출발 시각이다.
    매초 증가분을 더한 셈(delay_increment)도 함께 적는다(120절 처음 방식: 도착한 차의 마지막 1초 증가분이 빠지고,
    진입 대기를 초 단위로 셈). 엿봄 확인: 지나간 차 가운데 실제 차량 파일에서 출발 시각이 t0 뒤인 차를 센다(0이어야 함)."""

    def __init__(self, env, t0: float, vehicles: dict, injected=(), marks=(H_MAIN, H_SENS)):
        import traci.constants as tc
        self.tc, self.env, self.t0, self.vehicles, self.marks = tc, env, t0, vehicles, set(marks)
        self.desired = {v: dep for v, (dep, _, _) in vehicles.items()}
        self.desired.update({vid: dep for vid, _, _, dep in injected})
        s = env.sumo
        self.tl0 = {}
        for v in s.vehicle.getIDList():
            self.tl0[v] = s.vehicle.getTimeLoss(v)
            s.vehicle.subscribe(v, [tc.VAR_TIMELOSS])
        self.prev = dict(self.tl0)
        self.departed: dict[str, float] = {}
        self.increment, self.at, self.leaks = 0.0, {}, set()
        self._marked: dict = {}

    def step(self) -> None:
        from focused_rl import ALL_ZONE_EDGES
        tc, s = self.tc, self.env.sumo
        for v in s.simulation.getDepartedIDList():
            s.vehicle.subscribe(v, [tc.VAR_TIMELOSS])
            self.departed[v] = s.vehicle.getDeparture(v)
            if not v.startswith(SAMPLE_PREFIX) and self.vehicles.get(v, (-1,))[0] > self.t0:
                self.leaks.add(v)
        now_tl = {}
        for v, d in s.vehicle.getAllSubscriptionResults().items():
            if tc.VAR_TIMELOSS in d:
                tl = d[tc.VAR_TIMELOSS]
                self.increment += tl - self.prev.get(v, 0.0)
                self.prev[v] = tl
                now_tl[v] = tl
        pending = s.simulation.getPendingVehicles()
        self.increment += len(pending)
        for v in pending:
            if not v.startswith(SAMPLE_PREFIX) and self.vehicles.get(v, (-1,))[0] > self.t0:
                self.leaks.add(v)
        h = int(round(s.simulation.getTime() - self.t0))
        if h in self.marks:
            residual = sum(s.edge.getLastStepHaltingNumber(e) for e in ALL_ZONE_EDGES) + len(pending)
            # 이 초에 도착한 차는 구독 결과에 마지막 값이 남아 있다. 도로 위 차는 지금 실제로 도로 위인 차만 센다(121절:
            # 정확히 표시 시점에 도착한 차를 도로 위로도, 도착으로도 두 번 센 일이 있었다).
            on_road = set(s.vehicle.getIDList())
            self._marked[h] = {"tl": {v: tl for v, tl in now_tl.items() if v in on_road}, "pending": list(pending),
                               "departed": dict(self.departed),
                               "residual": residual, "increment": self.increment}

    def finish(self, tripinfo: Path) -> dict:
        """SUMO를 닫은 뒤 차량별 기록으로 표시 시점마다 정확한 지연을 계산해 self.at에 둔다."""
        arrived = {t.get("id"): (float(t.get("arrival")), float(t.get("timeLoss")))
                   for t in ET.parse(tripinfo).getroot().iter("tripinfo")} if Path(tripinfo).exists() else {}
        for h, m in sorted(self._marked.items()):
            M = self.t0 + h
            done = {v for v, (arr, _) in arrived.items() if 0 <= arr <= M}
            loss = sum(tl - self.tl0.get(v, 0.0) for v, tl in m["tl"].items() if v not in done)
            loss += sum(tl - self.tl0.get(v, 0.0) for v, (arr, tl) in arrived.items() if 0 <= arr <= M)   # 끝나지 않은 차(-1) 제외
            wait = sum(dep - max(self.desired.get(v, self.t0), self.t0) for v, dep in m["departed"].items())
            wait += sum(M - max(self.desired.get(v, self.t0), self.t0) for v in m["pending"])
            delay = loss + wait
            self.at[h] = {"delay": round(delay, 3), "time_loss": round(loss, 3), "depart_wait": round(wait, 3),
                          "delay_increment": round(m["increment"], 3), "residual": m["residual"],
                          "terminal": round(TERMINAL_S * m["residual"], 3), "cost": round(delay + TERMINAL_S * m["residual"], 3)}
        return self.at


def drive(env, cons, ctl, first: int, t0: float, horizon: int, on_second=None) -> tuple[list[int], int]:
    """첫 행동 뒤 P0로 horizon초. 매초 on_second()를 부른다. (실행된 행동들, 규칙이 바꾼 횟수)"""
    original = env._sumo_step

    def step():
        original()
        if on_second:
            on_second()
    env._sumo_step = step
    actions, changed = [], 0
    try:
        a = cons(env, first)
        changed += a != first
        while env.sim_step < t0 + horizon:
            env.step(a)
            actions.append(int(a))
            allowed = cons.allowed(env)
            want = ctl(env, allowed)
            a = cons(env, want)
            changed += a != want
    finally:
        env._sumo_step = original
    return actions, changed


def measured(stem: Path, route_dir: Path, candidate: int, seed: int, horizon: int = H_SENS, on_second=None):
    """표본 미래 seed로 복원해 candidate → P0로 horizon초 돌리고 정확한 지연을 잰다. (DelayMeter, 규칙이 바꾼 횟수, 넣은 차 수, 시작 준비 초)"""
    t = time.perf_counter()
    trip = Path(tempfile.mkstemp(prefix="trip_", suffix=".xml")[1])
    try:
        env, cons, ts, ctl, snap, injected, vehicles = restore(stem, route_dir, "sampled", seed, horizon, tripinfo=trip)
        setup = time.perf_counter() - t
        try:
            meter = DelayMeter(env, snap["t0"], vehicles, injected)

            def each():
                meter.step()
                if on_second:
                    on_second(env)
            _, changed = drive(env, cons, ctl, candidate, snap["t0"], horizon, each)
        finally:
            env.close()
        meter.finish(trip)
    finally:
        trip.unlink(missing_ok=True)
    return meter, changed, len(injected), setup


def rollout(stem: Path, route_dir: Path, candidate: int, seed: int, horizon: int = H_SENS) -> dict:
    t = time.perf_counter()
    meter, changed, injected, setup = measured(stem, route_dir, candidate, seed, horizon)
    return {"state": stem.name, "candidate": candidate, "seed": seed, "marks": {str(k): v for k, v in sorted(meter.at.items())},
            "injected": injected, "leaks": len(meter.leaks), "changed_by_rules": changed, "code": code_fingerprint(),
            "setup_s": round(setup, 2), "seconds": round(time.perf_counter() - t, 2)}


# ------------------------------------------------------------------ 119.1 복원 검사
def trace_fingerprint(sumo) -> str:
    from focused_data import TARGET_TLS
    rows = [(v, sumo.vehicle.getLaneID(v), sumo.vehicle.getLanePosition(v), sumo.vehicle.getSpeed(v))
            for v in sorted(sumo.vehicle.getIDList())]
    pend = tuple(sorted(sumo.simulation.getPendingVehicles()))
    return hashlib.sha256(repr((sumo.simulation.getTime(), sumo.trafficlight.getRedYellowGreenState(TARGET_TLS), rows, pend))
                          .encode()).hexdigest()[:16]


def _recorded(env, cons, ctl, first, t0, secs) -> tuple[list[str], list[int]]:
    trace: list[str] = []
    actions, _ = drive(env, cons, ctl, first, t0, secs, lambda: trace.append(trace_fingerprint(env.sumo)))
    return trace, actions


def check(args) -> None:
    work = Path(args.work) if args.work else Path(tempfile.mkdtemp(prefix="rollout_check_"))
    snaps, originals = [], {}
    for date, period in CHECK_EPISODES:
        ep = Episode(date, period, work / "routes")
        targets = [ep.handover + off for off in CHECK_OFFSETS_S]
        try:
            while targets and not ep.done:
                allowed, a = ep.decide()
                if ep.env.sim_step >= targets[0] and len(allowed) > 1:
                    stem = work / "states" / f"check_{date}_{period}_{int(ep.env.sim_step)}"
                    snap = ep.snapshot(stem, allowed, a)
                    snap["served_now"] = ep.cons._served_now(ep.ts, snap["t0"])
                    trace, actions = _recorded(ep.env, ep.cons, ep.ctl, a, snap["t0"], CHECK_TRACE_S)
                    originals[stem.name] = (trace, actions, snap)
                    snaps.append(stem)
                    targets.pop(0)
                    continue                         # _recorded가 이미 300초 나아갔다
                ep.apply(a)
        finally:
            ep.close()
    report, problems = {"plan": plan_record(), "code": code_fingerprint(), "states": []}, []
    for stem in snaps:
        trace_o, act_o, snap = originals[stem.name]
        row = {"state": stem.name, "t0": snap["t0"], "allowed": snap["allowed"], "p0_action": snap["p0_action"],
               "future_vehicles_in_state": len(snap["future_ids"]), "rng_in_state": "<rngState" in stem.with_suffix(".xml").read_text(),
               "idle_program_patched": snap["idle_program_patched"]}
        # 1·2. 실제 미래로 복원 → 같은 행동
        env, cons, ts, ctl, _, _, _ = restore(stem, work / "routes", "real", horizon=CHECK_TRACE_S)
        try:
            row["allowed_after_restore"] = cons.allowed(env)
            row["served_after_restore_same"] = cons._served_now(ts, snap["t0"]) == snap["served_now"]
            trace_r, act_r = _recorded(env, cons, ctl, snap["p0_action"], snap["t0"], CHECK_TRACE_S)
        finally:
            env.close()
        row["real_same_seconds"] = sum(a == b for a, b in zip(trace_o, trace_r))
        row["real_seconds"] = len(trace_o)
        row["real_first_diff"] = next((i for i, (a, b) in enumerate(zip(trace_o, trace_r)) if a != b), None)
        row["real_actions_same"] = act_o == act_r
        # 3·4. 표본 미래: 같은 시나리오 두 번, 다른 시나리오 한 번(정확한 지연도 잰다)
        sampled = []
        for seed in (SELECT_SEEDS[0], SELECT_SEEDS[0], SELECT_SEEDS[1]):
            trace: list[str] = []
            meter, _, injected, _ = measured(stem, work / "routes", snap["p0_action"], seed, CHECK_TRACE_S,
                                             lambda env: trace.append(trace_fingerprint(env.sumo)))
            sampled.append((trace, len(meter.leaks), injected, meter.at))
        row["sampled_same_seed_same_seconds"] = sum(a == b for a, b in zip(sampled[0][0], sampled[1][0]))
        row["sampled_other_seed_same_seconds"] = sum(a == b for a, b in zip(sampled[0][0], sampled[2][0]))
        row["sampled_seconds"] = len(sampled[0][0])
        row["sampled_leaks"] = sum(s[1] for s in sampled)
        row["injected"] = [s[2] for s in sampled]
        row["sampled_cost_300"] = [s[3].get(H_MAIN, {}).get("cost") for s in sampled]
        # 5. 지연 셈 대조(121절): 후보 둘(P0 선택 + 다른 하나) × 시나리오 둘, 600초. 정확한 셈 대 매초 증가분 셈
        cands = [snap["p0_action"]] + [c for c in snap["allowed"] if c != snap["p0_action"]][:1]
        runs = {}
        for c in cands:
            for seed in SELECT_SEEDS[:2]:
                runs[(c, seed)] = measured(stem, work / "routes", c, seed, H_SENS)[0].at
        row["cost_check"] = cost_check(runs, cands)
        report["states"].append(row)
        name = stem.name
        if row["real_same_seconds"] != row["real_seconds"] or not row["real_actions_same"]:
            problems.append(f"{name}: 원본 대 복원 {row['real_same_seconds']}/{row['real_seconds']}초 같음, 행동 같음 {row['real_actions_same']}")
        if row["allowed_after_restore"] != snap["allowed"] or not row["served_after_restore_same"]:
            problems.append(f"{name}: 복원 직후 허용 녹색 {row['allowed_after_restore']} ≠ {snap['allowed']} 또는 보행 기록 다름")
        if row["sampled_same_seed_same_seconds"] != row["sampled_seconds"]:
            problems.append(f"{name}: 표본 미래 같은 시나리오 두 번이 {row['sampled_same_seed_same_seconds']}/{row['sampled_seconds']}초만 같음")
        if row["sampled_other_seed_same_seconds"] == row["sampled_seconds"]:
            problems.append(f"{name}: 다른 시나리오인데 매초 같음(표본 미래가 교통을 바꾸지 않음)")
        if row["sampled_leaks"]:
            problems.append(f"{name}: 표본 미래 실행에 실제 미래 차 {row['sampled_leaks']}대")
        if not row["rng_in_state"]:
            problems.append(f"{name}: 저장 파일에 난수 상태 없음")
    report.update(problems=problems, ok=not problems and len(snaps) == len(CHECK_EPISODES) * len(CHECK_OFFSETS_S))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if not report["ok"]:
        raise SystemExit("복원 검사 실패: 후보 비교를 하지 않는다.")
    print("복원 검사 통과", flush=True)


# ------------------------------------------------------------------ 119.3 선택 시점 고르기
def _features(ep: Episode, allowed: list[int], a: int) -> dict:
    t0 = float(ep.env.sim_step)
    counts = waiting_counts(ep.env, ep.movement_of)
    p = {"state": f"{ep.date}_{ep.period}_{int(t0)}", "episode": f"{ep.date}_{ep.period}", "t0": t0,
         "since_handover": t0 - ep.handover, "W": int(sum(counts.values())), "allowed": list(allowed), "p0_action": int(a),
         "green": int(ep.ts.green_phase)}
    for name, m in LEFTS.items():
        p[name] = int(counts[m])
        p[f"{name}_served"] = served(ep.ts, m)
    return p


def _sample_episode(job: tuple) -> dict:
    """1차: 선택 시점마다 특징만 적는다. 2차(chosen이 있을 때): 같은 에피소드를 다시 돌려 고른 시점에서만 상태를 저장하고,
    다시 돌린 특징이 1차와 같은지 확인한다(다르면 그 시점을 '어긋남'으로 돌려준다)."""
    date, period, work, chosen = job
    work = Path(work)
    ep = Episode(date, period, work / "routes")
    points, mismatched, saved = [], [], []
    entered = sum(1 for dep, _, _ in ep.vehicles.values() if ep.handover <= dep < ep.handover + 3600)
    want = {p["t0"]: p for p in chosen or []}
    try:
        while not ep.done:
            allowed, a = ep.decide()
            if len(allowed) > 1:
                p = _features(ep, allowed, a)
                points.append(p)
                if p["t0"] in want:
                    keys = ("W", "allowed", "p0_action", "green", *LEFTS, *[f"{n}_served" for n in LEFTS])
                    if any(p[k] != want[p["t0"]][k] for k in keys):
                        mismatched.append(p["state"])
                    ep.snapshot(work / "states" / p["state"], allowed, a)
                    saved.append(p["state"])
            ep.apply(a)
    finally:
        ep.close()
    return {"episode": f"{date}_{period}", "points": points, "vehicles_per_hour": entered,
            "choice_points_per_hour": len(points), "mismatched": mismatched, "saved": saved}


def sample(args) -> None:
    from concurrent.futures import ProcessPoolExecutor
    import focused_train as ft
    from focused_scenario import dqn_template_network
    work, out = Path(args.work), Path(args.out)
    done = out / "sample.json"
    if done.exists():
        print(f"이미 있음 → 건너뜀: {done}")
        return
    dqn_template_network(ft.CACHE)                 # 틀 지도는 먼저 한 번 만든다(동시 생성 충돌 방지)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        episodes = list(pool.map(_sample_episode, [(d, p, str(work), None) for d, p in EPISODES]))
    points = [p for e in episodes for p in e["points"]]
    cut = classify(points)
    chosen = pick(points)
    print(f"1차: 선택 시점 {len(points)}개, 고른 상태 {dict(Counter(p['class'] for p in chosen))}", flush=True)
    jobs = [(d, p, str(work), [c for c in chosen if c["episode"] == f"{d}_{p}"]) for d, p in EPISODES]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        again = list(pool.map(_sample_episode, [j for j in jobs if j[3]]))
    mismatched = [s for e in again for s in e["mismatched"]]
    saved = {s for e in again for s in e["saved"]}
    lost = [c["state"] for c in chosen if c["state"] not in saved]
    if mismatched or lost:
        raise SystemExit(f"2차에서 1차와 다른 시점 {mismatched[:5]}, 저장 못 한 시점 {lost[:5]} → 멈춤(같은 에피소드가 같게 돌지 않음)")
    (out / "states").mkdir(parents=True, exist_ok=True)
    (out / "routes").mkdir(parents=True, exist_ok=True)
    for p in chosen:
        for suffix in (".xml", ".pkl"):
            shutil.copy(work / "states" / f"{p['state']}{suffix}", out / "states" / f"{p['state']}{suffix}")
    for e in episodes:
        shutil.copy(work / "routes" / f"{e['episode']}.rou.xml", out / "routes" / f"{e['episode']}.rou.xml")
    rec = {"plan": plan_record(), "code": code_fingerprint(), "cuts": cut, "rerun_same": True,
           "population": {c: sum(p["class"] == c for p in points) for c in (*CLASSES, None)} | {"all": len(points)},
           "episodes": {e["episode"]: {"vehicles_per_hour": e["vehicles_per_hour"], "choice_points_per_hour": e["choice_points_per_hour"]}
                        for e in episodes},
           "chosen": chosen, "points": points}
    rec["population"] = {str(k): v for k, v in rec["population"].items()}
    done.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    print(json.dumps({k: rec[k] for k in ("cuts", "population")}, ensure_ascii=False), "| 고른 상태",
          dict(Counter(p["class"] for p in chosen)), flush=True)


# ------------------------------------------------------------------ 119.4 후보 비교
def _rollout_job(job: tuple[str, str, int, int]) -> dict:
    stem, route_dir, candidate, seed = job
    return rollout(Path(stem), Path(route_dir), candidate, seed)


def compare(args) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    out, work = Path(args.out), Path(args.work)
    rec = json.loads((out / "sample.json").read_text())
    local = work / "chosen"                      # SUMO에 넘기는 경로(공백 없는 로컬)
    for sub in ("states", "routes"):
        (local / sub).mkdir(parents=True, exist_ok=True)
        for f in (out / sub).iterdir():
            if not (local / sub / f.name).exists():
                shutil.copy(f, local / sub / f.name)
    import focused_train as ft
    from focused_scenario import dqn_template_network
    dqn_template_network(ft.CACHE)                 # 동시 생성 충돌 방지: 틀 지도와 차종 파일은 먼저 만든다
    for f in sorted((local / "routes").glob("*.rou.xml")):
        if ".types." not in f.name:
            types_only_file(f)
    results = out / "rollouts.jsonl"
    seen = set()
    if results.exists():
        for line in results.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("code") != code_fingerprint():
                raise SystemExit(f"{results}에 다른 코드({r.get('code')})의 탐색 결과가 있다 → 새 결과 폴더를 쓴다.")
            seen.add((r["state"], r["candidate"], r["seed"]))
    jobs = [(str(local / "states" / p["state"]), str(local / "routes"), c, s) for p in rec["chosen"]
            for c in p["allowed"] for s in (*SELECT_SEEDS, *CONFIRM_SEEDS) if (p["state"], c, s) not in seen]
    print(f"탐색 {len(jobs)}번(이미 끝난 {len(seen)}번 건너뜀), 동시에 {args.workers}개", flush=True)
    started = time.time()
    pool = ProcessPoolExecutor(max_workers=args.workers)
    try:
        with results.open("a") as f:
            futures = [pool.submit(_rollout_job, j) for j in jobs]
            for n, fut in enumerate(as_completed(futures), 1):
                r = fut.result()                       # 하나라도 실패하면 남은 탐색을 취소하고 멈춘다
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                if n % 100 == 0 or n == len(futures):
                    print(f"  {n}/{len(futures)} ({time.time() - started:.0f}초)", flush=True)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


# ------------------------------------------------------------------ 119.5 정리
INFO_TABLE = [
    ("도로 위 차 하나하나의 위치·속도·차로", "안다", "모른다(차로별 밀도·대기, 진입로별 차·정지 비율만)"),
    ("차마다 가는 방향(이동)", "안다(경로)", "이동별 기다리는 차 수로만(114절 8칸)"),
    ("차종·운전자 값(속도 성향 등)", "안다", "모른다"),
    ("진입 대기 중인 차", "안다(차마다 경로까지)", "진입로별 비율과 이동별 수"),
    ("SUMO 난수(운전자 무작위가 이어지는 상태)", "안다(저장 시점부터)", "모른다"),
    ("신호 상태·운영 규칙 기록", "안다", "현시·경과·최소 녹색 충족·허용 녹색·보행 대기 비율"),
    ("최근 900초에 들어온 차(경로·차종)", "안다(표본 미래를 만드는 데 씀)", "모른다(지금 한 장면만 봄)"),
    ("앞으로 올 실제 차(시각·경로·차종)", "모른다(지움, 표본으로 바꿈)", "모른다"),
    ("교통이 어떻게 움직이는지(시뮬레이터)", "그대로 굴려 본다", "경험으로 배운 Q값뿐"),
]


def summary(args) -> None:
    out = Path(args.out)
    rec = json.loads((out / "sample.json").read_text())
    rows = [json.loads(l) for l in (out / "rollouts.jsonl").read_text().splitlines() if l.strip()]
    codes = sorted({r.get("code") for r in rows})
    if len(codes) != 1:
        raise SystemExit(f"서로 다른 코드의 탐색 결과가 섞였다: {codes} → 새 결과 폴더에서 다시 돈다.")
    by = {}
    for r in rows:
        m = r["marks"]
        by.setdefault(r["state"], {}).setdefault(r["candidate"], {})[r["seed"]] = {
            "main": m[str(H_MAIN)]["cost"], "main_delay": m[str(H_MAIN)]["delay"], "main_terminal": m[str(H_MAIN)]["terminal"],
            "sens": m[str(H_SENS)]["cost"], "sens_delay": m[str(H_SENS)]["delay"], "sens_terminal": m[str(H_SENS)]["terminal"]}
    states, missing = [], []
    for p in rec["chosen"]:
        costs = by.get(p["state"], {})
        need = [(c, s_) for c in p["allowed"] for s_ in (*SELECT_SEEDS, *CONFIRM_SEEDS) if s_ not in costs.get(c, {})]
        if need:
            missing.append((p["state"], len(need)))
            continue
        row = {"state": p["state"], "class": p["class"], "episode": p["episode"], "date": p["episode"][:8], "W": p["W"],
               "allowed": p["allowed"], "p0_action": p["p0_action"]}
        row["main"] = judge_state(costs, p["p0_action"], "main")
        row["main_at_600"] = {"best": row["main"]["best"], "differs": row["main"]["differs"],
                              **evaluate_choice(costs, p["p0_action"], row["main"]["best"], "sens")}
        row["sens"] = judge_state(costs, p["p0_action"], "sens")
        states.append(row)
    if missing:
        raise SystemExit(f"탐색이 빠진 상태가 있다: {missing[:5]} …")
    result = {"plan": plan_record(), "code": code_fingerprint(), "rollout_code": codes[0], "states": states}
    for key in ("main", "main_at_600", "sens"):
        cls = {c: read_class([s_[key] for s_ in states if s_["class"] == c]) for c in CLASSES}
        cls["all"] = read_class([s_[key] for s_ in states])
        result[key] = {"classes": cls, "reading": reading(cls)}
    result["by_date"] = by_group(states, lambda x: x["date"])
    result["by_episode"] = by_group(states, lambda x: x["episode"])
    eps = rec["episodes"]
    secs = [r["seconds"] for r in rows]
    setups = [r["setup_s"] for r in rows]
    cand = statistics.fmean(len(p["allowed"]) for p in rec["chosen"])
    cph = statistics.fmean(e["choice_points_per_hour"] for e in eps.values())
    per_rollout = statistics.fmean(secs)
    result["compute"] = {
        "rollouts": len(rows), "mean_rollout_s": round(per_rollout, 2), "mean_setup_s": round(statistics.fmean(setups), 2),
        "max_rollout_s": round(max(secs), 2), "mean_candidates": round(cand, 2), "choice_points_per_hour": round(cph, 1),
        "step1_day_cpu_hours_8_scenarios": round(24 * cph * cand * len(SELECT_SEEDS) * per_rollout / 3600, 1),
        "note": "1단계(하루를 이 탐색으로 제어)는 선택 시점마다 후보 × 고르는 용 8개를 굴린다고 본 CPU 시간이다(병렬이면 나눔)."}
    gap = [abs(m["delay"] - m["delay_increment"]) for r in rows for m in r["marks"].values()]
    result["safety"] = {"leaks": sum(r["leaks"] for r in rows), "changed_by_rules": sum(r["changed_by_rules"] for r in rows),
                        "injected_mean": round(statistics.fmean(r["injected"] for r in rows), 1),
                        "exact_vs_increment_mean_abs": round(statistics.fmean(gap), 1)}
    result["information"] = [{"정보": a_, "탐색기": b_, "최종 신경망(72칸 관측)": c_} for a_, b_, c_ in INFO_TABLE]
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    head = ("| 분류 | 상태 | P0와 다르게 고름 | 확인 이득 평균(대·초) | 95% 구간 | 그중 실제 지연 | 그중 끝 정체 벌점 | "
            "고르는 용에서 본 이득 | 유의하게 나음 | 나빠짐 | 유의하게 나빠짐 |")
    rule = "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|"
    fmt = lambda name, v: (f"| {name} | {v['states']} | {v['differs']} | {v['mean_confirm_gain']} | {v['ci95']} | {v['mean_delay_gain']} | "
                           f"{v['mean_terminal_gain']} | {v['mean_select_gain']} | {v['significantly_better']} | {v['worse']} | "
                           f"{v['significantly_worse']} |")
    lines = ["# 0단계 후보 비교 요약(진행 기록 119·121절)", "",
             "- 확인 이득 = 확인용 시나리오 8개에서 (P0 선택의 비용 − 고른 후보의 비용), 같은 시나리오끼리 짝. 비용 = 실제 지연 + 끝에 남은 정체 × 75초.",
             "- 상태 48개는 같은 3일·12개 에피소드에서 나와 서로 독립이 아니다. 아래 '읽는 법'은 참고이고, 날짜·에피소드별 표와 함께 본다.", ""]
    titles = {"main": f"주: 탐색 {H_MAIN}초로 고르고 {H_MAIN}초로 확인", "main_at_600": f"유지: {H_MAIN}초로 고른 후보를 그대로 {H_SENS}초 비용으로 확인",
              "sens": f"민감도: {H_SENS}초로 다시 고르고 {H_SENS}초로 확인"}
    for key in ("main", "main_at_600", "sens"):
        lines += [f"## {titles[key]} — {result[key]['reading']}", "", head, rule]
        lines += [fmt(c, result[key]["classes"][c]) for c in (*CLASSES, "all")]
        lines.append("")
    for name, field in (("날짜별", "by_date"), ("에피소드별", "by_episode")):
        lines += [f"## {name}(주 확인 이득 / {H_SENS}초 유지)", "",
                  "| 묶음 | 상태 | P0와 다르게 고름 | 주 확인 이득 | 주 유의하게 나음 | 주 유의하게 나빠짐 | 유지 이득 | 유지 유의하게 나음 | 유지 유의하게 나빠짐 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for g, v in result[field].items():
            m_, k_ = v["main"], v["main_at_600"]
            lines.append(f"| {g} | {m_['states']} | {m_['differs']} | {m_['mean_confirm_gain']} | {m_['significantly_better']} | "
                         f"{m_['significantly_worse']} | {k_['mean_confirm_gain']} | {k_['significantly_better']} | {k_['significantly_worse']} |")
        lines.append("")
    c = result["compute"]
    lines += ["## 계산 시간", "", f"- 탐색 {c['rollouts']}번, 한 번 평균 {c['mean_rollout_s']}초(시작 준비 {c['mean_setup_s']}초), 가장 길 때 {c['max_rollout_s']}초",
              f"- 후보 평균 {c['mean_candidates']}개, 시간당 선택 시점 {c['choice_points_per_hour']}개",
              f"- 1단계 하루 CPU 시간 추정 {c['step1_day_cpu_hours_8_scenarios']}시간", "",
              f"- 엿봄 확인: 실제 미래 차 {result['safety']['leaks']}대, 규칙이 바꾼 선택 {result['safety']['changed_by_rules']}번",
              f"- 지연 셈: 정확한 셈과 매초 증가분 셈의 차이 평균 {result['safety']['exact_vs_increment_mean_abs']}대·초(판정에는 정확한 셈을 씀)", "",
              "## 탐색기와 최종 신경망이 보는 정보", "", "| 정보 | 탐색기 | 최종 신경망(72칸 관측) |", "|---|---|---|"]
    lines += [f"| {a_} | {b_} | {c_} |" for a_, b_, c_ in INFO_TABLE]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--out", required=True)
    c.add_argument("--work")
    s = sub.add_parser("sample")
    s.add_argument("--work", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--workers", type=int, default=4)
    m = sub.add_parser("compare")
    m.add_argument("--work", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--workers", type=int, default=4)
    y = sub.add_parser("summary")
    y.add_argument("--out", required=True)
    args = ap.parse_args()
    {"check": check, "sample": sample, "compare": compare, "summary": summary}[args.cmd](args)


if __name__ == "__main__":
    main()
