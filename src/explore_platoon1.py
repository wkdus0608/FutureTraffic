"""[추가 탐색] 1단계 PLATOON1 — 무리 도착 세계(P)에서 정책 반복 1라운드(강화학습, 진행 기록 241·242절).

사전 등록: results/explore/platoon1_20261101/사전등록.md(10/31 23:56, 코드 전 고정). PLATOON0(241절)에서 무리 도착 세계 P의
이웃 시간표 정보가 M4보다 3.82초를 줄였다(TM4). 이번에는 그 정보를 보는 학생 신경망을 정책 반복(롤아웃 분류 정책 반복,
PI R1과 같은 방법)으로 배워, 같은 정보를 쓰는 계획기 TM4보다 짧아지는지(학습의 몫) 잰다.

PI R1(arena3_pi.py)과 달라지는 것 세 가지:
  1. 지금 정책 = TM4(PLATOON0 그대로). 결정 지점 = 허용 녹색이 2개 이상인 제어 단계 가운데 해시 규칙으로 25%.
  2. 롤아웃 이어 달리기 = 저장 순간의 TM4 기록(본 차·진입 기록·동쪽 상태별 기록)을 그대로 이어받은 TM4.
  3. 표본 미래 = PI R1 표본기와 같은 난수 + 동쪽 시작 도로 차만 P 세계 규칙으로 204822 녹색 초에 다시 놓는다.
학생 관측 = ARENA3 119칸 + 시간표 관측 48칸(TM4 녹색 원핫 4, 녹색별 TM4 계획 비용 차 4, 동쪽 예측 도착 16, 204822 시간표 24).
잠긴 파일·PI R1·PLATOON0 코드는 고치지 않고 가져다 쓴다.

명령(FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240):
  python src/explore_platoon1.py make-config --workers 8                            (맥, SUMO 없음: P 세계 차량 파일 SHA 고정)
  python src/explore_platoon1.py smoke   --out <Drive>/explore_platoon1/smoke/shard0 --work /content/pt1s [--platoon0 <Drive>/explore_platoon0]
  python src/explore_platoon1.py collect --out <Drive>/explore_platoon1/data --work /content/pt1 --shard 0 --shards 3
  python src/explore_platoon1.py offline|train|evaluate|baseline|judge|confirm …     (explore_platoon1_learn.py)
"""
from __future__ import annotations

import argparse
import bisect
import contextlib
import functools
import gzip
import hashlib
import json
import math
import multiprocessing
import os
import pickle
import platform
import random
import shutil
import statistics
import sys
import tempfile
import time
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import arena3_bprime as bp  # noqa: E402
import arena3_common as ac  # noqa: E402
import arena3_referee as ref  # noqa: E402
import explore_platoon0 as p0  # noqa: E402
import explore_pred as ep  # noqa: E402
import explore_weight as w  # noqa: E402
import explore_weight_core as wc  # noqa: E402
import focused_platoon as fp  # noqa: E402
import focused_scenario as fs  # noqa: E402
import focused_train as ft  # noqa: E402
from focused_data import TARGET_TLS  # noqa: E402

EXPERIMENT = "PLATOON1_V1"
ROUND = "PT1"
CONFIG = ROOT / "config/explore_platoon1_v1.json"
PREREG = ROOT / "results/explore/platoon1_20261101/사전등록.md"
PI_R1_CONFIG = ROOT / "config/arena3_pi_r1.json"
POLICY = "PT1_DATA"
# 수집 결과를 정하는 코드·자료(자료 폴더 지문). 학생·평가 코드(explore_platoon1_learn.py)는 평가 폴더에만 더한다.
SOURCES = ("src/explore_platoon1.py", "src/explore_platoon0.py", "src/explore_ceiling.py", "src/focused_platoon.py",
           "src/arena3_bprime.py", "config/east_platoon.json", "data/derived/east_upstream_204822.csv",
           "config/explore_platoon0_v1.json", "config/explore_platoon1_v1.json")
BASE_OBS = ac.OBSERVATION_SIZE               # ARENA3 119칸
EXTRA = 48
OBS = BASE_OBS + EXTRA                       # 167칸
GREENS = 4
ALLOWED = slice(60, 64)                      # 119칸 안의 '지금 실행할 수 있는 녹색'(arena3_pi와 같음)
TM4_ONEHOT = slice(119, 123)
TM4_GAPS = slice(123, 127)
FORECAST = slice(127, 143)
TIMETABLE = slice(143, 167)
FORECAST_STEPS = 6                           # 30초 칸(5초 × 6)
TIMETABLE_BIN_S, TIMETABLE_BINS = 10, 12     # 앞으로 120초를 10초 칸으로
CONTROL_S = 3600
KEEP = ("delay", "time_loss", "depart_wait", "residual", "cost", "left_east", "left_south", "injected",
        "east_platoon", "east_kept", "collisions_unexplained", "collisions_at_204820", "future_seed", "seconds")


# ----------------------------------------------------------------------------- 설정
def config() -> dict:
    cfg = w.read(CONFIG)
    if cfg.get("experiment") != EXPERIMENT or cfg.get("round") != ROUND:
        raise ValueError("PLATOON1 설정이 아닙니다.")
    for key, path in (("prereg_sha256", PREREG), ("lock_sha256", ac.CONFIG), ("platoon0_config_sha256", p0.CONFIG),
                      ("pi_r1_config_sha256", PI_R1_CONFIG)):
        if cfg.get(key) != w.file_hash(path):
            raise ValueError(f"PLATOON1 설정을 만든 때와 파일이 다릅니다: {path.relative_to(ROOT)}")
    return cfg


def future_seed(cfg: dict, state_id: str, index: int) -> int:
    return ac.seed32(cfg["rollout"]["future_salt"], state_id, index)


def point_sampled(cfg: dict, sid: str, step: int) -> bool:
    """결정 지점 표본(결과와 무관한 해시 규칙): 같은 상황·같은 단계는 언제나 같은 답."""
    p = cfg["points"]
    value = int(hashlib.sha256(f"{p['salt']}|{sid}|{int(step)}".encode()).hexdigest()[:8], 16) % 10000
    return value < round(float(p["fraction"]) * 10000)


def keep_state(cfg: dict, state_id: str) -> bool:
    value = int(hashlib.sha256(f"PLATOON1_KEEP|{state_id}".encode()).hexdigest()[:8], 16) % 1000
    return value < round(float(cfg["rollout"]["keep_state_fraction"]) * 1000)


def origin_of(situation: dict) -> float:
    """차량 파일 시각 0의 하루 초(준비 30분 앞)."""
    return float(fs.period_start_s(situation["period"]) - ft.WARMUP_S)


# ----------------------------------------------------------------------------- 무리 도착 표본 미래(순수 계산 + 감싸개)
@functools.lru_cache(maxsize=None)
def _lead(edges: tuple[str, ...]) -> float:
    fn, _, _ = p0.east_geometry()
    return fn.travel_time_to_stopline(list(edges), fp.EAST_APPROACH)


def _origin_weights(date: str, period_start: int) -> list[int]:
    counts = p0._upstream_counts()
    release, _ = p0._release()
    return [sum(counts.get((date, m, o), 0) for m in range(period_start // 60, period_start // 60 + 15)) for o in release]


def retime_sampled(sampled: list, t0: float, horizon: int, date: str, origin_s: float, seed: int) -> tuple[list, Counter]:
    """PI R1 표본 미래에서 동쪽 시작 도로 차만 P 세계 규칙으로 다시 놓는다(사전 등록 바꿈 3).

    차마다 그 차의 정지선 도착 15분의 상류 통과 비율로 204822 이동을 뽑고, 롤아웃 (t0, t0 + horizon]에 출발하게 되는
    그 이동의 204822 녹색 초 하나 + 0~1초를 고른다. 이동에 녹색이 없거나 그 창에 녹색 초가 없으면 원래 시각 그대로.
    """
    _, start, link_tt = p0.east_geometry()
    release, main_origin = p0._release()
    origins = list(release)
    windows = p0._green_windows(str(date))
    rng = random.Random(ac.seed32("PLATOON1_FUTURE", int(seed)))
    lo, hi = origin_s + t0 - link_tt, origin_s + t0 + horizon - link_tt
    out, kinds = [], Counter()
    for vid, edges, vt, dep in sampled:
        if edges[0] != start:
            out.append((vid, edges, vt, dep))
            continue
        period_start = int((origin_s + dep + _lead(tuple(edges))) // 900 * 900)
        weights = _origin_weights(str(date), period_start)
        origin = rng.choices(origins, weights=weights)[0] if sum(weights) else main_origin
        state = release.get(origin)
        seconds = windows.get(state, []) if state else []
        a, b = bisect.bisect_left(seconds, lo), bisect.bisect_left(seconds, hi)
        if b > a:
            u = seconds[rng.randrange(a, b)] + rng.random()
            dep = min(max(round(u + link_tt - origin_s, 2), round(t0 + 0.01, 2)), float(t0 + horizon))
            kinds["east_platoon"] += 1
        else:
            kinds["east_kept"] += 1
        out.append((vid, edges, vt, dep))
    out.sort(key=lambda r: (r[3], r[0]))
    return out, kinds


@contextlib.contextmanager
def platoon_futures(date: str, origin_s: float, stats: Counter | None = None):
    """with 안에서 explore_rollout의 표본 미래를 무리 도착 규칙으로 바꾼다(롤아웃 작업자 프로세스 안에서만 쓴다)."""
    import explore_rollout as er
    original = er.sample_future
    if getattr(original, "platoon1", False):
        raise RuntimeError("무리 도착 표본 미래가 이미 켜져 있습니다.")

    def sample_future(hist, t0, seed, horizon=er.H_SENS):
        out, kinds = retime_sampled(original(hist, t0, seed, horizon), t0, horizon, date, origin_s, seed)
        if stats is not None:
            stats.update(kinds)
        return out

    sample_future.platoon1 = True
    er.sample_future = sample_future
    try:
        yield stats
    finally:
        er.sample_future = original


# ----------------------------------------------------------------------------- 시간표 관측 48칸(순수 계산)
@functools.lru_cache(maxsize=1)
def east_moves() -> tuple[tuple[str, str], ...]:
    from focused_rl import MOVEMENT_LINKS
    moves = tuple(sorted(m for m in MOVEMENT_LINKS if m[0] == fp.EAST_APPROACH))
    if len(moves) != 4:
        raise ValueError(f"동쪽 이동이 4개가 아닙니다: {moves}")
    return moves


@functools.lru_cache(maxsize=1)
def release_states() -> tuple[str, str]:
    """동쪽 차를 내보내는 204822 상태: (주 이동의 상태, 다른 하나)."""
    release, main = p0._release()
    others = sorted({s for s in release.values() if s and s != release[main]})
    if not release[main] or len(others) != 1:
        raise ValueError(f"204822 내보내는 상태가 둘이 아닙니다: {release}")
    return release[main], others[0]


def east_forecast(arrivals: list[list[float]], index: dict) -> list[float]:
    """TM4가 예측한 동쪽 이동 4개의 정지선 도착, 30초 칸 4개(÷10, 0~10) → 16칸."""
    out = []
    for m in east_moves():
        i = index.get(m)
        for k in range(0, len(arrivals), FORECAST_STEPS):
            n = 0.0 if i is None else sum(arrivals[j][i] for j in range(k, min(k + FORECAST_STEPS, len(arrivals))))
            out.append(min(10.0, max(0.0, float(n) / 10.0)))
    if len(out) != 16:
        raise ValueError("동쪽 예측 칸이 16개가 아닙니다(계획 길이 120초가 아님).")
    return out


def timetable_view(state_at, now: float, travel: float) -> list[float]:
    """두 내보내는 상태가 정지선 도착 기준으로 앞으로 120초의 10초 칸마다 켜진 비율 → 24칸.

    정지선에 τ초에 닿는 차는 τ − 구간 통과 시간에 구간에 들고, state_at(그 시각)은 그 차가 204822를 떠날 때의 상태다(TM4와 같은 시간 축)."""
    out = []
    for s in release_states():
        for k in range(TIMETABLE_BINS):
            on = sum(state_at(now + TIMETABLE_BIN_S * k + d + 0.5 - travel) == s for d in range(TIMETABLE_BIN_S))
            out.append(on / TIMETABLE_BIN_S)
    return out


def extra_cells(green: int, costs: list, forecast: list[float], table: list[float]) -> list[float]:
    finite = [float(c) for c in costs if c is not None]
    if not finite or not 0 <= int(green) < GREENS or costs[int(green)] is None:
        raise ValueError("TM4 녹색 또는 녹색별 비용이 올바르지 않습니다.")
    low = min(finite)
    gaps = [0.0 if c is None else min(10.0, max(0.0, (float(c) - low) / 1000.0)) for c in costs]
    cells = [float(g == int(green)) for g in range(GREENS)] + gaps + list(forecast) + list(table)
    if len(cells) != EXTRA:
        raise ValueError("시간표 관측이 48칸이 아닙니다.")
    return cells


def observation167(base_obs, extra) -> np.ndarray:
    obs = np.asarray([*np.asarray(base_obs, np.float32), *extra], dtype=np.float32)
    if obs.shape != (OBS,) or not np.all(np.isfinite(obs)):
        raise ValueError("PLATOON1 관측 167칸이 올바르지 않습니다.")
    return obs


# ----------------------------------------------------------------------------- TM4(PLATOON0 InfoM4와 같은 계산 + 기록 넘기기)
class TM4Core:
    """PLATOON0 InfoM4('TM4')와 같은 판단. 기록(본 차·진입·동쪽 상태별)을 꺼내고 넣을 수 있고, 시간표 관측 48칸을 낸다."""

    def __init__(self):
        self.ctl, self.weights = None, None

    def build(self, movement_of: dict, greens: list[str], date: str, origin_s: float, env, movements=None) -> "TM4Core":
        cls = p0.timetable_class(p0.HORIZON_S)
        ctl = cls(movement_of, greens)
        ctl.configure(str(date), float(origin_s))
        ctl._setup(env)
        if movements is not None and list(ctl.movements) != list(movements):
            raise RuntimeError("TM4의 이동 순서가 M4 계획기와 다릅니다.")
        self.ctl, self.weights = ctl, ac.lefts_weights(ac.MODES6[ac.M4], ctl.movements)
        return self

    def observe(self, env) -> None:
        self.ctl._observe(env)

    def _inputs(self, env):
        ts, now = env.traffic_signals[TARGET_TLS], float(env.sim_step)
        served = env.signal_constraints._served_now(ts, now)
        state = ep.SignalState(ts.green_phase, now - ts.time_since_last_phase_change, float(ts.yellow_time),
                               tuple(served[c] for c in self.ctl._model.crosswalks))
        q0, arrivals = self.ctl.queues_and_arrivals(env)
        return state, now, q0, arrivals

    def _plan(self, state, now, q0, arrivals, allowed) -> int:
        action, _ = wc.weighted_plan(self.ctl._model, state, now, q0, arrivals, list(allowed), self.weights)
        if action not in allowed:
            raise RuntimeError("TM4가 허용 밖 녹색을 냈습니다.")
        return int(action)

    def decide(self, env, allowed: list[int]) -> int:
        return self._plan(*self._inputs(env), allowed)

    def view(self, env, allowed: list[int]) -> dict:
        """decide와 같은 녹색 + 녹색별 비용(그 녹색만 허용한 같은 계획) + 동쪽 예측 + 시간표."""
        state, now, q0, arrivals = self._inputs(env)
        forecast = east_forecast(arrivals, self.ctl.index)
        green = self._plan(state, now, q0, arrivals, allowed)
        costs = [None] * GREENS
        for a in allowed:
            _, info = wc.weighted_plan(self.ctl._model, state, now, q0, arrivals, [int(a)], self.weights)
            costs[int(a)] = float(info["cost"])
        table = timetable_view(self.ctl._state_at, now, self.ctl.geo["travel"][fp.EAST_APPROACH])
        return {"green": green, "costs": costs, "extra": extra_cells(green, costs, forecast, table)}

    def export_state(self) -> dict:
        c = self.ctl
        return {"seen": sorted(c._seen), "entries": list(c._entries), "first_time": c._first_time,
                "east_log": list(c._east_log)}

    def import_state(self, state: dict) -> None:
        c = self.ctl
        c._seen = set(state["seen"])
        c._entries = deque(tuple(e) for e in state["entries"])
        c._first_time = state["first_time"]
        c._east_log = deque(tuple(e) for e in state["east_log"])


def episode_core(holder, ctx) -> TM4Core:
    """본 에피소드: 첫 판단에서 TM4를 만든다(PLATOON0 InfoM4._build와 같은 차량 이동표·시각)."""
    if holder.core is None:
        e = ctx.ep
        greens = [p.state for p in e.ts.green_phases]
        cls = p0.timetable_class(p0.HORIZON_S)
        holder.core = TM4Core().build(cls.movements(e.route), greens, e.date, fs.period_start_s(e.period) - ft.WARMUP_S,
                                      e.env, movements=ctx.planner.p0.movements)
    return holder.core


# ----------------------------------------------------------------------------- 학생 신경망·정책
class PlatoonNet(nn.Module):
    """관측 167칸 → 녹색 4개의 값(TM4 녹색 = 0). 이점 머리를 0으로 시작해 학습 전 탐욕 정책은 TM4다."""

    def __init__(self, hidden=(64, 64)):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(OBS, hidden[0]), nn.ReLU(), nn.Linear(hidden[0], hidden[1]), nn.ReLU())
        self.advantage = nn.Linear(hidden[1], GREENS)
        nn.init.zeros_(self.advantage.weight)
        nn.init.zeros_(self.advantage.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.reshape(-1, OBS)
        adv = self.advantage(self.trunk(obs))
        return adv - adv.gather(1, obs[:, TM4_ONEHOT].argmax(dim=1, keepdim=True))


def allowed_mask(obs) -> np.ndarray:
    return np.asarray(obs, np.float32)[..., ALLOWED] > 0.5


def anchor_green(obs) -> np.ndarray:
    return np.asarray(obs, np.float32)[..., TM4_ONEHOT].argmax(axis=-1)


def greedy_green(net: PlatoonNet, obs) -> int:
    """허용 녹색 중 가장 큰 값(같으면 TM4 녹색). 고를 것이 없으면 신경망을 부르지 않는다."""
    obs = np.asarray(obs, np.float32)
    mask = allowed_mask(obs)
    anchor = int(anchor_green(obs))
    if not mask[anchor]:
        raise ValueError("TM4 녹색이 허용 녹색이 아닙니다.")
    if mask.sum() == 1:
        return anchor
    with torch.no_grad():
        rel = net(torch.from_numpy(obs).view(1, -1))[0].numpy().astype(np.float64)
    if not np.all(np.isfinite(rel)):
        raise ValueError("녹색 값이 유한하지 않습니다.")
    rel = np.where(mask, rel, -np.inf)
    rel[anchor] += 1e-6
    return int(rel.argmax())


class TM4Policy(ac.P0Policy):
    """TM4 그대로(PLATOON0 InfoM4('TM4')와 같은 판단). 확인 기준선·작은 실행용."""

    def __init__(self, name: str = "TM4"):
        self.name, self.core = name, None
        self.stats = Counter()

    def decide(self, snap, ctx) -> int:
        core = episode_core(self, ctx)
        core.observe(ctx.ep.env)
        allowed = [int(a) for a in ctx.allowed]
        if len(allowed) == 1:
            return allowed[0]
        self.stats["planned"] += 1
        return core.decide(ctx.ep.env, allowed)

    def info(self) -> dict:
        return {"kind": "timetable", **self.stats}


class StudentPolicy(ac.P0Policy):
    """심판 평가용 학생: 매 제어 단계 TM4가 진입을 기록하고, 고를 수 있으면 시간표 관측을 만들어 신경망이 고른다."""

    def __init__(self, net: PlatoonNet, name: str):
        self.net, self.name, self.core = net, name, None
        self.stats = Counter()

    def decide(self, snap, ctx) -> int:
        core = episode_core(self, ctx)
        env = ctx.ep.env
        core.observe(env)
        allowed = [int(a) for a in ctx.allowed]
        if len(allowed) == 1:
            return allowed[0]
        view = core.view(env, allowed)
        action = greedy_green(self.net, observation167(snap.observation, view["extra"]))
        self.stats["planned"] += 1
        self.stats["differs_from_tm4"] += int(action != view["green"])
        return action

    def info(self) -> dict:
        return {"kind": "student", **self.stats}


# ----------------------------------------------------------------------------- 롤아웃(작업자 프로세스)
def _pool_init() -> None:
    ref._worker_init()
    torch.set_num_threads(1)


def rollout_pool(workers: int):
    return ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"), initializer=_pool_init)


def rollout_worker(job: dict) -> dict:
    """저장 상태에서 무리 도착 표본 미래 하나로 300초: 첫 5초 후보, 이후 TM4(저장 기록 이어받음), 제어가 끝나면 P0."""
    import explore_rollout as er
    from focused_eval import judge_collisions
    started = time.monotonic()
    env = None
    fd, trip_name = tempfile.mkstemp(prefix="pt1_trip_", suffix=".xml")
    os.close(fd)
    fd, coll_name = tempfile.mkstemp(prefix="pt1_coll_", suffix=".xml")
    os.close(fd)
    trip, coll = Path(trip_name), Path(coll_name)
    try:
        w.configure()
        horizon = int(job["horizon_s"])
        kinds = Counter()
        with platoon_futures(job["date"], float(job["origin_s"]), kinds):
            env, cons, ts, ctl, snap, injected, vehicles = bp._restore_env(job, "sampled", int(job["future_seed"]), horizon,
                                                                             trip, coll)
        control_end = float(snap["handover"]) + CONTROL_S
        greens = [p.state for p in ts.green_phases]
        planner = ac._planner_class()(ctl.movement_of, greens, control_end)
        check = planner.arena_snapshot(env)
        if list(check.greens) != job["greens"] or list(check.allowed) != job["allowed"]:
            raise ValueError("표본 미래를 넣은 뒤 여섯 모드의 첫 녹색 또는 허용 녹색이 저장 순간과 다릅니다.")
        saved = np.asarray(job["observation"], np.float32)
        if np.max(np.abs(check.observation - saved[:BASE_OBS])) > 1e-6:
            raise ValueError("표본 미래를 넣은 뒤 관측 119칸이 저장 순간과 다릅니다(미래가 관측에 샜거나 복원 오차).")
        core = TM4Core().build(ctl.movement_of, greens, job["date"], float(job["origin_s"]), env,
                               movements=planner.p0.movements)
        core.import_state(snap["tm4_state"])
        view = core.view(env, list(check.allowed))
        if view["green"] != int(job["tm4"]) or np.max(np.abs(np.asarray(view["extra"], np.float32) - saved[BASE_OBS:])) > 1e-6:
            raise ValueError("TM4 기록을 넣은 뒤 TM4 녹색 또는 시간표 관측 48칸이 저장 순간과 다릅니다.")
        if int(job["candidate"]) not in check.allowed:
            raise ValueError("후보가 허용 녹색이 아닙니다.")
        meter = bp._meter_class()(env, snap["t0"], vehicles, injected, marks=(horizon,))

        def follow(e, allowed):
            if float(e.sim_step) >= control_end - 1e-9:
                return int(ctl(e, allowed))                  # 본 에피소드처럼 제어 60분 뒤는 P0
            core.observe(e)
            allowed = [int(a) for a in allowed]
            return allowed[0] if len(allowed) == 1 else core.decide(e, allowed)

        _, changed = er.drive(env, cons, follow, int(job["candidate"]), snap["t0"], horizon, meter.step)
        env.close()
        env = None
        meter.finish(trip)
        if horizon not in meter.at:
            raise ValueError("300초 비용이 없습니다.")
        collision = judge_collisions(coll)
        r = dict(meter.at[horizon])
        r.update({"candidate": int(job["candidate"]), "future": int(job["index"]), "future_seed": int(job["future_seed"]),
                  "leaks": len(meter.leaks), "changed_by_rules": int(changed), "injected": len(injected),
                  "east_platoon": int(kinds["east_platoon"]), "east_kept": int(kinds["east_kept"]),
                  "collisions_unexplained": len(collision["collisions_unexplained"]),
                  "collisions_at_204820": int(collision["collisions_at_204820"]), "seconds": time.monotonic() - started})
        return {"ok": True, **r}
    except BaseException as error:
        return {"ok": False, "candidate": job.get("candidate"), "future": job.get("index"),
                "error": f"{type(error).__name__}: {error}", "seconds": time.monotonic() - started}
    finally:
        if env is not None:
            env.close()
        trip.unlink(missing_ok=True)
        coll.unlink(missing_ok=True)


# ----------------------------------------------------------------------------- 수집기(본 에피소드)
class Collector(ac.P0Policy):
    """본 에피소드: 행동은 언제나 TM4. 표본 지점에서는 상태(SUMO + TM4 기록)를 저장하고 롤아웃을 맡긴 뒤 기다리지 않는다.

    limit(점검용) = 굴릴 최대 지점 수.
    """

    def __init__(self, pool, cfg: dict, situation: dict, state_dir: Path, keep_dir: Path | None = None, *,
                 limit: int | None = None):
        self.pool, self.cfg = pool, cfg
        self.situation, self.state_dir, self.keep_dir = situation, Path(state_dir), keep_dir
        self.sid = ac.situation_id(situation)
        self.futures, self.horizon = int(cfg["rollout"]["futures"]), int(cfg["rollout"]["horizon_s"])
        self.limit = limit
        self.name = f"{ROUND}_COLLECT"
        self.core = None
        self.points: list[dict] = []
        self.rollouts: list[tuple] = []          # (지점 번호, 후보 녹색, 미래 번호, Future)
        self.verifies: list[tuple] = []          # (지점 번호, Future)
        self.counts = Counter({"forced": 0, "planned": 0, "hash_sampled": 0, "sampled": 0})

    def decide(self, snap, ctx) -> int:
        core = episode_core(self, ctx)
        env = ctx.ep.env
        core.observe(env)
        allowed = [int(a) for a in ctx.allowed]
        if len(allowed) == 1:
            self.counts["forced"] += 1
            return allowed[0]
        self.counts["planned"] += 1
        if not point_sampled(self.cfg, self.sid, ctx.step):
            return core.decide(env, allowed)
        self.counts["hash_sampled"] += 1
        if self.limit is not None and len(self.points) >= self.limit:
            return core.decide(env, allowed)
        self.counts["sampled"] += 1
        view = core.view(env, allowed)
        self._submit(snap, ctx, view)
        return int(view["green"])

    def _submit(self, snap, ctx, view: dict) -> None:
        import explore_m4probe_pending as pend
        import explore_rollout as er
        e = ctx.ep
        k = len(self.points)
        state_id = f"{self.sid}_p{k:03d}"
        stem = self.state_dir / state_id
        saved = er.Episode.snapshot(e, stem, snap.allowed, snap.p0_action)
        saved["pending_snapshot"] = pend.capture_pending(e.env.sumo)
        saved["tm4_state"] = self.core.export_state()
        stem.with_suffix(".pkl").write_bytes(pickle.dumps(saved))
        er.types_only_file(e.route)          # 작업자들이 동시에 만들지 않게 미리 만든다
        obs = observation167(snap.observation, view["extra"])
        base = {"stem": str(stem), "route_dir": str(e.route.parent), "sumo_seed": int(self.situation["sumo_seed"]),
                "greens": list(snap.greens), "allowed": list(snap.allowed), "horizon_s": self.horizon,
                "observation": obs.tolist(), "tm4": int(view["green"]), "date": str(self.situation["date"]),
                "origin_s": origin_of(self.situation)}
        self.points.append({"state": state_id, "step": int(ctx.step), "now": float(snap.now), "allowed": list(snap.allowed),
                            "greens": list(snap.greens), "p0": int(snap.p0_action), "m4": int(snap.greens[ac.M4]),
                            "tm4": int(view["green"]), "executed": int(view["green"]), "observation": obs.tolist(),
                            "tm4_costs": view["costs"]})
        for a in snap.allowed:
            for j in range(self.futures):
                job = {**base, "candidate": int(a), "index": j, "future_seed": future_seed(self.cfg, state_id, j)}
                self.rollouts.append((k, int(a), j, self.pool.submit(rollout_worker, job)))
        self.verifies.append((k, self.pool.submit(bp.verify_worker, {**base, "green": int(view["green"])})))

    def after_step(self, ctx, snap, action, executed) -> None:
        """표본 지점 실행 직후 원래 실행의 표를 남긴다(복원 확인용)."""
        if self.points and self.points[-1]["step"] == ctx.step and "main_after" not in self.points[-1]:
            self.points[-1]["main_after"] = bp.trace_table(ctx.ep.env.sumo)

    def outstanding(self) -> int:
        return sum(not f.done() for *_, f in self.rollouts) + sum(not f.done() for _, f in self.verifies)

    def futures_list(self) -> list:
        return [f for *_, f in self.rollouts] + [f for _, f in self.verifies]

    def spent(self) -> float:
        done = [f for f in self.futures_list() if f.done() and not f.cancelled() and f.exception() is None]
        return sum(float(f.result().get("seconds") or 0.0) for f in done)

    def finish(self) -> dict:
        """모든 롤아웃이 끝난 뒤: 비용을 지점에 붙이고 복원을 확인한다(PI R1과 같은 규칙)."""
        tol = float(self.cfg["rollout"]["restore_check"]["tolerance"])
        cpu, errors, failures = 0.0, [], []
        east = Counter()
        for p in self.points:
            p["rollouts"] = {str(a): [None] * self.futures for a in p["allowed"]}
        for k, a, j, future in self.rollouts:
            r = future.result()
            cpu += float(r.get("seconds") or 0.0)
            if not r["ok"]:
                failures.append(f"{self.points[k]['state']} 녹색 {a} 미래 {j}: {r['error']}")
                self.points[k].setdefault("rollout_errors", []).append({"green": a, "future": j, "error": r["error"][:300]})
            elif r["leaks"] or r["changed_by_rules"]:
                errors.append(f"{self.points[k]['state']} 녹색 {a} 미래 {j}: 미래 엿봄 또는 규칙 변경")
            else:
                self.points[k]["rollouts"][str(a)][j] = {key: r.get(key) for key in KEEP}
                east.update({"east_platoon": r["east_platoon"], "east_kept": r["east_kept"]})
        allowed_failures = max(3, int(float(self.cfg["rollout"]["max_failed_rollout_fraction"]) * len(self.rollouts)))
        if len(failures) > allowed_failures:
            errors.insert(0, f"롤아웃 실패 {len(failures)}/{len(self.rollouts)}개(허용 {allowed_failures}), 첫 번째: {failures[0]}")
        checked = mismatched = 0
        for k, future in self.verifies:
            r = future.result()
            cpu += float(r.get("seconds") or 0.0)
            point = self.points[k]
            main = point.pop("main_after", None)
            ok = bool(r["ok"]) and main is not None and bp.same_table(r["table"], main, tol)
            point["restore_check"] = {"ok": ok, "error": r.get("error")}
            checked += 1
            mismatched += not ok
        for p in self.points:
            p.pop("main_after", None)
            p["valid_futures"] = len(valid_futures(p))
        self.cleanup()
        if errors:
            raise RuntimeError(f"수집 판 실패({len(errors)}건): {errors[0]}")
        return {"checked": checked, "mismatched": mismatched, "rollout_cpu_seconds": cpu,
                "rollout_failures": len(failures), "rollouts": len(self.rollouts), "east_future": dict(east)}

    def cleanup(self) -> None:
        for p in self.points:
            stem = self.state_dir / p["state"]
            for ext in (".xml", ".pkl"):
                src = stem.with_suffix(ext)
                if self.keep_dir is not None and keep_state(self.cfg, p["state"]) and src.exists():
                    self.keep_dir.mkdir(parents=True, exist_ok=True)
                    with src.open("rb") as fin, gzip.open(self.keep_dir / (src.name + ".gz"), "wb") as fout:
                        shutil.copyfileobj(fin, fout)
                src.unlink(missing_ok=True)


def valid_futures(point: dict) -> list[int]:
    """모든 후보에서 비용이 있고, 어느 후보에서도 미확인 충돌·204820 충돌이 없는 미래 번호(B′·PI R1과 같은 규칙)."""
    rows = list(point["rollouts"].values())
    out = []
    for j in range(len(rows[0])):
        samples = [r[j] for r in rows]
        if any(s is None or s.get("collisions_unexplained") or s.get("collisions_at_204820") for s in samples):
            continue
        out.append(j)
    return out


# ----------------------------------------------------------------------------- 수집 명령
def write_points(target: Path, points: list[dict]) -> str:
    target.mkdir(parents=True, exist_ok=True)
    path = target / "points.jsonl.gz"
    with gzip.open(path, "wt") as f:
        for p in points:
            f.write(json.dumps(w.plain(p), ensure_ascii=False, allow_nan=False) + "\n")
    return w.file_hash(path)


def read_points(target: Path) -> list[dict]:
    m = w.read(Path(target) / "metrics.json")
    path = Path(target) / "points.jsonl.gz"
    if w.file_hash(path) != m.get("points_sha256"):
        raise ValueError(f"결정 지점 자료 해시가 다릅니다: {target}")
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


def restore_totals(out: Path) -> tuple[int, int]:
    """모든 조각의 복원 확인 수를 합한다(조각마다 따로 세면 한 조각이 우연히 넘을 수 있다, 210절)."""
    checked = mismatched = 0
    for path in sorted((Path(out) / "status").glob("collect_shard*.json")):
        s = w.read(path)
        checked += int(s.get("restore_checked", 0))
        mismatched += int(s.get("restore_mismatched", 0))
    return checked, mismatched


def run_episode(situation: dict, work: Path, policy, **kwargs):
    """P 세계 에피소드(PLATOON0과 같은 무리 도착 차량 파일)."""
    with p0.platoon_world(situation["tag"]) as stats:
        result = ac.run_arena_episode(situation, work, policy, **kwargs)
    result.metrics["world"] = "P"
    result.metrics["world_changes"] = dict(stats)
    return result


def finalize(cfg: dict, out: Path, fp_: str, item: dict) -> dict:
    """끝난 수집 판 하나를 쓴다: points.jsonl.gz + metrics.json(+ decisions·cache, ARENA3 실행 형식)."""
    policy, result = item["policy"], item["result"]
    rc = policy.finish()
    target = out / "runs" / POLICY / item["sid"]
    sha = write_points(target, policy.points)
    result.metrics.update({"pt_round": ROUND, "pt_base_policy": "TM4", "pt_points": len(policy.points),
                           "pt_counts": dict(policy.counts), "pt_restore_check": {k: rc[k] for k in ("checked", "mismatched")},
                           "pt_rollouts": rc["rollouts"], "pt_rollout_failures": rc["rollout_failures"],
                           "pt_rollout_cpu_seconds": rc["rollout_cpu_seconds"], "pt_main_seconds": item["main_seconds"],
                           "pt_east_future": rc["east_future"]})
    ac.write_run(target, result, {"policy": POLICY, "fingerprint": fp_, "model_identity": "TM4", "points_sha256": sha,
                                  "role": "data"})
    return {**rc, "cpu_seconds": rc["rollout_cpu_seconds"] + item["main_seconds"], "points": len(policy.points)}


def collect(cfg: dict, out: Path, work: Path, shard: int, shards: int, workers: int, *, executor=None,
            episode=None) -> int:
    episode = episode or run_episode
    out, work = Path(out), Path(work)
    fp_ = ref.register(out, extra_files=SOURCES)
    r = cfg["rollout"]
    mine = [s for i, s in enumerate(cfg["situations_data"]) if i % shards == shard]
    ledger = ref.Ledger(out, f"collect_shard{shard}")
    status_path = out / "status" / f"collect_shard{shard}.json"
    rc_cfg = r["restore_check"]
    checked = mismatched = 0
    failed, inflight = [], []
    stopped = False

    def write_status():
        done = sum((out / "runs" / POLICY / ac.situation_id(x) / "cache.json").exists() for x in mine)
        w.write(status_path, {"done": done, "total": len(mine), "failed": failed, "restore_checked": checked,
                              "restore_mismatched": mismatched, "inflight": [i["sid"] for i in inflight],
                              "stopped_by_restore_rule": stopped, "updated_unix": time.time()})

    def harvest():
        nonlocal checked, mismatched
        for item in [i for i in inflight if i["policy"].outstanding() == 0]:
            inflight.remove(item)
            try:
                got = finalize(cfg, out, fp_, item)
                checked += got["checked"]
                mismatched += got["mismatched"]
                ledger.add(got["cpu_seconds"])
                c = item["policy"].counts
                ref.stamp(f"PLATOON1 자료 {item['sid']}: 지연 {item['result'].metrics['mean_delay_s']:.3f}초, "
                          f"지점 {got['points']}개(계획 {c['planned']}·강제 {c['forced']}), 복원 불일치 "
                          f"{got['mismatched']}/{got['checked']}, 동쪽 표본 {got['east_future']}, "
                          f"{got['cpu_seconds'] / 3600:.2f} CPU-시간")
            except BaseException as error:
                failed.append({"situation": item["sid"], "error": f"{type(error).__name__}: {error}"})
                ref.stamp(f"PLATOON1 자료 실패 {item['sid']}: {failed[-1]['error']}")
                ledger.add(item["main_seconds"] + item["policy"].spent())
                if isinstance(error, KeyboardInterrupt):
                    raise
            finally:
                shutil.rmtree(work / item["sid"], ignore_errors=True)
            write_status()

    def backlog() -> int:
        return sum(i["policy"].outstanding() for i in inflight)

    def wait_some():
        pending = [f for i in inflight for f in i["policy"].futures_list() if not f.done()]
        if pending:
            wait(pending, timeout=30, return_when=FIRST_COMPLETED)

    p0.prepare_caches()
    pool = executor or rollout_pool(workers)
    with pool:
        for s in mine:
            sid = ac.situation_id(s)
            target = out / "runs" / POLICY / sid
            if ac.read_run(target, fp_, "TM4") is not None:
                m = w.read(target / "metrics.json")
                read_points(target)
                checked += m["pt_restore_check"]["checked"]
                mismatched += m["pt_restore_check"]["mismatched"]
                continue
            while backlog() > int(r["max_backlog_jobs"]):
                wait_some()
                harvest()
            began = time.monotonic()
            shutil.rmtree(work / sid, ignore_errors=True)
            policy = Collector(pool, cfg, s, work / sid / "states", out / "kept_states")
            try:
                result = episode(s, work / sid / "episode", policy, max_seconds=float(r["main_episode_max_seconds"]),
                                 on_step=policy.after_step)
                if result.metrics["route_sha256"] != s["route_sha256"]:
                    raise ValueError("차량 파일 SHA가 설정과 다릅니다.")
                dirty = ref.quality_errors(result.metrics)            # 심판과 같은 품질 검사(충돌은 자료를 읽을 때 뺀다)
                if dirty:
                    raise RuntimeError(f"수집 본 에피소드 품질 오류: {dirty}")
                inflight.append({"situation": s, "sid": sid, "policy": policy, "result": result,
                                 "main_seconds": time.monotonic() - began})
            except BaseException as error:
                failed.append({"situation": sid, "error": f"{type(error).__name__}: {error}"})
                ref.stamp(f"PLATOON1 본 에피소드 실패 {sid}: {failed[-1]['error']}")
                for future in policy.futures_list():
                    future.cancel()
                wait([f for f in policy.futures_list() if not f.cancelled()])
                policy.cleanup()
                shutil.rmtree(work / sid, ignore_errors=True)
                ledger.add(time.monotonic() - began + policy.spent())
                if isinstance(error, KeyboardInterrupt):
                    raise
            harvest()
            write_status()
            total_checked, total_mismatched = restore_totals(out)          # 이 조각의 방금 쓴 상태 포함
            rate = total_mismatched / total_checked if total_checked else 0.0
            if total_checked >= int(rc_cfg["min_points"]) and rate > float(rc_cfg["max_mismatch_rate"]):
                stopped = True
                ref.stamp(f"모든 조각 합 복원 불일치 {rate:.1%}({total_mismatched}/{total_checked})가 "
                          f"{float(rc_cfg['max_mismatch_rate']):.0%}를 넘어 새 판을 시작하지 않습니다.")
                break
        while inflight:
            wait_some()
            harvest()
    write_status()
    if stopped:
        return 3
    return 1 if failed else 0


# ----------------------------------------------------------------------------- 작은 실행(실행 전 점검)
def smoke(cfg: dict, out: Path, work: Path, workers: int, platoon0_out: Path | None = None, points: int = 3) -> dict:
    """(1) 상태 저장·롤아웃을 해도 본 에피소드가 TM4 그대로인가, (2) 학습 전 학생 = TM4, (3) 복원·TM4 기록 넘기기,
    (4) 동쪽 표본 미래가 녹색에 놓이는가, (5) 시간 어림, (6) 평가 상황에서 학생(학습 전)이 PLATOON0 TM4와 같은가(폴더를 주면)."""
    out, work = Path(out), Path(work)
    fp_ = ref.register(out, extra_files=SOURCES)
    p0.prepare_caches()
    s = cfg["situations_data"][0]
    shutil.rmtree(work, ignore_errors=True)
    keys = ("mean_delay_s", "actual_total_delay", "executed_sha16", "route_sha256", "cohort_sha256")
    plain_policy = TM4Policy()
    plain = run_episode(s, work / "plain", plain_policy, max_seconds=7200.0)
    student = run_episode(s, work / "student", StudentPolicy(PlatoonNet().eval(), "PT1_untrained"), max_seconds=7200.0)
    with rollout_pool(workers) as pool:
        policy = Collector(pool, cfg, s, work / "same" / "states", None, limit=points)
        same = run_episode(s, work / "same" / "episode", policy, max_seconds=float(cfg["rollout"]["main_episode_max_seconds"]),
                           on_step=policy.after_step)
        wait(policy.futures_list())
        rc = policy.finish()
    secs = [x["seconds"] for p in policy.points for v in p["rollouts"].values() for x in v if x]
    cands = [len(p["allowed"]) for p in policy.points]
    c = policy.counts
    per_episode = None
    if secs and cands:
        rollouts = c["hash_sampled"] * statistics.fmean(cands) * policy.futures
        per_episode = (rollouts * statistics.fmean(secs) + same.metrics["seconds"]) / 3600
    checks = {
        "route_sha": plain.metrics["route_sha256"] == s["route_sha256"],
        "unchanged_by_saving": all(plain.metrics[k] == same.metrics[k] for k in keys),
        "untrained_student_is_tm4": all(plain.metrics[k] == student.metrics[k] for k in keys),
        "quality_clean": not (ref.quality_errors(plain.metrics) or ref.quality_errors(same.metrics)
                              or ref.quality_errors(student.metrics)),
        "rolled_points": 0 < len(policy.points) == min(points, c["hash_sampled"]),
        "restore_clean": rc["mismatched"] == 0 and rc["rollout_failures"] == 0,
        "east_future_on_green": rc["east_future"].get("east_platoon", 0) > 0,
        "estimate": per_episode is not None,
    }
    if platoon0_out is not None:
        e = cfg["situations_eval"][0]
        sid = ac.situation_id(e)
        fp0 = w.read(Path(platoon0_out) / "experiment.json")["fingerprint"]
        ref_tm4 = ac.read_run(Path(platoon0_out) / "runs" / "P" / "TM4" / sid, fp0, "P/TM4")
        evaluated = run_episode(e, work / "eval_student", StudentPolicy(PlatoonNet().eval(), "PT1_untrained"),
                                max_seconds=7200.0)
        checks["eval_student_equals_platoon0_tm4"] = ref_tm4 is not None and all(
            ref_tm4[k] == evaluated.metrics[k] for k in keys)
    report = {"passed": all(checks.values()), "checks": checks, "situation": ac.situation_id(s),
              "plain": {k: plain.metrics[k] for k in keys}, "collector": {k: same.metrics[k] for k in keys},
              "student_untrained": {k: student.metrics[k] for k in keys}, "counts_per_hour": dict(c),
              "rolled_points": len(policy.points), "candidates": cands,
              "rollout_seconds_mean": statistics.fmean(secs) if secs else None, "restore_check": rc,
              "episode_seconds": {"plain": plain.metrics["seconds"], "student": student.metrics["seconds"],
                                  "collector": same.metrics["seconds"]},
              "estimate_cpu_hours_per_episode": per_episode,
              "estimate_cpu_hours_all": per_episode * len(cfg["situations_data"]) if per_episode else None,
              "fingerprint": fp_}
    w.write(out / "smoke.json", report)
    ref.stamp(f"PLATOON1 작은 실행: {'통과' if report['passed'] else '실패'} {checks} | 시간당 계획 {c['planned']}·강제 "
              f"{c['forced']} | 롤아웃 평균 {report['rollout_seconds_mean']}초 | 판당 약 {per_episode} CPU-시간")
    shutil.rmtree(work, ignore_errors=True)
    return report


# ----------------------------------------------------------------------------- 설정 만들기(맥, SUMO 없음)
def make_config(workers: int) -> dict:
    lock = ac.config()
    p0cfg = p0.config()
    pi_cfg = w.read(PI_R1_CONFIG)
    if pi_cfg.get("experiment") != "ARENA3_PI_R1":
        raise ValueError("PI R1 설정이 아닙니다.")
    pick = lambda s: {k: s[k] for k in ("date", "period", "tag", "sumo_seed")}  # noqa: E731
    data = [pick(s) for s in pi_cfg["situations_data"]]
    confirmation = [pick(s) for s in lock["situations_confirmation"]]
    with tempfile.TemporaryDirectory() as d, p0._pool(workers) as pool:
        shas = {sid: sha for sid, _, sha in pool.map(p0._sha_job, [(s, "P", d) for s in data + confirmation])}
    for s in data + confirmation:
        s["route_sha256"] = shas[ac.situation_id(s)]
    evaluation = [{**pick(s), "route_sha256": s["route_sha256"]["P"]} for s in p0cfg["situations"]]
    random_shas = ({s["route_sha256"] for s in pi_cfg["situations_data"]}
                   | {x["route_sha256"] for part in ("eval", "confirmation", "bprime_data") for x in lock[f"situations_{part}"]})
    p_shas = [s["route_sha256"] for s in data + confirmation + evaluation]
    if len(set(p_shas)) != len(p_shas) or set(p_shas) & random_shas:
        raise ValueError("P 세계 차량 파일이 서로 같거나 무작위 세계 파일과 같습니다.")
    if {s["date"] for s in data} & set(lock["dates"]["eval"]):
        raise ValueError("학습 자료에 평가 날짜가 들어 있습니다.")
    cfg = {
        "experiment": EXPERIMENT, "round": ROUND,
        "design": "PROJECT_PROGRESS.md 241·242절, results/explore/platoon1_20261101/사전등록.md",
        "prereg": PREREG.relative_to(ROOT).as_posix(), "prereg_sha256": w.file_hash(PREREG),
        "lock_sha256": w.file_hash(ac.CONFIG), "platoon0_config_sha256": w.file_hash(p0.CONFIG),
        "pi_r1_config_sha256": w.file_hash(PI_R1_CONFIG),
        "method": ("롤아웃 분류 정책 반복(RCPI·DPI 계열 강화학습, PI R1과 같은 방법). 무리 도착 세계 P에서 TM4가 운전하고, 결정 지점마다 "
                   "허용 녹색을 무리 도착 표본 미래로 굴려 짝 비교하며(정책 평가), 시간표 관측을 보는 신경망이 더 나은 녹색을 배운다(정책 개선)."),
        "base_policy": {"name": "TM4", "kind": "timetable", "source": "src/explore_platoon0.py InfoM4('TM4') 그대로(TM4Core)"},
        "world": {"name": "P", "rule": "PLATOON0과 같은 무리 도착(동쪽 시작 도로 차만 204822 녹색 초 + 0~1초), 난수 seed32(PLATOON0, 배치, 날짜, 차 번호)"},
        "points": {"fraction": 0.25, "salt": "PLATOON1_POINTS",
                   "rule": "허용 녹색이 2개 이상인 제어 단계(TM4가 계획하는 곳) 중 해시 규칙(상황·단계)으로 25%"},
        "rollout": {"horizon_s": 300, "first_action_s": 5, "terminal_s": 75.0, "history_s": 900, "futures": 8,
                    "future_salt": "PLATOON1_FUTURE_SEED", "retime_salt": "PLATOON1_FUTURE", "min_valid_futures": 4,
                    "continuation": "첫 5초 후보 녹색 → 저장 순간의 TM4 기록을 이어받은 TM4 → 제어 60분이 끝나면 P0",
                    "future": ("PI R1 표본기(최근 900초 차에서 대수 포아송·경로·차종·시각 고르게)와 같은 난수 + 동쪽 시작 도로 차만 "
                               "정지선 도착 15분의 상류 통과 비율로 204822 이동을 뽑아 롤아웃 300초 안의 그 이동 녹색 초 + 0~1초로 다시 놓음"
                               "(녹색이 없으면 그대로)"),
                    "cost": "300초 지연(시간 손실 + 진입 대기) + 남은 정체 × 75초(B′·PI R1과 같음)",
                    "checks": "표본 미래를 넣은 뒤 관측 119칸·여섯 모드·허용 녹색과 TM4 녹색·시간표 관측 48칸이 저장 순간과 같음(1e-6)",
                    "drop_future": "어느 후보에서든 미확인 충돌·204820 충돌이 난 미래는 모든 후보에서 뺀다",
                    "restore_check": {"tolerance": 1e-6, "max_mismatch_rate": 0.08, "min_points": 200,
                                      "scope": "모든 조각을 합해서", "use": "불일치 지점은 학습에서 뺀다"},
                    "max_failed_rollout_fraction": 0.02, "keep_state_fraction": 0.05, "max_backlog_jobs": 1500,
                    "main_episode_max_seconds": 10800},
        "data_quality": ("본 에피소드에 미확인 충돌·204820 충돌이 있으면 그 판의 지점은 모두 학습에서 뺀다(기록). "
                         "규칙 변경·행동 오류·P0 불일치·지연 재구성 실패는 멈춘다."),
        "observation": {"size": OBS, "base": [0, BASE_OBS], "tm4_onehot": [119, 123], "tm4_gaps": [123, 127],
                        "east_forecast": [127, 143], "timetable": [143, 167],
                        "east_moves": [list(m) for m in east_moves()], "release_states": list(release_states()),
                        "note": "TM4 녹색별 비용 = 그 녹색만 허용한 같은 계획의 비용(최소 대비 ÷1000, 0~10). 예측 = 30초 칸 4개 ÷10. 시간표 = 10초 칸 12개"},
        "student": {"seeds": [42, 43, 44], "network": [64, 64], "outputs": "녹색 4개의 TM4 녹색 대비 값(TM4 = 0)",
                    "target": "mean", "target_note": "짝 평균 이득(TM4 녹색 대비, 차량·초 ÷ 1000), 교사 규칙 c=0",
                    "loss": "PI R1과 같은 신뢰 가중 Huber, 가중 = clamp(|이득| / (표준오차 + 0.1), 0.25, 4), 허용 녹색 중 TM4가 아닌 것",
                    "confidence_floor": 0.1, "confidence_clip": [0.25, 4.0],
                    "epochs": 200, "batch_size": 64, "learning_rate": 0.001, "weight_decay": 0.0001, "grad_clip": 10.0,
                    "init": "이점 머리 0(학습 전 탐욕 = TM4)", "greedy": "허용 녹색 중 가장 큰 값, 같으면 TM4 녹색"},
        "offline": {"status": "참고 진단(판정·선택에 쓰지 않음)", "folds": "날짜 하나씩 보류, 미래 앞 절반으로 학습·뒤 절반으로 채점",
                    "targets": ["mean"]},
        "evaluation": {"student": "PT1", "models": 3, "situations": "situations_eval(84, P)",
                       "baseline": {"folder": "explore_platoon0", "fingerprint_prefix": "b307616114b4",
                                    "policies": ["M4", "TM4", "OM4", "P1M4"], "world": "P"},
                       "judge": "config/arena3_v1.json:judge (그대로)"},
        "rules": {
            "main": ("PT1이 P 세계 M4 대비 ARENA3 판정을 통과하고(날짜 평균 ≥ 0.3초, t95 하한 > 0, 양수 날짜 ≥ 5, 좌회전 +5초 이내, "
                     "300초 초과 ≤ M4, 정책 고유 충돌 0, 품질 오류 0), L = PT1 − TM4의 날짜 평균 ≥ 0.3초, t95 하한 > 0, 양수 날짜 ≥ 5"),
            "learning_min_gain_s": 0.3, "learning_min_positive_dates": 5,
            "reading": "L 하한 > 0 → TM4보다 짧다(학습의 몫), 상한 < 0 → 길다, 그 밖 구분 못 함",
            "paired_comparisons": [["PT1", "TM4"], ["PT1", "OM4"], ["PT1", "P1M4"], ["TM4", "P1M4"], ["OM4", "TM4"]],
            "confirmation": "주 판정 통과일 때만 확인 84상황(P)에서 M4·TM4(새로 돌림)와 PT1을 한 번: ARENA3 통과 + L 하한 > 0",
            "next_round": "확인 통과 시 PT1 난수 42가 2라운드 운전자(2라운드는 결과를 본 뒤 설계)"},
        "reporting": ("반복 사용 날짜의 SUMO 개발 비교. 무리 도착은 모형이고 시간표는 무리를 만든 것과 같다. "
                      "최종 제어기는 신경망 + TM4 관측 계산(실행 때 롤아웃 없음)."),
        "situations_data": data, "situations_eval": evaluation, "situations_confirmation": confirmation,
    }
    CONFIG.write_text(json.dumps(w.plain(cfg), ensure_ascii=False, indent=1) + "\n")
    return {"config": str(CONFIG), "sha256": w.file_hash(CONFIG), "data": len(data), "eval": len(evaluation),
            "confirmation": len(confirmation)}


# ----------------------------------------------------------------------------- 명령
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("make-config")
    p.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("smoke")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    p.add_argument("--platoon0", type=Path, help="PLATOON0 결과 폴더(평가 상황에서 학습 전 학생 = PLATOON0 TM4 확인)")
    p.add_argument("--points", type=int, default=3)
    p = sub.add_parser("collect")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    import explore_platoon1_learn as pl
    pl.add_commands(sub)
    args = parser.parse_args()
    w.configure()
    torch.set_num_threads(1)
    if args.cmd == "make-config":
        print(json.dumps(make_config(args.workers), ensure_ascii=False, indent=1))
        return 0
    cfg = config()
    if args.cmd == "smoke":
        report = smoke(cfg, args.out, args.work, args.workers, args.platoon0, args.points)
        return 0 if report["passed"] else 2
    if args.cmd == "collect":
        if platform.system() == "Darwin":
            parser.error("자료 모으기는 Colab에서만 실행합니다(맥은 작은 실행까지).")
        if not 0 <= args.shard < args.shards:
            raise SystemExit("--shard는 0 이상 --shards 미만이어야 합니다.")
        return collect(cfg, args.out, args.work, args.shard, args.shards, args.workers)
    return pl.run_command(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
