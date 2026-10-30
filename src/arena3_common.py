"""ARENA3 공통 부분 — 세 학습기를 같은 심판으로 겨루기(진행 기록 208절, docs/ARENA3_설계_20261027.md).

- 잠금 설정 config/arena3_v1.json을 읽고 검사한다. 잠금은 `python src/arena3_common.py lock`으로 한 번 만든다.
- 도착 배치: focused_scenario.route_seed를 실행할 때만 바꿔 끼운다. 환경 파일은 그대로라 환경 지문이 같다.
  배치 이름이 None이면 원래 함수를 그대로 쓴다(이전 실험과 같은 차량 파일).
- 6모드 계획기: WEIGHT1 계획기(모드 0~4)를 그대로 부르고, W4(4,4)의 첫 녹색과 선택 지점의 M4 녹색별 비용만 더한다.
- 공통 관측 119칸 = WEIGHT1 110칸 + W4 첫 녹색 4칸 + W4 유효 1칸 + M4 녹색별 비용 차이 4칸(선택 지점에서만, 아니면 0).
- 에피소드 실행기: WEIGHT1 run_episode와 같은 물리·정산·지표를 쓰고, 정책은 녹색 번호를 돌려준다.
  선택 지점이 아니면 여섯 모드가 같은 녹색을 내므로 정책을 부르지 않고 그 녹색을 실행한다.

잠긴 기존 파일(focused_*, explore_weight_*, explore_pred, explore_rollout)은 고치지 않고 불러 쓴다.
"""
from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import explore_weight as w  # noqa: E402

CONFIG = ROOT / "config/arena3_v1.json"
EXPERIMENT = "ARENA3_V1"
MODES6 = ((1.0, 1.0), (1.5, 1.0), (1.0, 1.5), (1.5, 1.5), (2.0, 2.0), (4.0, 4.0))
M4, W4 = 4, 5
N_MODES = 6
BASE_SIZE = 110
OBSERVATION_SIZE = 119
W4_ONEHOT = slice(110, 114)
W4_MASK = 114
M4_GAPS = slice(115, 119)
BASE_ONEHOT = slice(84, 104)
BASE_MASK = slice(104, 109)
SPLIT_SALT = "ARENA3_SPLIT_20261027"
# 학습 상황 15분 칸: 06:00~21:45(제어 60분 + 꼬리 15분이 23:00 안에 끝난다).
TRAIN_SLOTS = tuple(f"{h:02d}{m:02d}" for h in range(6, 22) for m in (0, 15, 30, 45))


# ----------------------------------------------------------------------------- 설정·난수
def config() -> dict:
    return w.read(CONFIG)


def seed32(*parts) -> int:
    """'ARENA3|부분|…'의 SHA-256 앞 4바이트(SUMO가 받는 31비트 범위)."""
    raw = "|".join(("ARENA3", *[str(p) for p in parts])).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big") % (2 ** 31 - 1)


def arrival_seed(tag: str, date: str, period: str) -> int:
    return seed32("ARRIVAL", tag, date, period)


def sumo_seed(tag: str, date: str, period: str) -> int:
    return seed32("SUMO", tag, date, period)


def situation_id(s: dict) -> str:
    return f"{s['date']}_{s['period']}_{s['tag']}"


def make_situation(date: str, period: str, tag: str | None, seed: int | None = None) -> dict:
    """tag=None은 원래 배치(이전 실험과 같은 차량 파일)이며, 이때는 SUMO 난수를 직접 준다."""
    if tag is None:
        if seed is None:
            raise ValueError("원래 배치에는 SUMO 난수를 직접 정해야 합니다.")
        return {"date": str(date), "period": str(period), "tag": None, "sumo_seed": int(seed)}
    return {"date": str(date), "period": str(period), "tag": str(tag),
            "sumo_seed": int(sumo_seed(tag, date, period) if seed is None else seed)}


def split_dates(split: dict, salt: str = SPLIT_SALT) -> dict:
    """날짜 종류별로 sha256(salt|날짜) 순서의 앞 k개를 평가 날짜로(평일 4·주말 2·추석 1)."""
    usable = set(split["train"]) | set(split["validation"])
    want = {"weekday": 4, "weekend": 2, "chuseok_holiday": 1}
    out = {"eval": [], "train": []}
    for kind, k in want.items():
        dates = sorted((d for d in split["day_types"][kind] if d in usable),
                       key=lambda d: hashlib.sha256(f"{salt}|{d}".encode()).hexdigest())
        out["eval"] += sorted(dates[:k])
        out["train"] += sorted(dates[k:])
    if set(out["eval"]) & set(out["train"]) or set(out["eval"]) | set(out["train"]) != usable:
        raise ValueError("평가·학습 날짜 나누기가 사용 가능한 15일과 맞지 않습니다.")
    return out


# ----------------------------------------------------------------------------- 도착 배치
w.configure()                     # focused_* 모듈이 환경 변수(pet1·북쪽 240 m)를 읽기 전에 정한다
import focused_scenario as _fs  # noqa: E402

ORIGINAL_ROUTE_SEED = _fs.route_seed


@contextlib.contextmanager
def arrival_placement(tag: str | None):
    """with 안에서만 (날짜, 15분 시간대)의 차량 난수를 배치 이름으로 바꾼다. 수요 기록·차종·경로 규칙은 같다."""
    if _fs.route_seed is not ORIGINAL_ROUTE_SEED:
        raise RuntimeError("도착 배치가 이미 바뀐 상태에서 다시 바꿀 수 없습니다.")
    if tag is None:
        yield
        return
    _fs.route_seed = lambda date, period: arrival_seed(tag, str(date), str(period))
    try:
        yield
    finally:
        _fs.route_seed = ORIGINAL_ROUTE_SEED


def route_file_sha(situation: dict, work: Path) -> str:
    """심판 에피소드와 같은 창(준비 30분 + 제어 60분 + 꼬리 15분)의 차량 파일 SHA-256. SUMO 없이 만든다."""
    import focused_train as ft
    from focused_data import COUNTS, load_counts
    from explore_weight_episode import CONTROL_S, TAIL_S
    origin = _fs.period_start_s(situation["period"]) - ft.WARMUP_S
    path = Path(work) / f"{situation_id(situation)}.rou.xml"
    with arrival_placement(situation["tag"]):
        _fs.window_route_file(situation["date"], origin, origin + ft.WARMUP_S + CONTROL_S + TAIL_S, path,
                              origin_s=origin, counts=load_counts(COUNTS))
    sha = w.file_hash(path)
    path.unlink()
    return sha


# ----------------------------------------------------------------------------- 6모드 계획기·관측
def canonical_mask6(greens) -> list[bool]:
    seen, mask = set(), []
    for g in greens:
        mask.append(g not in seen)
        seen.add(g)
    return mask


def lefts_weights(factors: tuple[float, float], moves) -> tuple[float, ...]:
    from explore_weight_core import LEFTS
    east, south = factors
    return tuple(east if m == LEFTS["left_east"] else south if m == LEFTS["left_south"] else 1.0 for m in moves)


def observation119(base_obs, greens, mask, m4_costs) -> np.ndarray:
    base_obs = np.asarray(base_obs, dtype=np.float32)
    if base_obs.shape != (BASE_SIZE,) or len(greens) != N_MODES or len(mask) != N_MODES:
        raise ValueError("ARENA3 관측은 WEIGHT1 110칸과 6모드가 필요합니다.")
    gaps = [0.0] * 4
    if m4_costs is not None:
        finite = [c for c in m4_costs if c is not None]
        low = min(finite)
        gaps = [0.0 if c is None else min(10.0, max(0.0, (float(c) - low) / 1000.0)) for c in m4_costs]
    w4 = [float(greens[W4] == g) for g in range(4)]
    obs = np.asarray([*base_obs, *w4, float(mask[W4]), *gaps], dtype=np.float32)
    if obs.shape != (OBSERVATION_SIZE,) or not np.all(np.isfinite(obs)):
        raise ValueError("ARENA3 관측 119칸이 올바르지 않습니다.")
    return obs


@dataclass
class ArenaSnapshot:
    base: object                      # WEIGHT1 ModeSnapshot(모드 0~4, 원래 그대로)
    greens: list[int]                 # 모드 0~5의 첫 녹색
    mask: list[bool]                  # 녹색마다 가장 작은 모드만 유효
    choice: bool                      # 여섯 모드의 첫 녹색이 둘 이상으로 갈림
    observation: np.ndarray           # 119칸
    m4_costs: list[float | None] | None = None
    w4_cost: float | None = None
    planning_seconds: float = 0.0

    @property
    def allowed(self) -> list[int]:
        return list(self.base.allowed)

    @property
    def p0_action(self) -> int:
        return int(self.base.p0_action)

    @property
    def now(self) -> float:
        return float(self.base.now)

    @property
    def candidates(self) -> list[int]:
        """서로 다른 첫 녹색(모드 순서)."""
        return [g for g, ok in zip(self.greens, self.mask) if ok]

    def green(self, mode: int) -> int:
        if not 0 <= int(mode) < N_MODES or not self.mask[int(mode)]:
            raise ValueError(f"모드 {mode}는 유효하지 않거나 다른 모드와 같은 녹색입니다.")
        return int(self.greens[int(mode)])


def _planner_class():
    import explore_pred as ep
    from explore_weight_core import ModePlanner, weighted_plan

    class ArenaPlanner(ModePlanner):
        """WEIGHT1 ModePlanner.snapshot을 그대로 부른 뒤, 같은 순간의 줄·도착으로 W4와 M4 녹색별 비용을 더한다."""

        def __init__(self, movement_of: dict, greens: list[str], control_end: float):
            super().__init__(movement_of, greens, control_end)
            self._arena_env = self._arena = None

        def arena_snapshot(self, env) -> ArenaSnapshot:
            from focused_data import TARGET_TLS
            base = self.snapshot(env)
            if self._arena is not None and self._arena_env is env and self._arena.now == base.now:
                return self._arena
            started = time.perf_counter()
            allowed = list(base.allowed)
            m4_costs, w4_cost = None, None
            if len(allowed) == 1:
                greens = [allowed[0]] * N_MODES
            else:
                # 부모 snapshot과 같은 계산: 읽기만 하는 함수라 같은 순간에 다시 불러도 값이 같다.
                ts, guard, now = env.traffic_signals[TARGET_TLS], env.signal_constraints, float(env.sim_step)
                q0, arrivals = self.p0.queues_and_arrivals(env)
                served = guard._served_now(ts, now)
                state = ep.SignalState(ts.green_phase, now - ts.time_since_last_phase_change,
                                       float(ts.yellow_time), tuple(served[c] for c in self.p0._model.crosswalks))
                moves = self.p0.movements
                w4, info = weighted_plan(self.p0._model, state, now, q0, arrivals, allowed,
                                         lefts_weights(MODES6[W4], moves))
                w4_cost = float(info["cost"])
                greens = [int(a) for a in base.mode_actions] + [int(w4)]
                if len(set(greens)) > 1:
                    m4_costs = [None] * 4
                    for a in allowed:
                        _, info = weighted_plan(self.p0._model, state, now, q0, arrivals, [a],
                                                lefts_weights(MODES6[M4], moves))
                        m4_costs[a] = float(info["cost"])
            if any(g not in allowed for g in greens):
                raise RuntimeError("6모드 계획기가 실행할 수 없는 녹색을 냈습니다.")
            mask = canonical_mask6(greens)
            choice = len(set(greens)) > 1
            obs = observation119(base.observation, greens, mask, m4_costs if choice else None)
            self._arena_env, self._arena = env, ArenaSnapshot(
                base=base, greens=greens, mask=mask, choice=choice, observation=obs, m4_costs=m4_costs,
                w4_cost=w4_cost, planning_seconds=float(base.planning_seconds) + time.perf_counter() - started)
            return self._arena

    return ArenaPlanner


def make_planner(ep_obj, control_end: float):
    return _planner_class()(ep_obj.movement_of, [p.state for p in ep_obj.ts.green_phases], control_end)


# ----------------------------------------------------------------------------- 정책
class FixedMode:
    """고정 모드(M4=4, W4=5)."""

    def __init__(self, mode: int, name: str):
        self.mode, self.name = int(mode), name

    def decide(self, snap: ArenaSnapshot, ctx) -> int:
        return int(snap.greens[self.mode])


class P0Policy:
    """원래 P0 제어기를 그대로 부른다(WEIGHT1 run_episode의 policy=None과 같음)."""
    name = "P0"

    def decide(self, snap: ArenaSnapshot, ctx) -> int:
        action = int(ctx.ep.ctl(ctx.ep.env, ctx.allowed))
        ctx.mismatch += int(action != snap.p0_action)
        return action


def fixed_policy(name: str):
    if name == "P0":
        return P0Policy()
    if name == "M4":
        return FixedMode(M4, "M4")
    if name == "W4":
        return FixedMode(W4, "W4")
    raise ValueError(f"고정 정책 이름이 아닙니다: {name}")


# ----------------------------------------------------------------------------- 에피소드
@dataclass
class ArenaResult:
    metrics: dict
    decisions: list[dict]
    rewards: list[float]              # 제어 720단계 보상(꼬리는 마지막 단계에 접힘, WEIGHT1과 같음)
    choices: list[dict] = field(default_factory=list)   # 선택 지점: step, observation(119), greens, mask, green
    seconds: float = 0.0


def run_arena_episode(situation: dict, work: Path, policy, *, max_seconds: float | None = None,
                      trace: bool = True, keep_observations: bool = False,
                      on_step: Callable | None = None) -> ArenaResult:
    """실제 신호 30분 → 정책 60분 → P0 꼬리 900초. WEIGHT1 run_episode와 같은 정산·지표.

    policy.decide(snap, ctx)는 선택 지점에서만 불린다(아니면 여섯 모드의 같은 녹색을 실행).
    ctx: ep(에피소드), allowed, step(0~719), situation, mismatch(P0 확인용 누적), planner.
    on_step(ctx, snap, green, executed)는 매 제어 단계 실행 직후 불린다(B′ 복원 확인용).
    """
    from explore_weight_episode import (CONTROL_S, STEP_S, TAIL_S, _Episode, _exact_metrics, cohort_fingerprint,
                                        cohort_vehicle_ids, fold_tail_rewards, read_tripinfo, reconcile_costs)
    from focused_eval import judge_collisions

    started = time.monotonic()
    if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
        raise TimeoutError("에피소드 계산 예산이 없습니다.")

    def check_limit():
        if max_seconds is not None and time.monotonic() - started > max_seconds:
            raise TimeoutError("ARENA3 에피소드 계산 예산을 초과했습니다.")

    work = Path(work).resolve()
    with arrival_placement(situation["tag"]):
        ep = _Episode(situation["date"], situation["period"], int(situation["sumo_seed"]), work,
                      deadline=None if max_seconds is None else started + max_seconds)
    ep.date, ep.period = situation["date"], situation["period"]
    decisions, choices = [], []
    mode_errors = 0
    boundaries = [ep.handover]
    cohort = cohort_vehicle_ids(ep.vehicles, ep.handover)
    cohort_set = set(cohort)
    snapshots: list[dict] = []
    departures: dict[str, float] = {}
    control_end = ep.handover + CONTROL_S
    end = control_end + TAIL_S
    ctx = SimpleNamespace(ep=ep, allowed=None, step=-1, situation=situation, mismatch=0, planner=None)

    def capture():
        s = ep.env.sumo
        snapshots.append({v: float(s.vehicle.getTimeLoss(v)) for v in s.vehicle.getIDList() if v in cohort_set})
        for v in s.vehicle.getIDList():
            if v in cohort_set and v not in departures:
                departures[v] = float(s.vehicle.getDeparture(v))

    original_step = ep.env._sumo_step

    def step_with_ledger():
        original_step()
        s = ep.env.sumo
        for v in s.simulation.getDepartedIDList():
            if v in cohort_set:
                departures[v] = float(s.vehicle.getDeparture(v))
        at = float(ep.env.sim_step)
        elapsed = at - ep.handover
        if elapsed > 0 and abs(elapsed / STEP_S - round(elapsed / STEP_S)) < 1e-9:
            boundaries.append(at)
            capture()

    ep.env._sumo_step = step_with_ledger
    planning = 0.0
    try:
        check_limit()
        planner = make_planner(ep, control_end)
        ctx.planner = planner
        capture()
        step = 0
        while float(ep.env.sim_step) < end - 1e-9:
            check_limit()
            now = float(ep.env.sim_step)
            in_control = now < control_end - 1e-9
            allowed = list(ep.cons.allowed(ep.env))
            if not allowed:
                raise RuntimeError("실행 가능한 현시가 없습니다.")
            snap = None
            if in_control:
                snap = planner.arena_snapshot(ep.env)
                planning += snap.planning_seconds
                ctx.allowed, ctx.step = allowed, step
                if list(snap.allowed) != allowed:
                    raise RuntimeError("계획기와 운영 규칙의 실행 가능 녹색이 다릅니다.")
                if snap.choice or isinstance(policy, P0Policy):
                    action = int(policy.decide(snap, ctx))
                else:
                    action = int(snap.greens[0])
            else:
                action = int(ep.ctl(ep.env, allowed))
            before = now
            corrected, executed = ep.apply(action)
            after = float(ep.env.sim_step)
            if abs(after - before - STEP_S) > 1e-9:
                raise RuntimeError(f"5초 전이가 아닙니다: {before} → {after}")
            mode_errors += int(corrected != action or executed != action or action not in allowed)
            if in_control:
                if snap.choice:
                    row = {"step": step, "now": now, "greens": list(snap.greens), "mask": list(snap.mask),
                           "green": int(executed)}
                    if keep_observations:
                        row["observation"] = snap.observation.copy()
                    choices.append(row)
                if trace:
                    decisions.append({"now": now, "allowed": [int(a) for a in allowed], "greens": list(snap.greens),
                                      "choice": bool(snap.choice), "p0_action": int(snap.p0_action),
                                      "executed_green": int(executed)})
                if on_step is not None:
                    on_step(ctx, snap, action, executed)
                step += 1
            if ep.done and after < end - 1e-9:
                raise RuntimeError("제어/꼬리 구간이 끝나기 전에 SUMO가 종료했습니다.")
        check_limit()
    finally:
        ep.env._sumo_step = original_step
        ep.close()

    rows = read_tripinfo(ep.tripinfo)
    costs = reconcile_costs(ep.vehicles, rows, boundaries, snapshots, handover=ep.handover,
                            actual_departures=departures)
    rewards, reward_parts = fold_tail_rewards(costs["increments"])
    metrics = _exact_metrics(rows, ep.vehicles, ep.handover)
    collision = judge_collisions(ep.collisions)
    executed = [d["executed_green"] for d in decisions]
    metrics.update({
        "experiment": EXPERIMENT, "situation": situation_id(situation) if situation["tag"] else
        f"{situation['date']}_{situation['period']}_s{situation['sumo_seed']}",
        "date": situation["date"], "period": situation["period"], "tag": situation["tag"],
        "sumo_seed": int(situation["sumo_seed"]), "policy": getattr(policy, "name", type(policy).__name__),
        "handover": ep.handover, "control_s": CONTROL_S, "tail_s": TAIL_S, "seconds": time.monotonic() - started,
        "actual_total_delay": costs["actual_total_delay"], "reconstructed_total_delay": costs["reconstructed_total_delay"],
        "cost_check": costs["cost_check"], "cost_check_passed": bool(costs["cost_check"]["passed"]), **reward_parts,
        "changed_by_rules": int(ep.cons.interventions), "rule_decided": dict(ep.cons.rule_decided),
        "mode_action_errors": int(mode_errors), "p0_mismatch": int(ctx.mismatch),
        "collisions_at_204820": int(collision["collisions_at_204820"]),
        "collisions_unexplained": len(collision["collisions_unexplained"]), "collision_details": collision,
        "route_sha256": hashlib.sha256(ep.route.read_bytes()).hexdigest(),
        "cohort_sha256": cohort_fingerprint(ep.vehicles, cohort),
        "choice_points": len(choices), "planning_seconds": planning,
        "executed_sha16": hashlib.sha256(json.dumps(executed).encode()).hexdigest()[:16] if trace else None,
        "code": {name: w.file_hash(ROOT / name)[:12] for name in ("src/arena3_common.py", "src/explore_weight_episode.py",
                                                                  "src/explore_weight_core.py")}})
    if trace and len(decisions) != CONTROL_S // STEP_S:
        raise RuntimeError(f"제어 결정이 720개가 아닙니다: {len(decisions)}")
    (work / "cost_ledger.json").write_text(json.dumps({k: v for k, v in costs.items() if k != "cohort"},
                                                      ensure_ascii=False, indent=1))
    return ArenaResult(metrics, decisions, list(rewards), choices, time.monotonic() - started)


def write_run(target: Path, result: ArenaResult, extra: dict) -> dict:
    """metrics.json + decisions.jsonl.gz + cache.json(해시). 같은 형식을 심판과 학습기가 함께 쓴다."""
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    metrics = {**result.metrics, **extra}
    with gzip.open(target / "decisions.jsonl.gz", "wt") as f:
        for row in result.decisions:
            f.write(json.dumps(w.plain(row), ensure_ascii=False, allow_nan=False) + "\n")
    w.write(target / "metrics.json", metrics)
    w.write(target / "cache.json", {"fingerprint": extra.get("fingerprint"), "model_identity": extra.get("model_identity"),
                                    "metrics_sha256": w.file_hash(target / "metrics.json"),
                                    "decisions_sha256": w.file_hash(target / "decisions.jsonl.gz")})
    return metrics


def read_run(target: Path, fingerprint: str, model_identity=None) -> dict | None:
    target = Path(target)
    cache = target / "cache.json"
    if not cache.exists() or not (target / "metrics.json").exists():
        return None
    saved = w.read(cache)
    if (saved.get("fingerprint") != fingerprint or saved.get("model_identity") != model_identity
            or saved.get("metrics_sha256") != w.file_hash(target / "metrics.json")
            or saved.get("decisions_sha256") != w.file_hash(target / "decisions.jsonl.gz")):
        raise ValueError(f"재사용할 실행의 지문·해시 불일치: {target}")
    return w.read(target / "metrics.json")


# ----------------------------------------------------------------------------- 잠금 만들기(맥, SUMO 없음)
def eval_situations(cfg: dict, part: str = "eval") -> list[dict]:
    tags = cfg["evaluation"]["placements"] if part == "eval" else cfg["evaluation"]["confirmation_placements"]
    return [make_situation(d, p, t) for d in cfg["dates"]["eval"] for p in cfg["evaluation"]["periods"] for t in tags]


def bprime_data_situations(train_dates: list[str], per_date: int = 8) -> list[dict]:
    out, n = [], 0
    for d in train_dates:
        slots = sorted(TRAIN_SLOTS, key=lambda p: hashlib.sha256(f"ARENA3_BPRIME_SLOTS|{d}|{p}".encode()).hexdigest())
        for p in sorted(slots[:per_date]):
            n += 1
            out.append(make_situation(d, p, f"T{n}"))
    return out


def build_lock(work: Path) -> dict:
    """설계(208절)의 값으로 잠금 설정을 만든다. 차량 파일 SHA는 맥에서 SUMO 없이 계산한다."""
    w.configure()
    from focused_data import COUNTS, environment_fingerprint, load_counts
    split = w.read(ROOT / "config/focused_split.json")
    dates = split_dates(split)
    counts = load_counts(COUNTS)
    cfg = {
        "experiment": EXPERIMENT,
        "design": "docs/ARENA3_설계_20261027.md, PROJECT_PROGRESS.md 208절",
        "environment": {"fingerprint": environment_fingerprint(), "rule_set": "pet1", "north_extension_m": 240,
                        "p0_code": w.file_hash(ROOT / "src/explore_pred.py")[:12]},
        "dates": {"salt": SPLIT_SALT, "eval": dates["eval"], "train": dates["train"],
                  "not_used": {"test": split["test"], "note": "시험 4일·새 12일은 이미 사용해 쓰지 않음"}},
        "modes": [list(m) for m in MODES6], "m4_mode": M4, "w4_mode": W4,
        "observation": {"size": OBSERVATION_SIZE, "base": [0, BASE_SIZE], "w4_onehot": [110, 114], "w4_mask": W4_MASK,
                        "m4_gaps": [115, 119], "m4_gap_scale": 1000.0, "m4_gap_clip": 10.0},
        "evaluation": {"periods": ["0730", "1000", "1230", "1500", "1730", "2100"], "placements": ["E1", "E2"],
                       "confirmation_placements": ["C1", "C2"], "episode": {"warmup_s": 1800, "control_s": 3600,
                                                                           "tail_s": 900, "step_s": 5, "tail_policy": "P0"},
                       "baselines": ["P0", "M4", "W4"], "reference": "M4", "sensitivity_reference": "P0"},
        "judge": {"min_gain_s": 0.3, "date_t_df": 6, "t975": 2.446911851144969, "min_positive_dates": 5,
                  "left_tolerance_s": 5.0, "long_wait_rule": "300초 초과 차량 수가 M4 이하",
                  "collision_rule": "정책 고유 미확인 충돌 0; M4·P0와 같은 사건은 정책 무관 후보로 따로 적고 재현 진단 전 통과 보류",
                  "learner_models": 3, "learner_aggregate": "상황마다 모델 3개의 총지연 평균",
                  "winner": "통과한 학습기 중 주 지표 최대 → 확인 배치 C1·C2 84상황에서 같은 조건 재통과",
                  "gate1_bprime_min_gain_s": 0.3},
        "reproduction": {"note": "원래 배치(배치 이름 없음)에서 이전 실행과 소수점까지 같아야 한다",
                         "cases": [
                             {"date": "20220812", "period": "0730", "sumo_seed": 42, "policy": "P0",
                              "source": "explore_fixed24_v1/runs/sweep/P0_20220812_0730_s42",
                              "mean_delay_s": 36.57310599606815, "actual_total_delay": 223242.239,
                              "executed_sha16": "a2872723dc24cf0c",
                              "route_sha256": "796f57316662f847950e5289580706a34ccd054a4385592cc80b0a4e4a9d742d"},
                             {"date": "20220812", "period": "0730", "sumo_seed": 42, "policy": "M4",
                              "source": "explore_fixed24_v1/runs/sweep/M4_20220812_0730_s42",
                              "mean_delay_s": 34.644466087811274, "actual_total_delay": 211469.821,
                              "executed_sha16": "58e89cf18dac75c0",
                              "route_sha256": "796f57316662f847950e5289580706a34ccd054a4385592cc80b0a4e4a9d742d"},
                             {"date": "20220812", "period": "0730", "sumo_seed": 42, "policy": "W4",
                              "source": "explore_fixed24_v1/runs/sweep/W4_20220812_0730_s42",
                              "mean_delay_s": 35.64195543905636, "actual_total_delay": 217558.496,
                              "executed_sha16": "04e378ba7fee628b",
                              "route_sha256": "796f57316662f847950e5289580706a34ccd054a4385592cc80b0a4e4a9d742d"}]},
        "bprime": {"horizon_s": 300, "terminal_s": 75.0, "futures": 8, "history_s": 900, "continuation": "M4",
                   "rule": "M4가 아닌 후보 a의 짝 이득 평균 g>0이고 g ≥ 1.0×표준오차이면 a(여럿이면 g 최대), 아니면 M4",
                   "se_multiplier": 1.0, "future_seed": "seed32(BPRIME_FUTURE, 상태 이름, 번호)",
                   "restore_check": {"tolerance": 1e-6, "max_mismatch_rate": 0.05, "min_points": 40},
                   "keep_state_fraction": 0.1, "data_per_date": 8},
        "g1": {"seeds": [42, 43, 44], "network": [64, 64], "dueling_anchor": "M4", "gamma_per_5s": 0.99,
               "n_choice_steps": 3, "learning_rate": 0.001, "batch_size": 64, "replay_size": 50000,
               "learning_starts": 500, "updates_per_transition": 8, "target_every_updates": 1250,
               "epsilon_start": 0.5, "epsilon_end": 0.1, "epsilon_decay_choices": 3000, "reward_scale": 1000.0,
               "cpu_hours_per_seed": 20.0, "parallel_episodes": 2, "max_discarded": 5,
               "milestone_control_steps": [200000, 400000], "train_slots": [TRAIN_SLOTS[0], TRAIN_SLOTS[-1]]},
        "budget": {"learner_cpu_hours": 60.0, "accounting": "시뮬레이션·롤아웃·경사 갱신·저장 포함, 심판 평가 제외",
                   "compute_limit": "해제(207절); 학습기 사이 공정성용 예산만 둔다"},
        "drive_folders": {"lock": "arena3/lock", "baseline": "arena3/referee/baseline", "bprime_eval": "arena3/referee/bprime",
                          "g1_eval": "arena3/referee/G1", "bprime_data": "arena3/bprime_data", "g1_train": "arena3/G1_train"},
    }
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    groups = {"eval": eval_situations(cfg, "eval"), "confirmation": eval_situations(cfg, "confirmation"),
              "bprime_data": bprime_data_situations(dates["train"], cfg["bprime"]["data_per_date"])}
    for name, sits in groups.items():
        for s in sits:
            if (s["date"], s["period"]) not in counts:
                raise ValueError(f"수요 자료에 없는 상황: {s}")
            s["arrival_seed_example"] = arrival_seed(s["tag"], s["date"], s["period"])
            s["route_sha256"] = route_file_sha(s, work)
        cfg[f"situations_{name}"] = sits
    shas = [s["route_sha256"] for sits in groups.values() for s in sits]
    if len(shas) != len(set(shas)):
        raise ValueError("서로 다른 상황의 차량 파일이 같습니다.")
    return cfg


def validate_lock(cfg: dict | None = None) -> dict:
    cfg = cfg or config()
    split = w.read(ROOT / "config/focused_split.json")
    if cfg.get("experiment") != EXPERIMENT or split_dates(split) != {k: cfg["dates"][k] for k in ("eval", "train")}:
        raise ValueError("잠금 설정의 실험 이름 또는 날짜 나누기가 다릅니다.")
    if [tuple(m) for m in cfg["modes"]] != list(MODES6):
        raise ValueError("잠금 설정의 6모드가 코드와 다릅니다.")
    ev, cf, bd = cfg["situations_eval"], cfg["situations_confirmation"], cfg["situations_bprime_data"]
    if len(ev) != 84 or len(cf) != 84 or len(bd) != 64:
        raise ValueError("상황 수가 84·84·64가 아닙니다.")
    if {s["date"] for s in ev} != set(cfg["dates"]["eval"]) or {s["date"] for s in bd} != set(cfg["dates"]["train"]):
        raise ValueError("평가·자료 상황의 날짜가 잠근 나누기와 다릅니다.")
    for s in [*ev, *cf, *bd]:
        if s["sumo_seed"] != sumo_seed(s["tag"], s["date"], s["period"]):
            raise ValueError(f"SUMO 난수가 규칙과 다릅니다: {s}")
    if len({situation_id(s) for s in [*ev, *cf, *bd]}) != 232:
        raise ValueError("상황 이름이 겹칩니다.")
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("lock", help="잠금 설정을 만든다(이미 있으면 같은지만 확인)")
    p.add_argument("--work", type=Path, required=True)
    p = sub.add_parser("check-routes", help="잠금 설정의 차량 파일 SHA를 다시 계산해 대조한다(SUMO 없음)")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--part", choices=("eval", "confirmation", "bprime_data", "all"), default="eval")
    args = parser.parse_args()
    if args.cmd == "lock":
        cfg = build_lock(args.work)
        validate_lock(cfg)
        if CONFIG.exists():
            if w.read(CONFIG) != json.loads(json.dumps(w.plain(cfg), ensure_ascii=False)):
                raise SystemExit("이미 잠근 설정과 다릅니다. 잠금은 바꾸지 않습니다.")
            print("잠금 설정이 이미 있고 같습니다.")
        else:
            w.write(CONFIG, cfg)
            print(f"잠금 설정을 만들었습니다: {CONFIG} sha256={w.file_hash(CONFIG)}")
        return 0
    cfg = validate_lock()
    w.configure()
    parts = ("eval", "confirmation", "bprime_data") if args.part == "all" else (args.part,)
    bad = 0
    for part in parts:
        for s in cfg[f"situations_{part}"]:
            got = route_file_sha(s, args.work)
            bad += got != s["route_sha256"]
            if got != s["route_sha256"]:
                print(f"불일치 {situation_id(s)}: {got} ≠ {s['route_sha256']}")
    print(f"차량 파일 대조: {sum(len(cfg[f'situations_{p}']) for p in parts)}개 중 불일치 {bad}개")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
