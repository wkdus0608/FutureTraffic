"""WEIGHT1 SUMO episodes and exact, retrospective cohort reward accounting.

The controller only receives the 110 current-state features.  Route departure
times and final tripinfo are used by this module solely for reward/evaluation
bookkeeping.  No transition is returned to a learner until tripinfo has closed
and the final loss of arriving vehicles has been reconciled.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Callable
import xml.etree.ElementTree as ET

import numpy as np

STEP_S = 5
CONTROL_S = 3600
TAIL_S = 900
GAMMA = 0.99
REWARD_SCALE = 1000.0
ACCOUNTING_ATOL = 1e-6


class CostReconciliationError(ValueError):
    """A boundary snapshot cannot be reconciled with the vehicle record."""


@dataclass
class EpisodeResult:
    metrics: dict
    transitions: list[dict]
    decisions: list[dict]
    seconds: float


def read_tripinfo(path: Path) -> dict[str, dict]:
    return {t.get("id"): dict(t.attrib) for t in ET.parse(path).getroot().iter("tripinfo")}


def cohort_vehicle_ids(vehicles: dict, handover: float, control_s: float = CONTROL_S) -> list[str]:
    """A policy-independent cohort, selected by desired departure, not arrival."""
    return sorted(v for v, (dep, *_rest) in vehicles.items()
                  if handover <= float(dep) < handover + control_s)


def cohort_fingerprint(vehicles: dict, cohort: list[str]) -> str:
    raw = [(v, float(vehicles[v][0])) for v in sorted(cohort)]
    return hashlib.sha256(json.dumps(raw, separators=(",", ":")).encode()).hexdigest()


def reconcile_costs(vehicles: dict, rows: dict, boundaries: list[float],
                    live_time_loss: list[dict[str, float]], *,
                    handover: float, control_s: float = CONTROL_S,
                    actual_departures: dict[str, float] | None = None) -> dict:
    """Reconstruct exact cumulative delay at every boundary, then difference it.

    At a boundary an arrived vehicle contributes its *final* tripinfo timeLoss;
    an active vehicle contributes its live TraCI timeLoss.  Thus arriving at the
    boundary cannot drop a final second or count the vehicle twice.  Insertion
    wait is min(boundary, actual departure)-desired departure, including the
    fractional first second.  Never-inserted vehicles accrue wait until the end
    and are still flagged as missing by the evaluation metrics.
    """
    if len(boundaries) != len(live_time_loss) or len(boundaries) < 2:
        raise CostReconciliationError("경계 시각과 차량 손실 기록의 길이가 다릅니다.")
    times = [float(t) for t in boundaries]
    if abs(times[0] - handover) > ACCOUNTING_ATOL or any(b <= a for a, b in zip(times, times[1:])):
        raise CostReconciliationError("비용 경계 시각이 인계 시각부터 증가해야 합니다.")
    if any(not math.isfinite(t) for t in times):
        raise CostReconciliationError("비용 경계 시각에 비정상 값이 있습니다.")
    cohort = cohort_vehicle_ids(vehicles, handover, control_s)
    departures = dict(actual_departures or {})
    for v in cohort:
        row = rows.get(v)
        if row is not None:
            dep = float(row.get("depart", -1))
            if dep >= 0:
                if v in departures and abs(departures[v] - dep) > ACCOUNTING_ATOL:
                    raise CostReconciliationError(f"차량 {v}의 실제 출발 시각이 다릅니다.")
                departures[v] = dep
                if "departDelay" in row:
                    expected = dep - float(vehicles[v][0])
                    if abs(expected - float(row["departDelay"])) > ACCOUNTING_ATOL:
                        raise CostReconciliationError(f"차량 {v}의 진입 대기 정산이 다릅니다.")
    cumulative, tl_parts, insertion_parts = [], [], []
    for at, snapshot in zip(times, live_time_loss):
        losses, insertion = [], []
        for v in cohort:
            desired = float(vehicles[v][0])
            departure = departures.get(v, math.inf)
            insertion.append(max(0.0, min(at, departure) - desired))
            row = rows.get(v)
            arrival = float(row.get("arrival", -1)) if row is not None else -1.0
            if row is not None and 0 <= arrival <= at + 1e-9:
                loss = float(row.get("timeLoss", 0.0))
            elif v in snapshot:
                loss = float(snapshot[v])
            elif departure < at - 1e-9:
                raise CostReconciliationError(f"{at}초에 활동 중인 차량 {v}의 손실 기록이 없습니다.")
            else:
                loss = 0.0
            # SUMO can report small negative timeLoss, and the final tripinfo
            # value may be rounded differently from the preceding TraCI value.
            # Preserve both signs: clipping would change the evaluation target.
            if not math.isfinite(loss):
                raise CostReconciliationError(f"차량 {v}의 시간 손실이 올바르지 않습니다.")
            losses.append(loss)
        tl, dd = math.fsum(losses), math.fsum(insertion)
        tl_parts.append(tl)
        insertion_parts.append(dd)
        cumulative.append(tl + dd)
    if abs(cumulative[0]) > ACCOUNTING_ATOL:
        raise CostReconciliationError("제어 대상 차량의 인계 시점 비용이 0이 아닙니다.")
    increments = np.diff(np.asarray(cumulative, dtype=np.float64))
    reconstructed = float(np.sum(increments))
    missing = [v for v in cohort if v not in rows]
    unfinished = [v for v in cohort if v in rows and float(rows[v].get("arrival", -1)) < 0]
    final_vehicle_cost = math.fsum(float(rows[v].get("timeLoss", 0)) + float(rows[v].get("departDelay", 0))
                                   for v in cohort if v in rows)
    complete = not missing and not unfinished
    difference = reconstructed - final_vehicle_cost if complete else None
    if complete and abs(difference) > ACCOUNTING_ATOL:
        raise CostReconciliationError(f"차량별 평가와 보상 합이 다릅니다: {difference}")
    return {"cohort": cohort, "boundaries": times, "cumulative": cumulative,
            "increments": increments.tolist(), "time_loss": tl_parts, "depart_delay": insertion_parts,
            "actual_total_delay": final_vehicle_cost, "reconstructed_total_delay": reconstructed,
            "cost_check": {"passed": complete and abs(difference) <= ACCOUNTING_ATOL,
                           "difference": difference, "missing": len(missing), "unfinished": len(unfinished),
                           "tolerance": ACCOUNTING_ATOL}}


def fold_tail_rewards(costs: list[float], control_steps: int = CONTROL_S // STEP_S,
                      gamma: float = GAMMA, scale: float = REWARD_SCALE) -> tuple[list[float], dict]:
    """Fold autonomous P0 tail into the final control reward; no tail actions."""
    if control_steps <= 0 or len(costs) < control_steps:
        raise ValueError("제어 구간 비용이 부족합니다.")
    if not (0 < gamma <= 1 and scale > 0):
        raise ValueError("할인율과 보상 스케일이 올바르지 않습니다.")
    values = np.asarray(costs, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("실제 지연 비용이 올바르지 않습니다.")
    rewards = (-values[:control_steps] / scale).tolist()
    tail = values[control_steps:]
    folded = math.fsum(-(gamma ** (i + 1)) * float(c) / scale for i, c in enumerate(tail))
    rewards[-1] += folded
    return rewards, {"undiscounted_control_cost": float(np.sum(values[:control_steps])),
                     "tail_cost": float(np.sum(tail)), "folded_tail_reward": folded,
                     "tail_steps": len(tail), "gamma": gamma, "reward_scale": scale}


def canonical_fixed_mode(snapshot, requested: int) -> int:
    if not 0 <= int(requested) < 5:
        raise ValueError(f"모드 번호가 올바르지 않습니다: {requested}")
    action = snapshot.mode_actions[int(requested)]
    return next(i for i, (green, valid) in enumerate(zip(snapshot.mode_actions, snapshot.mode_mask))
                if valid and green == action)


def _deadline_step(original: Callable, deadline: float | None) -> Callable:
    """Enforce the job deadline during reset/warmup as well as control/tail."""
    if deadline is None:
        return original

    def checked_step():
        if time.monotonic() >= deadline:
            raise TimeoutError("WEIGHT1 에피소드 계산 예산을 초과했습니다.")
        result = original()
        if time.monotonic() >= deadline:
            raise TimeoutError("WEIGHT1 에피소드 계산 예산을 초과했습니다.")
        return result

    return checked_step


class _Episode:
    """Original OnlineEpisode physics with high precision evaluation output."""

    def __init__(self, date: str, period: str, seed: int, work: Path, *, deadline: float | None = None):
        import explore_rollout as er
        import focused_train as ft
        from focused_data import COUNTS, TARGET_TLS, load_counts
        from focused_rl import SignalConstraints, ZoneObservation, zone_queue_reward
        from focused_scenario import dqn_template_network, period_start_s, window_route_file
        from focused_sumo import make_environment

        if " " in str(work.resolve()):
            raise ValueError("SUMO 작업 경로에는 공백을 사용할 수 없습니다.")
        work.mkdir(parents=True, exist_ok=True)
        self.route = work / f"{date}_{period}.rou.xml"
        self.tripinfo, self.collisions = work / "tripinfo.xml", work / "collisions.xml"
        counts = load_counts(COUNTS)
        self.start = ft.warm_start(date, period, counts, self.route)
        origin = period_start_s(period) - ft.WARMUP_S
        window_route_file(date, origin, origin + ft.WARMUP_S + CONTROL_S + TAIL_S,
                          self.route, origin_s=origin, counts=counts)
        extra = (f"{er.STATE_OPTS} --tripinfo-output {self.tripinfo} "
                 f"--tripinfo-output.write-unfinished true --collision-output {self.collisions} "
                 f"--precision 12 --no-step-log true")
        self.env = make_environment(seed=seed, net_file=dqn_template_network(ft.CACHE), route_file=self.route,
                                    simulation_duration_sec=self.start["handover_s"] + CONTROL_S + TAIL_S,
                                    observation_class=ZoneObservation, reward_fn=zone_queue_reward,
                                    min_green=15, max_green=120, yellow_time=ft.AIHUB_YELLOW_S,
                                    additional_sumo_cmd=extra)
        try:
            # reset() itself runs the 30-minute warmup.  Install this before
            # reset so the deadline can interrupt that simulation per second.
            self.env._sumo_step = _deadline_step(self.env._sumo_step, deadline)
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("WEIGHT1 에피소드 준비 중 계산 예산을 초과했습니다.")
            self.env.warmup_states = self.start["warmup_states"]
            self.env.warmup_handover = self.start["handover_state"]
            self.cons = SignalConstraints().bind(self.env)
            self.cons.reset(self.start["last_served"])
            self.env.reset(seed=seed)
            self.ts = self.env.traffic_signals[TARGET_TLS]
            self.vehicles, _ = er.read_routes(self.route)
            P0 = er._p0()
            self.movement_of = P0.movements(self.route)
            self.ctl = P0(self.movement_of, [p.state for p in self.ts.green_phases])
            self.handover = float(self.env.sim_step)
            self.done = False
        except BaseException:
            self.env.close()
            raise

    def apply(self, action: int) -> tuple[int, int]:
        requested = int(action)
        corrected = int(self.cons(self.env, requested))
        _, _, terminated, truncated, _ = self.env.step(corrected)
        self.done = bool(terminated or truncated)
        return corrected, int(self.ts.green_phase)

    def close(self):
        self.env.close()


def _exact_metrics(rows: dict, vehicles: dict, handover: float) -> dict:
    import explore_online as eo
    import explore_rollout as er

    out = eo.episode_metrics(rows, vehicles, handover, CONTROL_S)
    by_movement: dict[str, list[float]] = {}
    measured = []
    for v in cohort_vehicle_ids(vehicles, handover):
        if v not in rows:
            continue
        r = rows[v]
        delay = float(r.get("timeLoss", 0)) + float(r.get("departDelay", 0))
        measured.append(delay)
        movement = er.movement_of_edges(vehicles[v][1])
        name = "->".join(movement) if movement else "none"
        by_movement.setdefault(name, []).append(delay)
    out["total_delay"] = math.fsum(measured)
    out["mean_delay_s"] = statistics.fmean(measured) if measured else None
    out["by_movement"] = {m: {"vehicles": len(v), "mean_delay_s": statistics.fmean(v)}
                           for m, v in sorted(by_movement.items())}
    for name, movement in er.LEFTS.items():
        out[name] = out["by_movement"].get("->".join(movement), {}).get("mean_delay_s")
    return out


def run_episode(date: str, period: str, eval_seed: int, work: Path,
                policy: Callable | int | None, *, collect: bool = False,
                train_seed: int | None = None, trace: bool = True,
                max_seconds: float | None = None) -> EpisodeResult:
    """Run one full episode, close SUMO, reconcile costs, then expose transitions.

    ``None`` executes the original P0; an integer requests one fixed mode; a
    callable receives (current observation, snapshot) and chooses a mode.
    max_seconds includes preparation, warmup, control, tail, and bookkeeping.
    Work is caller-owned and never removed by this function.
    """
    from explore_weight_core import ModePlanner, OBSERVATION_SIZE
    from focused_eval import judge_collisions

    started = time.monotonic()
    if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
        raise TimeoutError("에피소드 계산 예산이 없습니다.")

    def check_limit():
        if max_seconds is not None and time.monotonic() - started > max_seconds:
            raise TimeoutError("WEIGHT1 에피소드 계산 예산을 초과했습니다.")

    work = Path(work).resolve()
    ep = _Episode(str(date), str(period), int(eval_seed), work,
                  deadline=None if max_seconds is None else started + max_seconds)
    transitions, decisions = [], []
    mode_errors = 0
    boundaries = [ep.handover]
    cohort = cohort_vehicle_ids(ep.vehicles, ep.handover)
    cohort_set = set(cohort)
    snapshots: list[dict] = []
    departures: dict[str, float] = {}
    control_end = ep.handover + CONTROL_S
    end = control_end + TAIL_S

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
    try:
        check_limit()
        planner = ModePlanner(ep.movement_of, [p.state for p in ep.ts.green_phases], control_end)
        capture()
        while float(ep.env.sim_step) < end - 1e-9:
            check_limit()
            now = float(ep.env.sim_step)
            in_control = now < control_end - 1e-9
            allowed = list(ep.cons.allowed(ep.env))
            if not allowed:
                raise RuntimeError("실행 가능한 현시가 없습니다.")
            snap = planner.snapshot(ep.env) if in_control else None
            requested_mode = selected_mode = None
            if in_control:
                if policy is None:
                    selected_mode = requested_mode = 0
                    action = int(ep.ctl(ep.env, allowed))
                    mode_errors += action != snap.p0_action
                elif isinstance(policy, (int, np.integer)):
                    requested_mode = int(policy)
                    selected_mode = canonical_fixed_mode(snap, requested_mode)
                    action = int(snap.action(selected_mode))
                else:
                    requested_mode = selected_mode = int(policy(snap.observation.copy(), snap))
                    action = int(snap.action(selected_mode))
            else:
                action = int(ep.ctl(ep.env, allowed))
            before = now
            corrected, executed = ep.apply(action)
            after = float(ep.env.sim_step)
            if abs(after - before - STEP_S) > 1e-9:
                raise RuntimeError(f"5초 전이가 아닙니다: {before} → {after}")
            mode_errors += int(corrected != action or executed != action or action not in allowed)
            if in_control:
                decision = {"now": now, "time": now, "allowed": [int(a) for a in allowed],
                            "mode_actions": [int(a) for a in snap.mode_actions],
                            "mode_mask": [bool(v) for v in snap.mode_mask],
                            "mode_disagrees": bool(snap.mode_disagrees), "p0_action": int(snap.p0_action),
                            "requested_mode": int(requested_mode), "selected_mode": int(selected_mode),
                            "executed_green": executed}
                if trace:
                    decisions.append(decision)
                if collect:
                    next_obs = planner.snapshot(ep.env).observation.copy()
                    transitions.append({"obs": snap.observation.copy(), "next_obs": next_obs,
                                        "mode": int(selected_mode), "executed_green": executed,
                                        "reward": 0.0, "done": after >= control_end - 1e-9})
            if ep.done and after < end - 1e-9:
                raise RuntimeError("제어/꼬리 구간이 끝나기 전에 SUMO가 종료했습니다.")
        check_limit()
    finally:
        ep.env._sumo_step = original_step
        ep.close()

    check_limit()
    rows = read_tripinfo(ep.tripinfo)
    costs = reconcile_costs(ep.vehicles, rows, boundaries, snapshots, handover=ep.handover,
                            actual_departures=departures)
    rewards, reward_parts = fold_tail_rewards(costs["increments"])
    if collect:
        if len(transitions) != CONTROL_S // STEP_S:
            raise RuntimeError(f"제어 전이가 720개가 아닙니다: {len(transitions)}")
        for transition, reward in zip(transitions, rewards):
            if np.shape(transition["obs"]) != (OBSERVATION_SIZE,) or np.shape(transition["next_obs"]) != (OBSERVATION_SIZE,):
                raise RuntimeError("학습 관측이 110칸이 아닙니다.")
            transition["reward"] = float(reward)
        transitions[-1]["done"] = True
    metrics = _exact_metrics(rows, ep.vehicles, ep.handover)
    collision = judge_collisions(ep.collisions)
    seconds = time.monotonic() - started
    metrics.update({"date": str(date), "period": str(period), "seed": int(eval_seed), "eval_seed": int(eval_seed),
                    "train_seed": train_seed, "mode": "p0" if policy is None else (f"m{int(policy)}" if isinstance(policy, (int, np.integer)) else "rl"),
                    "handover": ep.handover, "control_s": CONTROL_S, "tail_s": TAIL_S,
                    "seconds": seconds, "actual_total_delay": costs["actual_total_delay"],
                    "reconstructed_total_delay": costs["reconstructed_total_delay"], "cost_check": costs["cost_check"],
                    "reconciled": bool(costs["cost_check"]["passed"]),
                    "cost_check_passed": bool(costs["cost_check"]["passed"]),
                    **reward_parts, "changed_by_rules": int(ep.cons.interventions),
                    "rule_decided": dict(ep.cons.rule_decided), "mode_action_errors": int(mode_errors),
                    "collisions_at_204820": int(collision["collisions_at_204820"]),
                    "collisions_unexplained": len(collision["collisions_unexplained"]),
                    "collision_details": collision,
                    "route_sha256": hashlib.sha256(ep.route.read_bytes()).hexdigest(),
                    "cohort_sha256": cohort_fingerprint(ep.vehicles, cohort),
                    "episode_code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "control_steps": CONTROL_S // STEP_S,
                    "mode_disagreement_points": sum(d["mode_disagrees"] for d in decisions),
                    "work": str(work)})
    ledger = {k: v for k, v in costs.items() if k != "cohort"}
    (work / "cost_ledger.json").write_text(json.dumps(ledger, ensure_ascii=False, indent=1))
    check_limit()
    return EpisodeResult(metrics, transitions, decisions, time.monotonic() - started)
