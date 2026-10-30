"""WEIGHT1: positive movement weights on the unchanged P0 planning model.

This module deliberately does not change ``explore_pred`` or ``focused_rl``.
Mode zero delegates to the original planner, including its exact tie-breaking.
The 110-value observation describes currently available information only.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

import explore_pred as ep

MODES = tuple(range(5))
LEFTS = {
    "left_east": ("477706392_02", "477706883_02"),
    "left_south": ("477706883_01", "477707013_01"),
}
MODE_FACTORS = ((1.0, 1.0), (1.5, 1.0), (1.0, 1.5), (1.5, 1.5), (2.0, 2.0))
OBSERVATION_SIZE = 110
MODE_MASK_SLICE = slice(104, 109)
HORIZON_S = 120
CONTROL_S = 3600
WAIT_SCALE_S = 165.0
COST_SCALE = 1000.0


def movement_weights(mode: int, movements: Sequence[tuple[str, str]]) -> tuple[float, ...]:
    """Weights attach to movement identifiers, never to phase numbers."""
    if isinstance(mode, bool) or not isinstance(mode, (int, np.integer)) or mode not in MODES:
        raise ValueError(f"Invalid weight mode: {mode!r}")
    east, south = MODE_FACTORS[mode]
    return tuple(east if m == LEFTS["left_east"] else south if m == LEFTS["left_south"] else 1.0
                 for m in movements)


def weighted_plan(model: ep.SignalModel, state: ep.SignalState, now: float, q0: list[float],
                  arrivals: list[list[float]], first_actions: list[int],
                  movement_weights: Sequence[float], merge: bool = True) -> tuple[int, dict]:
    """P0 with fixed, strictly positive weights in both running and terminal cost.

    Positive weights preserve P0's componentwise-queue dominance argument. They
    do not change capacities, transition timing, admissibility, or tie-breaking.
    """
    weights = tuple(float(w) for w in movement_weights)
    if len(weights) != len(q0) or not weights:
        raise ValueError("There must be one positive weight per queue movement.")
    if any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError("Movement weights must be finite and strictly positive.")
    if not arrivals or not first_actions or any(len(a) != len(weights) for a in arrivals):
        raise ValueError("The plan needs nonempty actions and matching arrival rows.")
    if all(w == 1.0 for w in weights):
        return ep.plan(model, state, now, q0, arrivals, first_actions, merge=merge)

    def queue_cost(q):
        return sum(w * x for w, x in zip(weights, q))

    current, step = state.phase, model.step
    level = {state: [(0.0, tuple(q0), (False, current, ()))]}
    expanded = 0
    for k, arriving in enumerate(arrivals):
        t = now + k * step
        nxt = {}
        for st, nodes in level.items():
            for a in (first_actions if k == 0 else model.actions(st, t)):
                st2, effective = model.advance(st, t, a)
                rate = model._rates[a] if effective > 0 else {}
                bucket = nxt.setdefault(st2, [])
                for cost, q, rank in nodes:
                    expanded += 1
                    q2 = ep.step_queue(q, arriving, rate, effective)
                    seq = rank[2] + (a,)
                    new_rank = (seq[0] != current, seq[0], seq)
                    c2 = cost + step * (queue_cost(q) + queue_cost(q2)) / 2
                    if merge:
                        ep._insert(bucket, c2, q2, new_rank)
                    else:
                        bucket.append((c2, q2, new_rank))
        level = nxt
    total, rank = min(((cost + model.terminal_s * queue_cost(q), rank)
                       for nodes in level.values() for cost, q, rank in nodes),
                      key=lambda x: (round(x[0], 6), x[1][0], x[1][1]))
    return rank[2][0], {"cost": total, "nodes": expanded, "plan": list(rank[2])}


def canonical_mode_mask(mode_actions: Sequence[int], forced: bool = False) -> list[bool]:
    """Retain the smallest mode for each distinct executable first green."""
    if len(mode_actions) != len(MODES):
        raise ValueError("Exactly five proposed first greens are required.")
    if forced:
        if len(set(mode_actions)) != 1:
            raise ValueError("A forced state must give the same green in every mode.")
        return [True, False, False, False, False]
    seen, mask = set(), []
    for a in mode_actions:
        mask.append(a not in seen)
        seen.add(a)
    return mask


def longest_wait_features(sumo, movement_of: dict, movements: Sequence[tuple[str, str]],
                          edges: Sequence[str], scale: float = WAIT_SCALE_S) -> list[float]:
    """The rule's departDelay + accumulatedWaitingTime, including served vehicles.

    Pending vehicles have no on-road accumulated waiting time yet. This does not
    reuse ``SignalConstraints.overdue``: that method omits currently served and
    below-threshold vehicles, which are important predictive observations here.
    """
    if scale <= 0 or not math.isfinite(scale):
        raise ValueError("Waiting-time scale must be positive and finite.")
    longest = dict.fromkeys(movements, 0.0)
    on_road = {v for e in edges for v in sumo.edge.getLastStepVehicleIDs(e)}
    pending = set(sumo.simulation.getPendingVehicles())
    for v in sorted(on_road | pending):
        movement = movement_of.get(v)
        if movement not in longest:
            continue
        wait = float(sumo.vehicle.getDepartDelay(v))
        if v in on_road:
            wait += float(sumo.vehicle.getAccumulatedWaitingTime(v))
        if not math.isfinite(wait) or wait < -ep.TOL:
            raise ValueError(f"Invalid cumulative vehicle waiting time for {v}: {wait}")
        longest[movement] = max(longest[movement], wait)
    return [min(2.0, longest[m] / scale) for m in movements]


def observation_vector(base: Sequence[float], queues: Sequence[float], waits: Sequence[float],
                       candidate_costs: Sequence[float | None], mode_actions: Sequence[int],
                       mode_mask: Sequence[bool], remaining: float) -> np.ndarray:
    """Assemble the frozen layout; a missing candidate is marked by the base mask."""
    if len(base) != 64 or len(queues) != 8 or len(waits) != 8:
        raise ValueError("WEIGHT1 needs the original 64 values and eight queue/wait values.")
    if len(candidate_costs) != 4 or len(mode_actions) != 5 or len(mode_mask) != 5:
        raise ValueError("Invalid green-candidate or mode observation dimensions.")
    finite_costs = [float(c) for c in candidate_costs if c is not None]
    if not finite_costs or any(not math.isfinite(c) for c in finite_costs):
        raise ValueError("At least one finite original P0 candidate cost is required.")
    low = min(finite_costs)
    gaps = [0.0 if c is None else min(10.0, max(0.0, (float(c) - low) / COST_SCALE))
            for c in candidate_costs]
    onehot = []
    for a in mode_actions:
        if isinstance(a, bool) or not isinstance(a, (int, np.integer)) or not 0 <= a < 4:
            raise ValueError("A mode must propose one of the four green identifiers.")
        onehot.extend(float(a == g) for g in range(4))
    obs = np.asarray([*base, *queues, *waits, *gaps, *onehot,
                      *[float(m) for m in mode_mask], min(1.0, max(0.0, remaining))], dtype=np.float32)
    if obs.shape != (OBSERVATION_SIZE,) or not np.isfinite(obs).all() or not any(mode_mask):
        raise ValueError("Invalid WEIGHT1 observation or empty mode mask.")
    return obs


@dataclass
class ModeSnapshot:
    observation: np.ndarray
    allowed: list[int]
    mode_actions: list[int]
    mode_mask: list[bool]
    p0_action: int
    candidate_costs: list[float | None]
    mode_disagrees: bool
    now: float
    mode_costs: list[float | None] = field(default_factory=list)
    p0_info: dict = field(default_factory=dict)
    planning_seconds: float = 0.0

    def action(self, mode: int) -> int:
        if isinstance(mode, bool) or not isinstance(mode, (int, np.integer)) or int(mode) not in MODES:
            raise ValueError(f"Invalid weight mode: {mode!r}")
        if not self.mode_mask[int(mode)]:
            raise ValueError(f"Mode {mode} duplicates another mode or is unavailable.")
        a = self.mode_actions[int(mode)]
        if a not in self.allowed:
            raise RuntimeError("The planned first green is outside the executable set.")
        return a


class ModePlanner:
    """One current-state planning snapshot shared by observation and execution."""

    def __init__(self, movement_of: dict, greens: list[str], control_end: float):
        if len(greens) != 4 or not math.isfinite(control_end):
            raise ValueError("WEIGHT1 expects four greens and a finite control end.")
        self.movement_of, self.greens, self.control_end = movement_of, list(greens), float(control_end)
        self.p0 = ep.controller_class("p0", HORIZON_S)(movement_of, greens)
        self._cached_env = None
        self._cached = None

    def snapshot(self, env) -> ModeSnapshot:
        from focused_data import TARGET_TLS
        from focused_rl import ALL_ZONE_EDGES

        now = float(env.sim_step)
        if self._cached is not None and self._cached_env is env and self._cached.now == now:
            return self._cached
        if float(env.delta_time) != 5.0:
            raise ValueError("WEIGHT1 requires the frozen five-second decision interval.")
        if self.p0._model is None:
            self.p0._setup(env)
        moves = self.p0.movements
        if len(moves) != 8 or not all(m in moves for m in LEFTS.values()):
            raise ValueError("WEIGHT1 requires the eight original protected movements, including both left turns.")
        ts, guard = env.traffic_signals[TARGET_TLS], env.signal_constraints
        allowed = list(guard.allowed(env))
        if not allowed or any(a not in range(4) for a in allowed):
            raise ValueError("Invalid executable green set.")
        base = np.asarray(ts.observation_fn(), dtype=np.float32)
        if base.shape != (64,):
            raise ValueError("The original observation is not 64 values; do not mix prior wrappers.")
        expected_mask = [a in allowed for a in range(4)]
        if list(base[-4:] > 0.5) != expected_mask:
            raise RuntimeError("Original observation and rule executable-green masks disagree.")
        q0, arrivals = self.p0.queues_and_arrivals(env)
        capacity = ts.observation_fn._capacity
        queues = [min(1.0, q / max(1.0, capacity[m[0]])) for m, q in zip(moves, q0)]
        scale = float(guard.max_wait) if guard.max_wait is not None else WAIT_SCALE_S
        if scale != WAIT_SCALE_S:
            raise ValueError("WEIGHT1 is frozen to the 165-second vehicle waiting rule.")
        waits = longest_wait_features(env.sumo, self.movement_of, moves, ALL_ZONE_EDGES, scale)

        started = time.perf_counter()
        candidate_costs = [None] * 4
        if len(allowed) == 1:
            p0_action = allowed[0]
            mode_actions, mode_costs = [p0_action] * 5, [None] * 5
            candidate_costs[p0_action] = 0.0
            p0_info = {"cost": None, "nodes": 0, "plan": [p0_action], "forced": True}
        else:
            served = guard._served_now(ts, now)
            state = ep.SignalState(ts.green_phase, now - ts.time_since_last_phase_change,
                                   float(ts.yellow_time), tuple(served[c] for c in self.p0._model.crosswalks))
            p0_action, p0_info = ep.plan(self.p0._model, state, now, q0, arrivals, allowed)
            mode_actions, mode_costs = [p0_action], [p0_info["cost"]]
            for a in allowed:
                _, info = ep.plan(self.p0._model, state, now, q0, arrivals, [a])
                candidate_costs[a] = float(info["cost"])
            for mode in MODES[1:]:
                a, info = weighted_plan(self.p0._model, state, now, q0, arrivals, allowed,
                                        movement_weights(mode, moves))
                mode_actions.append(a)
                mode_costs.append(float(info["cost"]))
        mode_mask = canonical_mode_mask(mode_actions, forced=len(allowed) == 1)
        obs = observation_vector(base, queues, waits, candidate_costs, mode_actions, mode_mask,
                                 (self.control_end - now) / CONTROL_S)
        result = ModeSnapshot(observation=obs, allowed=allowed, mode_actions=mode_actions, mode_mask=mode_mask,
                              p0_action=p0_action, candidate_costs=candidate_costs,
                              mode_disagrees=len(set(mode_actions)) > 1, now=now,
                              mode_costs=mode_costs, p0_info=p0_info,
                              planning_seconds=time.perf_counter() - started)
        if any(a not in allowed for a in mode_actions):
            raise RuntimeError("A weight mode proposed an inadmissible first green.")
        self._cached_env, self._cached = env, result
        return result


def observation_bounds() -> tuple[np.ndarray, np.ndarray]:
    """The bounded schema consumed by the generic five-mode DQN."""
    low = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
    high = np.ones(OBSERVATION_SIZE, dtype=np.float32)
    high[72:80] = 2.0
    high[80:84] = 10.0
    return low, high
