"""Diagnostic joint estimation of boundary departures from passage observations.

Variables are floating route counts in uniform 15-minute boundary-departure
cohorts. Posted-speed external-road travel propagates each cohort to successive
observed movements. Source counts remain integer observations, and absent cells
remain unknown. The kernel excludes signal/queue delays and is not historical
arrival reconstruction. Integer dispatch and actual SUMO passage validation are
separate stages; this module never certifies an environment for learning.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
import math
from pathlib import Path
import random
from typing import Mapping, Sequence

from network_demand import (CandidateRoute, DemandLedger, RouteVehicle, aggregate_counts,
                            audit_planned_routes, movement_specs, occurrence_indices,
                            _boundary_metadata, _lane_allows, _route_metadata, _topology)


COHORT_SECONDS = 900.0


def overlap_fraction(depart_begin: float, depart_end: float, lag: float,
                     observation_begin: float, observation_end: float) -> float:
    """Uniform cohort mass reaching a half-open observed passage interval."""
    if depart_end <= depart_begin or lag < 0:
        raise ValueError("Invalid departure cohort or travel lag")
    overlap = max(0.0, min(depart_end + lag, observation_end) -
                  max(depart_begin + lag, observation_begin))
    return overlap / (depart_end - depart_begin)


def _combine(ledgers: Sequence[DemandLedger]) -> DemandLedger:
    if not ledgers:
        raise ValueError("At least one common source window is required")
    for index, ledger in enumerate(ledgers):
        if not math.isclose(ledger.end - ledger.begin, COHORT_SECONDS,
                            rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("Diagnostic source windows must be exactly 900 seconds")
        if index and not math.isclose(ledgers[index - 1].end, ledger.begin,
                                     rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("Source windows must be ordered, contiguous, and non-overlapping")
    combined = aggregate_counts((r for ledger in ledgers for r in ledger.records),
                                ledgers[0].begin, ledgers[-1].end)
    expected_counts = Counter()
    for ledger in ledgers:
        expected_counts.update(ledger.counts)
        combined.source_hashes.update(ledger.source_hashes)
    if dict(expected_counts) != combined.counts:
        raise ValueError("Ledger records and recorded counts disagree; source totals cannot be inferred")
    # Individual rows retain their original per-window labels in the LP report.
    return combined


def _travel_events(route: CandidateRoute | RouteVehicle, kind: str, topology: dict, resolved: Mapping):
    """Entry to the final external edge; no waiting or green timing is added."""
    prefix = [0.0]
    for edge in route.edges:
        allowed = [lane for lane in topology["edges"][edge].findall("lane") if _lane_allows(lane, kind)]
        if not allowed:
            raise ValueError(f"Route contains an edge unavailable to {kind}: {edge}")
        prefix.append(prefix[-1] + min(float(lane.get("length")) / float(lane.get("speed")) for lane in allowed))
    events = []
    for pair, spec in resolved.items():
        path = spec["path"]
        for start in occurrence_indices(route.edges, path):
            events.append((pair, prefix[start + len(path) - 1]))
    return events


def _rows_and_totals(ledgers, source_rows, predictions):
    rows, groups = [], defaultdict(Counter)
    for (window, kind, a, b), prediction in zip(source_rows, predictions):
        ledger = ledgers[window]
        source_count = ledger.counts[(kind, a, b)]
        label = ledger.cell_provenance.get((kind, a, b), "provided")
        row = {"window_index": window, "source_begin": ledger.begin, "source_end": ledger.end,
               "vehicle_type": kind, "source_from": a, "source_to": b,
               "source_count": source_count, "expected_passages": float(prediction),
               "under": max(source_count - float(prediction), 0.0),
               "over": max(float(prediction) - source_count, 0.0),
               "provenance": label, "source_state": "positive" if source_count else "explicit_zero"}
        rows.append(row)
        for key in ("source_count", "expected_passages", "under", "over"):
            groups[label][key] += row[key]
    totals = {key: sum(row[key] for row in rows) for key in ("source_count", "expected_passages", "under", "over")}
    return {"scope": "source window x vehicle type x movement; expected nominal passage times",
            "totals": totals, "by_provenance": {label: dict(values) for label, values in groups.items()},
            "by_movement": rows}


def round_dispatch_allocations(expected_allocations: Sequence[Mapping]) -> list[dict]:
    """Hamilton rounding by class; preserve rounded class totals, not observations.

    Rounding cannot generally preserve every source passage constraint. Its
    effect must be reported through the separate nominal dispatch audit.
    """
    groups = defaultdict(list)
    for original in expected_allocations:
        item = dict(original)
        value = float(item["expected_count"])
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid expected route count")
        item["count"] = math.floor(value)
        groups[item["vehicle_type"]].append(item)
    output = []
    for kind, items in sorted(groups.items()):
        target = math.floor(sum(item["expected_count"] for item in items) + 0.5)
        remaining = target - sum(item["count"] for item in items)
        ranked = sorted(items, key=lambda item: (-(item["expected_count"] - item["count"]),
                                                  item["route_id"], item["cohort_start"]))
        for item in ranked[:remaining]:
            item["count"] += 1
        output.extend(item for item in items if item["count"])
    return output


def dispatch_nominal_audit(ledgers, allocations, specs, network_file, boundary_mode="terminal_turnaround", seed=42):
    """Sample the exact rounded dispatch at uniform boundary times, then propagate.

    These timestamps are kernel predictions. They are not actual SUMO events.
    Before/after-horizon movement events remain in the report.
    """
    combined = _combine(ledgers)
    resolved = movement_specs(combined, specs)
    topology = _topology(network_file, boundary_mode)
    starts = [ledger.begin - combined.begin for ledger in ledgers]
    ends = [ledger.end - combined.begin for ledger in ledgers]
    observed, unknown = Counter(), Counter()
    outside = Counter({"before_horizon": 0, "after_horizon": 0})
    rng, vehicles, cache = random.Random(seed), [], {}
    for item in allocations:
        kind = item["vehicle_type"]
        edges = tuple(item["edges"])
        identity = (kind, edges)
        if identity not in cache:
            cache[identity] = _travel_events(CandidateRoute(item["route_id"], edges), kind, topology, resolved)
        for _ in range(item["count"]):
            # Match network_environment.write_demand's actual XML precision.
            depart = round(item["cohort_start"] + rng.uniform(0, COHORT_SECONDS), 2)
            vehicles.append(RouteVehicle(f"temporal_{len(vehicles)}", kind, edges, depart))
            for (a, b), lag in cache[identity]:
                completion = depart + lag
                index = bisect_right(starts, completion) - 1
                if index < 0:
                    outside["before_horizon"] += 1
                elif completion >= ends[-1]:
                    outside["after_horizon"] += 1
                elif (kind, a, b) in ledgers[index].counts:
                    observed[(index, kind, a, b)] += 1
                else:
                    unknown[(index, kind, a, b)] += 1
    source_rows = sorted((window, *cell) for window, ledger in enumerate(ledgers) for cell in ledger.counts)
    report = _rows_and_totals(ledgers, source_rows, [observed[row] for row in source_rows])
    # Dispatch predictions are integer event counts, unlike LP expectations.
    for row in report["by_movement"]:
        row["nominal_dispatch_passages"] = int(row.pop("expected_passages"))
    report.update(scope="rounded uniform boundary dispatch propagated by nominal road travel; not actual SUMO",
                  generated_vehicle_count=len(vehicles), unrecorded_registered_movements=sum(unknown.values()),
                  outside_horizon=dict(outside), seed=seed, actual_passages_verified=False,
                  planned_path_audit=audit_planned_routes(combined, vehicles, specs, network_file=network_file,
                                                         boundary_mode=boundary_mode))
    return report


def fit_temporal_demand(
    ledgers: Sequence[DemandLedger], candidates: Sequence[CandidateRoute], specs: Sequence[Mapping],
    network_file: Path, boundary_mode: str = "terminal_turnaround", weights: Mapping[str, float] | None = None,
    cohort_starts: Sequence[float] | None = None, time_limit_s: float = 30, seed: int = 42,
    max_edge_repeats: int = 0, max_route_length_m: float | None = 10000,
) -> dict:
    """Joint sparse LP over route x uniform boundary-departure cohorts.

    Default cohorts are exactly the supplied observation windows; no unknown
    warmup-before/after cohorts are silently added. Unknown movement predictions
    receive a weak penalty. Explicit zeros are observed rows with over slack.
    No signal state, target green time, or historical departure time is used.
    """
    import numpy as np
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix, eye, hstack

    combined = _combine(ledgers)
    settings = {"collection_site": 1.0, "supplemental_site": 0.25, "provided": 1.0,
                "measured": 10.0, "estimated": 0.25, "under": 1000.0, "over": 1000.0,
                "unknown": 1.0, "outside": 1.0, "vehicle": 0.001}
    settings.update(weights or {})
    if any(not math.isfinite(value) or value < 0 for value in settings.values()) or min(settings["under"], settings["over"]) <= 0:
        raise ValueError("Invalid LP weights")
    if time_limit_s <= 0 or max_edge_repeats < 0:
        raise ValueError("Invalid temporal fit limits")
    starts = [ledger.begin - combined.begin for ledger in ledgers]
    ends = [ledger.end - combined.begin for ledger in ledgers]
    cohorts = list(starts if cohort_starts is None else cohort_starts)
    if not cohorts or len(set(cohorts)) != len(cohorts) or any(not math.isfinite(t) for t in cohorts):
        raise ValueError("Boundary cohorts must be nonempty, unique, finite starts")
    cohorts.sort()
    topology = _topology(network_file, boundary_mode)
    resolved = movement_specs(combined, specs)
    approaches = {a for a, b in resolved}
    eligible, rejected, candidate_ids = [], [], set()
    for route in candidates:
        if not route.edges or route.id in candidate_ids:
            raise ValueError("Empty candidate or duplicate route ID")
        candidate_ids.add(route.id)
        info = _route_metadata(route, topology, approaches)
        reason = None
        if not info["connected"]:
            reason = "disconnected_or_unknown_edge"
        elif max(route.edges.count(edge) - 1 for edge in set(route.edges)) > max_edge_repeats:
            reason = "repeated_edge_limit"
        elif not (info["origin_is_boundary"] and info["destination_is_boundary"]):
            reason = "internal_endpoint"
        elif max_route_length_m is not None and info["length_m"] > max_route_length_m:
            reason = "length_limit"
        if reason:
            rejected.append({"id": route.id, "reason": reason})
        else:
            eligible.append((route, info))
    expected_allocations, expected_counts, solver = [], Counter(), []
    expected_outside = Counter({"before_horizon": 0.0, "after_horizon": 0.0})
    expected_unknown = 0.0
    for kind in sorted({cell[0] for ledger in ledgers for cell in ledger.counts}):
        source_rows = sorted((window, *cell) for window, ledger in enumerate(ledgers)
                             for cell in ledger.counts if cell[0] == kind)
        row_index = {row: index for index, row in enumerate(source_rows)}
        targets = np.array([ledgers[w].counts[(k, a, b)] for w, k, a, b in source_rows], dtype=float)
        rr, cc, values, variables, costs = [], [], [], [], []
        for route, info in eligible:
            if any(not any(_lane_allows(a, kind) and _lane_allows(z, kind)
                           for a, z in topology["connection_lanes"][pair])
                   for pair in zip(route.edges, route.edges[1:])):
                continue
            events = _travel_events(route, kind, topology, resolved)
            for cohort in cohorts:
                column, unknown, before, after = Counter(), 0.0, 0.0, 0.0
                for (a, b), lag in events:
                    for window, (begin, end) in enumerate(zip(starts, ends)):
                        mass = overlap_fraction(cohort, cohort + COHORT_SECONDS, lag, begin, end)
                        if mass:
                            row = (window, kind, a, b)
                            if row in row_index:
                                column[row_index[row]] += mass
                            else:
                                unknown += mass
                    before += max(0.0, min(cohort + COHORT_SECONDS + lag, starts[0]) -
                                  min(cohort + lag, starts[0])) / COHORT_SECONDS
                    after += max(0.0, max(cohort + COHORT_SECONDS + lag, ends[-1]) -
                                 max(cohort + lag, ends[-1])) / COHORT_SECONDS
                if not any(value and targets[index] > 0 for index, value in column.items()):
                    continue
                index = len(variables)
                variables.append((route, info, cohort, unknown, before, after))
                costs.append(settings["unknown"] * unknown + settings["outside"] * (before + after) + settings["vehicle"])
                for row, value in column.items():
                    rr.append(row); cc.append(index); values.append(value)
        n, m = len(variables), len(source_rows)
        matrix = coo_matrix((values, (rr, cc)), shape=(m, n)).tocsc()
        equality = hstack((matrix, eye(m, format="csc"), -eye(m, format="csc")), format="csc")
        labels = [ledgers[w].cell_provenance.get((k, a, b), "provided") for w, k, a, b in source_rows]
        row_weights = [settings.get(label, settings["provided"]) for label in labels]
        objective = costs + [settings["under"] * weight for weight in row_weights] + [settings["over"] * weight for weight in row_weights]
        solution = linprog(objective, A_eq=equality, b_eq=targets, bounds=(0, None), method="highs",
                           options={"time_limit": time_limit_s})
        metadata = {"vehicle_type": kind, "status": int(solution.status), "message": solution.message,
                    "optimality_proven": solution.status == 0, "variables_route_cohort": n,
                    "source_rows": m, "objective": None if solution.fun is None else float(solution.fun)}
        if solution.x is None:
            route_counts = np.zeros(n)
            metadata["fallback"] = "zero route allocation; source deficit retained"
        else:
            if np.max(np.abs(equality @ solution.x - targets), initial=0) > 1e-5:
                raise RuntimeError("LP source equation validation failed")
            route_counts = np.maximum(solution.x[:n], 0)
        solver.append(metadata)
        for row, value in zip(source_rows, matrix @ route_counts):
            expected_counts[row] = float(value)
        for (route, info, cohort, unknown, before, after), value in zip(variables, route_counts):
            if value > 1e-8:
                expected_allocations.append({"vehicle_type": kind, "route_id": route.id, "edges": list(route.edges),
                                             "expected_count": float(value), "cohort_start": cohort,
                                             "cohort_duration": COHORT_SECONDS, **info})
                expected_unknown += value * unknown
                expected_outside["before_horizon"] += value * before
                expected_outside["after_horizon"] += value * after
    source_rows = sorted((window, *cell) for window, ledger in enumerate(ledgers) for cell in ledger.counts)
    expected = _rows_and_totals(ledgers, source_rows, [expected_counts[row] for row in source_rows])
    expected.update(unrecorded_registered_movements=float(expected_unknown),
                    outside_horizon={k: float(v) for k, v in expected_outside.items()})
    allocations = round_dispatch_allocations(expected_allocations)
    rounded = dispatch_nominal_audit(ledgers, allocations, specs, network_file, boundary_mode, seed)
    unresolved = sum(value for ledger in ledgers for (_, a, b), value in ledger.counts.items()
                     if resolved[(a, b)]["path"] is None)
    return {"status": "diagnostic_joint_boundary_demand_estimate", "source_total": combined.total,
            "source": combined.summary(), "unresolved_source_count": unresolved,
            "LPexpectedunder": expected["totals"]["under"], "LPexpectedover": expected["totals"]["over"],
            "rounddispatch_nominalunder": rounded["totals"]["under"],
            "rounddispatch_nominalover": rounded["totals"]["over"],
            "LP_expected": expected, "rounded_dispatch_nominal": rounded,
            "expected_allocations": expected_allocations, "allocations": allocations,
            "solver": solver, "weights": settings, "boundary": _boundary_metadata(topology),
            "cohort_count": len(cohorts), "cohort_starts": cohorts, "cohort_duration": COHORT_SECONDS,
            "automatic_unobserved_warmup_or_tail_cohorts": False,
            "candidate_routes_input": len(candidates), "candidate_routes_eligible": len(eligible),
            "rejected_candidates": rejected, "source_absence_is_zero": False,
            "actual_passages_verified": False, "historical_arrivals_recreated": False, "learning_ready": False,
            "assumptions": ["Boundary OD and departure cohorts are inferred jointly from passage observations.",
                            "The uniform boundary arrival kernel uses external roads at posted speeds.",
                            "Internal junction traversal, signal delays, insertion delay and queues are omitted.",
                            "No green timing or target intersection arrival schedule is used.",
                            "Absent source cells are unknown; explicit zeros retain observed over constraints.",
                            "Out-of-horizon predicted passages are reported, not dropped.",
                            "LP expectations and rounded sampled dispatch counts are distinct.",
                            "Actual SUMO completed passages must be validated before policy comparison."]}
