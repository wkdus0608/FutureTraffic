"""Preserve source passage counts and independently audit/fill network routes.

An observation is a junction passage, not a unique vehicle or a boundary arrival.
The fit below estimates integer route demand in a common time window; departure
times and historical passage times are separate problems. Missing XML relations
remain unknown, while an explicit zero is a recorded constraint. Source IDs,
counts, and denominators are retained even when a movement is unresolved.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
import hashlib
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET


TYPE_CODES = {"p": "passenger", "b": "bus", "t": "truck", "m": "motorcycle"}


@dataclass(frozen=True)
class TurnObservation:
    vehicle_type: str
    begin: float
    end: float
    source_from: str
    source_to: str
    count: int
    source_file: str = ""
    provenance: str = "provided"


@dataclass
class DemandLedger:
    begin: float
    end: float
    counts: dict[tuple[str, str, str], int]
    explicit_zero_records: list[dict]
    records: tuple[TurnObservation, ...]
    source_hashes: dict[str, str] = field(default_factory=dict)
    cell_provenance: dict[tuple[str, str, str], str] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def source_pairs(self) -> set[tuple[str, str]]:
        return {(a, b) for _, a, b in self.counts}

    def summary(self) -> dict:
        totals = Counter()
        for (kind, _, _), value in self.counts.items():
            totals[kind] += value
        return {
            "source_begin": self.begin, "source_end": self.end,
            "source_turn_count": self.total, "by_vehicle_type": dict(totals),
            "source_relation_rows": len(self.records),
            "explicit_zero_relation_rows": len(self.explicit_zero_records),
            "source_hashes": self.source_hashes,
            "absence_semantics": "unknown; no absence-to-zero imputation",
            "count_semantics": "junction passages, not unique vehicles",
        }


@dataclass(frozen=True)
class RouteVehicle:
    id: str
    vehicle_type: str
    edges: tuple[str, ...]
    depart: float = 0.0


@dataclass(frozen=True)
class CandidateRoute:
    id: str
    edges: tuple[str, ...]


def aggregate_counts(
    records: Iterable[TurnObservation], begin: float, end: float,
    provenance_by_pair: Mapping[tuple[str, str], str] | None = None,
) -> DemandLedger:
    """Aggregate fully contained source intervals without scaling/rounding.

    A boundary cutting a source interval is rejected: there is no evidence for
    distributing integer passages across the two pieces. Duplicate source cells
    are rejected rather than counted twice. Zeros retain their interval identity.
    """
    if not math.isfinite(begin) or not math.isfinite(end) or end <= begin:
        raise ValueError("A common time window must have positive duration")
    counts: Counter = Counter()
    zeros, selected, seen, provenance = [], [], set(), {}
    for row in records:
        if (not math.isfinite(row.begin) or not math.isfinite(row.end) or row.end <= row.begin or
                row.count < 0 or type(row.count) is not int):
            raise ValueError(f"Invalid source observation: {row}")
        if row.end <= begin or row.begin >= end:
            continue
        if row.begin < begin or row.end > end:
            raise ValueError("Common window cuts a source interval; no fractional count allocation")
        identity = (row.vehicle_type, row.begin, row.end, row.source_from, row.source_to)
        if identity in seen:
            raise ValueError(f"Duplicate source cell: {identity}")
        seen.add(identity)
        cell = (row.vehicle_type, row.source_from, row.source_to)
        label = (provenance_by_pair or {}).get((row.source_from, row.source_to), row.provenance)
        if cell in provenance and provenance[cell] != label:
            raise ValueError(f"Mixed provenance needs an explicit rule: {cell}")
        provenance[cell] = label
        counts[cell] += row.count  # also retains explicit zero cells
        selected.append(row)
        if row.count == 0:
            zeros.append({"vehicle_type": row.vehicle_type, "begin": row.begin,
                          "end": row.end, "source_from": row.source_from,
                          "source_to": row.source_to, "source_file": row.source_file})
    return DemandLedger(begin, end, dict(counts), zeros, tuple(selected), cell_provenance=provenance)


def read_turn_counts(
    turn_files: Mapping[str, Path], begin: float | None = None, end: float | None = None,
    provenance_by_pair: Mapping[tuple[str, str], str] | None = None,
) -> DemandLedger:
    """Read all vehicle classes onto one shared window, never per-class starts.

    The inferred window is the union of all source intervals. For a prefix's
    15-minute file set this preserves the common 05:00–05:15 window even if the
    motorcycle file first has a nonzero observation at 05:01.
    """
    records, intervals, hashes = [], [], {}
    for kind, filename in turn_files.items():
        kind = TYPE_CODES.get(kind, kind)
        filename = Path(filename)
        data = filename.read_bytes()
        hashes[str(filename.resolve())] = hashlib.sha256(data).hexdigest()
        for interval in ET.fromstring(data).iter("interval"):
            start, stop = float(interval.get("begin")), float(interval.get("end"))
            if not math.isfinite(start) or not math.isfinite(stop) or stop <= start:
                raise ValueError(f"Invalid interval in {filename}")
            intervals.append((start, stop))
            for relation in interval.findall("edgeRelation"):
                value = Decimal(relation.attrib["count"])
                if not value.is_finite() or value < 0 or value != value.to_integral_value():
                    raise ValueError(f"Source count must be a nonnegative integer: {value}")
                records.append(TurnObservation(kind, start, stop, relation.attrib["from"],
                                               relation.attrib["to"], int(value), str(filename.resolve())))
    if not intervals:
        raise ValueError("No source intervals found")
    ledger = aggregate_counts(records, min(a for a, _ in intervals) if begin is None else begin,
                              max(b for _, b in intervals) if end is None else end, provenance_by_pair)
    ledger.source_hashes = hashes
    return ledger


def _route_definitions(root: ET.Element) -> dict[str, tuple[str, ...]]:
    return {route.get("id"): tuple(route.attrib["edges"].split())
            for route in root.iter("route") if route.get("id") and route.get("edges")}


def read_route_vehicles(path: Path) -> list[RouteVehicle]:
    root = ET.parse(path).getroot()
    if root.findall("flow") or root.findall("trip"):
        raise ValueError("Audit requires explicit routed vehicles; flows/trips are not silently ignored")
    definitions = _route_definitions(root)
    vehicles, seen = [], set()
    for vehicle in root.findall("vehicle"):
        identifier = vehicle.attrib["id"]
        if identifier in seen:
            raise ValueError(f"Duplicate vehicle id: {identifier}")
        seen.add(identifier)
        inline = vehicle.find("route")
        edges = tuple(inline.attrib["edges"].split()) if inline is not None else definitions[vehicle.attrib["route"]]
        if not edges or any(e.startswith(":") for e in edges):
            raise ValueError("Audit routes must contain external edges")
        vehicles.append(RouteVehicle(identifier, TYPE_CODES.get(vehicle.get("type"), vehicle.get("type", "passenger")),
                                     edges, float(vehicle.attrib["depart"])))
    return vehicles


def read_candidate_routes(path: Path) -> list[CandidateRoute]:
    root = ET.parse(path).getroot()
    found = {}
    for index, route in enumerate(root.iter("route")):
        if route.get("edges"):
            edges = tuple(route.attrib["edges"].split())
            if edges and not any(e.startswith(":") for e in edges):
                found.setdefault(edges, CandidateRoute(route.get("id", f"candidate_{index}"), edges))
    return list(found.values())


def movement_specs(ledger: DemandLedger, specs: Sequence[Mapping] | None = None) -> dict:
    """Accept the network_movements schema; source IDs never change in the ledger."""
    supplied = {}
    for original in specs or []:
        item = dict(original)
        pair = (item["source_from"], item["source_to"])
        if pair in supplied:
            raise ValueError(f"Duplicate movement spec: {pair}")
        path = item.get("path")
        if path is not None and (isinstance(path, str) or len(path) < 2 or any(not isinstance(e, str) or not e for e in path)):
            raise ValueError(f"Invalid complete movement path: {path}")
        item["path"] = tuple(path) if path is not None else None
        supplied[pair] = item
    # Keep registered movements absent in the current window. Their complete
    # paths still need an unknown-demand audit, especially upstream bypasses
    # that do not traverse the raw incoming edge.
    result = dict(supplied)
    for pair in ledger.source_pairs:
        result[pair] = supplied.get(pair, {"id": "->".join(pair), "source_from": pair[0],
            "source_to": pair[1], "path": pair, "resolution": "direct", "evidence_level": "provided",
            "evidence": "raw consecutive-edge interpretation; network validity is checked separately"})
    used = {}
    for pair, item in result.items():
        path = item["path"]
        if path is not None and path in used:
            raise ValueError(f"Two source movements map to the same physical path; allocation rule required: {pair}, {used[path]}")
        if path is not None:
            used[path] = pair
    return result


def occurrence_indices(edges: Sequence[str], path: Sequence[str] | None) -> list[int]:
    if path is None:
        return []
    wanted = tuple(path)
    return [i for i in range(len(edges) - len(wanted) + 1) if tuple(edges[i:i + len(wanted)]) == wanted]


def _topology(network_file: Path | None, boundary_mode: str = "graph_terminal") -> dict | None:
    if boundary_mode not in {"graph_terminal", "terminal_turnaround"}:
        raise ValueError(f"Unknown boundary mode: {boundary_mode}")
    if network_file is None:
        return None
    root = ET.parse(network_file).getroot()
    edges = {e.get("id"): e for e in root.findall("edge") if not e.get("id", "").startswith(":")}
    junction_types = {j.get("id"): j.get("type", "") for j in root.findall("junction")}
    connections = {(c.get("from"), c.get("to")) for c in root.findall("connection")
                   if c.get("from") in edges and c.get("to") in edges}
    lanes = {lane.get("id"): lane for edge in edges.values() for lane in edge.findall("lane")}
    connection_lanes = defaultdict(list)
    non_turnaround = set()
    for connection in root.findall("connection"):
        a, b = connection.get("from"), connection.get("to")
        if (a, b) in connections:
            if connection.get("dir") != "t":
                non_turnaround.add((a, b))
            connection_lanes[(a, b)].append((lanes[f"{a}_{connection.get('fromLane')}"],
                                             lanes[f"{b}_{connection.get('toLane')}"]))
    graph_entries = set(edges) - {b for a, b in connections}
    graph_exits = set(edges) - {a for a, b in connections}
    entries, exits = graph_entries, graph_exits
    if boundary_mode == "terminal_turnaround":
        entries = {e for e, edge in edges.items() if e not in {b for a, b in non_turnaround}
                   and not junction_types.get(edge.get("from"), "").startswith("traffic_light")}
        exits = {e for e, edge in edges.items() if e not in {a for a, b in non_turnaround}
                 and not junction_types.get(edge.get("to"), "").startswith("traffic_light")}
    return {"edges": edges, "connections": connections, "incoming": {b for a, b in connections},
            "outgoing": {a for a, b in connections}, "connection_lanes": connection_lanes,
            "boundary_mode": boundary_mode, "entries": entries, "exits": exits,
            "graph_entries": graph_entries, "graph_exits": graph_exits}


def _boundary_metadata(topology: dict | None) -> dict | None:
    if topology is None:
        return None
    return {"mode": topology["boundary_mode"], "graph_entry_count": len(topology["graph_entries"]),
            "graph_exit_count": len(topology["graph_exits"]), "entry_count": len(topology["entries"]),
            "exit_count": len(topology["exits"]), "entry_edges": sorted(topology["entries"]),
            "exit_edges": sorted(topology["exits"]),
            "assumption": ("Graph endpoints are model boundaries; field OD is unknown." if
                           topology["boundary_mode"] == "graph_terminal" else
                           "A non-signal road end with only turnaround predecessors/successors is an inferred "
                           "external model boundary. Turnarounds at signal junctions remain internal.")}


def _lane_allows(lane: ET.Element, kind: str) -> bool:
    allow, disallow = lane.get("allow"), lane.get("disallow", "").split()
    return (allow is None or "all" in allow.split() or kind in allow.split()) and kind not in disallow and "all" not in disallow


def _route_metadata(route: CandidateRoute | RouteVehicle, topology: dict | None, approaches: set[str]) -> dict:
    edges = route.edges
    value = {"id": route.id, "edges": list(edges), "repeated_edge_count": len(edges) - len(set(edges)),
             "starts_on_observed_approach": edges[0] in approaches,
             "ends_on_observed_approach": edges[-1] in approaches}
    if topology is None:
        value.update(origin_is_boundary=None, destination_is_boundary=None, length_m=None, connected=None)
    else:
        value.update(origin_is_boundary=edges[0] in topology["entries"],
                     destination_is_boundary=edges[-1] in topology["exits"],
                     length_m=sum(float(topology["edges"][e].find("lane").get("length"))
                                  for e in edges if e in topology["edges"]),
                     connected=all(e in topology["edges"] for e in edges) and
                               all(pair in topology["connections"] for pair in zip(edges, edges[1:])))
    return value


def audit_planned_routes(
    ledger: DemandLedger, routes: Sequence[RouteVehicle], specs: Sequence[Mapping] | None = None,
    observed_approaches: Iterable[str] | None = None, network_file: Path | None = None,
    boundary_mode: str = "graph_terminal",
) -> dict:
    """Count the entire planned paths, ignoring departure and passage times.

    Explicit-zero cells get an overflow check. Unrecorded movements are reported
    separately, not certified errors. Internal endpoints and unresolved counts
    remain visible. This report must not be used as actual passage-time evidence.
    """
    resolved = movement_specs(ledger, specs)
    approaches = set(observed_approaches or ()) | {a for a, b in ledger.source_pairs}
    topology = _topology(network_file, boundary_mode)
    counted, unobserved, unobserved_movements = Counter(), Counter(), Counter()
    unrecorded_registered = Counter()
    path_counts, endpoint_totals = Counter(), Counter()
    invalid, examples = [], []
    known_for_type = defaultdict(set)
    for kind, a, b in ledger.counts:
        known_for_type[kind].add((a, b))
    for route in routes:
        metadata = _route_metadata(route, topology, approaches)
        if metadata["connected"] is False:
            invalid.append({"vehicle_id": route.id, "edges": list(route.edges)})
        claims, movement_claims = set(), set()
        for pair, item in resolved.items():
            starts = occurrence_indices(route.edges, item["path"])
            if not starts:
                continue
            count = len(starts)
            counted[(route.vehicle_type, *pair)] += count
            path_counts[pair] += count
            if pair in ledger.source_pairs:
                for start in starts:
                    movement_claims.update(range(start, start + len(item["path"]) - 1))
            if pair in known_for_type[route.vehicle_type]:
                for start in starts:
                    claims.update(range(start, start + len(item["path"]) - 1))
            else:
                unrecorded_registered[(route.vehicle_type, *pair)] += count
        for index, (a, b) in enumerate(zip(route.edges, route.edges[1:])):
            if a in approaches and index not in claims:
                unobserved[(route.vehicle_type, a, b)] += 1
            if a in approaches and index not in movement_claims:
                unobserved_movements[(a, b)] += 1
        for key in ("starts_on_observed_approach", "ends_on_observed_approach"):
            endpoint_totals[key] += int(metadata[key])
        for key in ("origin_is_boundary", "destination_is_boundary"):
            if metadata[key] is not None:
                endpoint_totals["internal_" + key.removesuffix("_is_boundary")] += int(not metadata[key])
        if (metadata["starts_on_observed_approach"] or metadata["ends_on_observed_approach"] or
                metadata["origin_is_boundary"] is False or metadata["destination_is_boundary"] is False) and len(examples) < 20:
            examples.append(metadata)
    rows, totals_by_type = [], defaultdict(Counter)
    for cell, source_count in sorted(ledger.counts.items()):
        kind, a, b = cell
        item = resolved[(a, b)]
        planned = counted[cell]
        row = {"vehicle_type": kind, "source_from": a, "source_to": b,
               "source_count": source_count, "planned_passages": planned,
               "under": max(source_count - planned, 0), "over": max(planned - source_count, 0),
               "source_state": "positive" if source_count else "explicit_zero",
               "provenance": ledger.cell_provenance.get(cell, "provided"), **item}
        rows.append(row)
        for key in ("source_count", "planned_passages", "under", "over"):
            totals_by_type[kind][key] += row[key]
    unresolved = sum(ledger.counts[cell] for cell in ledger.counts if resolved[cell[1:]]["path"] is None)
    totals = {key: sum(row[key] for row in rows) for key in ("source_count", "planned_passages", "under", "over")}
    source_by_movement = Counter()
    for (_, a, b), value in ledger.counts.items():
        source_by_movement[(a, b)] += value
    movement_rows = [{"source_from": a, "source_to": b, "source_count": count,
                      "planned_passages": path_counts[(a, b)], "under": max(count - path_counts[(a, b)], 0),
                      "over": max(path_counts[(a, b)] - count, 0)}
                     for (a, b), count in sorted(source_by_movement.items())]
    movement_totals = {key: sum(row[key] for row in movement_rows)
                       for key in ("source_count", "planned_passages", "under", "over")}
    return {"scope": "whole planned routes; passage times ignored", "source": ledger.summary(),
            "totals": totals, "totals_scope": "vehicle_type x source movement",
            "movement_totals_ignoring_vehicle_type": movement_totals,
            "by_movement_ignoring_vehicle_type": movement_rows,
            "by_vehicle_type": {k: dict(v) for k, v in totals_by_type.items()},
            "unresolved_source_count": unresolved, "by_movement": rows,
            "unobserved_planned_transitions": sum(unobserved.values()),
            "unrecorded_registered_complete_movements": sum(unrecorded_registered.values()),
            "unrecorded_registered_by_movement": [{"vehicle_type": k, "source_from": a,
                                                   "source_to": b, "count": n,
                                                   "path": resolved[(a, b)]["path"]}
                                                  for (k, a, b), n in sorted(unrecorded_registered.items())],
            "unobserved_scope_note": ("Raw-approach transitions and registered complete movements use different "
                                      "units and can overlap; do not add the two unknown-demand totals."),
            "unobserved_planned_movements_ignoring_vehicle_type": sum(unobserved_movements.values()),
            "unobserved_by_movement_ignoring_vehicle_type": [{"from": a, "to": b, "count": n}
                                                            for (a, b), n in sorted(unobserved_movements.items())],
            "unobserved_by_transition": [{"vehicle_type": k, "from": a, "to": b, "count": n}
                                          for (k, a, b), n in sorted(unobserved.items())],
            "observed_approaches": sorted(approaches), "generated_vehicle_count": len(routes),
            "boundary": _boundary_metadata(topology),
            "endpoint_counts": dict(endpoint_totals), "endpoint_examples": examples,
            "invalid_route_count": len(invalid), "invalid_route_examples": invalid[:20],
            "absence_is_zero": False, "actual_passage_times_verified": False}


def fit_route_demand(
    ledger: DemandLedger, candidates: Sequence[CandidateRoute], specs: Sequence[Mapping] | None = None,
    network_file: Path | None = None, observed_approaches: Iterable[str] | None = None,
    weights: Mapping[str, float] | None = None, time_limit_s: float = 30,
    boundary_only: bool = False, max_edge_repeats: int = 0,
    max_route_length_m: float | None = 10000,
    boundary_mode: str = "graph_terminal",
) -> dict:
    """Estimate integer route counts with separate under/over slack per source cell.

    Source provenance weights are explicit; supplied synthetic counts need not
    have the same weight as independent measurements. Unknown transitions and
    internal sources/sinks receive weak penalties, not forced-zero constraints.
    Boundary-only mode and route length/repetition limits are explicit choices.
    A best feasible time-limited result is labeled as such; fit is not validation
    of historical OD, arrivals, movement aliases, or actual passage counts.
    """
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    if time_limit_s <= 0 or max_edge_repeats < 0:
        raise ValueError("Invalid optimization limits")
    settings = {"measured": 10.0, "provided": 1.0, "estimated": 0.25,
                "under": 1000.0, "over": 1000.0, "unobserved": 1.0,
                "internal_endpoint": 0.25, "repeated_edge": 1.0, "vehicle": 0.001}
    settings.update(weights or {})
    if any(not np.isfinite(v) or v < 0 for v in settings.values()) or settings["under"] <= 0 or settings["over"] <= 0:
        raise ValueError("Weights must be finite/nonnegative; under/over must be positive")
    topology = _topology(network_file, boundary_mode)
    if boundary_only and topology is None:
        raise ValueError("Boundary-only fitting needs a network")
    resolved = movement_specs(ledger, specs)
    approaches = set(observed_approaches or ()) | {a for a, b in ledger.source_pairs}
    eligible, rejected, route_info = [], [], {}
    seen_candidate_ids = set()
    for route in candidates:
        if route.id in seen_candidate_ids:
            raise ValueError(f"Duplicate candidate id: {route.id}")
        seen_candidate_ids.add(route.id)
        if not route.edges:
            raise ValueError(f"Empty candidate route: {route.id}")
        info = _route_metadata(route, topology, approaches)
        reason = None
        if info["connected"] is False:
            reason = "disconnected_or_unknown_edge"
        elif max((route.edges.count(e) - 1 for e in set(route.edges)), default=0) > max_edge_repeats:
            reason = "repeated_edge_limit"
        elif max_route_length_m is not None and info["length_m"] is not None and info["length_m"] > max_route_length_m:
            reason = "length_limit"
        elif boundary_only and not (info["origin_is_boundary"] and info["destination_is_boundary"]):
            reason = "internal_endpoint"
        if reason:
            rejected.append({"id": route.id, "reason": reason})
        else:
            eligible.append(route)
            route_info[route.id] = info
    allocations, solver_reports, fitted = [], [], []
    for kind in sorted({cell[0] for cell in ledger.counts}):
        cells = sorted(cell for cell in ledger.counts if cell[0] == kind)
        b = np.array([ledger.counts[cell] for cell in cells], dtype=float)
        available, coefficients, costs = [], [], []
        for route in eligible:
            if topology is not None and any(not any(_lane_allows(a, kind) and _lane_allows(z, kind)
                                                    for a, z in topology["connection_lanes"][pair])
                                             for pair in zip(route.edges, route.edges[1:])):
                continue
            column = [len(occurrence_indices(route.edges, resolved[cell[1:]]["path"])) for cell in cells]
            if not any(n and target > 0 for n, target in zip(column, b)):
                continue
            claims = set()
            for cell in cells:
                path = resolved[cell[1:]]["path"]
                for start in occurrence_indices(route.edges, path):
                    claims.update(range(start, start + len(path) - 1))
            unknown = sum(a in approaches and i not in claims for i, (a, z) in enumerate(zip(route.edges, route.edges[1:])))
            # An absent bypass may never traverse its raw approach ID. Include
            # the complete registered movement when no raw transition penalty
            # already represents that occurrence, without forcing it to zero.
            recorded_pairs = {cell[1:] for cell in cells}
            for pair, item in resolved.items():
                if pair in recorded_pairs:
                    continue
                path = item["path"]
                for start in occurrence_indices(route.edges, path):
                    if not any(route.edges[i] in approaches and i not in claims
                               for i in range(start, start + len(path) - 1)):
                        unknown += 1
            info = route_info[route.id]
            internal = sum(info[k] is False for k in ("origin_is_boundary", "destination_is_boundary"))
            cost = (settings["unobserved"] * unknown + settings["internal_endpoint"] * internal +
                    settings["repeated_edge"] * info["repeated_edge_count"] + settings["vehicle"])
            available.append(route); coefficients.append(column); costs.append(cost)
        n, m = len(available), len(cells)
        rr, cc, vv = [], [], []
        for j, column in enumerate(coefficients):
            for i, value in enumerate(column):
                if value:
                    rr.append(i); cc.append(j); vv.append(value)
        for i in range(m):
            rr.extend((i, i)); cc.extend((n + i, n + m + i)); vv.extend((1, -1))
        matrix = coo_matrix((vv, (rr, cc)), shape=(m, n + 2 * m)).tocsc()
        cell_weights = [settings.get(ledger.cell_provenance.get(cell, "provided"), settings["provided"]) for cell in cells]
        objective = np.array(costs + [settings["under"] * w for w in cell_weights] +
                             [settings["over"] * w for w in cell_weights])
        solution = milp(objective, integrality=np.ones(n + 2 * m), bounds=Bounds(0, np.inf),
                        constraints=LinearConstraint(matrix, b, b), options={"time_limit": time_limit_s})
        report = {"vehicle_type": kind, "status": int(solution.status), "message": solution.message,
                  "candidate_count": n, "objective": None if solution.fun is None else float(solution.fun),
                  "optimality_proven": solution.status == 0, "mip_gap": getattr(solution, "mip_gap", None)}
        if solution.x is None:
            # Do not erase counts when no feasible incumbent was found.
            route_counts = np.zeros(n, dtype=int)
            report["fallback"] = "zero allocation; full source deficit retained"
        else:
            rounded = np.rint(solution.x)
            if np.max(np.abs(solution.x - rounded), initial=0) > 1e-5 or np.max(np.abs(matrix @ rounded - b), initial=0) > 1e-5:
                raise RuntimeError("Solver incumbent failed independent integer/equality validation")
            route_counts = rounded[:n].astype(int)
        solver_reports.append(report)
        for route, value in zip(available, route_counts):
            if value:
                allocations.append({"vehicle_type": kind, "route_id": route.id, "count": int(value), **route_info[route.id]})
                for i in range(int(value)):
                    fitted.append(RouteVehicle(f"estimated_{kind}_{len(fitted)}", kind, route.edges))
    return {"status": "estimated_route_demand", "source": ledger.summary(), "weights": settings,
            "limits": {"time_limit_s_per_type": time_limit_s, "boundary_only": boundary_only,
                       "boundary_mode": boundary_mode,
                       "max_edge_repeats": max_edge_repeats, "max_route_length_m": max_route_length_m},
            "candidate_routes_input": len(candidates), "candidate_routes_eligible": len(eligible),
            "rejected_candidates": rejected, "solver": solver_reports, "allocations": allocations,
            "boundary": _boundary_metadata(topology),
            "audit": audit_planned_routes(ledger, fitted, specs, approaches, network_file, boundary_mode),
            "historical_arrivals_recreated": False, "movement_assumptions_validated": False}
