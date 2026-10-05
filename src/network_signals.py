"""Explicit signal baselines for the complete provided SUMO network.

Public API:
    audit_signal_links(net_file, period_files) -> JSON-compatible inventory
    audit_priority_green_conflicts(net_file, programs=None) -> static foes audit
    build_network_signal_programs(net_file, period_files, start_s, duration_s,
        variant="embedded" | "reconstructed" | "reconstructed_mixed",
        column_maps=None, options=None) -> (programs, metadata)
    write_network_signal_additional(programs, output, metadata=None) -> Path

``period_files`` maps HHMM to source XML paths for one calendar day. Signal
records describe phases, never vehicle arrival times. Sources are not edited.
Width equality is only structural compatibility. Identity column mapping,
phase-row shifts, synthetic yellows, dropout interpolation and tail extension
are assumptions, and require explicit options or a supplied mapping.

``column_maps[tls]`` is {"indices": [source_index_for_each_network_link],
"reason": "...", "certified": False}. Indices may be repeated only when
``allow_shared_source`` is explicitly true. No padding or column guessing.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping

SIGNAL_VERSION = "network_signals_v2"
SIGNAL_STATES = frozenset("rRgGyYuUoOs")
EPSILON = 1e-7


class SignalGateError(ValueError):
    """A rejected replay; ``report`` preserves the precise unresolved gate."""

    def __init__(self, message: str, report: dict | None = None):
        super().__init__(message)
        self.report = report or {"error": message}


@dataclass(frozen=True)
class ReplayOptions:
    allow_identity_mapping: bool = False
    phase_shifts: dict[str, int] = field(default_factory=dict)
    interpolate_long_greens: bool = False
    repair_yellow: bool = False
    yellow_seconds: float = 3.0
    dropout_ratio: float = 1.6
    dropout_extra_seconds: float = 40.0
    # Missing final phases are inferred from the dominant source phase order.
    fill_period_tails: bool = True
    # A drain period repeats a source cycle, rather than jumping to midnight.
    extend_last_cycle: bool = False
    # Diagnostic execution only: a conflicting signal never becomes safe.
    allow_priority_conflicts: bool = False


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_phases(logic: ET.Element, label: str) -> list[tuple[float, str]]:
    phases = []
    for phase in logic.findall("phase"):
        duration = float(phase.get("duration", "nan"))
        state = phase.get("state", "")
        if not math.isfinite(duration) or duration < 0 or not state or set(state) - SIGNAL_STATES:
            raise SignalGateError(f"{label}: invalid phase {phase.attrib}")
        phases.append((duration, state))
    if not phases or not any(d > 0 for d, _ in phases):
        raise SignalGateError(f"{label}: no positive-duration phases")
    return phases


def _network(net_file: Path) -> dict:
    root = ET.parse(net_file).getroot()
    tls = {}
    for logic in root.findall("tlLogic"):
        key = logic.get("id")
        if key in tls:
            raise SignalGateError(f"{key}: select one network program before replay")
        phases = _read_phases(logic, f"network {key}")
        widths = sorted({len(s) for _, s in phases})
        if len(widths) != 1:
            raise SignalGateError(f"{key}: inconsistent embedded signal widths {widths}")
        tls[key] = {"width": widths[0], "phases": phases, "offset": logic.get("offset", "0"),
                    "program_id": logic.get("programID", "0"), "type": logic.get("type", "static"),
                    "links": []}
    for c in root.findall("connection"):
        key = c.get("tl")
        if not key:
            continue
        if key not in tls:
            raise SignalGateError(f"connection references unknown TLS {key}")
        idx = int(c.get("linkIndex", "-1"))
        if not 0 <= idx < tls[key]["width"]:
            raise SignalGateError(f"{key}: linkIndex {idx} outside embedded state")
        tls[key]["links"].append({"link_index": idx, "from_edge": c.get("from"),
            "to_edge": c.get("to"), "from_lane": c.get("fromLane"), "to_lane": c.get("toLane"),
            "direction": c.get("dir"), "via": c.get("via")})
    if not tls:
        raise SignalGateError("network contains no traffic light programs")
    _request_mappings(root, tls)
    return tls


def _request_mappings(root: ET.Element, tls: dict) -> None:
    """Map signals to right-of-way requests independently of TLS indices.

    SUMO orders requests by junction incLanes, then the connection elements
    belonging to each incoming lane (sumolib Node.getLinkIndex uses this too).
    Resolve lane IDs from the actual edge/lane inventory, never name suffixes.
    Trace split internal lanes to the destination as an additional structural
    check. In particular a joined TLS need not have the same ID as a junction.
    https://sumo.dlr.de/docs/Networks/SUMO_Road_Networks.html#indices-of-a-connection
    """
    edges = {e.get("id"): e for e in root.findall("edge")}
    lane_ids, lane_owner = {}, {}
    for edge_id, edge in edges.items():
        for lane in edge.findall("lane"):
            key, lane_id = (edge_id, lane.get("index")), lane.get("id")
            if key in lane_ids or lane_id in lane_owner:
                raise SignalGateError("duplicate lane ID/index prevents request mapping")
            lane_ids[key], lane_owner[lane_id] = lane_id, key
    outgoing = defaultdict(list)
    for conn in root.findall("connection"):
        outgoing[lane_ids.get((conn.get("from"), conn.get("fromLane")))].append(conn)
    mapped, matrices, node_errors = {}, {}, {}
    for junction in root.findall("junction"):
        requests = junction.findall("request")
        if not requests:
            continue
        node, inc = junction.get("id"), junction.get("incLanes", "").split()
        errors, ordered = [], []
        if len(set(inc)) != len(inc) or any(l not in lane_owner for l in inc):
            errors.append("incoming lane inventory is missing or duplicated")
        for lane_id in inc:
            for conn in outgoing[lane_id]:
                source, target = edges.get(conn.get("from")), edges.get(conn.get("to"))
                if source is None or target is None:
                    errors.append("request connection references an unknown edge")
                    continue
                # Pedestrian walking-area links are not right-of-way requests;
                # this is SUMO's own getLinkIndex filtering rule.
                if target.get("function") == "walkingarea" or (
                    source.get("function") == "walkingarea" and target.get("function") != "crossing"):
                    continue
                if source.get("function") not in ("internal", "walkingarea", "crossing") and source.get("to") != node:
                    errors.append(f"incoming edge {conn.get('from')} does not end at {node}")
                ordered.append(conn)
        rows = {}
        for req in requests:
            try:
                idx = int(req.get("index", "-1"))
            except ValueError:
                idx = -1
            if idx in rows:
                errors.append("duplicate request index")
            rows[idx] = {k: req.get(k, "") for k in ("foes", "response", "cont")}
        if set(rows) != set(range(len(ordered))):
            errors.append(f"request indices/count do not cover {len(ordered)} incoming connections")
        if any(len(row[k]) != len(ordered) or set(row[k]) - {"0", "1"}
               for row in rows.values() for k in ("foes", "response")):
            errors.append("foes/response bitstrings do not fit the request matrix")
        matrices[node], node_errors[node] = rows, errors
        for idx, conn in enumerate(ordered):
            if id(conn) in mapped:
                errors.append("a connection belongs to more than one request matrix")
            mapped[id(conn)] = (node, idx)

    junctions = {j.get("id"): j for j in root.findall("junction")}
    for key, info in tls.items():
        refs, errors = [], []
        for conn in (c for c in root.findall("connection") if c.get("tl") == key):
            ref = {"link_index": int(conn.get("linkIndex")), "from_edge": conn.get("from"),
                "to_edge": conn.get("to"), "from_lane": conn.get("fromLane"),
                "to_lane": conn.get("toLane"), "direction": conn.get("dir"), "via": conn.get("via")}
            node_index = mapped.get(id(conn))
            if node_index is None:
                errors.append(f"signal link {ref['link_index']} has no independently indexed request")
                continue
            node, idx = node_index
            errors += [f"{node}: {e}" for e in node_errors[node]]
            ref.update({"junction": node, "request_index": idx})
            if (conn.get("to"), conn.get("toLane")) not in lane_ids:
                errors.append(f"link {ref['link_index']}: destination lane is missing")
            # Follow actual via connections, including the second half after
            # an internal waiting position. No linkIndex/name-index shortcuts.
            trace, via = [], conn.get("via")
            while via:
                if via in trace or via not in lane_owner:
                    errors.append(f"link {ref['link_index']}: internal path missing/cyclic at {via}")
                    break
                trace.append(via)
                edge_id, _ = lane_owner[via]
                if edges[edge_id].get("function") != "internal":
                    errors.append(f"link {ref['link_index']}: via {via} is not an internal lane")
                    break
                followers = [c for c in outgoing[via] if c.get("to") == conn.get("to")
                             and c.get("toLane") == conn.get("toLane")]
                if len(followers) != 1:
                    errors.append(f"link {ref['link_index']}: internal continuation at {via} is not unique")
                    break
                via = followers[0].get("via")
            int_lanes = junctions[node].get("intLanes", "").split()
            matching = [l for l in trace if l in int_lanes]
            if len(matching) != 1:
                errors.append(f"link {ref['link_index']}: path has no unique parent-junction internal lane")
            ref["internal_lane_trace"] = trace
            ref["junction_internal_lane"] = matching[0] if len(matching) == 1 else None
            refs.append(ref)
        if not refs:
            errors.append("TLS has no mapped controlled connections")
        info["request_refs"] = refs
        info["request_mapping_errors"] = sorted(set(errors))
        info["request_matrices"] = {n: matrices[n] for n in sorted({r["junction"] for r in refs})}


def _priority_check(info: dict, phases: list[tuple[float, str]]) -> dict:
    states, elapsed = {}, 0.0
    errors = list(info["request_mapping_errors"])
    for phase_index, (duration, state) in enumerate(phases):
        if not math.isfinite(duration) or duration < 0 or set(state) - SIGNAL_STATES:
            errors.append("audited program contains an invalid duration/state")
            continue
        if duration <= 0:
            continue
        item = states.setdefault(state, {"first_phase_index": phase_index,
            "first_simulation_second": elapsed, "occurrences": 0, "total_duration_s": 0.0})
        item["occurrences"] += 1
        item["total_duration_s"] += duration
        elapsed += duration
    if not states:
        errors.append("audited program has no positive-duration states")
    if any(len(s) != info["width"] for s in states):
        errors.append("audited phase width does not match the network")
    conflicts = []
    if not errors:
        for state, occurrence in states.items():
            active = [r for r in info["request_refs"] if state[r["link_index"]] == "G"]
            pairs = []
            for i, left in enumerate(active):
                for right in active[i + 1:]:
                    if left["junction"] != right["junction"]:
                        continue
                    rows = info["request_matrices"][left["junction"]]
                    a, b = left["request_index"], right["request_index"]
                    # Foes strings start at the RIGHT. TLS states start left.
                    lr, rl = rows[a]["foes"][-1 - b] == "1", rows[b]["foes"][-1 - a] == "1"
                    if lr or rl:
                        pairs.append({"left": left, "right": right,
                            "left_request": rows[a], "right_request": rows[b],
                            "left_foes_right": lr, "right_foes_left": rl,
                            "left_yields_to_right": rows[a]["response"][-1 - b] == "1",
                            "right_yields_to_left": rows[b]["response"][-1 - a] == "1"})
            if pairs:
                conflicts.append({"state": state, **occurrence, "pairs": pairs})
    complete = not errors and len(info["request_refs"]) == len(info["links"])
    return {"mapping_complete": complete, "mapping_errors": sorted(set(errors)),
        "mapping_method": "junction incLanes plus XML connection order; actual internal-lane traces",
        "controlled_connections": len(info["links"]), "mapped_connections": len(info["request_refs"]),
        "request_mapping": info["request_refs"], "junctions": sorted(info["request_matrices"]),
        "unique_states_checked": len(states), "priority_green_states_checked": sum("G" in s for s in states),
        "conflicting_states": conflicts, "static_priority_green_safe": complete and not conflicts,
        "scope": "simultaneous priority G/G against this net's foes matrix; permissive g and transition/clearance safety are separate"}


def audit_priority_green_conflicts(net_file: Path, programs: dict | None = None) -> dict:
    """Check every TLS, including custom/grouped signal indices and joined TLS.

    Missing/extra TLS programs and incomplete request mappings fail the audit.
    This certifies a narrow model invariant, not physical road/signal safety.
    """
    net = _network(Path(net_file))
    selected = programs if programs is not None else {k: v["phases"] for k, v in net.items()}
    missing, extra = sorted(set(net) - set(selected)), sorted(set(selected) - set(net))
    checked = {k: _priority_check(v, selected[k]) for k, v in net.items() if k in selected}
    return {"version": SIGNAL_VERSION, "tls_count": len(net), "tls": checked,
        "missing_tls": missing, "unknown_tls": extra,
        "static_priority_green_safe": not missing and not extra and
            all(v["static_priority_green_safe"] for v in checked.values())}


def _period_start(key: str) -> int:
    if len(key) != 4 or not key.isdigit():
        raise SignalGateError(f"invalid period HHMM {key!r}")
    hour, minute = int(key[:2]), int(key[2:])
    if not 0 <= hour < 24 or minute not in (0, 15, 30, 45):
        raise SignalGateError(f"invalid quarter-hour {key}")
    return hour * 3600 + minute * 60


def _source(period_files: Mapping[str, Path]) -> dict:
    sources = {}
    for key in sorted(period_files):
        _period_start(key)
        path = Path(period_files[key])
        logics = {}
        for logic in ET.parse(path).getroot().findall("tlLogic"):
            tls = logic.get("id")
            if tls in logics:
                raise SignalGateError(f"{path}: duplicate TLS {tls}")
            logics[tls] = _read_phases(logic, f"{key} {tls}")
        sources[key] = {"path": path, "sha256": _sha(path), "tls": logics}
    return sources


def _audit(net: dict, sources: dict, net_file: Path) -> dict:
    inventory = {}
    for tls, info in net.items():
        missing = [key for key, src in sources.items() if tls not in src["tls"]]
        widths = sorted({len(s) for src in sources.values() for _, s in src["tls"].get(tls, [])})
        width_ok = widths == [info["width"]]
        inventory[tls] = {"network_width": info["width"], "source_widths": widths,
            "width_compatible": width_ok, "mapping_certified": False,
            "mapping_status": "unverified_identity" if width_ok else "unresolved_columns",
            "missing_periods": missing, "links": info["links"],
            "unused_network_columns": sorted(set(range(info["width"])) - {c["link_index"] for c in info["links"]}),
            "embedded_priority_check": _priority_check(info, info["phases"])}
    return {"version": SIGNAL_VERSION, "network": {"path": str(net_file), "sha256": _sha(net_file)},
        "tls_count": len(net), "periods": sorted(sources), "tls": inventory,
        "unknown_source_tls": sorted({k for src in sources.values() for k in src["tls"]} - set(net)),
        "source_files": [{"period": key, "path": str(src["path"]), "sha256": src["sha256"]}
                         for key, src in sources.items()]}


def audit_signal_links(net_file: Path, period_files: Mapping[str, Path]) -> dict:
    """Extract movement/linkIndex relations; no semantic match is certified."""
    net_file = Path(net_file)
    return _audit(_network(net_file), _source(period_files), net_file)


def _mapping(tls: str, info: dict, source_width: int, supplied: dict | None,
             options: ReplayOptions) -> tuple[list[int], dict]:
    if supplied is None:
        if info["width"] != source_width:
            raise SignalGateError(f"{tls}: network/source widths {info['width']}/{source_width}; explicit columns required")
        if not options.allow_identity_mapping:
            raise SignalGateError(f"{tls}: width equality does not certify column identity; opt in to that assumption")
        return list(range(source_width)), {"kind": "index_identity_assumption", "mapping_certified": False,
            "reason": "explicit allow_identity_mapping; original index semantics not independently certified"}
    indices = supplied.get("indices")
    if not isinstance(indices, list) or len(indices) != info["width"]:
        raise SignalGateError(f"{tls}: mapping needs exactly {info['width']} source indices")
    if any(type(i) is not int or not 0 <= i < source_width for i in indices):
        raise SignalGateError(f"{tls}: source column out of range; padding/red guessing is prohibited")
    if len(set(indices)) != len(indices) and not supplied.get("allow_shared_source", False):
        raise SignalGateError(f"{tls}: duplicate source columns require allow_shared_source")
    if not supplied.get("reason"):
        raise SignalGateError(f"{tls}: mapping must explain its evidence or assumption")
    certified = bool(supplied.get("certified", False))
    if certified and not supplied.get("evidence"):
        raise SignalGateError(f"{tls}: certified mapping needs an evidence reference")
    return indices, {"kind": "supplied_column_map", "mapping_certified": certified,
                     "specification": copy.deepcopy(supplied)}


def _order(periods: dict, tls: str) -> tuple[list[str], dict[str, str], list[dict]]:
    transitions = defaultdict(Counter)
    states = set()
    for phases in periods.values():
        # Zero-duration boundary records still identify the phase before yellow.
        states.update(s for _, s in phases)
        for (_, a), (_, b) in zip(phases, phases[1:]):
            if a != b:
                transitions[a][b] += 1
    if len(states) == 1:
        only = next(iter(states))
        return [only], {only: only}, []
    successor, notes = {}, []
    for s in states:
        counts = transitions[s]
        if not counts:
            raise SignalGateError(f"{tls}: phase order has no observed successor for {s}")
        ranked = counts.most_common()
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            raise SignalGateError(f"{tls}: ambiguous phase successor for {s}")
        successor[s] = ranked[0][0]
        if len(ranked) > 1:
            notes.append({"kind": "dominant_successor_assumption", "state": s,
                          "selected": ranked[0][0], "transition_counts": dict(counts)})
    order, s = [], next(iter(periods.values()))[0][1]
    while s not in order:
        order.append(s)
        s = successor[s]
    if s != order[0] or set(order) != states:
        raise SignalGateError(f"{tls}: source successors do not form one complete cycle")
    return order, successor, notes


def _append(segs: list, start: float, duration: float, state: str) -> None:
    if duration <= EPSILON:
        return
    if segs and abs(segs[-1][0] + segs[-1][1] - start) > EPSILON:
        raise SignalGateError(f"signal timeline gap/overlap at {start}")
    if segs and segs[-1][2] == state:
        segs[-1][1] += duration
    else:
        segs.append([start, duration, state])


def _interpolate(segs: list, order: list[str], successor: dict, options: ReplayOptions) -> tuple[list, list]:
    def median(state, around):
        vals = [d for t, d, s in segs if s == state and abs(t - around) < 1800 and d > 0]
        return statistics.median(vals) if vals else None
    out, notes = [], []
    for i, (start, duration, state) in enumerate(segs):
        med = median(state, start)
        if "y" in state.lower() or not any(c in "Gg" for c in state) or med is None or not (
            duration >= med * options.dropout_ratio and duration >= med + options.dropout_extra_seconds):
            _append(out, start, duration, state)
            continue
        nxt = segs[i + 1][2] if i + 1 < len(segs) else None
        target = order[order.index(nxt) - 1] if nxt is not None else None
        typical = {s: median(s, start) or options.yellow_seconds for s in order}
        seq, s, best = [], state, None
        for _ in range(10000):
            seq.append(s)
            length = sum(typical[x] for x in seq)
            if target is None or s == target:
                if best is None or abs(length - duration) < abs(best[0] - duration):
                    best = (length, list(seq))
                if length >= duration:
                    break
            s = successor[s]
        else:
            raise SignalGateError("dropout interpolation failed to find a bounded cycle")
        seq = best[1]
        green = [k for k, x in enumerate(seq) if "y" not in x.lower() and any(c in "Gg" for c in x)]
        lengths = [typical[x] for x in seq]
        fixed = sum(d for k, d in enumerate(lengths) if k not in green)
        if not green or fixed >= duration:
            raise SignalGateError("dropout interval too short to preserve inferred yellow lengths")
        scale = (duration - fixed) / sum(lengths[k] for k in green)
        for k in green:
            lengths[k] = float(round(lengths[k] * scale))
        lengths[green[-1]] += duration - sum(lengths)
        if min(lengths) <= 0:
            raise SignalGateError("dropout interpolation produced a nonpositive phase")
        t = start
        for s, d in zip(seq, lengths):
            _append(out, t, d, s)
            t += d
        notes.append({"kind": "long_green_interpolation_assumption", "time": start,
            "seconds": duration, "median_seconds": med, "phases_filled": len(seq)})
    return out, notes


def _day_timeline(periods: dict, tls: str, options: ReplayOptions) -> tuple[list, list, list[str]]:
    order, successor, notes = _order(periods, tls)
    keys = sorted(periods)
    if any(_period_start(b) - _period_start(a) != 900 for a, b in zip(keys, keys[1:])):
        raise SignalGateError(f"{tls}: missing quarter-hour source files")
    segs, carry = [], 0.0
    for i, key in enumerate(keys):
        start = _period_start(key)
        t = start + carry
        for duration, state in periods[key]:
            if duration <= 0:
                continue
            if carry:
                if "y" not in state.lower() and duration > carry:
                    duration -= carry
                elif segs and segs[-1][1] > carry:
                    t -= carry
                    segs[-1][1] -= carry
                else:
                    raise SignalGateError(f"{tls} {key}: cannot reconcile overlapping records")
                carry = 0.0
            _append(segs, t, duration, state)
            t += duration
        gap = start + 900 - t
        if gap < -EPSILON:
            carry = -gap
            notes.append({"kind": "period_overlap_trimmed", "time": start + 900, "seconds": carry})
            if i + 1 == len(keys):
                if segs[-1][1] <= carry:
                    raise SignalGateError(f"{tls}: overlap exceeds final phase")
                segs[-1][1] -= carry
                carry = 0.0
        elif gap > EPSILON:
            if not options.fill_period_tails:
                raise SignalGateError(f"{tls} {key}: missing {gap:g} seconds of source phases")
            state = successor[segs[-1][2]]
            _append(segs, t, gap, state)
            note = {"kind": "period_tail_successor_assumption", "time": t, "seconds": gap, "state": state}
            if i + 1 < len(keys):
                first = periods[keys[i + 1]][0][1]
                note["next_boundary_state"] = first
                note["boundary_agrees"] = first == state
            notes.append(note)
    if options.interpolate_long_greens:
        segs, changes = _interpolate(segs, order, successor, options)
        notes += changes
    if abs(sum(d for _, d, _ in segs) - len(keys) * 900) > EPSILON:
        raise SignalGateError(f"{tls}: reconstructed duration mismatch")
    return segs, notes, order


def _shift_map(order: list[str], shift: int, tls: str) -> dict[str, str]:
    if not shift:
        return {s: s for s in order}
    if len(order) % 2 or any(("y" in a.lower()) == ("y" in b.lower())
                            for a, b in zip(order, order[1:] + order[:1])):
        raise SignalGateError(f"{tls}: phase shift needs alternating distinct green/yellow pairs")
    return {s: order[(i + 2 * shift) % len(order)] for i, s in enumerate(order)}


def _extend_cycle(segs: list, order: list[str], until: float) -> tuple[list, dict]:
    end = segs[-1][0] + segs[-1][1]
    # Use median durations of recent occurrences, not a truncated boundary phase.
    recent = segs[-max(len(order) * 4, 1):]
    typical = {s: statistics.median([d for _, d, x in recent if x == s]) for s in order}
    idx = (order.index(segs[-1][2]) + 1) % len(order)
    extended = copy.deepcopy(segs)
    t = end
    while t < until - EPSILON:
        s = order[idx]
        d = min(typical[s], until - t)
        if d <= 0:
            raise SignalGateError("cannot extend a nonpositive source cycle")
        _append(extended, t, d, s)
        t += d
        idx = (idx + 1) % len(order)
    return extended, {"kind": "drain_cycle_extension_assumption", "time": end,
        "seconds": until - end, "phase_durations": typical}


def _yellow_repair(phases: list[tuple[float, str]], seconds: float) -> tuple[list, list]:
    out, notes, t = [], [], 0.0
    for i, (duration, state) in enumerate(phases):
        nxt = phases[i + 1][1] if i + 1 < len(phases) else None
        stop = [k for k, (a, b) in enumerate(zip(state, nxt or state)) if a in "Gg" and b in "rR"]
        if stop:
            yellow = "".join("y" if k in stop else c for k, c in enumerate(state))
            if "y" in state.lower():
                out.append((duration, yellow))
                kind = "permissive_yellow_assumption"
            else:
                if duration <= seconds:
                    raise SignalGateError(f"phase at {t:g} too short to insert {seconds:g}s yellow")
                out += [(duration - seconds, state), (seconds, yellow)]
                kind = "missing_yellow_assumption"
            notes.append({"kind": kind, "time": t + (duration - seconds if kind == "missing_yellow_assumption" else 0),
                          "indices": stop, "before": state, "after": yellow})
        else:
            out.append((duration, state))
        t += duration
    return out, notes


def _transition_counts(phases: list[tuple[float, str]]) -> dict:
    priority, all_green = 0, 0
    for (_, a), (_, b) in zip(phases, phases[1:]):
        priority += sum(x == "G" and y in "rR" for x, y in zip(a, b))
        all_green += sum(x in "Gg" and y in "rR" for x, y in zip(a, b))
    return {"priority_green_to_red_without_yellow": priority, "any_green_to_red_without_yellow": all_green}


def _repeat_embedded(info: dict, duration: float) -> list[tuple[float, str]]:
    if info["type"] != "static" or info["offset"] not in ("0", "0.0"):
        raise SignalGateError("embedded expansion currently requires a static program with offset=0")
    out, elapsed = [], 0.0
    while elapsed < duration - EPSILON:
        for d, s in info["phases"]:
            if d <= 0:
                continue
            d = min(d, duration - elapsed)
            if d > EPSILON:
                out.append((d, s))
                elapsed += d
            if elapsed >= duration - EPSILON:
                break
    return out


def build_network_signal_programs(net_file: Path, period_files: Mapping[str, Path], start_s: float,
        duration_s: float, variant: str = "embedded", column_maps: dict | None = None,
        options: ReplayOptions | None = None) -> tuple[dict[str, list[tuple[float, str]]], dict]:
    """Build every network TLS; strict replay gates or explicitly mixed fallback.

    Embedded starts at phase zero at simulation time zero, as the supplied net
    (all offsets zero) does. Reconstructed starts at absolute source day time
    ``start_s`` and lasts ``duration_s``; its output starts at simulation zero.
    No optional correction is implicitly inherited from focused experiments.
    """
    if variant not in ("embedded", "reconstructed", "reconstructed_mixed"):
        raise ValueError(f"unknown signal baseline {variant}")
    if not math.isfinite(start_s) or start_s < 0 or not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("start_s must be nonnegative and duration_s must be positive")
    options = options or ReplayOptions()
    if not math.isfinite(options.yellow_seconds) or options.yellow_seconds <= 0:
        raise ValueError("yellow_seconds must be positive")
    if (not math.isfinite(options.dropout_ratio) or options.dropout_ratio <= 1 or
            not math.isfinite(options.dropout_extra_seconds) or options.dropout_extra_seconds < 0):
        raise ValueError("dropout thresholds must be finite, ratio > 1 and extra seconds >= 0")
    net_file = Path(net_file)
    net, sources = _network(net_file), _source(period_files)
    unknown_options = (set(column_maps or {}) | set(options.phase_shifts)) - set(net)
    if unknown_options:
        raise SignalGateError(f"signal options reference unknown TLS {sorted(unknown_options)}")
    audit = _audit(net, sources, net_file)
    programs, details, errors = {}, {}, {}
    for tls, info in net.items():
        if variant == "embedded":
            phases = _repeat_embedded(info, duration_s)
            details[tls] = {"baseline": "embedded", "mapping_certified": False, "modifications": [],
                           "clock": "original embedded phase zero at simulation zero"}
        else:
            try:
                inventory = audit["tls"][tls]
                if not sources or inventory["missing_periods"] or len(inventory["source_widths"]) != 1:
                    raise SignalGateError(f"{tls}: missing/inconsistent source program")
                indices, mapping = _mapping(tls, info, inventory["source_widths"][0],
                    (column_maps or {}).get(tls), options)
                periods = {key: src["tls"][tls] for key, src in sources.items()}
                segs, notes, order = _day_timeline(periods, tls, options)
                if start_s < segs[0][0] - EPSILON:
                    raise SignalGateError(f"{tls}: source begins after requested start")
                end = start_s + duration_s
                if end > segs[-1][0] + segs[-1][1] + EPSILON:
                    if not options.extend_last_cycle:
                        raise SignalGateError(f"{tls}: source ends before requested interval; explicitly enable drain extension")
                    segs, note = _extend_cycle(segs, order, end)
                    notes.append(note)
                shift = options.phase_shifts.get(tls, 0)
                if type(shift) is not int:
                    raise SignalGateError(f"{tls}: phase shift must be an integer number of green phases")
                shifted = _shift_map(order, shift, tls)
                if shift:
                    notes.append({"kind": "phase_row_shift_assumption", "green_phases": shift,
                                  "state_map": shifted})
                phases = []
                for t, d, state in segs:
                    a, b = max(t, start_s), min(t + d, end)
                    if b > a + EPSILON:
                        state = shifted[state]
                        phases.append((b - a, "".join(state[i] for i in indices)))
                if options.repair_yellow:
                    phases, repairs = _yellow_repair(phases, options.yellow_seconds)
                    for note in repairs:
                        note["time"] += start_s
                    notes += repairs
                transitions = _transition_counts(phases)
                if transitions["priority_green_to_red_without_yellow"]:
                    raise SignalGateError(f"{tls}: {transitions['priority_green_to_red_without_yellow']} priority-green→red transitions; explicit yellow reconstruction required")
                details[tls] = {"baseline": "reconstructed", **mapping, "source_phase_order": order,
                    "modifications": notes, "source_window": [start_s, end], "transition_check": transitions,
                    "modification_time_basis": "source day seconds (phase shift applies to the whole program)",
                    "clock": "source day seconds sliced into simulation time zero"}
            except SignalGateError as exc:
                errors[tls] = str(exc)
                if variant == "reconstructed":
                    continue
                phases = _repeat_embedded(info, duration_s)
                details[tls] = {"baseline": "embedded_fallback", "mapping_certified": False,
                    "fallback_reason": str(exc), "modifications": [],
                    "clock": "original embedded phase zero at simulation zero"}
        if abs(sum(d for d, _ in phases) - duration_s) > EPSILON:
            raise SignalGateError(f"{tls}: emitted program duration mismatch")
        if any(d <= 0 or len(s) != info["width"] for d, s in phases):
            raise SignalGateError(f"{tls}: emitted phase does not fit network")
        programs[tls] = phases
    metadata = {"version": SIGNAL_VERSION, "implementation_sha256": _sha(Path(__file__)),
        "variant": variant, "start_s": start_s, "duration_s": duration_s,
        "options": asdict(options), "audit": audit, "tls": details, "gates": errors,
        "source_clock_assumption": "HHMM filenames identify the start of their 900-second record window",
        "claim": "SUMO baseline with recorded assumptions; historical field signal mapping is not certified"}
    if variant == "reconstructed" and errors:
        raise SignalGateError("strict signal reconstruction rejected; select explicit mappings/options or an explicitly mixed baseline", metadata)
    checks = {k: _priority_check(net[k], phases) for k, phases in programs.items()}
    metadata["priority_green_audit"] = {"tls_count": len(net), "tls": checks,
        "static_priority_green_safe": len(checks) == len(net) and
            all(c["static_priority_green_safe"] for c in checks.values())}
    metadata["static_priority_green_safe"] = metadata["priority_green_audit"]["static_priority_green_safe"]
    incomplete = [k for k, c in checks.items() if not c["mapping_complete"]]
    conflicting = [k for k, c in checks.items() if c["conflicting_states"]]
    if incomplete:
        raise SignalGateError(f"incomplete signal/request mapping at {incomplete}; diagnostics cannot bypass this structural gate", metadata)
    if conflicting:
        metadata["diagnostic_only"] = True
        metadata["learning_allowed"] = False
        if not options.allow_priority_conflicts:
            raise SignalGateError(f"priority G/G foes conflict at {conflicting}; explicitly allow_priority_conflicts for diagnostic execution only", metadata)
    else:
        metadata["diagnostic_only"] = False
        # Other field/data gates remain the caller's responsibility.
        metadata["learning_allowed"] = None
    metadata["fingerprint"] = hashlib.sha256(json.dumps(metadata, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return programs, metadata


def write_network_signal_additional(programs: dict[str, list[tuple[float, str]]], output: Path,
                                     metadata: dict | None = None) -> Path:
    """Write an additional-file containing all programs, never a source file."""
    output = Path(output)
    if metadata:
        inputs = [metadata["audit"]["network"]["path"]]
        inputs += [f["path"] for f in metadata["audit"]["source_files"]]
        if any(output.resolve() == Path(p).resolve() for p in inputs):
            raise ValueError("refusing to overwrite an original network/signal input")
    root = ET.Element("additional")
    for tls, phases in sorted(programs.items()):
        logic = ET.SubElement(root, "tlLogic", id=tls, type="static", programID=SIGNAL_VERSION, offset="0")
        for duration, state in phases:
            if duration <= 0:
                raise ValueError("cannot emit a nonpositive signal phase")
            ET.SubElement(logic, "phase", duration=f"{duration:.8g}", state=state)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=True)
    if metadata:
        output.with_suffix(".metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
