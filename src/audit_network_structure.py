"""도로망 전체의 참조·차로 연결·출구 도달·신호·후보 경로를 검사한다.

경계 dead_end와 내부 연결 부족을 구분한다. 보고서만 만들며 지도를 수정하지 않는다.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

CLASSES = ("passenger", "bus", "truck", "motorcycle")


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def permits(lane, kind):
    allowed, denied = lane.get("allow"), lane.get("disallow", "").split()
    return (allowed is None or kind in allowed.split() or "all" in allowed.split()) and kind not in denied and "all" not in denied


def audit(path, routes=None):
    root = ET.parse(path).getroot()
    errors = []; findings = []
    def indexed(tag):
        elements = root.findall(tag)
        counts = Counter(e.get("id") for e in elements)
        for key, n in counts.items():
            if key is None or n != 1:
                errors.append({"type": "duplicate_or_missing_id", "tag": tag, "id": key, "count": n})
        return {e.get("id"): e for e in elements}
    edges, junctions, logics = indexed("edge"), indexed("junction"), indexed("tlLogic")
    external = {k: e for k, e in edges.items() if e.get("function") != "internal"}
    lanes = {}; lane_index = {}
    for key, edge in edges.items():
        for ordinal, lane in enumerate(edge.findall("lane")):
            name = lane.get("id")
            if name in lanes:
                errors.append({"type": "duplicate_lane_id", "lane": name})
            # SUMO and sumolib resolve connection fromLane/toLane by edge lane
            # order. Some netconvert versions emit index="0" on the second lane
            # of a grouped internal continuation edge. Keep that producer
            # metadata discrepancy visible, without inventing a broken runtime
            # connection. A genuinely out-of-range ordinal still fails below.
            lanes[name] = lane; lane_index[key, str(ordinal)] = lane
            if lane.get("index") != str(ordinal):
                findings.append({"type": "lane_index_metadata_mismatch", "edge": key,
                    "lane": name, "declared_index": lane.get("index"), "runtime_ordinal": ordinal,
                    "interpretation": "producer metadata differs from SUMO/sumolib ordered-lane lookup"})
            if float(lane.get("length", "0")) <= 0 or float(lane.get("speed", "0")) <= 0:
                errors.append({"type": "nonpositive_lane_geometry", "lane": name})
        if key in external:
            for attr in ("from", "to"):
                if edge.get(attr) not in junctions:
                    errors.append({"type": "unknown_junction", "edge": key, "attribute": attr})
    for key, junction in junctions.items():
        for attr in ("incLanes", "intLanes"):
            for name in junction.get(attr, "").split():
                if name not in lanes:
                    errors.append({"type": "unknown_junction_lane", "junction": key, "field": attr, "lane": name})
    connections = root.findall("connection")
    lane_graph = defaultdict(set)
    for c in connections:
        la = lane_index.get((c.get("from"), c.get("fromLane")))
        lb = lane_index.get((c.get("to"), c.get("toLane")))
        destination = c.get("via") or (lb.get("id") if lb is not None else None)
        if la is not None and destination in lanes:
            lane_graph[la.get("id")].add(destination)

    def internal_reaches(start, target, kind=None):
        seen = set(); todo = [start]
        while todo:
            name = todo.pop()
            if name in seen or name not in lanes or (kind and not permits(lanes[name], kind)):
                continue
            if name == target:
                return True
            seen.add(name)
            if name.startswith(":"):
                todo.extend(lane_graph[name])
        return False
    pair_classes = defaultdict(set); lane_out = defaultdict(set); edge_out = defaultdict(set)
    used_indexes = defaultdict(set); connection_keys = Counter()
    for c in connections:
        a, b = c.get("from"), c.get("to")
        la, lb = lane_index.get((a, c.get("fromLane"))), lane_index.get((b, c.get("toLane")))
        via = c.get("via")
        bad = False
        for label, exists in (("from_edge", a in edges), ("to_edge", b in edges),
                              ("from_lane", la is not None), ("to_lane", lb is not None),
                              ("via_lane", not via or via in lanes)):
            if not exists:
                errors.append({"type": "invalid_connection_reference", "field": label, "connection": c.attrib}); bad = True
        if bad:
            continue
        signature = (a, b, c.get("fromLane"), c.get("toLane"), via)
        connection_keys[signature] += 1
        if a in external and b in external:
            if edges[a].get("to") != edges[b].get("from"):
                errors.append({"type": "connection_crosses_different_junctions", "connection": c.attrib})
            edge_out[a].add(b); lane_out[la.get("id")].add(lb.get("id"))
            if via and not internal_reaches(via, lb.get("id")):
                errors.append({"type": "broken_internal_lane_chain", "connection": c.attrib})
            used = [la, lb]
            for kind in CLASSES:
                if all(permits(l, kind) for l in used) and (not via or internal_reaches(via, lb.get("id"), kind)):
                    pair_classes[a, b].add(kind)
        if c.get("tl"):
            tls = c.get("tl")
            try:
                link = int(c.get("linkIndex", "-1"))
            except ValueError:
                link = -1
            used_indexes[tls].add(link)
            if tls not in logics or link < 0:
                errors.append({"type": "invalid_signal_reference", "connection": c.attrib})
            elif any(link >= len(p.get("state", "")) for p in logics[tls].findall("phase")):
                errors.append({"type": "signal_index_out_of_range", "connection": c.attrib})
    for k, n in connection_keys.items():
        if n > 1:
            errors.append({"type": "duplicate_connection", "connection": list(k), "count": n})
    signal_summary = []
    for key, logic in logics.items():
        widths = sorted({len(p.get("state", "")) for p in logic.findall("phase")})
        used = used_indexes[key]
        if len(widths) != 1 or any(float(p.get("duration", "0")) <= 0 for p in logic.findall("phase")):
            errors.append({"type": "invalid_signal_phases", "tls": key})
        if widths and used != set(range(widths[0])):
            findings.append({"type": "unused_signal_indexes", "tls": key, "unused": sorted(set(range(widths[0])) - used)})
        if len(widths) == 1:
            for index in used:
                if 0 <= index < widths[0] and not any(p.get("state", "")[index] in "Ggs" for p in logic.findall("phase")):
                    findings.append({"type": "signal_connection_never_served", "tls": key, "link_index": index})
        signal_summary.append({"tls": key, "widths": widths, "controlled_indexes": sorted(used)})
    boundary_exits = []; boundary_entries = []; internal_sinks = []; lane_sinks = []
    edge_in = defaultdict(set)
    for a, successors in edge_out.items():
        for b in successors:
            edge_in[b].add(a)
    for key, e in external.items():
        j = junctions.get(e.get("to"))
        if not edge_out[key]:
            if j is not None and j.get("type") == "dead_end":
                boundary_exits.append(key)
            else:
                internal_sinks.append(key)
        if not edge_in[key]:
            start = junctions.get(e.get("from"))
            if start is not None and start.get("type") == "dead_end":
                boundary_entries.append(key)
            else:
                findings.append({"type": "internal_edge_without_incoming", "edge": key})
        for lane in e.findall("lane"):
            if not lane_out[lane.get("id")] and edge_out[key]:
                lane_sinks.append({"lane": lane.get("id"), "edge": key, "junction": e.get("to"),
                                   "classes": [k for k in CLASSES if permits(lane, k)],
                                   "interpretation": "review lane termination/change; not automatic deletion"})
    class_summary = {}
    for kind in CLASSES:
        valid = {k for k, e in external.items() if any(permits(l, kind) for l in e.findall("lane"))}
        graph = defaultdict(set); reverse = defaultdict(set)
        for (a, b), kinds in pair_classes.items():
            if kind in kinds:
                graph[a].add(b); reverse[b].add(a)
        exits = set(boundary_exits) & valid
        reachable = set(exits); todo = deque(exits)
        while todo:
            for prev in reverse[todo.popleft()]:
                if prev not in reachable:
                    reachable.add(prev); todo.append(prev)
        class_summary[kind] = {"allowed_edges": len(valid), "cannot_reach_boundary_exit": sorted(valid - reachable),
                               "internal_edges_without_allowed_successor": sorted(k for k in valid - exits if not graph[k])}
    candidate_errors = []; route_count = 0; allowed_route_counts = Counter()
    if routes:
        route_root = ET.parse(routes).getroot()
        vehicle_classes = {v.get("id"): v.get("vClass", "passenger") for v in route_root.findall("vType")}
        route_parent = {id(v.find("route")): v for v in route_root.findall("vehicle") if v.find("route") is not None}
        for route in route_root.findall(".//route"):
            route_count += 1
            seq = route.get("edges", "").split()
            invalid_edges = [e for e in seq if e not in external]
            broken = [(a, b) for a, b in zip(seq, seq[1:]) if b not in edge_out[a]]
            allowed = [k for k in CLASSES if all(k in pair_classes[a, b] for a, b in zip(seq, seq[1:]))
                       and all(any(permits(l, k) for l in external[e].findall("lane")) for e in seq if e in external)]
            allowed_route_counts.update(allowed)
            vehicle = route_parent.get(id(route))
            required = vehicle_classes.get(vehicle.get("type"), "passenger") if vehicle is not None else None
            if required and required not in allowed:
                candidate_errors.append({"vehicle": vehicle.get("id"), "type": "vehicle_class_route_forbidden", "class": required})
            if invalid_edges or broken or not allowed:
                candidate_errors.append({"route": route.get("id"), "unknown_edges": invalid_edges,
                                         "broken_pairs": broken, "allowed_research_classes": allowed})
    return {"network": str(path.resolve()), "sha256": digest(path),
            "counts": {"edges": len(edges), "external_edges": len(external), "lanes": len(lanes),
                       "junctions": len(junctions), "connections": len(connections), "signals": len(logics)},
            "structural_errors": errors, "review_findings": findings,
            "boundary_exit_edges": boundary_exits, "boundary_entry_edges": boundary_entries,
            "terminal_interpretation": "dead_end is a declared terminal in the supplied model, not proof of a real-road boundary; compare source before any repair",
            "internal_sink_edges": internal_sinks, "lane_termination_review": lane_sinks,
            "vehicle_class_reachability": class_summary, "signals": signal_summary,
            "candidate_routes": route_count, "candidate_route_errors": candidate_errors,
            "routes_allowed_by_vehicle_class": dict(allowed_route_counts),
            "scope": "all XML references, external lane connections, four vehicle classes and supplied routes; not real-road legality"}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--network", type=Path, required=True)
    p.add_argument("--routes", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = audit(args.network, args.routes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"counts": report["counts"], "structural_errors": len(report["structural_errors"]),
                     "internal_sinks": report["internal_sink_edges"],
                     "lane_terminations": len(report["lane_termination_review"]),
                     "candidate_route_errors": len(report["candidate_route_errors"]),
                     "reachability": {k: len(v["cannot_reach_boundary_exit"]) for k,v in report["vehicle_class_reachability"].items()}}, ensure_ascii=False))
