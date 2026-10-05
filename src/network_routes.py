"""Whole-network route candidates with explicit inferred boundary origins/destinations.

Counts do not identify OD trips. These paths are model candidates, not recovered
individual journeys. A movement is covered by its complete external-edge path.
"""
from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def network_graph(network: Path, boundary_mode: str = "graph_terminal"):
    root = ET.parse(network).getroot()
    edges = {}
    for edge in root.findall("edge"):
        identifier = edge.get("id", "")
        if identifier.startswith(":"):
            continue
        lanes = [lane for lane in edge.findall("lane") if lane.get("disallow") != "all"]
        if lanes:
            edges[identifier] = max(float(lane.get("length")) for lane in lanes)
    successors = {edge: set() for edge in edges}
    predecessors = {edge: set() for edge in edges}
    uturns = set()
    for connection in root.findall("connection"):
        a, b = connection.get("from"), connection.get("to")
        if a in edges and b in edges:
            successors[a].add(b)
            predecessors[b].add(a)
            if connection.get("dir") == "t":
                uturns.add((a, b))
    if boundary_mode not in ("graph_terminal", "terminal_turnaround"):
        raise ValueError("Unknown model boundary mode")
    nodes = {node.get("id"): node.get("type", "") for node in root.findall("junction")}
    external = {edge.get("id"): edge for edge in root.findall("edge") if edge.get("id") in edges}
    entries = sorted(edge for edge in edges if not predecessors[edge])
    exits = sorted(edge for edge in edges if not successors[edge])
    if boundary_mode == "terminal_turnaround":
        entries = sorted(edge for edge in edges if
            not {other for other in predecessors[edge] if (other, edge) not in uturns} and
            not nodes.get(external[edge].get("from"), "").startswith("traffic_light"))
        exits = sorted(edge for edge in edges if
            not {other for other in successors[edge] if (edge, other) not in uturns} and
            not nodes.get(external[edge].get("to"), "").startswith("traffic_light"))
    return edges, successors, predecessors, uturns, entries, exits


def shortest_path(start, end, edges, successors, forbidden_pairs=(), forbidden_edges=()):
    if start not in edges or end not in edges:
        return None
    forbidden_pairs, forbidden_edges = set(forbidden_pairs), set(forbidden_edges)
    queue = [(0.0, (start,))]
    best = {start: 0.0}
    while queue:
        cost, path = heapq.heappop(queue)
        current = path[-1]
        if cost > best.get(current, float("inf")):
            continue
        if current == end:
            return path
        for following in sorted(successors[current]):
            if (current, following) in forbidden_pairs or following in forbidden_edges:
                continue
            candidate_cost = cost + edges[following]
            if candidate_cost < best.get(following, float("inf")):
                best[following] = candidate_cost
                heapq.heappush(queue, (candidate_cost, path + (following,)))
    return None


def contains_path(route, path):
    n = len(path)
    return n > 0 and any(tuple(route[i:i + n]) == tuple(path) for i in range(len(route) - n + 1))


def build_boundary_candidates(network: Path, specs: list[dict], provided_routes: Path | None = None,
                              per_movement: int = 12, boundary_mode: str = "graph_terminal") -> tuple[list[dict], dict]:
    """Preserve provided boundary paths and add shortest paths through each movement.

    U-turns outside the requested movement are avoided for generated candidates.
    No edges/connections are added and no observed junction is truncated.
    """
    if per_movement < 1:
        raise ValueError("per_movement must be positive")
    lengths, successors, _, uturns, entries, exits = network_graph(network, boundary_mode)
    entry_set, exit_set = set(entries), set(exits)
    candidates: dict[tuple, dict] = {}
    provided_count = 0
    provided_boundary_count = 0
    if provided_routes:
        for element in ET.parse(provided_routes).iter("route"):
            path = tuple(element.get("edges", "").split())
            if not path:
                continue
            provided_count += 1
            if path[0] not in entry_set or path[-1] not in exit_set:
                continue
            if len(set(path)) != len(path):
                continue
            if not all(a in successors and b in successors[a] for a, b in zip(path, path[1:])):
                continue
            # Unobserved artificial turnarounds are not required to preserve an OD.
            requested_uturns = {tuple(spec["path"]) for spec in specs if spec.get("path")
                               and len(spec["path"]) == 2 and tuple(spec["path"]) in uturns}
            if any((a, b) in uturns and (a, b) not in requested_uturns for a, b in zip(path, path[1:])):
                continue
            provided_boundary_count += 1
            candidates[path] = {"origin": "provided_boundary_candidate"}
    coverage = []
    for spec in specs:
        target = tuple(spec.get("path") or ())
        if not target:
            coverage.append({"movement_id": spec["id"], "covered": False, "reason": "unresolved_movement"})
            continue
        if not all(a in successors and b in successors[a] for a, b in zip(target, target[1:])):
            coverage.append({"movement_id": spec["id"], "covered": False, "reason": "disconnected_movement_path"})
            continue
        prefixes, suffixes = [], []
        for entry in entries:
            prefix = shortest_path(entry, target[0], lengths, successors, uturns, target[1:])
            if prefix:
                prefixes.append(prefix)
        for exit_edge in exits:
            suffix = shortest_path(target[-1], exit_edge, lengths, successors, uturns, target[:-1])
            if suffix:
                suffixes.append(suffix)
        possible = set()
        for prefix in prefixes:
            for suffix in suffixes:
                path = prefix[:-1] + target + suffix[1:]
                if len(set(path)) == len(path):
                    possible.add(path)
        selected = sorted(possible, key=lambda path: (sum(lengths[e] for e in path), path))[:per_movement]
        for path in selected:
            candidates.setdefault(path, {"origin": "inferred_boundary_shortest_candidate"})
        count = sum(contains_path(path, target) for path in candidates)
        coverage.append({"movement_id": spec["id"], "covered": count > 0,
                         "new_candidates_for_movement": len(selected), "candidate_count": count})
    output = []
    for index, (path, provenance) in enumerate(sorted(candidates.items())):
        output.append({"id": f"boundary_{index}", "edges": list(path), **provenance,
                       "length_m": round(sum(lengths[e] for e in path), 2),
                       "origin_is_boundary": True, "destination_is_boundary": True})
    metadata = {"method": "whole_network_boundary_candidates_v2", "boundary_mode": boundary_mode, "entry_edges": entries,
                "exit_edges": exits, "provided_routes": provided_count,
                "retained_provided_boundary_routes": provided_boundary_count,
                "candidate_count": len(output), "per_movement": per_movement,
                "movement_coverage": coverage,
                "od_is_inferred": True, "junctions_truncated": False,
                "assumptions": ["Graph terminals represent the model boundary, not verified field OD.",
                                "Internal sources/sinks are not reconstructed without supporting data.",
                                "Generated paths minimize length among sampled boundary pairs."]}
    return output, metadata


def write_candidates(candidates: list[dict], path: Path):
    root = ET.Element("routes")
    for candidate in candidates:
        ET.SubElement(root, "route", id=candidate["id"], edges=" ".join(candidate["edges"]))
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--provided-routes", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    specs = json.loads(args.specs.read_text())
    routes, metadata = build_boundary_candidates(args.network, specs, args.provided_routes)
    write_candidates(routes, args.out)
    args.out.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
