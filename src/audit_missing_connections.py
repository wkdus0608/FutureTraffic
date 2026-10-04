"""남은 회전 연결을 원본 교통량·도로망·차량 이동 자료와 자동 대조한다.

이 검사는 제공 자료 안의 정합성을 검사한다. 실제 도로의 회전 허용 여부를
확정하지 않으며 도로망·교통량·기존 시나리오를 수정하지 않는다.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
import csv
import json
import math
from pathlib import Path
import unicodedata
import xml.etree.ElementTree as ET

from prepare_checked_sample import PROJECT, digest, write_json
from prepare_sumo_scenario import find_one


def road_name(edge: ET.Element) -> str:
    name = edge.get("name", "")
    try:
        return bytes.fromhex(name).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return name


def direction(a: ET.Element, b: ET.Element) -> tuple[str, float]:
    p = [tuple(map(float, xy.split(","))) for xy in a.find("lane").get("shape").split()]
    q = [tuple(map(float, xy.split(","))) for xy in b.find("lane").get("shape").split()]
    inbound = math.atan2(p[-1][1]-p[-2][1], p[-1][0]-p[-2][0])
    outbound = math.atan2(q[1][1]-q[0][1], q[1][0]-q[0][0])
    angle = (math.degrees(outbound-inbound)+180) % 360 - 180
    label = "유턴" if abs(angle) > 150 else "직진" if abs(angle) < 30 else "좌회전" if angle > 0 else "우회전"
    return label, round(angle, 2)


def shortest_path(a: str, b: str, outgoing: dict) -> list[str]:
    queue = deque([[a]])
    seen = {a}
    while queue:
        path = queue.popleft()
        if path[-1] == b:
            return path
        for nxt in sorted(outgoing.get(path[-1], set())):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(path + [nxt])
    return []


def audit(bundle: Path, sample: Path, output: Path) -> dict:
    network_path = find_one(sample, "1200012n.net.xml")
    trajectory_path = find_one(sample, "90001220220810050000tra.csv")
    source_paths = [network_path, trajectory_path]
    hashes = {str(p): digest(p) for p in source_paths}
    network = ET.parse(network_path).getroot()
    all_edges = {e.get("id"): e for e in network.findall("edge")}
    edges = {k: e for k, e in all_edges.items() if not k.startswith(":")}
    lane_to_edge = {lane.get("id"): name for name, e in all_edges.items() for lane in e.findall("lane")}
    connections = {(c.get("from"), c.get("to")) for c in network.findall("connection")
                   if not c.get("from", "").startswith(":")}
    outgoing = {}
    for a, b in connections:
        outgoing.setdefault(a, set()).add(b)
    rows = json.loads((bundle / "scenario/data_quality_by_movement.json").read_text())
    missing = Counter()
    types = {}
    for r in rows:
        pair = r["from_edge"], r["to_edge"]
        if r["missing_turn_count"] > 0 and pair not in connections:
            missing[pair] += r["missing_turn_count"]
            types.setdefault(pair, Counter())[r["vehicle_type"]] += r["missing_turn_count"]
    candidate_pairs = {p for r in ET.parse(bundle / "inputs/possible_routes.rou.xml").findall(".//route")
                       for p in zip(r.get("edges").split(), r.get("edges").split()[1:])}
    trajectory_rows = Counter()
    direct_passages = Counter()
    last_external, last_time = {}, {}
    unknown_lanes = Counter()
    time_regressions = record_count = 0
    times = []
    with trajectory_path.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            record_count += 1
            vehicle, timestamp = row["vhcl_id"], float(row["unix_time"])
            times.append(timestamp)
            if vehicle in last_time and timestamp < last_time[vehicle]:
                time_regressions += 1
            last_time[vehicle] = timestamp
            edge = lane_to_edge.get(row["lane_id"])
            if edge is None:
                unknown_lanes[row["lane_id"]] += 1
                continue
            if edge.startswith(":"):
                continue
            trajectory_rows[edge] += 1
            if vehicle in last_external and edge != last_external[vehicle]:
                direct_passages[last_external[vehicle], edge] += 1
            last_external[vehicle] = edge
    signal_ids = set()
    signal_lengths = {}
    signal_file_count = 0
    for p in sample.rglob("*s.csv"):
        if "신호데이터" not in unicodedata.normalize("NFC", str(p)):
            continue
        signal_file_count += 1
        with p.open(encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                signal_ids.add(row["interid"])
                signal_lengths.setdefault(row["interid"], Counter())[len(row["signalstate"])] += 1
    network_signal_lengths = {t.get("id"): sorted({len(p.get("state")) for p in t.findall("phase")})
                              for t in network.findall("tlLogic")}
    signal_xml_path = find_one(sample, "90001220220810050000s.xml")
    xml_signal_lengths = {t.get("id"): sorted({len(p.get("state")) for p in t.findall("phase")})
                          for t in ET.parse(signal_xml_path).findall("tlLogic")}
    signal_comparison = []
    for junction in sorted(signal_ids):
        source_lengths = sorted(signal_lengths[junction])
        net_lengths = network_signal_lengths.get(junction, [])
        signal_comparison.append({"junction": junction, "source_csv_state_lengths": source_lengths,
                                  "source_xml_state_lengths": xml_signal_lengths.get(junction, []),
                                  "network_state_lengths": net_lengths,
                                  "length_mismatch": source_lengths != net_lengths})

    items = []
    for (a, b), count in sorted(missing.items()):
        junction = edges[a].get("to")
        label, angle = direction(edges[a], edges[b])
        path = shortest_path(a, b, outgoing)
        nodes = [edges[e].get("from") for e in path] + ([edges[path[-1]].get("to")] if path else [])
        same_from_family = [e.get("id") for e in edges.values()
                            if e.get("id").split("_")[0] == a.split("_")[0] and e.get("to") == junction]
        same_to_family = [e.get("id") for e in edges.values()
                          if e.get("id").split("_")[0] == b.split("_")[0] and e.get("from") == junction]
        item = {"from_edge": a, "to_edge": b, "from_road_name": road_name(edges[a]),
                "to_road_name": road_name(edges[b]), "junction": junction,
                "same_junction": junction == edges[b].get("from"),
                "missing_turn_count": count, "by_vehicle_type": dict(types[a, b]),
                "geometry_inferred_turn": label, "angle_deg": angle,
                "direct_network_connection": (a, b) in connections,
                "candidate_direct_movement": (a, b) in candidate_pairs,
                "from_edge_trajectory_rows": trajectory_rows[a],
                "to_edge_trajectory_rows": trajectory_rows[b],
                "direct_trajectory_passages": direct_passages[a, b],
                "same_family_incoming_candidates": same_from_family,
                "same_family_outgoing_candidates": same_to_family,
                "indirect_connection_path": path,
                "indirect_path_revisits_junction": nodes.count(junction) > 1,
                "real_road_permission": "not established by supplied sample",
                "decision": "keep pending; do not add a connection without source evidence"}
        items.append(item)
    if any(digest(Path(p)) != fingerprint for p, fingerprint in hashes.items()):
        raise AssertionError("검사 중 원본 자료가 변경됐습니다.")
    result = {"source_files": hashes, "trajectory_rows": record_count,
              "trajectory_vehicle_count": len(last_time), "source_time_range": [min(times), max(times)],
              "unknown_lanes": dict(unknown_lanes), "time_regressions": time_regressions,
              "signal_csv_file_count": signal_file_count, "signal_intersection_ids": sorted(signal_ids),
              "signal_length_comparison": signal_comparison,
              "remaining_missing_turn_count": sum(missing.values()), "movement_count": len(items),
              "junction_count": len({i['junction'] for i in items}), "items": items,
              "interpretation": "제공된 차량 이동 자료 안의 대조 결과이며, 실제 도로의 통행 허용 여부를 입증하지 않습니다."}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "remaining_connections_audit.json", result)
    fields = ["from_edge", "to_edge", "from_road_name", "to_road_name", "junction", "missing_turn_count",
              "geometry_inferred_turn", "direct_trajectory_passages", "real_road_permission"]
    with (output / "remaining_connections.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(items)
    print(json.dumps({k: v for k, v in result.items() if k not in ("items", "source_files")}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path,
                        default=PROJECT / "results/data_quality_audit_20261001/checked_sample_refined")
    parser.add_argument("--sample-root", type=Path, default=PROJECT / "Sample")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT / "results/data_quality_audit_20261001/remaining_checks")
    args = parser.parse_args()
    audit(args.bundle.resolve(), args.sample_root.resolve(), args.output_dir.resolve())
