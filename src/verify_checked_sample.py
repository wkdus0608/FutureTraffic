"""보완 사본의 원본 보존·경로 연결·관측량을 독립적으로 대조한다."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from prepare_checked_sample import PROJECT, digest, write_json
from prepare_sumo_scenario import VEHICLE_TYPES


def verify(bundle: Path) -> dict:
    manifest = json.loads((bundle / "correction_manifest.json").read_text())
    metadata = json.loads((bundle / "scenario/scenario_metadata.json").read_text())
    sources = manifest["source_files"]
    errors = []
    source_files_unchanged = True
    for record in sources.values():
        if digest(Path(record["path"])) != record["sha256"]:
            errors.append("원본 파일 지문 변경")
            source_files_unchanged = False
    network_path = bundle / "scenario/network.net.xml"
    if digest(network_path) != sources["1200012n.net.xml"]["sha256"]:
        errors.append("생성 도로망이 원본과 다름")
    network = ET.parse(network_path).getroot()
    edges = {e.get("id"): e for e in network.findall("edge") if not e.get("id", "").startswith(":")}
    connections = {(c.get("from"), c.get("to")) for c in network.findall("connection")
                   if not c.get("from", "").startswith(":")}
    incoming_target = {name for name, e in edges.items() if e.get("to") == "204820"}
    mapping = manifest["conditional_id_mapping"]
    raw = Counter()
    for code, (kind, _) in VEHICLE_TYPES.items():
        name = f"{metadata['source_turn_prefix']}{code}.xml"
        for interval in ET.parse(sources[name]["path"]).findall(".//interval"):
            begin = float(interval.get("begin"))
            for r in interval.findall("edgeRelation"):
                a, b = r.get("from"), r.get("to")
                if b == mapping["from"]:
                    b = mapping["to"]
                raw[kind, begin, a, b] += int(float(r.get("count", "0")))

    generated = Counter()
    vehicles = ET.parse(bundle / "scenario" / metadata["route_file"]).findall("vehicle")
    for vehicle in vehicles:
        route = vehicle.find("route").get("edges").split()
        if any(pair not in connections for pair in zip(route, route[1:])):
            errors.append(f"허용되지 않은 생성 경로: {vehicle.get('id')}")
        begin = math.floor(float(vehicle.get("depart")) / 60) * 60 + metadata["source_begin_sec"]
        # routeSampler는 경로가 관측 관계를 포함하는지를 차량마다 센다.
        for a, b in set(zip(route, route[1:])):
            generated[vehicle.get("type"), begin, a, b] += 1
    candidates = {pair for r in ET.parse(bundle / "inputs/possible_routes.rou.xml").findall(".//route")
                  for pair in zip(r.get("edges").split(), r.get("edges").split()[1:])}
    rows = json.loads((bundle / "scenario/data_quality_by_movement.json").read_text())
    row_keys = set()
    missing_categories = Counter()
    target_original = target_missing = overflow = missing = 0
    for r in rows:
        a, b = r["from_edge"], r["to_edge"]
        key = r["vehicle_type"], r["source_begin_sec"], a, b
        row_keys.add(key)
        if raw[key] != r["original_turn_count"] or generated[key] != r["generated_turn_count"]:
            errors.append(f"원본/실제 생성 경로와 보고서 불일치: {key}")
        deficit = raw[key] - generated[key]
        if deficit != r["missing_turn_count"]:
            errors.append(f"누락 계산 불일치: {key}")
        missing += max(deficit, 0)
        overflow += max(-deficit, 0)
        if a in incoming_target:
            target_original += raw[key]
            target_missing += max(deficit, 0)
        if deficit > 0:
            category = ("missing_edge_id" if a not in edges or b not in edges else
                        "no_direct_connection" if (a, b) not in connections else
                        "no_candidate_route" if (a, b) not in candidates else "sampling_shortfall")
            missing_categories[category] += deficit
    if any(raw[key] > 0 for key in raw.keys() - row_keys):
        errors.append("품질 보고서에서 빠진 원본 관측 관계")
    for route in manifest["added_candidate_routes"]:
        route_edges = route["edges"]
        if len(route_edges) < 3 or any(p not in connections for p in zip(route_edges, route_edges[1:])):
            errors.append("보충 후보 경로가 짧거나 연결되지 않음")
    summary = json.loads((bundle / "scenario/data_quality_summary.json").read_text())
    if summary["missing_turn_count"] != missing or summary["generated_vehicle_count"] != len(vehicles):
        errors.append("요약 수치 불일치")
    result = {"passed": not errors, "errors": errors, "source_files_unchanged": source_files_unchanged,
              "network_unchanged": digest(network_path) == sources["1200012n.net.xml"]["sha256"],
              "original_turn_count": sum(raw.values()), "generated_vehicle_count": len(vehicles),
              "positive_missing_turn_count": missing, "extra_turn_count": overflow,
              "missing_categories": dict(missing_categories),
              "target_original_turn_count": target_original, "target_missing_turn_count": target_missing,
              "added_full_candidate_routes": len(manifest["added_candidate_routes"])}
    write_json(bundle / "independent_verification.json", result)
    if errors:
        raise AssertionError(f"검사 실패 {len(errors)}건: {errors[:5]}. 자세한 내용은 independent_verification.json 참조")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path,
                        default=PROJECT / "results/data_quality_audit_20261001/checked_sample")
    print(json.dumps(verify(parser.parse_args().bundle.resolve()), ensure_ascii=False, indent=2))
