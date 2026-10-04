"""보완 사본의 작은 경로 선택 부족을 제한된 경로 교환으로 줄인다.

도로망·원본 관측·후보 경로를 바꾸지 않고, 같은 차종과 60초 출발 구간 안에서
기존 차량 한 대의 경로를 후보 경로 한두 개로 교환한다. 다른 관측의 반영량은
정확히 유지되는 교환만 허용한다. 기존 결과는 별도 출력 사본으로 보존한다.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
import math
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from prepare_checked_sample import PROJECT, digest, print_summary, write_json, write_xml
from prepare_sumo_scenario import (
    VEHICLE_TYPES, merge_and_rebase_routes, write_data_quality_report,
)
from verify_checked_sample import verify


def plan_exchange(source: Path) -> dict:
    metadata = json.loads((source / "scenario/scenario_metadata.json").read_text())
    rows = json.loads((source / "scenario/data_quality_by_movement.json").read_text())
    network = ET.parse(source / "scenario/network.net.xml").getroot()
    connections = {(c.get("from"), c.get("to")) for c in network.findall("connection")}
    route_candidates = [tuple(r.get("edges").split())
                        for r in ET.parse(source / "inputs/possible_routes.rou.xml").findall(".//route")]
    route_candidates = [r for r in route_candidates if all(p in connections for p in zip(r, r[1:]))]
    coverage = {p for route in route_candidates for p in zip(route, route[1:])}
    deficits = [r for r in rows if r["missing_turn_count"] > 0
                and (r["from_edge"], r["to_edge"]) in coverage]
    if len(deficits) != 1 or deficits[0]["missing_turn_count"] != 1:
        raise ValueError("이 보정은 후보 경로가 존재하는 관측 1건의 부족만 대상으로 합니다.")
    target = deficits[0]
    kind, begin, end = target["vehicle_type"], target["simulation_begin_sec"], target["simulation_end_sec"]
    active = [r for r in rows if r["vehicle_type"] == kind and r["simulation_begin_sec"] == begin]
    keys = {(r["from_edge"], r["to_edge"]) for r in active}
    deficit_pair = (target["from_edge"], target["to_edge"])

    def signature(route: tuple) -> frozenset:
        return frozenset(set(zip(route, route[1:])) & keys)

    # 같은 관측 조합을 만족하는 기존 후보 중 도로 수가 적은 것을 대표로 삼는다.
    candidates = {}
    for route in route_candidates:
        s = signature(route)
        if s and (s not in candidates or len(route) < len(candidates[s])):
            candidates[s] = route
    chosen = defaultdict(list)
    for v in ET.parse(source / "scenario" / metadata["route_file"]).findall("vehicle"):
        if v.get("type") == kind and begin <= float(v.get("depart")) < end:
            chosen[signature(tuple(v.find("route").get("edges").split()))].append(v.get("id"))
    for s, ids in chosen.items():
        if deficit_pair in s:
            continue
        residual = s | {deficit_pair}
        if residual in candidates:
            selected = [candidates[residual]]
        else:
            selected = None
            for r in candidates:
                other = residual - r
                if deficit_pair in r and r <= residual and other in candidates:
                    selected = [candidates[r], candidates[other]]
                    break
        if selected:
            return {"source_bundle": str(source), "vehicle_type": kind,
                    "simulation_begin_sec": begin, "simulation_end_sec": end,
                    "removed_vehicle_ids": [ids[0]], "new_routes": [list(r) for r in selected],
                    "deficit_pair": list(deficit_pair),
                    "method": "bounded exact exchange: at most one removed and two replacement routes"}
    raise ValueError("다른 관측량을 유지하는 작은 경로 교환을 찾지 못했습니다.")


def refine(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"기존 결과를 덮어쓰지 않습니다: {output}")
    if source == output or source in output.parents:
        raise ValueError("원본 결과 폴더 내부에는 출력 사본을 만들 수 없습니다.")
    before = {p: digest(p) for p in source.rglob("*") if p.is_file()}
    plan = plan_exchange(source)
    # 이전 주행 통계는 복사하지 않는다. 새 수요는 따로 주행 검증해야 한다.
    output.mkdir(parents=True)
    for name in ("inputs", "scenario"):
        shutil.copytree(source / name, output / name)
    shutil.copy2(source / "correction_manifest.json", output / "correction_manifest.json")
    metadata_path = output / "scenario/scenario_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    kind = plan["vehicle_type"]
    intermediate = output / "scenario/intermediate"
    tree = ET.parse(intermediate / f"{kind}.rou.xml")
    parent = tree.getroot()
    victim = next(v for v in parent.findall("vehicle") if v.get("id") == plan["removed_vehicle_ids"][0])
    parent.remove(victim)
    new_ids = []
    for i, route in enumerate(plan["new_routes"]):
        vehicle = copy.deepcopy(victim)
        if i:
            vehicle.set("id", f"{victim.get('id')}_count_refinement_{i}")
        new_ids.append(vehicle.get("id"))
        vehicle.find("route").set("edges", " ".join(route))
        parent.append(vehicle)
    all_vehicles = sorted(parent.findall("vehicle"), key=lambda v: float(v.get("depart")))
    for vehicle in parent.findall("vehicle"):
        parent.remove(vehicle)
    parent.extend(all_vehicles)
    write_xml(tree, intermediate / f"{kind}.rou.xml")
    plan["replacement_vehicle_ids"] = new_ids

    sampled = [intermediate / f"{name}.rou.xml" for name, _ in VEHICLE_TYPES.values()]
    count = merge_and_rebase_routes(output / "inputs/exp.rou.xml", sampled,
                                  output / "scenario" / metadata["route_file"], metadata["source_begin_sec"])
    actual = Counter()
    for vehicle in ET.parse(output / "scenario" / metadata["route_file"]).findall("vehicle"):
        begin = math.floor(float(vehicle.get("depart")) / 60) * 60 + metadata["source_begin_sec"]
        route = vehicle.find("route").get("edges").split()
        for a, b in set(zip(route, route[1:])):
            actual[vehicle.get("type"), begin, a, b] += 1
    mismatch_files = {}
    for name, _ in VEHICLE_TYPES.values():
        path = intermediate / f"{name}.mismatch.xml"
        tree = ET.parse(path)
        for interval in tree.findall(".//interval"):
            begin, end = float(interval.get("begin")), float(interval.get("end"))
            for r in interval.findall("edgeRelation"):
                measured = int(float(r.get("measuredCount")))
                generated = actual[name, begin, r.get("from"), r.get("to")]
                r.set("deficit", str(measured - generated))
                geh = abs(measured - generated) * math.sqrt(2 * 3600 / (end - begin) / (measured + generated)) if measured + generated else 0
                r.set("GEH", f"{geh:.6f}")
        write_xml(tree, path)
        mismatch_files[name] = path
    previous = json.loads((source / "scenario/data_quality_by_movement.json").read_text())
    quality = write_data_quality_report(output / "scenario", mismatch_files,
                                        metadata["source_begin_sec"], count, 0.10)
    current = json.loads((output / "scenario/data_quality_by_movement.json").read_text())
    changed = []
    for old, new in zip(previous, current, strict=True):
        if old["generated_turn_count"] != new["generated_turn_count"]:
            changed.append(new)
    if len(changed) != 1 or changed[0]["missing_turn_count"] != 0 or quality["missing_turn_count"] != 59:
        raise AssertionError("목표 1건 이외의 관측량이 변경됐습니다. 새 사본은 채택하지 않습니다.")
    metadata["mismatch_rate"] = quality["mismatch_rate"]
    metadata["data_quality_status"] = quality["status"]
    metadata["count_refinement"] = plan["method"]
    write_json(metadata_path, metadata)
    manifest_path = output / "correction_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["count_refinement"] = plan
    write_json(manifest_path, manifest)
    write_json(output / "route_exchange_plan.json", plan)
    verify(output)
    if any(digest(path) != fingerprint for path, fingerprint in before.items()):
        raise AssertionError("원래 보완 사본의 파일이 변경됐습니다.")
    print_summary(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path,
                        default=PROJECT / "results/data_quality_audit_20261001/checked_sample")
    parser.add_argument("--output-bundle", type=Path,
                        default=PROJECT / "results/data_quality_audit_20261001/checked_sample_refined")
    parser.add_argument("--show-summary", action="store_true")
    args = parser.parse_args()
    if args.show_summary:
        print_summary(args.output_bundle.expanduser().resolve())
    else:
        refine(args.source_bundle.expanduser().resolve(), args.output_bundle.expanduser().resolve())
