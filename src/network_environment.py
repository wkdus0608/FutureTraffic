"""Build and inspect a continuous 12-junction environment before any RL training.

The source ledger is immutable. Route demand and boundary arrival times are
estimated. Unresolved observations and signal assumptions are explicit gates;
successful SUMO execution alone never marks the environment ready for learning.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

from focused_data import find_network, iter_turn_files, signal_xml_files, VEHICLE_CODES
from network_demand import (aggregate_counts, audit_planned_routes, fit_route_demand,
                            read_candidate_routes, read_route_vehicles, read_turn_counts)
from network_routes import build_boundary_candidates, write_candidates

PROJECT = Path(__file__).resolve().parents[1]
VERSION = "network_environment_v5"
SIMULATION_OUTPUTS = ("vehroute.xml", "tripinfo.xml", "stats.xml", "sumo.log", "collisions.xml")
SUPPLEMENTAL_NODES = {"201300", "201329", "201330"}


def implementation_fingerprints():
    files = ["src/network_environment.py", "src/network_routes.py", "src/network_demand.py",
             "src/network_temporal.py", "src/network_movements.py", "src/network_signals.py",
             "config/network_movements.json"]
    files.extend(["src/network_geometry.py", "src/network_join.py"])
    return {name: digest(PROJECT / name) for name in files}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def date_window(date: str, start: str, minutes: int, warmup: int):
    if minutes <= 0 or minutes % 15 or warmup < 0 or warmup % 15:
        raise ValueError("Duration and warmup must use whole 15-minute windows")
    requested = datetime.strptime(date + start, "%Y%m%d%H%M").replace(tzinfo=ZoneInfo("Asia/Seoul"))
    if requested.minute % 15:
        raise ValueError("Start must align with a 15-minute boundary")
    first = requested - timedelta(minutes=warmup)
    end = requested + timedelta(minutes=minutes)
    if first.date() != requested.date() or end > requested.replace(hour=0, minute=0) + timedelta(days=1):
        raise ValueError("This version requires all demand windows in one source day")
    windows = []
    current = first
    while current < end:
        windows.append({"period": current.strftime("%H%M"), "begin": int(current.timestamp()),
                        "end": int((current + timedelta(minutes=15)).timestamp()),
                        "simulation_begin": int((current - first).total_seconds()),
                        "warmup": current < requested})
        current += timedelta(minutes=15)
    return windows


def extract_inputs(date: str, periods: list[str], output: Path):
    """Extract only requested immutable XMLs from the CRC-checked local archive."""
    wanted, files, inventory = set(periods), defaultdict(dict), []
    for name, data in iter_turn_files():
        stem = Path(name).stem
        if stem[6:14] != date or stem[14:18] not in wanted:
            continue
        period, code = stem[14:18], stem[-1]
        if code not in VEHICLE_CODES:
            continue
        destination = output / period / Path(name).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        files[period][VEHICLE_CODES[code]] = destination
        inventory.append({"source_archive_path": name, "path": str(destination),
                          "sha256": digest(destination), "crc_verified_by_archive_reader": True})
    for period in periods:
        if set(files[period]) != set(VEHICLE_CODES.values()):
            raise ValueError(f"Missing original vehicle-type files for {date} {period}")
    return dict(files), inventory


def provenance_labels(network: Path, pairs):
    edges = {e.get("id"): e for e in ET.parse(network).getroot().findall("edge")}
    return {pair: "supplemental_site" if edges[pair[0]].get("to") in SUPPLEMENTAL_NODES
            else "collection_site" for pair in pairs}


def write_demand(allocations: list[dict], vtypes: Path, output: Path, seed: int,
                 driver: dict | None = None):
    root = ET.Element("routes")
    for element in ET.parse(vtypes).getroot().findall("vType"):
        # Parse returns fresh elements; supplier vType source is never changed.
        if driver:
            for key, value in driver.items():
                element.set(key, str(value))
        root.append(element)
    rng, paths, vehicles = random.Random(seed), {}, []
    for allocation in allocations:
        edges = tuple(allocation["edges"])
        if edges not in paths:
            paths[edges] = f"route_{len(paths)}"
        for _ in range(allocation["count"]):
            depart = allocation["simulation_begin"] + rng.uniform(0, 900)
            vehicles.append((depart, allocation["vehicle_type"], paths[edges], allocation["period"]))
    for edges, identifier in paths.items():
        ET.SubElement(root, "route", id=identifier, edges=" ".join(edges))
    intended = {}
    for index, (depart, kind, route, period) in enumerate(sorted(vehicles)):
        identifier = f"network_{index}"
        element = ET.SubElement(root, "vehicle", id=identifier, type=kind, route=route,
                               depart=f"{depart:.2f}", departLane="best", departSpeed="max")
        ET.SubElement(element, "param", key="source_period", value=period)
        intended[identifier] = {"depart": round(depart, 2), "type": kind, "period": period}
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return intended


def build(date: str, start: str, minutes: int, warmup: int, output: Path,
          variant: str = "conservative", time_limit_s: float = 10, seed: int = 42,
          signal_variant: str = "embedded", driver_model: Path | None = None,
          boundary_mode: str = "terminal_turnaround", candidates_per_movement: int = 64,
          assumed_signal_columns: bool = False, repair_yellow: bool = False,
          phase_shifts: dict | None = None, allow_priority_conflicts: bool = False,
          demand_mode: str = "joint_freeflow", geometry_retreat_m: float | None = None,
          join_priority_diagnostic: bool = False):
    from network_movements import load_movement_specs, build_central_assumption_variant
    implementation = implementation_fingerprints()
    if output.exists():
        raise FileExistsError(f"Existing results are preserved; choose a new output: {output}")
    output.mkdir(parents=True)
    windows = date_window(date, start, minutes, warmup)
    original_network = find_network()
    original_hash = digest(original_network)
    map_metadata = {"variant": "original", "map_changed": False}
    network = original_network
    if variant.startswith("central_assumed_embedded"):
        from rl_environment import configure_sumo
        configure_sumo()
        map_metadata = build_central_assumption_variant(original_network, output / "map_variant")
        network = Path(map_metadata["network_file"])
    if geometry_retreat_m is not None:
        from network_geometry import build_internal_waiting_position_variant
        from rl_environment import configure_sumo
        configure_sumo()
        geometry = build_internal_waiting_position_variant(network, output / "geometry_variant",
                                                           retreat_m=geometry_retreat_m)
        map_metadata = {**map_metadata, "geometry_variant": geometry}
        network = Path(geometry["network_file"])
    if join_priority_diagnostic:
        from network_join import build_joined_priority_variant
        from rl_environment import configure_sumo
        configure_sumo()
        joined = build_joined_priority_variant(network, output / "join_variant")
        if not joined["success"]:
            raise ValueError(f"Priority join failed preservation checks: {joined.get('gates')}")
        map_metadata = {**map_metadata, "map_changed": True, "priority_join_variant": joined}
        network = Path(joined["network_file"])
    network_hash = digest(network)
    files, source_inventory = extract_inputs(date, [w["period"] for w in windows], output / "inputs")
    # Source schemas use epoch seconds; the filename alone is not used as the ledger clock.
    ledgers, records = [], []
    for window in windows:
        ledger = read_turn_counts(files[window["period"]], window["begin"], window["end"])
        labels = provenance_labels(network, ledger.source_pairs)
        ledger.cell_provenance = {(kind, a, b): labels[(a, b)] for kind, a, b in ledger.counts}
        ledgers.append(ledger)
        records.extend(ledger.records)
    all_pairs = set().union(*(ledger.source_pairs for ledger in ledgers))
    # Keep all registered source locations, including those without this window's records.
    specs = load_movement_specs(network, variant=variant)
    registered_pairs = {(spec["source_from"], spec["source_to"]) for spec in specs}
    if all_pairs - registered_pairs:
        specs.extend(load_movement_specs(network, source_pairs=all_pairs - registered_pairs, variant=variant))
    registered_approaches = {spec["source_from"] for spec in specs}
    write_json(output / "movement_specs.json", specs)
    labels = provenance_labels(network, all_pairs)
    combined = aggregate_counts(records, windows[0]["begin"], windows[-1]["end"], labels)
    original_candidates = next((PROJECT / "Sample").rglob("possible_routes.rou.xml"))
    candidates, coverage = build_boundary_candidates(network, specs, original_candidates,
                                                     per_movement=candidates_per_movement,
                                                     boundary_mode=boundary_mode)
    candidate_path = output / "candidate_routes.rou.xml"
    write_candidates(candidates, candidate_path)
    write_json(output / "candidate_manifest.json", coverage)
    allocations, window_reports = [], []
    if demand_mode == "joint_freeflow":
        from network_temporal import fit_temporal_demand
        print(f"Jointly estimating {date}: {len(windows)} continuous source windows", flush=True)
        estimate = fit_temporal_demand(ledgers, read_candidate_routes(candidate_path), specs, network,
                                      boundary_mode=boundary_mode, time_limit_s=time_limit_s, seed=seed,
                                      weights={"collection_site": 1.0, "supplemental_site": 0.25})
        write_json(output / "temporal_demand_fit.json", estimate)
        for allocation in estimate["allocations"]:
            matching = next(w for w in windows if w["simulation_begin"] == allocation["cohort_start"])
            allocations.append({**allocation, **matching})
        unresolved_pairs = {(s["source_from"], s["source_to"]) for s in specs if not s.get("path")}
        for index, (window, ledger) in enumerate(zip(windows, ledgers)):
            rows = [row for row in estimate["rounded_dispatch_nominal"]["by_movement"]
                    if row["window_index"] == index]
            totals = {key: sum(row[key] for row in rows) for key in ("source_count", "under", "over")}
            totals["planned_passages"] = sum(row["nominal_dispatch_passages"] for row in rows)
            held = sum(n for (_, a, b), n in ledger.counts.items() if (a, b) in unresolved_pairs)
            report = {"audit": {"by_movement": rows, "totals": totals},
                      "scope": "Uniform boundary dispatch plus nominal road travel; actual passages not yet verified"}
            write_json(output / "demand_fit" / f"{window['period']}.json", report)
            window_reports.append({**window, "source_count": ledger.total, "fit_totals": totals,
                                   "unresolved_source_count": held, "solver": estimate["solver"]})
    elif demand_mode == "same_window_diagnostic":
        for window, ledger in zip(windows, ledgers):
            print(f"Estimating {date} {window['period']}: source passages {ledger.total}", flush=True)
            estimate = fit_route_demand(ledger, read_candidate_routes(candidate_path), specs,
                                       network_file=network, observed_approaches=registered_approaches,
                                       boundary_only=True, boundary_mode=boundary_mode, time_limit_s=time_limit_s,
                                       weights={"collection_site": 1.0, "supplemental_site": 0.25})
            for allocation in estimate["allocations"]:
                allocations.append({**allocation, **window})
            write_json(output / "demand_fit" / f"{window['period']}.json", estimate)
            window_reports.append({**window, "source_count": ledger.total,
                                   "fit_totals": estimate["audit"]["totals"],
                                   "unresolved_source_count": estimate["audit"]["unresolved_source_count"],
                                   "solver": estimate["solver"]})
    else:
        raise ValueError("Unknown boundary demand estimation mode")
    scenario_network = output / "network.net.xml"
    shutil.copy2(network, scenario_network)
    route_file = output / "traffic.rou.xml"
    driver, driver_metadata = None, {"mode": "provided_vtypes", "field_calibrated": False}
    if driver_model:
        driver_metadata = json.loads(driver_model.read_text())
        driver = driver_metadata.get("vtype_attributes")
        if not driver:
            raise ValueError("Driver model must contain vtype_attributes")
    vtypes = next((PROJECT / "Sample").rglob("exp.rou.xml"))
    intended = write_demand(allocations, vtypes, route_file, seed, driver)
    write_json(output / "intended_vehicles.json", intended)
    planned = audit_planned_routes(combined, read_route_vehicles(route_file), specs,
                                  observed_approaches=registered_approaches,
                                  network_file=scenario_network, boundary_mode=boundary_mode)
    write_json(output / "planned_passage_audit.json", planned)
    duration = minutes * 60 + warmup * 60
    from network_signals import (build_network_signal_programs, write_network_signal_additional,
                                 ReplayOptions, SignalGateError)
    options = ReplayOptions(allow_identity_mapping=assumed_signal_columns, repair_yellow=repair_yellow,
                            extend_last_cycle=True, phase_shifts=phase_shifts or {},
                            allow_priority_conflicts=allow_priority_conflicts)
    # Rebuilt model columns must never be mistaken for the old 17 source columns.
    column_maps = None
    if variant.startswith("central_assumed_embedded") and signal_variant != "embedded":
        column_maps = {"201306": {"indices": list(range(15)) + [None, None],
                                   "reason": "New model links have no verified historical source column"}}
    try:
        programs, signal_metadata = build_network_signal_programs(
            scenario_network, signal_xml_files(date),
            int(start[:2]) * 3600 + int(start[2:]) * 60 - warmup * 60, duration + 900,
            variant=signal_variant, options=options, column_maps=column_maps)
    except SignalGateError as error:
        write_json(output / "signal_gate_failure.json", error.report)
        raise
    additional = output / "signals.add.xml"
    write_network_signal_additional(programs, additional, signal_metadata)
    write_json(output / "signal_audit.json", signal_metadata)
    timed_totals = {key: sum(w["fit_totals"][key] for w in window_reports)
                    for key in ("source_count", "planned_passages", "under", "over")}
    build_gates = {"source_denominator_preserved": sum(w["source_count"] for w in window_reports) == combined.total,
                   "unresolved_observations_zero": planned["unresolved_source_count"] == 0,
                   "planned_under_zero": timed_totals["under"] == 0,
                   "planned_over_zero": timed_totals["over"] == 0,
                   "invalid_routes_zero": planned["invalid_route_count"] == 0,
                   "continuous_12_junction_network": len(ET.parse(scenario_network).findall("tlLogic")) == 12,
                   "movement_correspondence_certified": not any(
                       spec.get("variant_certification_blocked", False) for spec in specs),
                   "priority_green_conflicts_zero": signal_metadata.get("static_priority_green_safe") is True,
                   "actual_passages_validated": False,
                   "historical_signal_alignment_validated": False}
    manifest = {"version": VERSION, "date": date, "start": start, "duration_s": duration,
                "evaluation_start_s": warmup * 60, "tail_s": 900, "seed": seed,
                "movement_variant": variant, "signal_variant": signal_variant,
                "demand_mode": demand_mode,
                "boundary_mode": boundary_mode, "map_variant": map_metadata,
                "source_network": {"path": str(original_network), "sha256": original_hash},
                "scenario_network_source": {"path": str(network), "sha256": network_hash},
                "source_inputs": source_inventory, "source_summary": combined.summary(),
                "planned_totals_by_source_window": timed_totals,
                "window_reports": window_reports, "driver_model": driver_metadata,
                "implementation_fingerprints": implementation,
                "build_gates": build_gates, "learning_ready": False,
                "input_fingerprints": {str(p.relative_to(output)): digest(p)
                    for p in output.rglob("*") if p.is_file() and p.name != "manifest.json"},
                "assumptions": ["Boundary OD is inferred from aggregate passages.",
                                "Uniform arrival times are sampled at external model boundaries.",
                                "Nominal boundary travel omits queues and signal delay; actual passage audit is required.",
                                "Supplemental junction source counts receive an explicit reduced fit weight.",
                                "Warmup and initial storage are synthetic; field queues are unavailable."]}
    if digest(network) != network_hash or digest(original_network) != original_hash:
        raise RuntimeError("Original network changed during preparation")
    if implementation_fingerprints() != implementation:
        raise RuntimeError("Implementation changed during preparation; do not mix versions in one bundle")
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"source_turn_count": combined.total, "generated_vehicles": len(intended),
                      "planned_totals": planned["totals"], "learning_ready": False}, ensure_ascii=False), flush=True)
    return manifest


def movement_completion_events(vehicle: ET.Element, specs: list[dict]):
    """Use entry to the final external edge, including every preceding internal edge.

    A positive upstream exitTime only proves entry into the junction. A negative
    internal exitTime cannot prove a completed turn. Intentionally unfinished
    planned suffixes are never counted as completed movements.
    """
    routes = vehicle.findall("route") + vehicle.findall("routeDistribution/route")
    route = next((r for r in reversed(routes) if r.get("exitTimes") is not None), None)
    if route is None:
        return []
    edges = route.get("edges", "").split()
    times = [float(t) for t in route.get("exitTimes", "").split()]
    if len(edges) != len(times):
        raise ValueError("Internal edge output must have one exitTime per planned edge")
    external = [(edge, i) for i, edge in enumerate(edges) if not edge.startswith(":" )]
    external_edges = [edge for edge, _ in external]
    events = []
    for spec in specs:
        path = spec.get("path")
        if not path or len(path) < 2:
            continue
        for start in range(len(external_edges) - len(path) + 1):
            if external_edges[start:start + len(path)] != list(path):
                continue
            first_index = external[start][1]
            last_index = external[start + len(path) - 1][1]
            # All preceding sections of the semantic corridor must have exited.
            traversed_times = times[first_index:last_index]
            if not traversed_times or any(t < 0 for t in traversed_times):
                continue
            if any(a > b for a, b in zip(traversed_times, traversed_times[1:])):
                raise ValueError("Non-monotone route exit times")
            events.append({"vehicle_id": vehicle.get("id"), "vehicle_type": vehicle.get("type", "DEFAULT_VEHTYPE"),
                           "source_from": spec["source_from"], "source_to": spec["source_to"],
                           "movement_id": spec["id"], "completion_s": times[last_index - 1]})
    return events


def verify_run_inputs(bundle: Path, run: Path):
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    execution = json.loads((run / "execution.json").read_text())
    if manifest.get("version") != VERSION or execution.get("manifest_sha256") != digest(manifest_path):
        raise ValueError("Unsupported or changed manifest; previous output cannot certify this environment")
    if execution.get("input_fingerprints") != manifest["input_fingerprints"]:
        raise ValueError("Run input inventory does not match its manifest")
    if manifest.get("implementation_fingerprints") != implementation_fingerprints():
        raise ValueError("Implementation changed since preparation; do not certify with a different validator")
    for name, fingerprint in manifest["input_fingerprints"].items():
        if digest(bundle / name) != fingerprint:
            raise ValueError(f"Run input changed: {name}")
    outputs = execution.get("output_fingerprints", {})
    if set(outputs) != set(SIMULATION_OUTPUTS):
        raise ValueError("Simulation output fingerprints are incomplete")
    for name, fingerprint in outputs.items():
        if digest(run / name) != fingerprint:
            raise ValueError(f"Run output changed: {name}")
    return manifest


def inspect_run(bundle: Path, run: Path):
    manifest = verify_run_inputs(bundle, run)
    command = json.loads((run / "execution.json").read_text()).get("command", [])
    specs = json.loads((bundle / "movement_specs.json").read_text())
    intended = json.loads((bundle / "intended_vehicles.json").read_text())
    expected, provenance = Counter(), {}
    independent_source_total = 0
    for window in manifest["window_reports"]:
        report = json.loads((bundle / "demand_fit" / f"{window['period']}.json").read_text())
        originals = {VEHICLE_CODES[p.stem[-1]]: p for p in (bundle / "inputs" / window["period"]).glob("*.xml")}
        ledger = read_turn_counts(originals, window["begin"], window["end"])
        fit_cells = {(row["vehicle_type"], row["source_from"], row["source_to"]): row["source_count"]
                     for row in report["audit"]["by_movement"]}
        if ledger.counts != fit_cells or ledger.total != window["source_count"]:
            raise ValueError("Fit denominator or cells differ from independent source XML")
        independent_source_total += ledger.total
        labels = provenance_labels(bundle / "network.net.xml", ledger.source_pairs)
        for (kind, a, b), count in ledger.counts.items():
            key = (window["simulation_begin"], kind, a, b)
            expected[key] = count
            provenance[key] = labels[(a, b)]
    if independent_source_total != manifest["source_summary"]["source_turn_count"]:
        raise ValueError("Manifest source total differs from independent source XML")
    completed, raw_transitions = Counter(), Counter()
    observed_approaches = {spec["source_from"] for spec in specs}
    network_root = ET.parse(bundle / "network.net.xml").getroot()
    physical_pairs = {(c.get("from"), c.get("to")) for c in network_root.findall("connection")
                      if c.get("from") in observed_approaches and not c.get("to", "").startswith(":" )}
    physical_specs = [{"id": "physical_" + a + "_" + b, "source_from": a,
                       "source_to": b, "path": [a, b]} for a, b in sorted(physical_pairs)]
    registered_direct = {tuple(spec["path"]) for spec in specs if spec.get("path") and len(spec["path"]) == 2}
    all_events, within_events, entered_ids, finished_ids, seen_ids = 0, 0, set(), set(), set()
    for _, vehicle in ET.iterparse(run / "vehroute.xml", events=("end",)):
        if vehicle.tag != "vehicle":
            continue
        identifier = vehicle.get("id")
        if identifier in seen_ids or identifier not in intended:
            raise ValueError(f"Duplicate or unknown simulation vehicle ID: {identifier}")
        seen_ids.add(identifier)
        if float(vehicle.get("depart", "-1")) >= 0:
            entered_ids.add(identifier)
        if float(vehicle.get("arrival", "-1")) >= 0:
            finished_ids.add(identifier)
        for event in movement_completion_events(vehicle, specs):
            all_events += 1
            when = event["completion_s"]
            if 0 <= when < manifest["duration_s"]:
                within_events += 1
                key = (int(when // 900) * 900, event["vehicle_type"], event["source_from"], event["source_to"])
                completed[key] += 1
        for event in movement_completion_events(vehicle, physical_specs):
            when = event["completion_s"]
            if manifest["evaluation_start_s"] <= when < manifest["duration_s"]:
                raw_transitions[(event["vehicle_type"], event["source_from"], event["source_to"])] += 1
        vehicle.clear()
    rows = []
    for key, observed in sorted(expected.items()):
        window, kind, a, b = key
        if window < manifest["evaluation_start_s"]:
            continue
        actual = completed[key]
        observed_rate, actual_rate = observed * 4, actual * 4
        geh = math.sqrt(2 * (actual_rate - observed_rate) ** 2 / (actual_rate + observed_rate)) if actual_rate + observed_rate else 0
        rows.append({"simulation_begin": window, "vehicle_type": kind, "source_from": a, "source_to": b,
                     "source_count": observed, "actual_completed": actual,
                     "under": max(observed - actual, 0), "over": max(actual - observed, 0),
                     "GEH_hourly_rate": geh, "provenance": provenance[key]})
    health = ET.parse(run / "stats.xml").getroot()
    safety = health.find("safety")
    teleports = health.find("teleports")
    vehicles_stats = health.find("vehicles")
    safety_values = dict(safety.attrib) if safety is not None else {}
    teleport_values = dict(teleports.attrib) if teleports is not None else {}
    health_gates = {"all_vehicles_inserted": entered_ids == set(intended),
                    "all_vehicles_arrived": finished_ids == set(intended),
                    "collisions_zero": safety is not None and int(safety.get("collisions", "-1")) == 0,
                    "teleports_zero": teleports is not None and int(teleports.get("total", "-1")) == 0,
                    "emergency_stops_zero": safety is not None and int(safety.get("emergencyStops", "-1")) == 0,
                    "emergency_braking_zero": safety is not None and int(safety.get("emergencyBraking", "-1")) == 0}
    health_gates["junction_collision_checks_enabled"] = any(
        command[i:i + 2] == ["--collision.check-junctions", "true"] for i in range(len(command)))
    health_gates["collision_teleport_disabled"] = any(
        command[i:i + 2] == ["--collision.action", "warn"] for i in range(len(command)))
    health_gates["xml_matches_SUMO_inserted_count"] = (vehicles_stats is not None and
        int(vehicles_stats.get("inserted", "-1")) == len(entered_ids))
    insertion_delay, observed_time_loss, observed_waiting = 0.0, 0.0, 0.0
    trip_ids = set()
    for _, trip in ET.iterparse(run / "tripinfo.xml", events=("end",)):
        if trip.tag != "tripinfo":
            continue
        identifier = trip.get("id")
        if identifier in trip_ids or identifier not in intended:
            raise ValueError("Duplicate or unknown tripinfo vehicle")
        trip_ids.add(identifier)
        insertion_delay += max(0.0, float(trip.get("departDelay", "0")))
        observed_time_loss += max(0.0, float(trip.get("timeLoss", "0")))
        observed_waiting += max(0.0, float(trip.get("waitingTime", "0")))
        trip.clear()
    health_gates["tripinfo_matches_inserted_vehicle_ids"] = trip_ids == entered_ids
    primary = [r for r in rows if r["provenance"] == "collection_site"]
    geh_share = sum(r["GEH_hourly_rate"] < 5 for r in primary) / len(primary) if primary else 0
    source_total = sum(r["source_count"] for r in primary)
    residual = sum(r["under"] + r["over"] for r in primary) / source_total if source_total else 1
    quality_gates = {"collection_site_GEH_lt5_share_at_least_95pct": geh_share >= .95,
                     "collection_site_total_absolute_residual_at_most_5pct": residual <= .05}
    gates = {**manifest["build_gates"], **health_gates, **quality_gates}
    gates["actual_passages_validated"] = all(health_gates.values()) and all(quality_gates.values())
    result = {"version": VERSION, "count_scope": "actual entry into final external edge of complete semantic corridor",
              "intended_vehicles": len(intended), "entered_vehicles": len(entered_ids),
              "arrived_vehicles": len(finished_ids), "never_inserted": len(set(intended) - entered_ids),
              "unfinished_or_uninserted": len(set(intended) - finished_ids),
              "completed_movement_events_all_times": all_events, "completed_events_before_demand_end": within_events,
              "safety": safety_values, "teleports": teleport_values,
              "sumo_vehicles": dict(vehicles_stats.attrib) if vehicles_stats is not None else {},
              "collection_site_GEH_lt5_share": geh_share, "collection_site_absolute_residual_fraction": residual,
              "by_movement_and_period": rows,
              "unobserved_completed": sum(value for key, value in completed.items() if key not in expected and
                                             manifest["evaluation_start_s"] <= key[0] < manifest["duration_s"]),
              "physical_transitions_not_in_registered_direct_movements": [
                  {"vehicle_type": kind, "from": a, "to": b, "count": count}
                  for (kind, a, b), count in sorted(raw_transitions.items()) if (a, b) not in registered_direct],
              "unknown_transition_scope": "Physical approach-pair events; do not add to semantic full-path events.",
              "source_turn_count_independently_recomputed": independent_source_total,
              "insertion_delay_s": insertion_delay, "observed_time_loss_s": observed_time_loss,
              "observed_waiting_s": observed_waiting,
              "delay_scope": "Observed so far for inserted trips; unfinished trips are censored, not final performance.",
              "gates": gates, "learning_ready": all(gates.values()),
              "acceptance_scope": "Project engineering checks; not independent field delay validation."}
    write_json(run / "validation.json", result)
    return result


def simulate(bundle: Path, run_name: str = "fixed_diagnostic"):
    from rl_environment import configure_sumo
    configure_sumo()
    import sumolib
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest.get("version") != VERSION:
        raise ValueError("Older bundle version cannot be reused; prepare a new result bundle")
    if manifest.get("implementation_fingerprints") != implementation_fingerprints():
        raise ValueError("Implementation changed since preparation; build a new bundle")
    for name, fingerprint in manifest["input_fingerprints"].items():
        if digest(bundle / name) != fingerprint:
            raise ValueError(f"Input changed since preparation: {name}; rebuild in a new bundle")
    run = bundle / run_name
    if run.exists():
        raise FileExistsError(f"Existing run preserved: {run}")
    run.mkdir()
    command = [sumolib.checkBinary("sumo"), "-n", str(bundle / "network.net.xml"),
               "-r", str(bundle / "traffic.rou.xml"), "-a", str(bundle / "signals.add.xml"),
               "--seed", str(manifest["seed"]), "--end", str(manifest["duration_s"] + manifest["tail_s"]),
               "--time-to-teleport", "-1", "--no-step-log", "--duration-log.statistics", "true",
               "--collision.check-junctions", "true", "--collision.action", "warn",
               "--collision-output", str(run / "collisions.xml"),
               "--tripinfo-output", str(run / "tripinfo.xml"), "--tripinfo-output.write-unfinished", "true",
               "--vehroute-output", str(run / "vehroute.xml"), "--vehroute-output.exit-times", "true",
               "--vehroute-output.internal", "true", "--vehroute-output.write-unfinished", "true",
               "--statistic-output", str(run / "stats.xml")]
    execution = {"command": command, "input_fingerprints": manifest["input_fingerprints"],
                 "manifest_sha256": digest(bundle / "manifest.json")}
    write_json(run / "execution.json", execution)
    with (run / "sumo.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    execution["output_fingerprints"] = {name: digest(run / name) for name in SIMULATION_OUTPUTS}
    write_json(run / "execution.json", execution)
    result = inspect_run(bundle, run)
    print(json.dumps({k: result[k] for k in ("intended_vehicles", "arrived_vehicles", "never_inserted",
                                            "collection_site_GEH_lt5_share", "learning_ready")}, ensure_ascii=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("build")
    prepare.add_argument("--date", default="20220811")
    prepare.add_argument("--start", default="0500")
    prepare.add_argument("--minutes", type=int, default=60)
    prepare.add_argument("--warmup", type=int, default=30)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--variant", choices=["conservative", "uturn_as_straight",
                         "central_assumed_embedded", "central_assumed_embedded_uturn_as_straight"], default="conservative")
    prepare.add_argument("--signal-variant", choices=["embedded", "reconstructed", "reconstructed_mixed"], default="embedded")
    prepare.add_argument("--driver-model", type=Path)
    prepare.add_argument("--boundary-mode", choices=["graph_terminal", "terminal_turnaround"], default="terminal_turnaround")
    prepare.add_argument("--candidates-per-movement", type=int, default=64)
    prepare.add_argument("--demand-mode", choices=["joint_freeflow", "same_window_diagnostic"],
                         default="joint_freeflow")
    prepare.add_argument("--geometry-retreat", type=float,
                         help="Explicit waiting-position hypothesis in metres; not historical geometry certification")
    prepare.add_argument("--join-priority-diagnostic", action="store_true",
                         help="Explicit regional join of audited 204922/605257; never certifies historical geometry")
    prepare.add_argument("--assumed-signal-columns", action="store_true")
    prepare.add_argument("--repair-yellow", action="store_true")
    prepare.add_argument("--phase-shift", action="append", default=[], metavar="TLS:SHIFT")
    prepare.add_argument("--allow-priority-conflicts-diagnostic", action="store_true",
                         help="Preserve known source signal conflicts for diagnosis; readiness remains false")
    prepare.add_argument("--fit-time-limit", type=float, default=10)
    prepare.add_argument("--seed", type=int, default=42)
    run = sub.add_parser("simulate")
    run.add_argument("--bundle", type=Path, required=True)
    run.add_argument("--run-name", default="fixed_diagnostic")
    args = parser.parse_args()
    if args.command == "build":
        build(args.date, args.start, args.minutes, args.warmup, args.out.resolve(), args.variant,
              args.fit_time_limit, args.seed, args.signal_variant, args.driver_model,
              args.boundary_mode, args.candidates_per_movement, args.assumed_signal_columns,
              args.repair_yellow, {key: int(value) for key, value in
                                  (item.split(":", 1) for item in args.phase_shift)},
              args.allow_priority_conflicts_diagnostic, args.demand_mode, args.geometry_retreat,
              args.join_priority_diagnostic)
    else:
        simulate(args.bundle.resolve(), args.run_name)


if __name__ == "__main__":
    main()
