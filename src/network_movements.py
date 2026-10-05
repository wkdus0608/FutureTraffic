"""Traceable raw-to-semantic movements for the complete supplied network.

Raw count keys never change. A semantic path is counted only as a complete,
contiguous sequence. Map connectivity is a model fact; remapping an observation
onto a bypass or another edge ID is explicitly an assumption. Unknown movements
remain in the raw denominator. No resolver edits a network or source count file.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT / "config/network_movements.json"


class MovementResolutionError(ValueError):
    """A proposed movement cannot be represented safely in the supplied map."""


@dataclass
class NetworkTopology:
    path: Path
    root: ET.Element
    edges: dict[str, ET.Element]
    connections: set[tuple[str, str]]

    @classmethod
    def read(cls, network_file: Path | str) -> "NetworkTopology":
        path = Path(network_file)
        root = ET.parse(path).getroot()
        edges = {e.get("id"): e for e in root.findall("edge") if e.get("function") != "internal"}
        connections = {(c.get("from"), c.get("to")) for c in root.findall("connection")
                       if c.get("from") in edges and c.get("to") in edges}
        return cls(path, root, edges, connections)

    def validate_path(self, path: list[str]) -> None:
        if len(path) < 2 or any(e not in self.edges for e in path):
            raise MovementResolutionError(f"Unknown or incomplete movement path: {path}")
        broken = [(a, b) for a, b in zip(path, path[1:]) if (a, b) not in self.connections]
        if broken:
            raise MovementResolutionError(f"Movement path has missing connections: {broken}")


def load_resolution_config(config_path: Path | str | dict | None = None) -> dict:
    config = deepcopy(config_path) if isinstance(config_path, dict) else json.loads(
        Path(config_path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise MovementResolutionError("Unsupported movement configuration version")
    seen = set()
    for row in config["overrides"]:
        key = row["source_from"], row["source_to"]
        if key in seen:
            raise MovementResolutionError(f"Duplicate movement override: {key}")
        seen.add(key)
    return config


def movement_id(source_from: str, source_to: str) -> str:
    # Deterministic across variants; the ID refers to the raw observed movement.
    payload = json.dumps([source_from, source_to], separators=(",", ":"), ensure_ascii=False)
    return "movement_" + hashlib.sha256(payload.encode()).hexdigest()[:20]


def resolve_movement(source_from: str, source_to: str, network_file: Path | str | NetworkTopology,
                     config: dict | Path | str | None = None, variant: str = "conservative") -> dict:
    topology = network_file if isinstance(network_file, NetworkTopology) else NetworkTopology.read(network_file)
    settings = load_resolution_config(config)
    if variant not in settings["variants"]:
        raise MovementResolutionError(f"Unknown movement variant: {variant}")
    key = source_from, source_to
    override = next((x for x in settings["overrides"] if (x["source_from"], x["source_to"]) == key), None)
    result = {"id": movement_id(*key), "source_from": source_from, "source_to": source_to,
              "variant": variant, "path": None, "resolution": "unresolved", "status": "on_hold",
              "evidence_level": "provided", "evidence": "No connected representation is verified",
              "source_refs": [], "raw_denominator_preserved": True,
              "variant_certification_blocked": not settings["variants"][variant].get("learning_ready", False)}
    if override is None:
        if key in topology.connections:
            result.update(path=list(key), resolution="direct", status="model_fact",
                          evidence="Raw IDs are directly connected in the supplied model")
        return result
    result.update({k: deepcopy(v) for k, v in override.items() if k not in {"variants"}})
    # The original pair and stable ID cannot be redefined by config metadata.
    result.update(id=movement_id(*key), source_from=source_from, source_to=source_to, variant=variant)
    if variant in override.get("variants", {}):
        result.update(deepcopy(override["variants"][variant]))
    if result["path"] is not None:
        try:
            topology.validate_path(result["path"])
        except MovementResolutionError as exc:
            # An assumption variant requiring a patched connection must not work
            # on the original map by silently generating the nonexistent turn.
            if result.get("requires_network_patch"):
                result.update(path=None, resolution="unresolved", status="on_hold", evidence=str(exc))
            else:
                raise
    if result["resolution"] == "unresolved" and result["path"] is not None:
        raise MovementResolutionError("An unresolved movement cannot have an active path")
    if result["resolution"] != "unresolved" and result["path"] is None:
        raise MovementResolutionError("An active movement must have a verified connected path")
    result["raw_denominator_preserved"] = True
    # A complete hypothesis path is not itself a research-ready environment.
    # Keep the config's unresolved historical provenance gate visible even when
    # a combined sensitivity variant can represent every raw count mechanically.
    result["variant_certification_blocked"] = not settings["variants"][variant].get("learning_ready", False)
    return result


def load_movement_specs(network_file: Path | str | NetworkTopology, source_pairs=None,
                        config_path: Path | str | dict | None = None, variant: str = "conservative") -> list[dict]:
    topology = network_file if isinstance(network_file, NetworkTopology) else NetworkTopology.read(network_file)
    config = load_resolution_config(config_path)
    if source_pairs is None:
        source_pairs = [(x["source_from"], x["source_to"]) for x in config["inventory"]]
    keys = []
    for row in source_pairs:
        if isinstance(row, dict):
            key = (row.get("source_from", row.get("from_edge", row.get("from"))),
                   row.get("source_to", row.get("to_edge", row.get("to"))))
        else:
            key = tuple(row)
        if len(key) != 2 or not all(isinstance(e, str) and e for e in key):
            raise MovementResolutionError(f"Invalid source movement key: {row}")
        keys.append(key)
    return [resolve_movement(a, b, topology, config, variant) for a, b in sorted(set(keys))]


def count_complete_path(route_edges: list[str] | tuple[str, ...] | str, path: list[str] | None) -> int:
    if path is None:
        return 0
    if len(path) < 2:
        raise MovementResolutionError("A semantic movement must include entry and exit")
    sequence = route_edges.split() if isinstance(route_edges, str) else list(route_edges)
    width = len(path)
    return sum(sequence[i:i + width] == path for i in range(len(sequence) - width + 1))


def unresolved_gate(specs: list[dict], raw_counts: dict[tuple[str, str], int] | None = None) -> dict:
    unresolved = [deepcopy(s) for s in specs if s["resolution"] == "unresolved"]
    counts = raw_counts or {}
    defined = {(s["source_from"], s["source_to"]) for s in specs}
    for key,n in counts.items():
        if n < 0 or int(n) != n:
            raise MovementResolutionError(f"Invalid raw count: {key}={n}")
        if n and key not in defined:
            unresolved.append({"id":movement_id(*key),"source_from":key[0],"source_to":key[1],
                "path":None,"resolution":"unresolved","status":"on_hold","evidence_level":"provided",
                "evidence":"Raw count has no movement specification","raw_denominator_preserved":True})
    total = sum(counts.values())
    held = sum(counts.get((s["source_from"], s["source_to"]), 0) for s in unresolved)
    certification_blocked = any(s.get("variant_certification_blocked", False) for s in specs)
    return {"learning_ready": not unresolved and not certification_blocked,
            "movement_coverage_complete": not unresolved,
            "blocked_by_variant_certification": certification_blocked, "unresolved_movements": unresolved,
            "unresolved_movement_count": len(unresolved), "unresolved_turn_count": held,
            "raw_total_turn_count": total, "represented_target_turn_count": total - held,
            "raw_denominator_preserved": True,
            "interpretation": "A diagnostic subset can run; unresolved observations remain unrepresented, not deleted or corrected."}


def audit_inventory(network_file: Path | str, routes_file: Path | str,
                    config_path: Path | str | dict | None = None) -> dict:
    """Reproduce raw source/map/candidate gaps independently of any sampler."""
    topology = NetworkTopology.read(network_file)
    settings = load_resolution_config(config_path)
    coverage = {p for r in ET.parse(routes_file).iter("route")
                for p in zip(r.get("edges").split(), r.get("edges").split()[1:])}
    categories = {name: {"pair_count": 0, "sample_turn_count": 0, "all_19_day_turn_count": 0}
                  for name in ("covered", "missing_edge_id", "no_direct_connection", "no_candidate")}
    inventory = []
    for row in settings["inventory"]:
        a, b = row["source_from"], row["source_to"]
        category = "missing_edge_id" if a not in topology.edges or b not in topology.edges else (
            "no_direct_connection" if (a, b) not in topology.connections else (
                "no_candidate" if (a, b) not in coverage else "covered"))
        summary = categories[category]
        summary["pair_count"] += 1
        summary["sample_turn_count"] += row["sample_20220810_0500_count"]
        summary["all_19_day_turn_count"] += row["all_19_day_count"]
        inventory.append({**row, "category": category})
    return {"categories": categories, "inventory": inventory,
            "unique_positive_pairs": len(inventory),
            "raw_sample_turn_count": sum(x["sample_20220810_0500_count"] for x in inventory),
            "raw_all_19_day_turn_count": sum(x["all_19_day_count"] for x in inventory),
            "raw_sample_gap": sum(v["sample_turn_count"] for k,v in categories.items() if k != "covered")}


def build_central_assumption_variant(network_file: Path | str, output_dir: Path | str,
                                     config_path: Path | str | dict | None = None,
                                     netconvert_binary: Path | str | None = None) -> dict:
    """Rebuild a *hypothetical* 201306 map using netconvert, never raw XML edits.

    The right turn uses source lane 0 and downstream lane 0; the left turn uses
    source lane 1 and downstream lane 1. Their permissive greens follow the same
    approach's embedded through phase. Historical source-column identity remains
    unknown: reconstructed signal replay and learning certification stay blocked.
    The conservative configuration does not activate this separate variant.
    Only netconvert's complete rebuilt central subgraph is imported into a copy
    of the original. This avoids unrelated junction trimming by newer compilers.
    """
    from audit_network_structure import audit
    settings = load_resolution_config(config_path)
    source = Path(network_file).resolve()
    output = Path(output_dir).resolve()
    approved_root = (PROJECT / "results/network_environment_20261005").resolve()
    if not output.is_relative_to(approved_root):
        raise MovementResolutionError(f"Write map variants only below {approved_root}")
    if output.exists():
        raise FileExistsError(f"Refusing to replace an existing variant: {output}")
    topology = NetworkTopology.read(source)
    originals = [c for c in topology.root.findall("connection") if c.get("tl") == "201306"]
    logic = next(t for t in topology.root.findall("tlLogic") if t.get("id") == "201306")
    if {len(p.get("state")) for p in logic.findall("phase")} != {15}:
        raise MovementResolutionError("Expected the audited original 15-column 201306 program")
    additions = [x["required_patch"] for x in settings["overrides"] if x.get("required_patch")]
    if len(additions) != 2:
        raise MovementResolutionError("Exactly the two reviewed central movements are allowed")
    def xml_root(tag, schema):
        return ET.Element(tag, {"xmlns:xsi":"http://www.w3.org/2001/XMLSchema-instance",
                               "xsi:noNamespaceSchemaLocation":"http://sumo.dlr.de/xsd/"+schema})
    con_root = xml_root("connections", "connections_file.xsd")
    for c in originals:
        ET.SubElement(con_root, "connection", {k: c.get(k) for k in ("from", "to", "fromLane", "toLane")})
    for addition in additions:
        a,b = addition["from"], addition["to"]
        if (a,b) in topology.connections or topology.edges[a].get("to") != "201306" or topology.edges[b].get("from") != "201306":
            raise MovementResolutionError("A proposed central connection does not match the audited missing movement")
        for edge,index in ((a,addition["fromLane"]),(b,addition["toLane"])):
            if not any(l.get("index")==str(index) for l in topology.edges[edge].findall("lane")):
                raise MovementResolutionError(f"Unknown proposed connection lane: {edge}:{index}")
        ET.SubElement(con_root, "connection", {"from":a,"to":b,"fromLane":str(addition["fromLane"]),
            "toLane":str(addition["toLane"])})
    # TLS link-index assignments belong to plain tllogic input, not .con.xml.
    # Keeping these separate also lets XML validation reject invalid patch data.
    tl_root=xml_root("tlLogics", "tllogic_file.xsd")
    copy_logic=deepcopy(logic)
    for phase in copy_logic.findall("phase"):
        old_state=phase.get("state")
        extra="".join("g" if old_state[a["embedded_through_linkIndex"]] in "Gg" else
                      "y" if old_state[a["embedded_through_linkIndex"]] in "yY" else "r" for a in additions)
        phase.set("state", old_state+extra)
    tl_root.append(copy_logic)
    for c in originals:
        ET.SubElement(tl_root,"connection",{k:c.get(k) for k in ("from","to","fromLane","toLane","tl","linkIndex")})
    for addition in additions:
        ET.SubElement(tl_root,"connection",{"from":addition["from"],"to":addition["to"],
            "fromLane":str(addition["fromLane"]),"toLane":str(addition["toLane"]),"tl":"201306",
            "linkIndex":str(addition["variant_linkIndex"])})
    before_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    output.mkdir(parents=True)
    connection_file,signal_file,target=output/"central.con.xml",output/"central_embedded_assumption.tll.xml",output/"network.net.xml"
    compiled_target=output/"netconvert_rebuilt_all.net.xml"
    for path,root in ((connection_file,con_root),(signal_file,tl_root)):
        ET.indent(root,space="  ");ET.ElementTree(root).write(path,encoding="utf-8",xml_declaration=True)
    if netconvert_binary is None:
        import sumolib
        netconvert_binary=sumolib.checkBinary("netconvert")
    command=[str(netconvert_binary),"--sumo-net-file",str(source),"--connection-files",str(connection_file),
             "--tllogic-files",str(signal_file),"--output-file",str(compiled_target),"--no-turnarounds","true",
             "--offset.disable-normalization","true","--tls.rebuild","false","--xml-validation","always"]
    run=subprocess.run(command,capture_output=True,text=True)
    (output/"netconvert.log").write_text(run.stdout+run.stderr)
    if run.returncode:
        raise MovementResolutionError(f"netconvert rejected the assumption variant; see {output/'netconvert.log'}")
    compiled=NetworkTopology.read(compiled_target)
    central_edges={k for k,e in topology.edges.items() if e.get("from")=="201306" or e.get("to")=="201306"}
    def central_element(element):
        identifier=element.get("id", "")
        if element.tag=="edge":
            return identifier.startswith(":201306_") or identifier in central_edges
        if element.tag=="junction":
            return identifier=="201306" or identifier.startswith(":201306_")
        if element.tag=="tlLogic":
            return identifier=="201306"
        if element.tag=="connection":
            return element.get("tl")=="201306" or element.get("from", "").startswith(":201306_")
        return False
    # Compiler-produced connections, internal lanes and request matrices travel
    # together. Never append guessed connection records to the original net.
    # All other elements (including outside lane geometry) remain source copies.
    merged=ET.Element(topology.root.tag,deepcopy(topology.root.attrib))
    for tag in ("location","type","edge","tlLogic","junction","connection","roundabout"):
        elements=[deepcopy(e) for e in topology.root.findall(tag) if not central_element(e)]
        elements.extend(deepcopy(e) for e in compiled.root.findall(tag) if central_element(e))
        if tag=="edge":elements.sort(key=lambda e:(e.get("function")!="internal",e.get("id")))
        merged.extend(elements)
    unsupported_tags={e.tag for e in topology.root}-{e.tag for e in merged}
    if unsupported_tags:
        raise MovementResolutionError(f"Regional map merge would drop source elements: {unsupported_tags}")
    ET.indent(merged,space="    ")
    ET.ElementTree(merged).write(target,encoding="utf-8",xml_declaration=True)
    rebuilt=NetworkTopology.read(target)
    expected={ (a["from"],a["to"]) for a in additions }
    added=rebuilt.connections-topology.connections;removed=topology.connections-rebuilt.connections
    def controller_signature(root,identifier):
        t=next(t for t in root.findall("tlLogic") if t.get("id")==identifier)
        return (t.attrib,[(float(p.get("duration")),p.get("state")) for p in t.findall("phase")],
                sorted((c.get("from"),c.get("to"),c.get("fromLane"),c.get("toLane"),c.get("linkIndex"))
                       for c in root.findall("connection") if c.get("tl")==identifier))
    changed_controllers=[t.get("id") for t in topology.root.findall("tlLogic") if t.get("id")!="201306"
        and controller_signature(topology.root,t.get("id"))!=controller_signature(rebuilt.root,t.get("id"))]
    def external_geometry(topology,exclude_central=False):
        return {k:[dict(l.attrib) for l in e.findall("lane")] for k,e in topology.edges.items()
                if not exclude_central or k not in central_edges}
    outside_geometry_unchanged=external_geometry(topology,True)==external_geometry(rebuilt,True)
    central_existing_indexes_unchanged=all(
        any(all(c.get(k)==old.get(k) for k in ("from","to","fromLane","toLane","tl","linkIndex"))
            for c in rebuilt.root.findall("connection")) for old in originals)
    schema=Path(netconvert_binary).parent.parent/"data/xsd/net_file.xsd"
    schema_check=subprocess.run(["/usr/bin/xmllint","--nonet","--noout","--schema",str(schema),str(target)],
                                capture_output=True,text=True)
    (output/"schema_validation.log").write_text(schema_check.stdout+schema_check.stderr)
    structure=audit(target)
    original_unchanged=hashlib.sha256(source.read_bytes()).hexdigest()==before_hash
    safe=(added==expected and not removed and not changed_controllers and original_unchanged
          and outside_geometry_unchanged and central_existing_indexes_unchanged
          and schema_check.returncode==0 and not structure["structural_errors"])
    manifest={"variant":"central_assumed_embedded","network_file":str(target),"source_network_file":str(source),
        "source_sha256":before_hash,"source_unchanged":original_unchanged,"added_pairs":sorted(added),
        "removed_pairs":sorted(removed),"other_changed_controllers":changed_controllers,
        "outside_central_lane_geometry_unchanged":outside_geometry_unchanged,
        "central_existing_signal_indexes_unchanged":central_existing_indexes_unchanged,
        "schema_validation_returncode":schema_check.returncode,
        "structural_errors":structure["structural_errors"],"map_variant_valid":safe,
        "review_findings":structure["review_findings"],
        "compiler_output_file":str(compiled_target),
        "assembly":"Only netconvert's entire 201306 subgraph replaces corresponding source elements; all outside elements are preserved.",
        "assumptions":additions,"evidence_level":"assumption","historical_source_column_mapping":None,
        "reconstructed_signal_ready":False,"learning_ready":False,
        "blocked_reason":"Historical signal columns are unverified; 201293 U-turn remains unresolved.",
        "netconvert_command":command}
    (output/"patch_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    if not safe:
        raise MovementResolutionError(f"Variant failed preservation/structure gates: {output/'patch_manifest.json'}")
    return manifest
