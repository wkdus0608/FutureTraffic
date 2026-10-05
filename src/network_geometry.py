"""Explicit, isolated hypotheses for internal vehicle waiting positions.

This module never changes source maps, demand, signals or driver settings. The
three reviewed bus-body overlaps motivate a conservative geometric hypothesis:
retreat an internal waiting point by one metre along its existing corridor.
These are not surveyed historical stop positions and do not certify road safety.
netconvert builds the complete affected internal topology and request matrices;
source corridor geometry is then restored to avoid its unrelated boundary trims.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

PROJECT = Path(__file__).resolve().parents[1]
TARGETS = (
    ("203019", "477649508_0202_01", "477655738_01", "2", "0"),
    ("201305", "477605619_02", "477605619_01", "5", "5"),
    ("201330", "477605419_02", "477657424_01", "4", "1"),
)
KEY = ("from", "to", "fromLane", "toLane")


class GeometryVariantError(ValueError):
    """A diagnostic map failed identification or preservation requirements."""


class _Map:
    def __init__(self, path):
        self.root = ET.parse(path).getroot()
        self.edges = {e.get("id"): e for e in self.root.findall("edge")}
        self.lanes = {l.get("id"): l for e in self.edges.values() for l in e.findall("lane")}
        self.lane_refs = {l.get("id"): (e.get("id"), str(i))
                          for e in self.edges.values() for i, l in enumerate(e.findall("lane"))}
        self.connections = self.root.findall("connection")
        self.successors = {(c.get("from"), c.get("fromLane")): c
                           for c in self.connections if c.get("from", "").startswith(":")}
        self.external = {tuple(c.get(k) for k in KEY): c for c in self.connections
                         if not c.get("from", "").startswith(":")}

    def trace(self, connection):
        via, answer = connection.get("via"), []
        while via:
            if via in answer or via not in self.lanes:
                raise GeometryVariantError(f"Unknown/cyclic internal lane: {via}")
            answer.append(via)
            next_connection = self.successors.get(self.lane_refs[via])
            if next_connection is None:
                raise GeometryVariantError(f"Missing internal successor: {via}")
            via = next_connection.get("via")
        if not answer:
            raise GeometryVariantError("Expected an existing internal corridor")
        return answer


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _points(shape):
    return [tuple(float(v) for v in p.split(",")) for p in shape.split()]


def _shape(points):
    return " ".join(",".join(f"{v:.10f}".rstrip("0").rstrip(".") for v in p) for p in points)


def _joined(model, trace):
    answer = []
    for identifier in trace:
        points = _points(model.lanes[identifier].get("shape"))
        if answer and math.dist(answer[-1], points[0]) > 1e-7:
            raise GeometryVariantError(f"Disconnected source corridor: {identifier}")
        answer.extend(points[1:] if answer else points)
    return answer


def _split(points, fraction):
    """Split without changing the source polyline, including curved first stages."""
    lengths = [math.dist(a, b) for a, b in zip(points, points[1:])]
    remaining = sum(lengths) * fraction
    for i, length in enumerate(lengths):
        if remaining <= length + 1e-10:
            share = min(1.0, max(0.0, remaining / length))
            q = tuple(a + share * (b - a) for a, b in zip(points[i], points[i + 1]))
            return points[:i + 1] + [q], [q] + points[i + 1:]
        remaining -= length
    raise GeometryVariantError("Invalid geometric split")


def inventory_waiting_position_targets(network_file, retreat_m=1.0):
    """Identify the three exact reviewed lane pairs in either original/central map.

    SUMO's contPos=0 removes internal waiting rather than creating a zero-length
    stage. Such clamped requests are reported explicitly; they are never silently
    described as a moved internal waiting point.
    """
    if not math.isfinite(retreat_m) or retreat_m <= 0:
        raise GeometryVariantError("retreat_m must be finite and positive")
    model, answer = _Map(network_file), []
    for junction, a, b, i, k in TARGETS:
        pair = (a, b, i, k)
        connection = model.external.get(pair)
        if connection is None or model.edges[a].get("to") != junction:
            raise GeometryVariantError(f"Reviewed connection missing: {pair}")
        trace = model.trace(connection)
        if len(trace) != 2:
            raise GeometryVariantError(f"Expected two source stages for {pair}, got {trace}")
        node = model.root.find(f"junction[@id='{junction}']")
        request_index = node.get("intLanes").split().index(trace[-1])
        request = node.find(f"request[@index='{request_index}']")
        if request is None or request.get("cont") != "1":
            raise GeometryVariantError(f"Reviewed internal waiting permission missing: {pair}")
        first_length = float(model.lanes[trace[0]].get("length"))
        new_position = round(max(0.0, first_length - retreat_m), 12)
        answer.append({"junction": junction, "from": a, "to": b, "fromLane": i, "toLane": k,
                       "via": trace[0], "trace_before": trace, "request_index_before": request_index,
                       "tl": connection.get("tl"), "linkIndex": connection.get("linkIndex"),
                       "contPos_before_m": first_length, "contPos_requested_m": new_position,
                       "internal_wait_removed": new_position == 0,
                       "interpretation": "conservative_geometry_hypothesis"})
    return answer


def _xml(tag, schema):
    return ET.Element(tag, {"xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
                           "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/" + schema})


def _write(root, path):
    ET.indent(root, space="    ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _canonical(element):
    return (element.tag, tuple(sorted(element.attrib.items())),
            tuple(_canonical(child) for child in element))


def _path_length(points):
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


def _same_polyline(original, actual):
    def distance(p, a, b):
        vector = tuple(y - x for x, y in zip(a, b))
        norm = sum(x * x for x in vector)
        share = max(0.0, min(1.0, sum((x-y)*z for x,y,z in zip(p,a,vector))/norm)) if norm else 0
        return math.dist(p, tuple(x + share*y for x,y in zip(a,vector)))
    return (abs(_path_length(original) - _path_length(actual)) < 1e-7 and
            all(min(distance(p,a,b) for a,b in zip(original,original[1:])) < 1e-7 for p in actual) and
            all(min(distance(p,a,b) for a,b in zip(actual,actual[1:])) < 1e-7 for p in original))


def build_internal_waiting_position_variant(network_file, output_dir, retreat_m=1.0,
                                            netconvert_binary=None, preserve_requests=True):
    """Build a separate diagnostic map; preserve external geometry/all signals.

    netconvert may clip other legacy short corridors while recreating requests.
    We restore every original physical corridor and lane length, speed, width and
    permission, with only the reviewed internal split positions changed. A tiny
    compiler-only split is used if clipping would delete an existing non-target
    waiting node; its final physical position is restored to the source position.
    By default source foes/response/cont are retained after proving identical
    request-index-to-lane-pair correspondence. Internal waiting eligibility and
    internal connection semantics are held fixed as well, after topology checks.
    This deliberate counterfactual
    isolates waiting-position geometry; it does not certify that the source
    conflict model is correct. Set preserve_requests=False to retain the separate
    compiler-response hypothesis. Both versions remain diagnostic-only.
    """
    from audit_network_structure import audit
    import sumolib
    source, output = Path(network_file).resolve(), Path(output_dir).resolve()
    approved = (PROJECT / "results/network_environment_20261005").resolve()
    if not output.is_relative_to(approved):
        raise GeometryVariantError(f"Write diagnostic maps only below {approved}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    targets = inventory_waiting_position_targets(source, retreat_m)
    if preserve_requests and any(t["internal_wait_removed"] for t in targets):
        raise GeometryVariantError("contPos=0 removes internal waiting; source cont=1 cannot be preserved. "
                                   "Use preserve_requests=False only for an explicit removal diagnostic.")
    source_hash, original = _sha(source), _Map(source)
    junctions = {t["junction"] for t in targets}
    target_by_pair = {tuple(t[k] for k in KEY): t for t in targets}
    affected = {key: c for key,c in original.external.items()
                if original.edges[c.get("from")].get("to") in junctions}
    traces = {key: original.trace(c) for key,c in affected.items()}
    output.mkdir(parents=True)
    connections, edges, signals = _xml("connections", "connections_file.xsd"), _xml("edges", "edges_file.xsd"), _xml("tlLogics", "tllogic_file.xsd")
    patch_entries = {}
    for key, c in affected.items():
        trace = traces[key]
        attrs = {k: c.get(k) for k in KEY}
        attrs.update({"shape": _shape(_joined(original, trace)),
                      "contPos": str(target_by_pair[key]["contPos_requested_m"] if key in target_by_pair else
                                     float(original.lanes[trace[0]].get("length")) if len(trace) == 2 else 0),
                      "speed": original.lanes[trace[0]].get("speed")})
        for k in ("keepClear", "pass", "allow", "disallow", "visibility", "uncontrolled"):
            if c.get(k) is not None:
                attrs[k] = c.get(k)
        patch_entries[key] = ET.SubElement(connections, "connection", attrs)
    for e in original.edges.values():
        if e.get("function") == "internal" or not {e.get("from"),e.get("to")} & junctions:
            continue
        new = ET.SubElement(edges, "edge", {k:e.get(k) for k in ("id", "from", "to")})
        for i,lane in enumerate(e.findall("lane")):
            ET.SubElement(new, "lane", {"index":str(i), "shape":lane.get("shape")})
    for logic in original.root.findall("tlLogic"):
        signals.append(deepcopy(logic))
    for c in original.external.values():
        if c.get("tl"):
            ET.SubElement(signals,"connection",{k:c.get(k) for k in (*KEY,"tl","linkIndex")})
    con_file, edge_file, signal_file = output/"waiting.con.xml", output/"preserve.edg.xml", output/"preserve.tll.xml"
    _write(edges, edge_file); _write(signals, signal_file)
    binary = str(netconvert_binary or sumolib.checkBinary("netconvert"))
    compiled_path = output/"netconvert_rebuilt_all.net.xml"
    command = [binary,"--sumo-net-file",str(source),"--connection-files",str(con_file),
               "--edge-files",str(edge_file),"--tllogic-files",str(signal_file),"--output-file",str(compiled_path),
               "--offset.disable-normalization","true","--tls.rebuild","false","--xml-validation","always"]
    compiler_helpers = []
    for attempt in range(2):
        _write(connections,con_file)
        run = subprocess.run(command,capture_output=True,text=True)
        (output/f"netconvert_{attempt+1}.log").write_text(run.stdout+run.stderr)
        if run.returncode:
            raise GeometryVariantError(f"netconvert failed; see {output}")
        compiled = _Map(compiled_path)
        collapsed = [key for key in affected if key not in target_by_pair and
                     len(traces[key])==2 and len(compiled.trace(compiled.external[key]))!=2]
        if not collapsed:
            break
        if attempt:
            raise GeometryVariantError(f"Compiler could not preserve non-target waiting topology: {collapsed}")
        for key in collapsed:
            compiler_helpers.append({"connection":dict(zip(KEY,key)),"compiler_only_contPos_m":0.01,
                                     "final_contPos_m":float(original.lanes[traces[key][0]].get("length"))})
            patch_entries[key].set("contPos","0.01")
    # Restore each physical corridor within netconvert's complete topology. No
    # guessed connection, request, lane or waiting junction is appended manually.
    internal_nodes = {n.get("id"):n for n in compiled.root.findall("junction") if n.get("type")=="internal"}
    restored_lane_attrs = {}
    for key,c in affected.items():
        old_trace = traces[key]; new_connection = compiled.external[key]
        new_trace = compiled.trace(new_connection)
        old_lanes = [original.lanes[x] for x in old_trace]
        new_position = target_by_pair[key]["contPos_requested_m"] if key in target_by_pair else None
        expected_stages = 1 if new_position == 0 else len(old_trace)
        if len(new_trace)!=expected_stages:
            raise GeometryVariantError(f"Unexpected compiler stage count for {key}: {new_trace}")
        replacements = [dict(l.attrib) for l in old_lanes]
        if new_position == 0:
            replacements = [dict(old_lanes[0].attrib)]
            replacements[0].update(shape=_shape(_joined(original,old_trace)),
                                   length=str(sum(float(l.get("length")) for l in old_lanes)))
        elif new_position is not None:
            first, remainder = _split(_points(old_lanes[0].get("shape")),
                                      new_position/float(old_lanes[0].get("length")))
            rest = remainder + _points(old_lanes[1].get("shape"))[1:]
            replacements[0].update(shape=_shape(first),length=str(new_position))
            replacements[1].update(shape=_shape(rest),
                                   length=str(sum(float(l.get("length")) for l in old_lanes)-new_position))
        for identifier,attrs in zip(new_trace,replacements):
            lane = compiled.lanes[identifier]
            attrs.update(id=identifier,index=lane.get("index"))
            if identifier in restored_lane_attrs and restored_lane_attrs[identifier] != attrs:
                raise GeometryVariantError(f"Inconsistent shared internal lane restore: {identifier}")
            restored_lane_attrs[identifier]=attrs
            lane.attrib.clear();lane.attrib.update(attrs)
        if len(new_trace)==2:
            node=internal_nodes.get(new_trace[1])
            if node is None:
                raise GeometryVariantError(f"Compiler omitted waiting junction: {new_trace[1]}")
            point=_points(compiled.lanes[new_trace[0]].get("shape"))[-1]
            node.set("x",str(point[0]));node.set("y",str(point[1]))
        # Retain source external semantic attributes plus compiler-produced via.
        new_via=new_connection.get("via")
        new_connection.attrib.clear();new_connection.attrib.update(c.attrib)
        new_connection.set("via",new_via)
        if key in target_by_pair:
            new_connection.set("contPos",str(new_position))
    def request_lane_pairs(model, junction):
        node=model.root.find(f"junction[@id='{junction}']")
        int_lanes=node.get("intLanes").split()
        answer={}
        for key,c in model.external.items():
            if model.edges[c.get("from")].get("to")==junction:
                index=int_lanes.index(model.trace(c)[-1])
                if index in answer:
                    raise GeometryVariantError(f"Ambiguous request index {junction}:{index}")
                answer[index]=key
        if set(answer)!={int(q.get("index")) for q in node.findall("request")}:
            raise GeometryVariantError(f"Incomplete request correspondence at {junction}")
        return answer
    mapping_equal=all(request_lane_pairs(original,j)==request_lane_pairs(compiled,j) for j in junctions)
    if not mapping_equal:
        raise GeometryVariantError("Compiler changed request-index/lane-pair correspondence")
    compiler_request_changes=[]
    for j in sorted(junctions):
        old=original.root.find(f"junction[@id='{j}']");new=compiled.root.find(f"junction[@id='{j}']")
        before={q.get("index"):q.attrib for q in old.findall("request")}
        after={q.get("index"):q.attrib for q in new.findall("request")}
        compiler_request_changes.extend({"junction":j,"index":idx,"before":dict(before.get(idx,{})),
                                         "after":dict(after.get(idx,{}))}
                                        for idx in sorted(before.keys()|after.keys(),key=int)
                                        if before.get(idx)!=after.get(idx))
        # Main-junction polygon/position is not part of the waiting experiment.
        # Retain compiler internal-lane membership but preserve source geometry.
        compiled_int_lanes=new.get("intLanes")
        new.attrib.clear();new.attrib.update(old.attrib)
        new.set("intLanes",compiled_int_lanes)
        chosen_requests=old.findall("request") if preserve_requests else new.findall("request")
        requests_copy=[deepcopy(q) for q in chosen_requests]
        for child in list(new):
            new.remove(child)
        new.extend(requests_copy)
        new.extend(deepcopy(child) for child in old if child.tag!="request")
    original_internal_nodes={n.get("id"):n for n in original.root.findall("junction")
                             if n.get("type")=="internal" and
                             n.get("id","").startswith(tuple(":"+j+"_" for j in junctions))}
    compiler_internal_rule_changes=[]
    for identifier,old in original_internal_nodes.items():
        new=internal_nodes.get(identifier)
        if new is not None:
            differences={k:{"before":old.get(k),"after":new.get(k)} for k in ("incLanes","intLanes")
                         if old.get(k)!=new.get(k)}
            if differences:
                compiler_internal_rule_changes.append({"junction":identifier,"changes":differences})
    if preserve_requests:
        if any(traces[key]!=compiled.trace(compiled.external[key]) for key in affected):
            raise GeometryVariantError("Cannot hold internal waiting rules fixed with changed corridor topology")
        moved_nodes={traces[key][1] for key in target_by_pair}
        for identifier,old in original_internal_nodes.items():
            new=internal_nodes.get(identifier)
            if new is None:
                raise GeometryVariantError(f"Source internal waiting node disappeared: {identifier}")
            new_position={k:new.get(k) for k in ("x","y")}
            new.attrib.clear();new.attrib.update(old.attrib)
            if identifier in moved_nodes:
                new.attrib.update(new_position)
            for child in list(new):new.remove(child)
            new.extend(deepcopy(child) for child in old)
        original_internal_connections={tuple(c.get(k) for k in KEY):c for c in original.connections
                                       if c.get("from","").startswith(tuple(":"+j+"_" for j in junctions))}
        actual_internal_connections={tuple(c.get(k) for k in KEY):c for c in compiled.connections
                                     if c.get("from","").startswith(tuple(":"+j+"_" for j in junctions))}
        if original_internal_connections.keys()!=actual_internal_connections.keys():
            raise GeometryVariantError("Internal lane-to-lane topology changed")
        for key,c in original_internal_connections.items():
            new=actual_internal_connections[key]
            new.attrib.clear();new.attrib.update(c.attrib)
    def regional(e):
        if e.tag in ("edge","junction"):
            return any(e.get("id","").startswith(":"+j+"_") for j in junctions) or (e.tag=="junction" and e.get("id") in junctions)
        if e.tag=="connection":
            return e.get("from","").startswith(tuple(":"+j+"_" for j in junctions)) or tuple(e.get(k) for k in KEY) in affected
        return False
    merged=ET.Element(original.root.tag,dict(original.root.attrib))
    # Preserve arbitrary source top-level elements; only the reviewed region is
    # replaced. Keep SUMO's required grouping/order of element types.
    for tag in dict.fromkeys(e.tag for e in original.root):
        elements=[deepcopy(e) for e in original.root.findall(tag) if not regional(e)]
        elements.extend(deepcopy(e) for e in compiled.root.findall(tag) if regional(e))
        if tag=="edge":elements.sort(key=lambda e:(e.get("function")!="internal",e.get("id")))
        merged.extend(elements)
    target=output/"network.net.xml";_write(merged,target)
    result=_Map(target)
    outside_before=sorted((_canonical(e) for e in original.root if not regional(e)),key=repr)
    outside_after=sorted((_canonical(e) for e in result.root if not regional(e)),key=repr)
    signal_signature=lambda m:[_canonical(t) for t in m.root.findall("tlLogic")]
    assignments=lambda m:{key:(c.get("tl"),c.get("linkIndex")) for key,c in m.external.items()}
    corridor_results=[]
    for key,c in affected.items():
        old_trace=traces[key];new_trace=result.trace(result.external[key])
        corridor_results.append({"connection":dict(zip(KEY,key)),"trace_before":old_trace,"trace_after":new_trace,
            "physical_corridor_preserved":_same_polyline(_joined(original,old_trace),_joined(result,new_trace)),
            "total_lane_length_preserved":abs(sum(float(original.lanes[x].get("length")) for x in old_trace)-
                                               sum(float(result.lanes[x].get("length")) for x in new_trace))<1e-9})
    for t in targets:
        key=tuple(t[k] for k in KEY); trace=result.trace(result.external[key]);t["trace_after"]=trace
        t["first_lane_length_after_m"]=float(result.lanes[trace[0]].get("length"))
        if len(trace)==2:
            point=_points(result.lanes[trace[0]].get("shape"))[-1];node=result.root.find(f"junction[@id='{trace[1]}']")
            old_first=_points(original.lanes[traces[key][0]].get("shape"))
            t["waiting_point_before"]=list(old_first[-1])
            t["waiting_point_after"]=list(point)
            t["physical_retreat_along_source_shape_m"]=_path_length(old_first)-_path_length(
                _points(result.lanes[trace[0]].get("shape")))
            t["waiting_node_matches_split"]=math.dist(point,(float(node.get("x")),float(node.get("y"))))<1e-9
            t["requested_position_matches"]=abs(t["first_lane_length_after_m"]-t["contPos_requested_m"])<1e-9
        else:
            t["waiting_node_matches_split"]=True;t["requested_position_matches"]=t["internal_wait_removed"]
    request_changes=[]
    for j in sorted(junctions):
        old=original.root.find(f"junction[@id='{j}']");new=result.root.find(f"junction[@id='{j}']")
        before={q.get("index"):q.attrib for q in old.findall("request")};after={q.get("index"):q.attrib for q in new.findall("request")}
        request_changes.extend({"junction":j,"index":idx,"before":before.get(idx),"after":after.get(idx)}
                               for idx in sorted(before.keys()|after.keys(),key=int) if before.get(idx)!=after.get(idx))
    schema=Path(binary).resolve().parent.parent/"data/xsd/net_file.xsd"
    check=subprocess.run(["/usr/bin/xmllint","--nonet","--noout","--schema",str(schema),str(target)],capture_output=True,text=True)
    (output/"schema_validation.log").write_text(check.stdout+check.stderr)
    structure=audit(target)
    internal_rule_changes=[]
    for identifier,old in original_internal_nodes.items():
        new=result.root.find(f"junction[@id='{identifier}']")
        differences={k:{"before":old.get(k),"after":new.get(k) if new is not None else None}
                     for k in ("incLanes","intLanes") if new is None or old.get(k)!=new.get(k)}
        if differences:internal_rule_changes.append({"junction":identifier,"changes":differences})
    checks={"source_unchanged":_sha(source)==source_hash,"outside_region_unchanged":outside_before==outside_after,
            "external_lane_geometry_unchanged":all(_canonical(e)==_canonical(result.edges[k]) for k,e in original.edges.items() if e.get("function")!="internal"),
            "all_tls_programs_unchanged":signal_signature(original)==signal_signature(result),
            "external_lane_pairs_and_signal_indices_unchanged":assignments(original)==assignments(result),
            "all_corridor_geometries_preserved":all(x["physical_corridor_preserved"] for x in corridor_results),
            "all_corridor_lengths_preserved":all(x["total_lane_length_preserved"] for x in corridor_results),
            "request_index_lane_pair_mapping_unchanged":mapping_equal,
            "requested_request_policy_realized":not preserve_requests or not request_changes,
            "requested_internal_waiting_policy_realized":not preserve_requests or not internal_rule_changes,
            "all_main_junction_shapes_unchanged":all(
                e.get("shape")==result.root.find(f"junction[@id='{e.get('id')}']").get("shape")
                for e in original.root.findall("junction") if e.get("type")!="internal"),
            "requested_waiting_positions_realized":all(t["waiting_node_matches_split"] and t["requested_position_matches"] for t in targets),
            "schema_valid":check.returncode==0,"structural_errors_zero":not structure["structural_errors"]}
    manifest={"variant":"internal_waiting_retreat_preserved_requests_diagnostic" if preserve_requests else
                        "internal_waiting_retreat_compiler_requests_diagnostic",
              "network_file":str(target),"source_network_file":str(source),
              "source_sha256":source_hash,"source_unchanged":checks["source_unchanged"],"retreat_m":retreat_m,
              "targets":targets,"checks":checks,"map_variant_valid":all(checks.values()),
              "corridor_inventory":corridor_results,"compiler_topology_helpers":compiler_helpers,
              "preserve_requests":preserve_requests,"source_requests_preserved":not request_changes,
              "compiler_request_changes":compiler_request_changes,
              "compiler_internal_waiting_rule_changes":compiler_internal_rule_changes,
              "internal_waiting_rule_changes":internal_rule_changes,
              "regenerated_request_changes":request_changes,"structural_errors":structure["structural_errors"],
              "review_findings":structure["review_findings"],"netconvert_command":command,
              "evidence_level":"assumption","learning_ready":False,"real_world_waiting_positions_verified":False,
              "hypothesis":"Retreat three internal waiting positions, preserve physical corridors. Bus body/corner overlap depends on position and angle; this is not surveyed geometry.",
              "request_hypothesis":"Source priority/conflict matrices deliberately held fixed to isolate waiting geometry, without certification of their correctness." if preserve_requests else
                                   "Compiler rebuild changes priority responses as listed; this is not a geometry-only causal comparison.",
              "runtime_collisions_validated":False,"source":"https://sumo.dlr.de/docs/Networks/PlainXML.html#lane-to-lane-connectivity"}
    (output/"patch_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    if not manifest["map_variant_valid"]:
        raise GeometryVariantError(f"Preservation/structure gate failed; see {output/'patch_manifest.json'}")
    return manifest
