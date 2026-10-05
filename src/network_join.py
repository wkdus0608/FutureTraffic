"""Diagnostic join of the two audited, closely spaced priority junctions.

netconvert generates the complete joined right-of-way subgraph. Only that
compiler-produced region is transplanted into a copy of the supplied network;
outside geometry, connections, and every traffic-light program remain original.
This is a SUMO modelling hypothesis, not a verified historical road correction.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


JOIN_NODES = ('204922', '605257')
JOIN_ID = 'joined_priority_204922_605257'
CONNECTORS = frozenset(('477771616_01', '477771616_02'))
PROJECT = Path(__file__).resolve().parents[1]
JOIN_DOCUMENTATION = 'https://sumo.dlr.de/docs/Networks/PlainXML.html#joining-nodes'


class JoinValidationError(ValueError):
    """A joined variant cannot preserve the reviewed source scope."""


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _signature(element):
    # Ignore indentation only; preserve every substantive attribute and child.
    return (element.tag, tuple(sorted(element.attrib.items())), (element.text or '').strip(),
            tuple(_signature(child) for child in element))


def _prefix(identifier, nodes):
    return any(identifier.startswith(':' + node + '_') for node in nodes)


def _region_predicate(root, nodes):
    external = {edge.get('id') for edge in root.findall('edge')
                if edge.get('function') != 'internal' and
                (edge.get('from') in nodes or edge.get('to') in nodes)}
    incoming = {edge.get('id') for edge in root.findall('edge') if edge.get('to') in nodes}
    def inside(element):
        identifier = element.get('id', '')
        if element.tag == 'edge':
            return identifier in external or _prefix(identifier, nodes)
        if element.tag == 'junction':
            return identifier in nodes or _prefix(identifier, nodes)
        if element.tag == 'connection':
            return element.get('from') in incoming or _prefix(element.get('from', ''), nodes)
        return False
    return inside, external


def _outside_inventory(root, predicate):
    return {_signature(element) for element in root if not predicate(element)}


def _lane_outer_endpoint_checks(original, assembled, original_nodes):
    edges_a = {e.get('id'): e for e in original.findall('edge') if e.get('function') != 'internal'}
    edges_b = {e.get('id'): e for e in assembled.findall('edge') if e.get('function') != 'internal'}
    failures = []
    for identifier, before in edges_a.items():
        if identifier in CONNECTORS or not ({before.get('from'), before.get('to')} & set(original_nodes)):
            continue
        after = edges_b[identifier]
        lanes_after = {lane.get('index'): lane for lane in after.findall('lane')}
        for lane in before.findall('lane'):
            other = lanes_after.get(lane.get('index'))
            if other is None:
                failures.append({'edge': identifier, 'lane': lane.get('id'), 'reason': 'lane removed'})
                continue
            # The endpoint next to an outside junction must not shift, or its
            # preserved internal geometry would cease to join this region.
            endpoint = 0 if before.get('to') in original_nodes else -1
            old_point = lane.get('shape', '').split()[endpoint]
            new_point = other.get('shape', '').split()[endpoint]
            if old_point != new_point:
                failures.append({'edge': identifier, 'lane': lane.get('id'),
                                 'old_outer_endpoint': old_point, 'new_outer_endpoint': new_point})
    return failures


def _connections(root):
    return {(c.get('from'), c.get('to')) for c in root.findall('connection')}


def build_joined_priority_variant(network_file: Path | str, output_dir: Path | str,
                                  source_pairs=None, netconvert_binary=None) -> dict:
    """Generate and gate a fresh regional priority join without editing inputs.

    Returns a manifest even when a compiler/preservation gate fails. A failed
    manifest cannot be used to rewrite routes. Default raw inventory is the
    independently audited 162 positive source pairs; lack of a stub observation
    is not a claim that real traffic on that road is zero.
    """
    source, output = Path(network_file).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to replace an existing diagnostic variant: {output}')
    if source.is_relative_to(output):
        raise JoinValidationError('Output directory must not contain the source network')
    original_hash = _digest(source)
    original = ET.parse(source).getroot()
    nodes = {j.get('id'): j for j in original.findall('junction')}
    if any(nodes.get(node) is None or nodes[node].get('type') != 'priority' for node in JOIN_NODES):
        raise JoinValidationError('Join is restricted to the two audited priority junctions')
    if JOIN_ID in nodes:
        raise JoinValidationError('The requested joined junction already exists')
    inside_original, affected_external = _region_predicate(original, JOIN_NODES)
    external_original = {edge.get('id'): edge for edge in original.findall('edge') if edge.get('function') != 'internal'}
    connecting_edges = {edge.get('id') for edge in external_original.values()
                        if edge.get('from') in JOIN_NODES and edge.get('to') in JOIN_NODES}
    if connecting_edges != CONNECTORS:
        raise JoinValidationError('Only the two audited 0.29m connectors may be removed')
    if source_pairs is None:
        inventory = json.loads((PROJECT / 'config/network_movements.json').read_text())['inventory']
        pairs = {(r['source_from'], r['source_to']) for r in inventory}
        inventory_evidence = 'Audited complete 19-day positive source-pair inventory'
    else:
        pairs = {(r['source_from'], r['source_to']) if isinstance(r, dict) else tuple(r) for r in source_pairs}
        inventory_evidence = 'Caller-supplied source-pair inventory'
    stub_observations = sorted(pair for pair in pairs if set(pair) & CONNECTORS)
    output.mkdir(parents=True)
    patch = ET.Element('nodes', {'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
                                'xsi:noNamespaceSchemaLocation': 'http://sumo.dlr.de/xsd/nodes_file.xsd'})
    ET.SubElement(patch, 'join', nodes=' '.join(JOIN_NODES), id=JOIN_ID, type='priority', reset='false')
    ET.indent(patch)
    patch_file, compiled_file, assembled_file = output / 'join.nod.xml', output / 'netconvert_all.net.xml', output / 'network.net.xml'
    ET.ElementTree(patch).write(patch_file, encoding='utf-8', xml_declaration=True)
    if netconvert_binary is None:
        import sumolib
        netconvert_binary = sumolib.checkBinary('netconvert')
    command = [str(netconvert_binary), '--sumo-net-file', str(source), '--node-files', str(patch_file),
               '--output-file', str(compiled_file), '--offset.disable-normalization', 'true',
               '--tls.rebuild', 'false', '--xml-validation', 'always']
    result = subprocess.run(command, capture_output=True, text=True)
    (output / 'netconvert.log').write_text(result.stdout + result.stderr)
    manifest = {'variant': 'joined_priority_diagnostic', 'network_file': str(assembled_file),
                'source_network_file': str(source), 'source_sha256': original_hash,
                'joined_nodes': list(JOIN_NODES), 'joined_id': JOIN_ID,
                'removed_connectors': sorted(CONNECTORS), 'affected_external_edges': sorted(affected_external - CONNECTORS),
                'raw_source_pair_count': len(pairs), 'stub_observed_source_pairs': stub_observations,
                'source_inventory_evidence': inventory_evidence, 'netconvert_command': command,
                'netconvert_returncode': result.returncode, 'success': False, 'learning_ready': False,
                'documentation': JOIN_DOCUMENTATION,
                'assumptions': ['Nearby priority nodes are represented as one model junction.',
                                'This removes artificial connector storage and recomputes local right-of-way.',
                                'Historical field geometry and priority remain unverified.',
                                'Source observations and boundary vehicle demand are never deleted.']}
    if result.returncode == 0 and compiled_file.exists():
        compiled = ET.parse(compiled_file).getroot()
        inside_compiled, compiled_external = _region_predicate(compiled, (JOIN_ID,))
        merged = ET.Element(original.tag, deepcopy(original.attrib))
        tags = list(dict.fromkeys(element.tag for element in original))
        for tag in tags:
            elements = [deepcopy(e) for e in original.findall(tag) if not inside_original(e)]
            elements.extend(deepcopy(e) for e in compiled.findall(tag) if inside_compiled(e))
            if tag == 'edge':
                elements.sort(key=lambda e: (e.get('function') != 'internal', e.get('id')))
            merged.extend(elements)
        ET.indent(merged, space='    ')
        ET.ElementTree(merged).write(assembled_file, encoding='utf-8', xml_declaration=True)
        original_outside = _outside_inventory(original, inside_original)
        outside_unchanged = original_outside == _outside_inventory(merged, inside_compiled)
        # The predicates differ only in local node IDs; outside elements cannot
        # be exempted by the compiler from preservation checks.
        source_tls = [_signature(t) for t in original.findall('tlLogic')]
        merged_tls = [_signature(t) for t in merged.findall('tlLogic')]
        source_pairs_external = {(a,b) for a,b in _connections(original) if a in external_original and b in external_original}
        external_merged = {e.get('id'): e for e in merged.findall('edge') if e.get('function') != 'internal'}
        merged_pairs_external = {(a,b) for a,b in _connections(merged) if a in external_merged and b in external_merged}
        endpoints = _lane_outer_endpoint_checks(original, merged, JOIN_NODES)
        from audit_network_structure import audit
        structure = audit(assembled_file)
        schema = Path(netconvert_binary).parent.parent / 'data/xsd/net_file.xsd'
        validation = subprocess.run(['/usr/bin/xmllint', '--nonet', '--noout', '--schema', str(schema), str(assembled_file)],
                                    capture_output=True, text=True)
        (output / 'schema_validation.log').write_text(validation.stdout + validation.stderr)
        gates = {'compiler_join_has_exact_external_region': compiled_external == affected_external - CONNECTORS,
                 'exactly_two_connectors_removed': set(external_original) - set(external_merged) == CONNECTORS,
                 'no_external_edge_added': not (set(external_merged) - set(external_original)),
                 'outside_region_elements_preserved': outside_unchanged,
                 'all_TLS_programs_preserved': source_tls == merged_tls,
                 'TLS_count_preserved': len(source_tls) == len(merged_tls),
                 'outside_endpoint_geometry_preserved': not endpoints,
                 'removed_stubs_unobserved_in_source_inventory': not stub_observations,
                 'schema_valid': validation.returncode == 0,
                 'structure_valid': not structure['structural_errors'],
                 'source_network_unchanged': _digest(source) == original_hash}
        manifest.update(gates=gates, success=all(gates.values()),
                        outside_endpoint_failures=endpoints,
                        removed_pairs=sorted(source_pairs_external - merged_pairs_external),
                        added_pairs=sorted(merged_pairs_external - source_pairs_external),
                        structural_errors=structure['structural_errors'],
                        schema_validation_returncode=validation.returncode,
                        source_TLS_count=len(source_tls), joined_TLS_count=len(merged_tls),
                        network_sha256=_digest(assembled_file),
                        compiled_outside_changed_elements=sum(_signature(e) not in original_outside
                            for e in compiled if not inside_compiled(e)))
    manifest['source_unchanged'] = _digest(source) == original_hash
    (output / 'join_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return manifest


def rewrite_routes_for_join(routes_file: Path | str, join_manifest: dict | Path | str,
                            output_routes: Path | str) -> dict:
    """Delete only unobserved connector IDs and preserve dispatch/driver identity."""
    manifest = json.loads(Path(join_manifest).read_text()) if not isinstance(join_manifest, dict) else join_manifest
    if not manifest.get('success'):
        raise JoinValidationError('Cannot rewrite routes for a failed join variant')
    network = Path(manifest['network_file'])
    if _digest(network) != manifest.get('network_sha256'):
        raise JoinValidationError('Joined network changed after validation')
    source, output = Path(routes_file).resolve(), Path(output_routes).resolve()
    if source == output or output.exists():
        raise FileExistsError('Refusing to replace a route source or existing output')
    before_hash = _digest(source)
    root = ET.parse(source).getroot()
    if any(root.findall(tag) for tag in ('trip', 'flow')):
        raise JoinValidationError('Route rewriting requires explicit planned vehicle routes')
    removed = set(manifest['removed_connectors'])
    if removed != CONNECTORS:
        raise JoinValidationError('Unexpected connector deletion set')
    vehicle_identity = [(v.get('id'), _signature(v)) for v in root.findall('vehicle')]
    vtype_identity = [_signature(v) for v in root.findall('vType')]
    net_root = ET.parse(network).getroot()
    available = {e.get('id') for e in net_root.findall('edge') if e.get('function') != 'internal'}
    connections = _connections(net_root)
    total_removed, changed_routes = 0, 0
    before_after = []
    for route in root.iter('route'):
        if 'exitTimes' in route.attrib or not route.get('edges'):
            raise JoinValidationError('Only planned explicit edge sequences can be rewritten')
        before = route.get('edges').split()
        after = [edge for edge in before if edge not in removed]
        if len(after) < 1 or any(edge not in available for edge in after):
            raise JoinValidationError('A rewritten route contains no edges or unknown edges')
        missing = [(a,b) for a,b in zip(after, after[1:]) if (a,b) not in connections]
        if missing:
            raise JoinValidationError(f'Joined route is disconnected: {missing}')
        count = len(before) - len(after)
        total_removed += count
        changed_routes += bool(count)
        route.set('edges', ' '.join(after))
        if count:
            before_after.append({'route_id': route.get('id'), 'removed_stub_occurrences': count,
                                 'retained_external_order': after})
    # Top-level route definitions change; vehicle attributes, params and vTypes
    # are unchanged. Inline routes retain vehicle attrs but their edges change.
    after_identity = [(v.get('id'), _signature(v)) for v in root.findall('vehicle')]
    if vehicle_identity != after_identity:
        raise JoinValidationError('Vehicle identity changed; inline routes require explicit handling')
    if vtype_identity != [_signature(v) for v in root.findall('vType')]:
        raise JoinValidationError('Driver or vehicle type definitions changed')
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space='    ')
    ET.ElementTree(root).write(output, encoding='utf-8', xml_declaration=True)
    if _digest(source) != before_hash:
        raise JoinValidationError('Original route source changed')
    report = {'source_routes': str(source), 'source_sha256': before_hash, 'output_routes': str(output),
              'output_sha256': _digest(output), 'vehicles_before': len(vehicle_identity),
              'vehicles_after': len(after_identity), 'vehicle_ids_types_depart_params_preserved': True,
              'vtypes_preserved': True, 'removed_stub_occurrences_in_route_definitions': total_removed,
              'changed_route_definitions': changed_routes, 'routes': before_after,
              'all_new_pairs_connected': True, 'source_unchanged': True,
              'interpretation': 'Same vehicles and retained external order; only unobserved join stubs removed.'}
    output.with_suffix(output.suffix + '.rewrite.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report
