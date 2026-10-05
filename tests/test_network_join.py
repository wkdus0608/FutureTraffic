"""A regional join may remove model stubs, never observations or vehicles."""
from pathlib import Path
import hashlib
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from network_join import (CONNECTORS, JOIN_ID, JOIN_NODES, JoinValidationError,
                          build_joined_priority_variant, rewrite_routes_for_join)


class JoinChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = next((ROOT / 'Sample').rglob('1200012n.net.xml'))
        cls.source_hash = hashlib.sha256(cls.source.read_bytes()).hexdigest()
        cls.temp = tempfile.TemporaryDirectory()
        cls.parent = Path(cls.temp.name)
        cls.manifest = build_joined_priority_variant(cls.source, cls.parent / 'joined')

    @classmethod
    def tearDownClass(cls):
        try:
            if hashlib.sha256(cls.source.read_bytes()).hexdigest() != cls.source_hash:
                raise AssertionError('Source map changed')
        finally:
            cls.temp.cleanup()

    def test_only_local_region_changes_and_source_inventory_is_retained(self):
        m = self.manifest
        self.assertTrue(m['success'])
        self.assertTrue(all(m['gates'].values()))
        self.assertFalse(m['learning_ready'])
        self.assertEqual(m['raw_source_pair_count'], 162)
        self.assertEqual(m['stub_observed_source_pairs'], [])
        self.assertEqual(m['source_TLS_count'], 12)
        self.assertEqual(m['joined_TLS_count'], 12)
        self.assertEqual(m['outside_endpoint_failures'], [])
        net = ET.parse(m['network_file']).getroot()
        self.assertIsNotNone(net.find(f"junction[@id='{JOIN_ID}']"))
        self.assertTrue(all(net.find(f"junction[@id='{n}']") is None for n in JOIN_NODES))

    def test_join_connections_equal_original_reachable_compositions(self):
        source = ET.parse(self.source).getroot()
        external = {e.get('id') for e in source.findall('edge') if e.get('function') != 'internal'}
        pairs = {(c.get('from'), c.get('to')) for c in source.findall('connection')
                 if c.get('from') in external and c.get('to') in external}
        composed = {(a, b) for a, stub in pairs if stub in CONNECTORS
                    for stub2, b in pairs if stub2 == stub and b not in CONNECTORS}
        self.assertEqual(set(map(tuple, self.manifest['added_pairs'])), composed)
        self.assertTrue(all(set(pair) & CONNECTORS for pair in self.manifest['removed_pairs']))

    def make_routes(self, name):
        path = self.parent / name
        path.write_text('''<routes><vType id="bus" length="12" sigma="0.5"/>
<route id="r" edges="477603177_02 477771616_01 477771614_02"/>
<vehicle id="same_bus" type="bus" route="r" depart="12.34" departLane="best"><param key="source_period" value="0500"/></vehicle>
</routes>''')
        return path

    def test_rewrite_preserves_vehicle_driver_dispatch_and_remaining_route_order(self):
        source = self.make_routes('routes.xml')
        before_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        output = self.parent / 'rewritten.xml'
        report = rewrite_routes_for_join(source, self.manifest, output)
        self.assertEqual(report['vehicles_before'], 1)
        self.assertEqual(report['vehicles_after'], 1)
        self.assertTrue(report['vehicle_ids_types_depart_params_preserved'])
        self.assertTrue(report['vtypes_preserved'])
        self.assertTrue(report['all_new_pairs_connected'])
        self.assertEqual(report['removed_stub_occurrences_in_route_definitions'], 1)
        after = ET.parse(output).getroot()
        self.assertEqual(after.find('route').get('edges'), '477603177_02 477771614_02')
        self.assertEqual(after.find('vehicle').attrib,
                         ET.parse(source).getroot().find('vehicle').attrib)
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before_hash)

    def test_observed_stub_prevents_approval_or_route_rewrite(self):
        m = build_joined_priority_variant(self.source, self.parent / 'observed_stub',
                                         source_pairs=[('477603177_02', '477771616_01')])
        self.assertFalse(m['success'])
        self.assertFalse(m['gates']['removed_stubs_unobserved_in_source_inventory'])
        with self.assertRaises(JoinValidationError):
            rewrite_routes_for_join(self.make_routes('blocked.xml'), m, self.parent / 'blocked_out.xml')

    def test_changed_join_map_or_existing_outputs_are_rejected(self):
        with self.assertRaises(FileExistsError):
            build_joined_priority_variant(self.source, self.parent / 'joined')
        copy = self.parent / 'changed.net.xml'
        copy.write_bytes(Path(self.manifest['network_file']).read_bytes() + b'\n')
        m = dict(self.manifest, network_file=str(copy))
        with self.assertRaises(JoinValidationError):
            rewrite_routes_for_join(self.make_routes('changed_routes.xml'), m,
                                    self.parent / 'changed_out.xml')


if __name__ == '__main__':
    unittest.main()
