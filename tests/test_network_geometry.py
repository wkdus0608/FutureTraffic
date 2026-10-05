"""Geometry hypotheses preserve demand paths, signals and source maps."""
import hashlib
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from network_geometry import (
    GeometryVariantError, build_internal_waiting_position_variant,
    inventory_waiting_position_targets,
)


class GeometryChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = next((ROOT / "Sample").rglob("1200012n.net.xml"))
        cls.source_hash = hashlib.sha256(cls.source.read_bytes()).hexdigest()
        parent = ROOT / "results/network_environment_20261005/geometry_tests"
        parent.mkdir(parents=True, exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(dir=parent)
        cls.manifest = build_internal_waiting_position_variant(cls.source, Path(cls.temp.name) / "original")

    @classmethod
    def tearDownClass(cls):
        try:
            if hashlib.sha256(cls.source.read_bytes()).hexdigest() != cls.source_hash:
                raise AssertionError("Source map changed")
        finally:
            cls.temp.cleanup()

    def test_reviewed_exact_connections_and_retreats(self):
        targets = inventory_waiting_position_targets(self.source)
        self.assertEqual([t["junction"] for t in targets], ["203019", "201305", "201330"])
        self.assertEqual([t["request_index_before"] for t in targets], [3, 19, 15])
        self.assertEqual([t["linkIndex"] for t in targets], [None, "19", "15"])
        for target, expected in zip(targets, [2.64, 0.44, 2.63]):
            self.assertAlmostEqual(target["contPos_requested_m"], expected)
            self.assertFalse(target["internal_wait_removed"])

    def test_zero_position_is_not_mislabeled_as_waiting_point(self):
        targets = inventory_waiting_position_targets(self.source, retreat_m=10)
        self.assertTrue(all(t["internal_wait_removed"] for t in targets))
        self.assertTrue(all(t["contPos_requested_m"] == 0 for t in targets))
        for invalid in [0, -1, float("inf"), float("nan")]:
            with self.assertRaises(GeometryVariantError):
                inventory_waiting_position_targets(self.source, invalid)

    def test_all_corridors_external_geometry_signals_and_source_preserved(self):
        manifest = self.manifest
        self.assertTrue(manifest["map_variant_valid"])
        self.assertTrue(all(manifest["checks"].values()))
        self.assertFalse(manifest["learning_ready"])
        self.assertFalse(manifest["real_world_waiting_positions_verified"])
        self.assertFalse(manifest["runtime_collisions_validated"])
        self.assertEqual(manifest["structural_errors"], [])
        self.assertEqual(len(manifest["corridor_inventory"]), 55)
        self.assertTrue(all(c["physical_corridor_preserved"] and c["total_lane_length_preserved"]
                            for c in manifest["corridor_inventory"]))
        self.assertEqual(len(ET.parse(manifest["network_file"]).findall("tlLogic")), 12)
        self.assertEqual(manifest["evidence_level"], "assumption")

    def test_new_waiting_node_coordinates_and_split_lane_lengths(self):
        root = ET.parse(self.manifest["network_file"]).getroot()
        for target in self.manifest["targets"]:
            trace = target["trace_after"]
            first = root.find(f".//lane[@id='{trace[0]}']")
            second = root.find(f".//lane[@id='{trace[1]}']")
            junction = root.find(f"junction[@id='{trace[1]}']")
            point = tuple(map(float, first.get("shape").split()[-1].split(",")))
            start = tuple(map(float, second.get("shape").split()[0].split(",")))
            self.assertEqual(point, start)
            self.assertAlmostEqual(math.dist(point, (float(junction.get("x")), float(junction.get("y")))), 0)
            self.assertAlmostEqual(float(first.get("length")), target["contPos_requested_m"])
            self.assertEqual(first.get("width"), "3.20")
        self.assertTrue(self.manifest["compiler_topology_helpers"])
        self.assertTrue(self.manifest["compiler_request_changes"])
        self.assertEqual(self.manifest["regenerated_request_changes"], [])
        self.assertTrue(self.manifest["source_requests_preserved"])
        self.assertEqual(self.manifest["internal_waiting_rule_changes"], [])

    def test_source_requests_and_mapping_stay_identical_and_compiler_variant_is_explicit(self):
        original=ET.parse(self.source).getroot()
        rebuilt=ET.parse(self.manifest["network_file"]).getroot()
        for node in original.findall("junction"):
            after=rebuilt.find(f"junction[@id='{node.get('id')}']")
            self.assertEqual([r.attrib for r in node.findall("request")],
                             [r.attrib for r in after.findall("request")])
            if node.get("type")!="internal":
                self.assertEqual(node.get("shape"),after.get("shape"))
        self.assertTrue(self.manifest["checks"]["request_index_lane_pair_mapping_unchanged"])
        compiler=build_internal_waiting_position_variant(self.source,Path(self.temp.name)/"compiler_variant",
                                                        preserve_requests=False)
        self.assertTrue(compiler["map_variant_valid"])
        self.assertFalse(compiler["source_requests_preserved"])
        self.assertFalse(compiler["learning_ready"])
        self.assertEqual(len(compiler["regenerated_request_changes"]),8)
        with self.assertRaises(GeometryVariantError):
            build_internal_waiting_position_variant(self.source,Path(self.temp.name)/"unsafe_clamp",retreat_m=10)

    def test_non_target_internal_waiting_eligibility_and_connection_semantics_preserved(self):
        before=ET.parse(self.source).getroot();after=ET.parse(self.manifest["network_file"]).getroot()
        moved={t["trace_after"][1] for t in self.manifest["targets"]}
        for node in before.findall("junction"):
            if node.get("type")=="internal":
                actual=after.find(f"junction[@id='{node.get('id')}']")
                if node.get("id") in moved:
                    self.assertEqual({k:v for k,v in node.attrib.items() if k not in ["x","y"]},
                                     {k:v for k,v in actual.attrib.items() if k not in ["x","y"]})
                else:self.assertEqual(node.attrib,actual.attrib)
        keys=("from","to","fromLane","toLane")
        actual={tuple(c.get(k) for k in keys):c.attrib for c in after.findall("connection")}
        for c in before.findall("connection"):
            if c.get("from").startswith(":"):
                self.assertEqual(c.attrib,actual[tuple(c.get(k) for k in keys)])

    def test_central_patched_input_keeps_added_connections_and_17_column_program(self):
        from network_movements import build_central_assumption_variant
        base = Path(self.temp.name)
        central = build_central_assumption_variant(self.source, base / "central_input")
        digest = hashlib.sha256(Path(central["network_file"]).read_bytes()).hexdigest()
        result = build_internal_waiting_position_variant(central["network_file"], base / "central_retreat")
        self.assertTrue(result["map_variant_valid"])
        self.assertTrue(all(result["checks"].values()))
        root = ET.parse(result["network_file"]).getroot()
        logic = root.find("tlLogic[@id='201306']")
        self.assertEqual({len(p.get("state")) for p in logic.findall("phase")}, {17})
        self.assertEqual(hashlib.sha256(Path(central["network_file"]).read_bytes()).hexdigest(), digest)
        for idx in ["15", "16"]:
            self.assertIsNotNone(root.find(f"connection[@tl='201306'][@linkIndex='{idx}']"))

    def test_actual_sumo_accepts_the_restored_internal_topology(self):
        import sumolib
        out = Path(self.manifest["network_file"]).parent
        run = subprocess.run([sumolib.checkBinary("sumo"), "--net-file", self.manifest["network_file"],
                              "--begin", "0", "--end", "0", "--xml-validation", "always",
                              "--collision.check-junctions", "true"], capture_output=True, text=True)
        (out / "sumo_load.log").write_text(run.stdout + run.stderr)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertNotIn("Error:", run.stderr)

    def test_output_reuse_rejected(self):
        with self.assertRaises(FileExistsError):
            build_internal_waiting_position_variant(self.source, Path(self.manifest["network_file"]).parent)
        with self.assertRaises(GeometryVariantError):
            build_internal_waiting_position_variant(self.source, ROOT / "config/unsafe_geometry_output")


if __name__ == "__main__":
    unittest.main()
