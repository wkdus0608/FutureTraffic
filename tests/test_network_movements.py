"""Complete-network source preservation, semantic paths and assumption gates."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from network_movements import (
    MovementResolutionError, NetworkTopology, audit_inventory, build_central_assumption_variant,
    count_complete_path, load_movement_specs, load_resolution_config, resolve_movement, unresolved_gate,
)


class MovementChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.network = next((ROOT / "Sample").rglob("1200012n.net.xml"))
        cls.routes = next((ROOT / "Sample").rglob("possible_routes.rou.xml"))
        cls.topology = NetworkTopology.read(cls.network)
        cls.config = load_resolution_config()
        cls.digest = hashlib.sha256(cls.network.read_bytes()).hexdigest()

    @classmethod
    def tearDownClass(cls):
        if hashlib.sha256(cls.network.read_bytes()).hexdigest() != cls.digest:
            raise AssertionError("Source network changed during movement tests")

    def test_independent_inventory_reproduces_162_movements_and_103_gap(self):
        result = audit_inventory(self.network, self.routes)
        self.assertEqual(result["unique_positive_pairs"], 162)
        self.assertEqual(result["raw_sample_turn_count"], 1628)
        self.assertEqual(result["raw_all_19_day_turn_count"], 13336507)
        self.assertEqual(result["raw_sample_gap"], 103)
        self.assertEqual({k:v["pair_count"] for k,v in result["categories"].items()},
                         {"covered":140,"missing_edge_id":3,"no_direct_connection":9,"no_candidate":10})
        self.assertEqual([result["categories"][k]["sample_turn_count"] for k in
                          ("missing_edge_id","no_candidate","no_direct_connection")], [14,30,59])

    def test_aliases_preserve_raw_keys_and_explicit_assumption(self):
        specs = load_movement_specs(self.topology)
        aliases = [s for s in specs if s["resolution"] == "alias"]
        self.assertEqual(len(aliases), 3)
        for s in aliases:
            self.assertEqual(s["source_to"], "477706911_02")
            self.assertEqual(s["path"], [s["source_from"], "477706911_02_02"])
            self.assertEqual(s["evidence_level"], "assumption")
            self.assertTrue(s["raw_denominator_preserved"])

    def test_all_bypasses_require_complete_corridor_not_just_entry(self):
        bypasses = [s for s in load_movement_specs(self.topology) if s["resolution"] == "bypass"]
        self.assertEqual(len(bypasses), 6)
        counts = {(x["source_from"],x["source_to"]):x["sample_20220810_0500_count"] for x in self.config["inventory"]}
        self.assertEqual(sum(counts[s["source_from"],s["source_to"]] for s in bypasses), 37)
        for s in bypasses:
            self.assertEqual(len(s["path"]), 3)
            self.assertEqual(count_complete_path(s["path"][:2], s["path"]), 0)
            self.assertEqual(count_complete_path(s["path"], s["path"]), 1)
            self.assertEqual(count_complete_path(s["path"] + ["separator"] + s["path"], s["path"]), 2)
            self.assertEqual(count_complete_path(s["path"][::-1], s["path"]), 0)
            self.topology.validate_path(s["path"])

    def test_raw_denominator_and_conservative_learning_gate(self):
        specs = load_movement_specs(self.topology)
        self.assertEqual(Counter(s["resolution"] for s in specs),
                         {"direct":150,"alias":3,"bypass":6,"unresolved":3})
        counts = {(x["source_from"],x["source_to"]):x["sample_20220810_0500_count"] for x in self.config["inventory"]}
        gate = unresolved_gate(specs, counts)
        self.assertFalse(gate["learning_ready"])
        self.assertEqual(gate["raw_total_turn_count"], 1628)
        self.assertEqual(gate["unresolved_turn_count"], 22)
        self.assertEqual(gate["represented_target_turn_count"], 1606)
        self.assertEqual(gate["unresolved_movement_count"], 3)
        direct=[s for s in specs if s["resolution"]=="direct"][:1]
        direct_gate=unresolved_gate(direct)
        self.assertTrue(direct_gate["movement_coverage_complete"])
        self.assertFalse(direct_gate["learning_ready"])

    def test_uturn_sensitivity_is_separate_and_not_ground_truth(self):
        raw = resolve_movement("477670332_01", "477670332_02", self.topology)
        hypothesis = resolve_movement("477670332_01", "477670332_02", self.topology, variant="uturn_as_straight")
        self.assertIsNone(raw["path"])
        self.assertEqual(hypothesis["path"], ["477670332_01", "477603993_02_02"])
        self.assertEqual(raw["id"], hypothesis["id"])
        self.assertEqual(hypothesis["source_to"], "477670332_02")
        self.assertEqual(hypothesis["evidence_level"], "assumption")
        self.assertTrue(hypothesis["raw_denominator_preserved"])
        self.assertIsNone(resolve_movement("477670332_01", "477670332_02", self.topology)["path"])

    def test_missing_new_movement_is_held_not_invented(self):
        spec=resolve_movement("unknown_road", "another_road", self.topology)
        self.assertEqual(spec["resolution"], "unresolved")
        self.assertIsNone(spec["path"])
        self.assertEqual(unresolved_gate([spec], {("unknown_road","another_road"):17})["raw_total_turn_count"],17)
        omitted=unresolved_gate([], {("unknown_road","another_road"):17})
        self.assertFalse(omitted["learning_ready"])
        self.assertEqual(omitted["unresolved_turn_count"],17)

    def test_broken_alias_or_incomplete_path_rejected(self):
        changed=deepcopy(self.config)
        changed["overrides"][0]["path"]=["477706392_02","477603993_02_02"]
        with self.assertRaises(MovementResolutionError):
            resolve_movement("477706392_02","477706911_02",self.topology,changed)
        with self.assertRaises(MovementResolutionError):
            count_complete_path(["a"],["a"])

    def test_duplicate_override_and_unknown_variant_rejected(self):
        changed=deepcopy(self.config);changed["overrides"].append(deepcopy(changed["overrides"][0]))
        with self.assertRaises(MovementResolutionError):load_resolution_config(changed)
        with self.assertRaises(MovementResolutionError):
            resolve_movement("477670332_01","477670332_02",self.topology,variant="pretend_confirmed")

    def test_central_variant_does_not_activate_missing_original_connection(self):
        for pair in [("477610076_02","477649580_01_01"),("477651439_02","477649578_02_02")]:
            spec=resolve_movement(*pair,self.topology,variant="central_assumed_embedded")
            self.assertIsNone(spec["path"])
            self.assertEqual(spec["resolution"],"unresolved")
            self.assertEqual(spec["status"],"on_hold")

    def test_rebuild_variant_preserves_source_and_other_controllers(self):
        parent=ROOT/"results/network_environment_20261005/movement_tests"
        parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as tmp:
            output=Path(tmp)/"rebuilt"
            manifest=build_central_assumption_variant(self.network,output)
            self.assertTrue(manifest["map_variant_valid"])
            self.assertTrue(manifest["source_unchanged"])
            self.assertFalse(manifest["learning_ready"])
            self.assertFalse(manifest["reconstructed_signal_ready"])
            self.assertEqual(manifest["other_changed_controllers"],[])
            self.assertEqual(manifest["removed_pairs"],[])
            self.assertEqual(manifest["structural_errors"],[])
            self.assertTrue(manifest["outside_central_lane_geometry_unchanged"])
            self.assertTrue(manifest["central_existing_signal_indexes_unchanged"])
            self.assertEqual(manifest["schema_validation_returncode"],0)
            specs=load_movement_specs(manifest["network_file"],variant="central_assumed_embedded")
            self.assertEqual(sum(s["resolution"]=="unresolved" for s in specs),1)
            central=[s for s in specs if s.get("required_patch")]
            self.assertEqual(len(central),2)
            self.assertTrue(all(s["path"] and s["evidence_level"]=="assumption" for s in central))
            combined=load_movement_specs(manifest["network_file"],variant="central_assumed_embedded_uturn_as_straight")
            combined_gate=unresolved_gate(combined)
            self.assertTrue(combined_gate["movement_coverage_complete"])
            self.assertFalse(combined_gate["learning_ready"])
            self.assertTrue(combined_gate["blocked_by_variant_certification"])
            # Preserve every original 201306 index and append exactly the two
            # declared model indices, without claiming source-column identity.
            rebuilt=NetworkTopology.read(manifest["network_file"])
            original_links={(c.get("from"),c.get("to"),c.get("fromLane"),c.get("toLane")):c.get("linkIndex")
                            for c in self.topology.root.findall("connection") if c.get("tl")=="201306"}
            actual_links={(c.get("from"),c.get("to"),c.get("fromLane"),c.get("toLane")):c.get("linkIndex")
                          for c in rebuilt.root.findall("connection") if c.get("tl")=="201306"}
            self.assertTrue(all(actual_links[k]==index for k,index in original_links.items()))
            self.assertEqual({actual_links[k] for k in actual_links.keys()-original_links.keys()},{"15","16"})
            # Actual SUMO passage is separate from graph/XML validity. A tiny
            # source-blind probe checks both generated turns with teleport off.
            import sumolib
            vehicle_root=ET.Element("routes")
            ET.SubElement(vehicle_root,"vType",id="probe",vClass="passenger",sigma="0")
            for identifier,a,b,lane in [("right","477610076_02","477649580_01_01",0),
                                        ("left","477651439_02","477649578_02_02",1)]:
                vehicle=ET.SubElement(vehicle_root,"vehicle",id=identifier,type="probe",depart="0",departLane=str(lane))
                ET.SubElement(vehicle,"route",edges=a+" "+b)
            route_file=output/"turn_probe.rou.xml"
            ET.ElementTree(vehicle_root).write(route_file,encoding="utf-8",xml_declaration=True)
            run=subprocess.run([sumolib.checkBinary("sumo"),"--net-file",manifest["network_file"],
                "--route-files",str(route_file),"--end","150","--time-to-teleport","-1", "--no-step-log","true",
                "--tripinfo-output",str(output/"turn_probe.tripinfo.xml"),"--statistic-output",str(output/"turn_probe.statistics.xml")],
                capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            trips=ET.parse(output/"turn_probe.tripinfo.xml").findall("tripinfo")
            self.assertEqual({x.get("id") for x in trips},{"right","left"})
            statistics=ET.parse(output/"turn_probe.statistics.xml").getroot()
            self.assertEqual(int(statistics.find("safety").get("collisions")),0)
            self.assertEqual(int(statistics.find("teleports").get("total")),0)
            with self.assertRaises(FileExistsError):build_central_assumption_variant(self.network,output)


if __name__ == "__main__":
    unittest.main()
