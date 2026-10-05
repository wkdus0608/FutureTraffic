import sys
import json
import tempfile
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from network_environment import (VERSION, SIMULATION_OUTPUTS, date_window, movement_completion_events,
                                 digest, implementation_fingerprints, inspect_run,
                                 verify_run_inputs)


class ActualMovementCompletion(unittest.TestCase):
    spec = {"id": "turn", "source_from": "a", "source_to": "b", "path": ["a", "b"]}

    def test_internal_entry_is_not_turn_completion(self):
        vehicle = ET.fromstring('<vehicle id="v" type="bus"><route edges="a :j b" exitTimes="2 -1 -1"/></vehicle>')
        self.assertEqual(movement_completion_events(vehicle, [self.spec]), [])

    def test_internal_exit_proves_entry_to_destination_without_route_arrival(self):
        vehicle = ET.fromstring('<vehicle id="v" type="bus"><route edges="a :j b" exitTimes="2 5 -1"/></vehicle>')
        self.assertEqual(movement_completion_events(vehicle, [self.spec])[0]["completion_s"], 5)

    def test_bypass_requires_whole_corridor(self):
        spec = {**self.spec, "path": ["a", "curve", "b"]}
        vehicle = ET.fromstring('<vehicle id="v"><route edges="a :j curve" exitTimes="2 3 4"/></vehicle>')
        self.assertEqual(movement_completion_events(vehicle, [spec]), [])

    def test_common_windows_include_warmup_without_reset(self):
        windows = date_window("20220811", "0700", 60, 30)
        self.assertEqual(windows[0]["period"], "0630")
        self.assertEqual(windows[-1]["period"], "0745")
        self.assertEqual([w["simulation_begin"] for w in windows], [0, 900, 1800, 2700, 3600, 4500])
        self.assertEqual(sum(w["warmup"] for w in windows), 2)


class IndependentValidation(unittest.TestCase):
    def fixture(self, directory, vehicle_id="v", emergency_braking=0, fit_count=1):
        bundle = Path(directory)
        run = bundle / "run"
        run.mkdir()
        inputs = bundle / "inputs" / "0500"
        inputs.mkdir(parents=True)
        (inputs / "originalp.xml").write_text('<data><interval begin="0" end="900">'
            '<edgeRelation from="a" to="b" count="1"/></interval></data>')
        (bundle / "network.net.xml").write_text('<net><edge id="a" to="J"/>'
            '<edge id="b" from="J"/><connection from="a" to="b"/></net>')
        (bundle / "movement_specs.json").write_text(json.dumps([
            {"id": "turn", "source_from": "a", "source_to": "b", "path": ["a", "b"]}]))
        (bundle / "intended_vehicles.json").write_text('{"v": {"type": "passenger", "depart": 0}}')
        fit = bundle / "demand_fit"
        fit.mkdir()
        (fit / "0500.json").write_text(json.dumps({"audit": {"by_movement": [
            {"vehicle_type": "passenger", "source_from": "a", "source_to": "b", "source_count": fit_count}]}}))
        (run / "vehroute.xml").write_text(f'<routes><vehicle id="{vehicle_id}" type="passenger" '
            'depart="0" arrival="10"><route edges="a :J b" exitTimes="1 2 10"/></vehicle></routes>')
        (run / "tripinfo.xml").write_text(f'<tripinfos><tripinfo id="{vehicle_id}" '
            'departDelay="0" timeLoss="1" waitingTime="0"/></tripinfos>')
        (run / "stats.xml").write_text(f'<statistics><safety collisions="0" emergencyStops="0" '
            f'emergencyBraking="{emergency_braking}"/><teleports total="0"/>'
            '<vehicles inserted="1"/></statistics>')
        (run / "sumo.log").write_text("fixture execution")
        (run / "collisions.xml").write_text("<collisions/>")
        manifest = {"version": VERSION, "implementation_fingerprints": implementation_fingerprints(),
            "input_fingerprints": {str(p.relative_to(bundle)): digest(p) for p in bundle.rglob("*")
                                   if p.is_file() and run not in p.parents},
            "window_reports": [{"period": "0500", "begin": 0, "end": 900,
                                "simulation_begin": 0, "source_count": 1}],
            "source_summary": {"source_turn_count": 1}, "evaluation_start_s": 0,
            "duration_s": 900, "build_gates": {"historical_signal_alignment_validated": False}}
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        execution = {"manifest_sha256": digest(bundle / "manifest.json"),
            "command": ["sumo", "--collision.check-junctions", "true", "--collision.action", "warn"],
            "input_fingerprints": manifest["input_fingerprints"],
            "output_fingerprints": {name: digest(run / name) for name in SIMULATION_OUTPUTS}}
        (run / "execution.json").write_text(json.dumps(execution))
        return bundle, run

    def test_equal_counts_do_not_hide_replaced_vehicle_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle, run = self.fixture(directory, vehicle_id="unplanned")
            with self.assertRaisesRegex(ValueError, "unknown simulation vehicle ID"):
                inspect_run(bundle, run)

    def test_fit_denominator_is_recomputed_from_source_xml(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle, run = self.fixture(directory, fit_count=99)
            with self.assertRaisesRegex(ValueError, "independent source XML"):
                inspect_run(bundle, run)

    def test_changed_input_or_output_cannot_reuse_validation(self):
        for filename, message in (("movement_specs.json", "Run input changed"),
                                  ("run/stats.xml", "Run output changed")):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                bundle, run = self.fixture(directory)
                self.assertEqual(verify_run_inputs(bundle, run)["version"], VERSION)
                with (bundle / filename).open("a") as output:
                    output.write("\n")
                with self.assertRaisesRegex(ValueError, message):
                    verify_run_inputs(bundle, run)

    def test_emergency_braking_is_a_failure_even_with_arrivals_and_counts_matching(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle, run = self.fixture(directory, emergency_braking=10)
            result = inspect_run(bundle, run)
            self.assertEqual(result["arrived_vehicles"], 1)
            self.assertEqual(result["collection_site_absolute_residual_fraction"], 0)
            self.assertFalse(result["gates"]["emergency_braking_zero"])
            self.assertFalse(result["gates"]["actual_passages_validated"])
            self.assertFalse(result["learning_ready"])

    def test_zero_collision_count_without_junction_checks_is_not_certified(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle, run = self.fixture(directory)
            execution = json.loads((run / "execution.json").read_text())
            execution["command"] = ["sumo"]
            (run / "execution.json").write_text(json.dumps(execution))
            result = inspect_run(bundle, run)
            self.assertEqual(result["safety"]["collisions"], "0")
            self.assertFalse(result["gates"]["junction_collision_checks_enabled"])
            self.assertFalse(result["gates"]["actual_passages_validated"])


if __name__ == "__main__":
    unittest.main()
