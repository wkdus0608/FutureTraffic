"""Check source preservation and route accounting, including prior counterexamples."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from network_demand import (CandidateRoute, RouteVehicle, TurnObservation, aggregate_counts,
                            audit_planned_routes, fit_route_demand, movement_specs,
                            read_route_vehicles, read_turn_counts)


class LedgerChecks(unittest.TestCase):
    def test_common_window_preserves_late_motorcycle_and_explicit_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            passenger = directory / "p.xml"
            passenger.write_text('<data><interval begin="0" end="60"><edgeRelation from="A" to="B" count="8"/>'
                                 '<edgeRelation from="A" to="C" count="0"/></interval>'
                                 '<interval begin="840" end="900"><edgeRelation from="A" to="B" count="2"/></interval></data>')
            motorcycle = directory / "m.xml"
            motorcycle.write_text('<data><interval begin="60" end="120"><edgeRelation from="A" to="B" count="25.0"/></interval></data>')
            ledger = read_turn_counts({"p": passenger, "m": motorcycle})
            self.assertEqual((ledger.begin, ledger.end, ledger.total), (0, 900, 35))
            self.assertEqual(ledger.counts[("motorcycle", "A", "B")], 25)
            self.assertEqual(ledger.counts[("passenger", "A", "C")], 0)
            self.assertNotIn(("motorcycle", "A", "C"), ledger.counts)
            self.assertEqual(len(ledger.explicit_zero_records), 1)
            self.assertEqual(len(ledger.source_hashes), 2)

    def test_partial_interval_and_duplicate_are_rejected(self):
        row = TurnObservation("passenger", 0, 60, "A", "B", 7)
        with self.assertRaisesRegex(ValueError, "cuts"):
            aggregate_counts([row], 0, 30)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            aggregate_counts([row, row], 0, 60)

    def test_zero_does_not_become_unknown(self):
        ledger = aggregate_counts([TurnObservation("passenger", 0, 60, "A", "B", 0)], 0, 60)
        routes = [RouteVehicle("one", "passenger", ("A", "B")), RouteVehicle("two", "passenger", ("A", "C"))]
        audit = audit_planned_routes(ledger, routes)
        self.assertEqual(audit["totals"]["over"], 1)
        self.assertEqual(audit["unobserved_planned_transitions"], 1)
        self.assertEqual(audit["by_movement"][0]["source_state"], "explicit_zero")
        self.assertFalse(audit["absence_is_zero"])


class PlannedRouteChecks(unittest.TestCase):
    def test_passages_are_not_unique_vehicles_and_short_bypass_does_not_complete(self):
        ledger = aggregate_counts([TurnObservation("passenger", 0, 60, "A", "B", 1),
                                   TurnObservation("passenger", 0, 60, "X", "Y", 1)], 0, 60)
        specs = [{"id": "bypass", "source_from": "X", "source_to": "Y", "path": ["U", "C", "D"],
                  "resolution": "bypass", "evidence_level": "assumption", "evidence": "fixture"}]
        routes = [RouteVehicle("complete", "passenger", ("A", "B", "U", "C", "D")),
                  RouteVehicle("partial", "passenger", ("U", "C"))]
        audit = audit_planned_routes(ledger, routes, specs)
        self.assertEqual(audit["totals"]["planned_passages"], 2)
        self.assertEqual(audit["totals"]["under"], 0)
        bypass = next(row for row in audit["by_movement"] if row["id"] == "bypass")
        self.assertEqual(bypass["planned_passages"], 1)
        self.assertFalse(audit["actual_passage_times_verified"])

    def test_movement_occurrences_are_all_counted_and_absent_type_is_unknown(self):
        ledger = aggregate_counts([TurnObservation("passenger", 0, 60, "A", "B", 1)], 0, 60)
        routes = [RouteVehicle("loop", "passenger", ("A", "B", "A", "B")),
                  RouteVehicle("unrecorded_bus", "bus", ("A", "B"))]
        audit = audit_planned_routes(ledger, routes)
        self.assertEqual(audit["totals"]["planned_passages"], 2)
        self.assertEqual(audit["totals"]["over"], 1)
        self.assertEqual(audit["unobserved_planned_transitions"], 1)
        self.assertEqual(audit["movement_totals_ignoring_vehicle_type"]["planned_passages"], 3)
        self.assertEqual(audit["unobserved_planned_movements_ignoring_vehicle_type"], 0)

    def test_raw_alias_and_unresolved_denominator_are_retained(self):
        ledger = aggregate_counts([TurnObservation("passenger", 0, 60, "A", "old", 14),
                                   TurnObservation("passenger", 0, 60, "U", "V", 17)], 0, 60)
        specs = [{"id": "alias", "source_from": "A", "source_to": "old", "path": ["A", "new"]},
                 {"id": "unknown", "source_from": "U", "source_to": "V", "path": None}]
        audit = audit_planned_routes(ledger, [RouteVehicle(str(i), "passenger", ("A", "new")) for i in range(14)], specs)
        self.assertEqual(audit["totals"]["source_count"], 31)
        self.assertEqual(audit["unresolved_source_count"], 17)
        self.assertEqual(audit["totals"]["under"], 17)
        self.assertEqual(ledger.counts[("passenger", "A", "old")], 14)

    def test_colliding_aliases_need_explicit_allocation_rule(self):
        ledger = aggregate_counts([TurnObservation("passenger", 0, 60, "A", "old", 1),
                                   TurnObservation("passenger", 0, 60, "A", "new", 1)], 0, 60)
        with self.assertRaisesRegex(ValueError, "same physical path"):
            movement_specs(ledger, [{"source_from": "A", "source_to": "old", "path": ["A", "new"]}])

    def test_registered_bypass_absent_in_window_is_still_counted_as_unknown(self):
        ledger = aggregate_counts([TurnObservation("passenger", 0, 60, "A", "B", 1)], 0, 60)
        specs = [{"source_from": "X", "source_to": "Y", "path": ["U", "C", "D"]}]
        routes = [RouteVehicle("complete", "passenger", ("A", "B", "U", "C", "D"))]
        audit = audit_planned_routes(ledger, routes, specs)
        self.assertEqual(audit["totals"]["source_count"], 1)
        self.assertEqual(audit["unobserved_planned_transitions"], 0)
        self.assertEqual(audit["unrecorded_registered_complete_movements"], 1)
        self.assertEqual(audit["unrecorded_registered_by_movement"][0]["source_from"], "X")

    def test_vehicle_route_references_are_read_and_flows_not_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "routes.xml"
            path.write_text('<routes><route id="r" edges="A B"/><vehicle id="v" type="bus" depart="900" route="r"/></routes>')
            route = read_route_vehicles(path)[0]
            self.assertEqual((route.edges, route.depart, route.vehicle_type), (("A", "B"), 900, "bus"))
            path.write_text('<routes><flow id="f" number="100"/></routes>')
            with self.assertRaisesRegex(ValueError, "flows/trips"):
                read_route_vehicles(path)


class IntegerFitChecks(unittest.TestCase):
    def test_terminal_turnaround_frontier_preserves_signal_junctions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "net.xml"
            edge_nodes = {"A": ("entry", "J"), "B": ("J", "exit"), "O": ("J", "stub"),
                          "I": ("stub", "J"), "L": ("J", "other_signal"), "M": ("other_signal", "J")}
            edges = ''.join(f'<edge id="{e}" from="{a}" to="{b}"><lane id="{e}_0" length="100" speed="10"/></edge>'
                            for e, (a, b) in edge_nodes.items())
            junctions = '<junction id="J" type="traffic_light"/><junction id="other_signal" type="traffic_light_right_on_red"/>'
            junctions += '<junction id="stub" type="priority"/><junction id="entry" type="dead_end"/><junction id="exit" type="dead_end"/>'
            connections = ''.join(f'<connection from="{a}" to="{b}" fromLane="0" toLane="0" dir="{d}"/>'
                                  for a, b, d in [("A", "O", "s"), ("O", "I", "t"), ("I", "B", "s"),
                                                  ("A", "L", "s"), ("L", "M", "t"), ("M", "B", "s")])
            path.write_text('<net>'+edges+junctions+connections+'</net>')
            ledger = aggregate_counts([TurnObservation("passenger", 0, 60, "I", "B", 3)], 0, 60)
            candidates = [CandidateRoute("stub_entry", ("I", "B"))]
            old = fit_route_demand(ledger, candidates, network_file=path, boundary_only=True, time_limit_s=2)
            new = fit_route_demand(ledger, candidates, network_file=path, boundary_only=True,
                                   boundary_mode="terminal_turnaround", time_limit_s=2)
            self.assertEqual(old["audit"]["totals"]["under"], 3)
            self.assertEqual(new["audit"]["totals"]["under"], 0)
            self.assertEqual(new["boundary"]["entry_count"], 2)
            self.assertEqual(new["boundary"]["exit_count"], 2)
            self.assertEqual(new["boundary"]["graph_entry_count"], 1)
            self.assertIn("I", new["boundary"]["entry_edges"])
            self.assertIn("O", new["boundary"]["exit_edges"])
            self.assertNotIn("M", new["boundary"]["entry_edges"])
            self.assertNotIn("L", new["boundary"]["exit_edges"])

    def test_unknown_transitions_not_forced_zero_and_unresolved_remains(self):
        ledger = aggregate_counts([TurnObservation("passenger", 0, 60, "A", "B", 5),
                                   TurnObservation("passenger", 0, 60, "U", "V", 17)], 0, 60)
        specs = [{"source_from": "U", "source_to": "V", "path": None}]
        candidates = [CandidateRoute("r", ("A", "B", "X", "Y"))]
        result = fit_route_demand(ledger, candidates, specs, observed_approaches=["X"], time_limit_s=2)
        self.assertEqual(result["allocations"][0]["count"], 5)
        self.assertEqual(result["audit"]["unobserved_planned_transitions"], 5)
        self.assertEqual(result["audit"]["totals"]["under"], 17)
        self.assertEqual(result["audit"]["source"]["source_turn_count"], 22)

    def test_conflicting_passage_counts_reveal_over_and_provenance_weights(self):
        rows = [TurnObservation("bus", 0, 60, "A", "B", 5, provenance="measured"),
                TurnObservation("bus", 0, 60, "B", "C", 2, provenance="estimated")]
        ledger = aggregate_counts(rows, 0, 60)
        result = fit_route_demand(ledger, [CandidateRoute("r", ("A", "B", "C"))], time_limit_s=2)
        self.assertEqual(result["allocations"][0]["count"], 5)
        self.assertEqual(result["audit"]["totals"]["over"], 3)
        self.assertEqual(result["audit"]["totals"]["under"], 0)
        self.assertEqual(result["audit"]["source"]["source_turn_count"], 7)

    def test_loops_are_excluded_and_empty_fit_keeps_source(self):
        ledger = aggregate_counts([TurnObservation("passenger", 0, 60, "A", "B", 5)], 0, 60)
        result = fit_route_demand(ledger, [CandidateRoute("loop", ("A", "B", "A", "B"))], time_limit_s=2)
        self.assertEqual(result["allocations"], [])
        self.assertEqual(result["audit"]["totals"]["under"], 5)
        self.assertEqual(result["rejected_candidates"][0]["reason"], "repeated_edge_limit")

    def test_boundary_only_mode_and_vehicle_lane_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "net.xml"
            edges = ''.join(f'<edge id="{e}" from="{i}" to="{i+1}"><lane id="{e}_0" length="100" speed="10" '
                            f'{"allow=\"passenger\"" if e == "C" else ""}/></edge>' for i, e in enumerate(("A", "B", "C")))
            path.write_text('<net>'+edges+'<connection from="A" to="B" fromLane="0" toLane="0"/>'
                            '<connection from="B" to="C" fromLane="0" toLane="0"/></net>')
            candidates = [CandidateRoute("boundary", ("A", "B", "C")), CandidateRoute("internal", ("B", "C"))]
            ledger = aggregate_counts([TurnObservation("passenger", 0, 60, "B", "C", 3),
                                       TurnObservation("bus", 0, 60, "B", "C", 2)], 0, 60)
            result = fit_route_demand(ledger, candidates, network_file=path, boundary_only=True, time_limit_s=2)
            self.assertEqual(result["allocations"][0]["vehicle_type"], "passenger")
            self.assertEqual(result["allocations"][0]["count"], 3)
            self.assertEqual(result["audit"]["totals"]["under"], 2)
            self.assertEqual(result["audit"]["endpoint_counts"]["internal_origin"], 0)
            self.assertEqual(result["rejected_candidates"][0]["reason"], "internal_endpoint")


if __name__ == "__main__":
    unittest.main()
