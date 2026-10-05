"""Verify passage-to-boundary timing, denominator preservation and dispatch separation."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from network_demand import CandidateRoute, TurnObservation, aggregate_counts
from network_temporal import (dispatch_nominal_audit, fit_temporal_demand, overlap_fraction,
                              round_dispatch_allocations)


class TemporalChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.network = Path(self.temp.name) / "network.xml"

    def tearDown(self):
        self.temp.cleanup()

    def make_network(self, length=9000, bus_allowed=True):
        edges = ''.join(f'<edge id="{edge}" from="{i}" to="{i+1}"><lane id="{edge}_0" speed="10" length="{length}"'
                        f'{" allow=\"passenger\"" if not bus_allowed and edge == "C" else ""}/></edge>'
                        for i, edge in enumerate(("A", "B", "C")))
        junctions = '<junction id="0" type="dead_end"/><junction id="1" type="traffic_light"/>'
        junctions += '<junction id="2" type="traffic_light"/><junction id="3" type="dead_end"/>'
        links = '<connection from="A" to="B" fromLane="0" toLane="0" dir="s"/>'
        links += '<connection from="B" to="C" fromLane="0" toLane="0" dir="s"/>'
        self.network.write_text('<net>'+edges+junctions+links+'</net>')

    def ledger(self, index, entries):
        begin, end = index * 900, (index + 1) * 900
        rows = [TurnObservation(kind, begin, end, a, b, count) for kind, a, b, count in entries]
        return aggregate_counts(rows, begin, end)

    def test_uniform_kernel_conserves_mass_across_windows(self):
        masses = [overlap_fraction(0, 900, 450, begin, begin + 900) for begin in (0, 900)]
        self.assertEqual(masses, [0.5, 0.5])
        self.assertEqual(sum(masses), 1)

    def test_later_passage_infers_earlier_boundary_departure(self):
        self.make_network()
        ledgers = [self.ledger(0, [("passenger", "A", "B", 0)]),
                   self.ledger(1, [("passenger", "A", "B", 3)])]
        # C is necessary to keep the route's destination at the model boundary.
        fit = fit_temporal_demand(ledgers, [CandidateRoute("r", ("A", "B", "C"))], [], self.network,
                                  max_route_length_m=None, time_limit_s=2)
        self.assertEqual(fit["source_total"], 3)
        self.assertAlmostEqual(fit["LPexpectedunder"], 0)
        self.assertAlmostEqual(fit["LPexpectedover"], 0)
        self.assertEqual(fit["allocations"][0]["cohort_start"], 0)
        self.assertEqual(fit["allocations"][0]["count"], 3)
        self.assertEqual(fit["rounddispatch_nominalunder"], 0)
        self.assertFalse(fit["learning_ready"])

    def test_one_vehicle_satisfies_successive_movements_in_different_windows(self):
        self.make_network()
        ledgers = [self.ledger(0, []), self.ledger(1, [("bus", "A", "B", 3)]),
                   self.ledger(2, [("bus", "B", "C", 3)])]
        fit = fit_temporal_demand(ledgers, [CandidateRoute("r", ("A", "B", "C"))], [], self.network,
                                  max_route_length_m=None, time_limit_s=2)
        self.assertEqual(fit["source_total"], 6)
        self.assertEqual(sum(a["count"] for a in fit["allocations"]), 3)
        self.assertAlmostEqual(fit["LPexpectedunder"], 0)
        self.assertEqual(fit["rounddispatch_nominalunder"], 0)
        self.assertEqual(fit["cohort_count"], 3)

    def test_unresolved_and_after_horizon_counts_are_not_dropped(self):
        self.make_network(length=100)
        ledger = self.ledger(0, [("passenger", "A", "B", 2), ("passenger", "U", "V", 17)])
        specs = [{"source_from": "U", "source_to": "V", "path": None}]
        fit = fit_temporal_demand([ledger], [CandidateRoute("r", ("A", "B", "C"))], specs, self.network,
                                  time_limit_s=2)
        self.assertEqual(fit["source_total"], 19)
        self.assertEqual(fit["unresolved_source_count"], 17)
        self.assertAlmostEqual(fit["LPexpectedunder"], 17)
        self.assertGreater(fit["LP_expected"]["outside_horizon"]["after_horizon"], 0)
        self.assertEqual(fit["cohort_starts"], [0])
        self.assertFalse(fit["automatic_unobserved_warmup_or_tail_cohorts"])

    def test_explicit_zero_is_recorded_and_absence_has_no_zero_row(self):
        self.make_network(length=100)
        ledgers = [self.ledger(0, [("passenger", "A", "B", 0)]),
                   self.ledger(1, [("passenger", "A", "B", 1)])]
        fit = fit_temporal_demand(ledgers, [CandidateRoute("r", ("A", "B", "C"))], [], self.network,
                                  time_limit_s=2)
        rows = fit["LP_expected"]["by_movement"]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["source_state"], "explicit_zero")
        self.assertNotIn(("B", "C"), [(r["source_from"], r["source_to"]) for r in rows])
        self.assertFalse(fit["source_absence_is_zero"])

    def test_rounding_preserves_rounded_class_total_and_not_fractional_expectation(self):
        expected = [{"vehicle_type": "passenger", "route_id": "a", "cohort_start": 0, "expected_count": 1.6},
                    {"vehicle_type": "passenger", "route_id": "b", "cohort_start": 900, "expected_count": .6},
                    {"vehicle_type": "bus", "route_id": "c", "cohort_start": 0, "expected_count": .7}]
        rounded = round_dispatch_allocations(expected)
        self.assertEqual(sum(r["count"] for r in rounded if r["vehicle_type"] == "passenger"), 2)
        self.assertEqual(sum(r["count"] for r in rounded if r["vehicle_type"] == "bus"), 1)
        self.assertTrue(all(isinstance(r["count"], int) for r in rounded))

    def test_dispatch_uses_the_same_two_decimal_departure_clock_as_xml(self):
        self.make_network(length=.01)
        ledgers = [self.ledger(0, [("passenger", "A", "B", 1)]),
                   self.ledger(1, [("passenger", "A", "B", 0)])]
        allocations = [{"vehicle_type": "passenger", "route_id": "r", "edges": ["A", "B", "C"],
                        "cohort_start": 0, "count": 1}]
        class NearBoundaryRandom:
            def __init__(self, seed): pass
            def uniform(self, a, b): return 899.996
        with patch('network_temporal.random.Random', NearBoundaryRandom):
            report = dispatch_nominal_audit(ledgers, allocations, [], self.network)
        # XML rounds depart to 900.00; nominal completion belongs to window 1.
        rows = report['by_movement']
        self.assertEqual(rows[0]['nominal_dispatch_passages'], 0)
        self.assertEqual(rows[1]['nominal_dispatch_passages'], 1)

    def test_lane_permission_filter_matches_static_engine(self):
        self.make_network(length=100, bus_allowed=False)
        ledger = self.ledger(0, [("bus", "A", "B", 2)])
        fit = fit_temporal_demand([ledger], [CandidateRoute("r", ("A", "B", "C"))], [], self.network,
                                  time_limit_s=2)
        self.assertEqual(fit["allocations"], [])
        self.assertEqual(fit["LPexpectedunder"], 2)
        self.assertEqual(fit["source_total"], 2)

    def test_absent_registered_complete_movement_is_reported(self):
        self.make_network(length=100)
        ledger = self.ledger(0, [("passenger", "A", "B", 2)])
        specs = [{"source_from": "X", "source_to": "Y", "path": ["B", "C"]}]
        fit = fit_temporal_demand([ledger], [CandidateRoute("r", ("A", "B", "C"))], specs, self.network,
                                  time_limit_s=2)
        self.assertGreater(fit["LP_expected"]["unrecorded_registered_movements"], 0)
        self.assertEqual(fit["source_total"], 2)

    def test_noncontiguous_windows_and_source_record_mismatch_are_rejected(self):
        self.make_network(length=100)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            fit_temporal_demand([self.ledger(0, []), self.ledger(2, [])], [], [], self.network)
        ledger = self.ledger(0, [("passenger", "A", "B", 2)])
        ledger.counts[("passenger", "A", "B")] = 3
        with self.assertRaisesRegex(ValueError, "records"):
            fit_temporal_demand([ledger], [], [], self.network)

    def test_epoch_windows_reject_one_second_gap_or_overlap(self):
        self.make_network(length=100)
        epoch = 1660176000.0
        first = aggregate_counts([], epoch, epoch + 900)
        # Relative tolerance at epoch scale would accept both discontinuities.
        for offset in (-1.0, 1.0):
            with self.subTest(offset=offset):
                second = aggregate_counts([], epoch + 900 + offset, epoch + 1800 + offset)
                with self.assertRaisesRegex(ValueError, "contiguous"):
                    fit_temporal_demand([first, second], [], [], self.network)


if __name__ == "__main__":
    unittest.main()
