"""Exact reward reconstruction, tail discount, and mode identity tests."""
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import explore_weight_episode as ew


class ExactLedgerTest(unittest.TestCase):
    def setUp(self):
        self.vehicles = {"a": (1.25, ["in", "out"], "car"),
                         "b": (6.5, ["in", "out"], "car"),
                         "outside": (-10.0, ["in", "out"], "car")}
        self.rows = {
            "a": {"depart": "3", "arrival": "10", "timeLoss": "4.75", "departDelay": "1.75"},
            "b": {"depart": "12", "arrival": "17", "timeLoss": "1.5", "departDelay": "5.5"},
            "outside": {"depart": "-10", "arrival": "15", "timeLoss": "1000", "departDelay": "0"}}
        self.boundaries = [0., 5., 10., 15., 20.]
        self.live = [{}, {"a": 1.2, "outside": 900}, {}, {"b": .9}, {}]

    def test_final_arrival_loss_and_fractional_departure_wait(self):
        result = ew.reconcile_costs(self.vehicles, self.rows, self.boundaries, self.live,
                                    handover=0, control_s=10)
        np.testing.assert_allclose(result["cumulative"], [0., 2.95, 10., 12.9, 13.5])
        self.assertEqual(result["cohort"], ["a", "b"])
        self.assertAlmostEqual(sum(result["increments"]), 13.5)
        self.assertTrue(result["cost_check"]["passed"])

    def test_vehicle_arriving_on_boundary_is_not_counted_twice(self):
        self.live[2]["a"] = 4.0  # stale subscription must lose to final arrival value
        result = ew.reconcile_costs(self.vehicles, self.rows, self.boundaries, self.live,
                                    handover=0, control_s=10)
        self.assertAlmostEqual(result["cumulative"][2], 10.0)

    def test_unfinished_vehicle_keeps_its_live_loss_and_is_not_complete(self):
        self.rows["b"]["arrival"] = "-1"
        self.rows["b"]["timeLoss"] = "2.0"
        self.live[-1] = {"b": 2.0}
        result = ew.reconcile_costs(self.vehicles, self.rows, self.boundaries, self.live,
                                    handover=0, control_s=10)
        self.assertFalse(result["cost_check"]["passed"])
        self.assertEqual(result["cost_check"]["unfinished"], 1)
        self.assertAlmostEqual(result["reconstructed_total_delay"], 14.0)

    def test_never_inserted_vehicle_accrues_wait_but_missing_record_fails_check(self):
        del self.rows["b"]
        self.live[3] = {}
        result = ew.reconcile_costs(self.vehicles, self.rows, self.boundaries, self.live,
                                    handover=0, control_s=10)
        self.assertFalse(result["cost_check"]["passed"])
        self.assertEqual(result["cost_check"]["missing"], 1)
        self.assertAlmostEqual(result["reconstructed_total_delay"], 6.5 + 13.5)

    def test_missing_live_snapshot_for_an_active_vehicle_is_rejected(self):
        self.live[1] = {}
        with self.assertRaises(ew.CostReconciliationError):
            ew.reconcile_costs(self.vehicles, self.rows, self.boundaries, self.live,
                                handover=0, control_s=10)

    def test_inconsistent_insertion_delay_or_actual_departure_is_rejected(self):
        self.rows["a"]["departDelay"] = "2.0"
        with self.assertRaises(ew.CostReconciliationError):
            ew.reconcile_costs(self.vehicles, self.rows, self.boundaries, self.live,
                                handover=0, control_s=10)
        self.rows["a"]["departDelay"] = "1.75"
        with self.assertRaises(ew.CostReconciliationError):
            ew.reconcile_costs(self.vehicles, self.rows, self.boundaries, self.live,
                                handover=0, control_s=10, actual_departures={"a": 4.0})

    def test_future_arrivals_do_not_supply_loss_before_the_vehicle_exists(self):
        result = ew.reconcile_costs(self.vehicles, self.rows, self.boundaries, self.live,
                                    handover=0, control_s=10)
        self.assertEqual(result["time_loss"][0], 0)
        self.assertEqual(result["time_loss"][1], 1.2)
        self.assertAlmostEqual(result["depart_delay"][2], 1.75 + 3.5)

    def test_cohort_excludes_right_boundary_and_fingerprint_is_order_independent(self):
        self.vehicles["edge"] = (10.0, ["in", "out"], "car")
        self.assertEqual(ew.cohort_vehicle_ids(self.vehicles, 0, 10), ["a", "b"])
        self.assertEqual(ew.cohort_fingerprint(self.vehicles, ["a", "b"]),
                         ew.cohort_fingerprint(self.vehicles, ["b", "a"]))

    def test_sumo_negative_time_loss_is_preserved_without_clipping(self):
        vehicles = {"v": (0., ["in", "out"], "car")}
        rows = {"v": {"depart": "0", "arrival": "10", "timeLoss": "-0.007", "departDelay": "0"}}
        result = ew.reconcile_costs(vehicles, rows, [0., 5., 10.], [{}, {"v": .001}, {}],
                                    handover=0, control_s=10)
        np.testing.assert_allclose(result["increments"], [.001, -.008])
        self.assertAlmostEqual(result["actual_total_delay"], -.007)
        self.assertAlmostEqual(result["reconstructed_total_delay"], -.007)
        self.assertTrue(result["cost_check"]["passed"])
        rewards, _ = ew.fold_tail_rewards(result["increments"], control_steps=2)
        self.assertAlmostEqual(rewards[-1], .000008)

    def test_arrival_rounding_can_reduce_cumulative_cost_without_corrupting_total(self):
        vehicles = {"v": (0., ["in", "out"], "car")}
        rows = {"v": {"depart": "0", "arrival": "10", "timeLoss": "0.666", "departDelay": "0"}}
        result = ew.reconcile_costs(vehicles, rows, [0., 5., 10.], [{}, {"v": .66649}, {}],
                                    handover=0, control_s=10)
        self.assertLess(result["increments"][-1], 0)
        self.assertAlmostEqual(sum(result["increments"]), .666)
        self.assertTrue(result["cost_check"]["passed"])


class TailAndModeTest(unittest.TestCase):
    def test_deadline_prevents_starting_another_simulation_second(self):
        steps = []
        step = ew._deadline_step(lambda: steps.append(1), deadline=10.)
        with patch.object(ew.time, "monotonic", return_value=10.):
            with self.assertRaises(TimeoutError):
                step()
        self.assertEqual(steps, [])

    def test_deadline_detects_expiry_inside_a_simulation_second(self):
        steps = []
        step = ew._deadline_step(lambda: steps.append(1), deadline=10.)
        with patch.object(ew.time, "monotonic", side_effect=[9., 10.]):
            with self.assertRaises(TimeoutError):
                step()
        self.assertEqual(steps, [1])

    def test_unlimited_step_preserves_the_original_callable(self):
        original = lambda: 7
        self.assertIs(ew._deadline_step(original, None), original)
        self.assertEqual(ew._deadline_step(original, None)(), 7)

    def test_tail_is_discounted_from_last_control_step_and_only_folded_once(self):
        rewards, parts = ew.fold_tail_rewards([10, 20, 30, 40], control_steps=2, gamma=.5, scale=10)
        self.assertEqual(rewards, [-1, -4.5])
        self.assertEqual(parts["undiscounted_control_cost"], 30)
        self.assertEqual(parts["tail_cost"], 70)
        self.assertEqual(parts["tail_steps"], 2)
        # Discounted full return equals the two-control-action return.
        self.assertAlmostEqual(rewards[0] + .5 * rewards[1], -1 - .5 * 2 - .25 * 3 - .125 * 4)

    def test_complete_720_control_and_180_tail_rewards(self):
        rewards, parts = ew.fold_tail_rewards([1.] * 900)
        self.assertEqual(len(rewards), 720)
        self.assertEqual(parts["tail_steps"], 180)
        self.assertAlmostEqual(parts["tail_cost"], 180)
        self.assertAlmostEqual(rewards[-1], -.001 - sum(.99 ** (i + 1) * .001 for i in range(180)))
        self.assertEqual(rewards[:-1], [-.001] * 719)

    def test_fixed_duplicate_mode_uses_canonical_mode_but_preserves_physical_green(self):
        snapshot = SimpleNamespace(mode_actions=[2, 1, 1, 3, 2], mode_mask=[True, True, False, True, False])
        self.assertEqual(ew.canonical_fixed_mode(snapshot, 2), 1)
        self.assertEqual(ew.canonical_fixed_mode(snapshot, 4), 0)
        forced = SimpleNamespace(mode_actions=[3] * 5, mode_mask=[True, False, False, False, False])
        self.assertEqual(ew.canonical_fixed_mode(forced, 4), 0)

    def test_invalid_rewards_and_mode_are_rejected(self):
        for costs in ([float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                ew.fold_tail_rewards(costs, control_steps=1)
        with self.assertRaises(ValueError):
            ew.canonical_fixed_mode(SimpleNamespace(), 5)


class EpisodeContractTest(unittest.TestCase):
    """Exercise the full 720-control/180-tail API without launching SUMO."""

    class FakeEpisode:
        instances = []

        def __init__(self, date, period, seed, work, *, deadline=None):
            self.__class__.instances.append(self)
            self.closed = False
            self.handover = 0.
            self.done = False
            self.vehicles = {"a": (0., ["in", "out"], "car")}
            self.movement_of = {"a": ("in", "out")}
            self.route = Path(work) / "routes.xml"
            self.route.parent.mkdir(parents=True, exist_ok=True)
            self.route.write_text("<routes/>")
            self.tripinfo = Path(work) / "tripinfo.xml"
            self.collisions = Path(work) / "collisions.xml"
            self.ts = SimpleNamespace(green_phase=2, green_phases=[SimpleNamespace(state=str(i)) for i in range(4)])
            self.env = SimpleNamespace(sim_step=0.)
            vehicle = SimpleNamespace(
                getIDList=lambda: ["a"] if self.env.sim_step < 4500 else [],
                getTimeLoss=lambda v: self.env.sim_step / 5.,
                getDeparture=lambda v: 0.)
            simulation = SimpleNamespace(getDepartedIDList=lambda: [])
            self.env.sumo = SimpleNamespace(vehicle=vehicle, simulation=simulation)
            self.env._sumo_step = self.one_second
            self.cons = SimpleNamespace(allowed=lambda env: [2], interventions=0, rule_decided={"hold": 900})
            self.ctl = lambda env, allowed: 2

        def one_second(self):
            self.env.sim_step += 1.

        def apply(self, action):
            for _ in range(5):
                self.env._sumo_step()
            self.ts.green_phase = action
            self.done = self.env.sim_step >= 4500
            return action, action

        def close(self):
            self.closed = True

    class FakePlanner:
        def __init__(self, movement_of, greens, end):
            self.end = end

        def snapshot(self, env):
            obs = np.zeros(110, np.float32)
            obs[0] = env.sim_step
            obs[104] = 1  # five-mode mask begins before the last time field
            return SimpleNamespace(observation=obs, allowed=[2], mode_actions=[2] * 5,
                                   mode_mask=[True, False, False, False, False],
                                   mode_disagrees=False, p0_action=2, now=env.sim_step,
                                   action=lambda mode: 2 if mode == 0 else self.fail_mode())

        @staticmethod
        def fail_mode():
            raise ValueError("masked mode")

    def _patches(self):
        core = SimpleNamespace(ModePlanner=self.FakePlanner, OBSERVATION_SIZE=110)
        evaluation = SimpleNamespace(judge_collisions=lambda path: {
            "collisions_at_204820": 0, "collisions_unexplained": []})
        return patch.dict(sys.modules, {"explore_weight_core": core, "focused_eval": evaluation})

    def test_policy_called_for_all_control_steps_and_rewards_reconciled_before_return(self):
        calls = []

        def policy(obs, snapshot):
            calls.append(snapshot.now)
            return 0

        rows = {"a": {"depart": "0", "arrival": "4500", "timeLoss": "900", "departDelay": "0"}}
        base_metrics = {"vehicles": 1, "total_delay": 900., "mean_delay_s": 900., "missing": 0, "unfinished": 0}
        with tempfile.TemporaryDirectory() as td, self._patches(), \
                patch.object(ew, "_Episode", self.FakeEpisode), \
                patch.object(ew, "read_tripinfo", return_value=rows), \
                patch.object(ew, "_exact_metrics", return_value=base_metrics):
            result = ew.run_episode("20220811", "1300", 42, Path(td), policy, collect=True)
            self.assertTrue((Path(td) / "cost_ledger.json").is_file())
        self.assertEqual(calls, list(range(0, 3600, 5)))
        self.assertEqual(len(result.transitions), 720)
        self.assertEqual(len(result.decisions), 720)
        self.assertEqual(result.transitions[0]["mode"], 0)
        self.assertEqual(result.transitions[0]["executed_green"], 2)
        for current, after in zip(result.transitions, result.transitions[1:]):
            np.testing.assert_array_equal(current["next_obs"], after["obs"])
        self.assertTrue(result.transitions[-1]["done"])
        self.assertEqual(sum(t["done"] for t in result.transitions), 1)
        self.assertTrue(result.metrics["cost_check_passed"])
        self.assertEqual(result.metrics["tail_cost"], 180.)
        self.assertEqual(result.metrics["changed_by_rules"], 0)
        self.assertEqual(result.metrics["mode_action_errors"], 0)
        self.assertTrue(self.FakeEpisode.instances[-1].closed)

    def test_timeout_closes_sumo_and_does_not_return_partly_reconciled_transitions(self):
        clock = iter(i * .01 for i in range(10000))
        with tempfile.TemporaryDirectory() as td, self._patches(), \
                patch.object(ew, "_Episode", self.FakeEpisode), \
                patch.object(ew.time, "monotonic", side_effect=lambda: next(clock)):
            with self.assertRaises(TimeoutError):
                ew.run_episode("20220811", "1300", 42, Path(td), 0, collect=True, max_seconds=.2)
        self.assertTrue(self.FakeEpisode.instances[-1].closed)

    def test_planner_initialization_failure_also_closes_sumo(self):
        core = SimpleNamespace(ModePlanner=lambda *_: (_ for _ in ()).throw(ValueError("bad map")),
                               OBSERVATION_SIZE=110)
        evaluation = SimpleNamespace(judge_collisions=lambda path: {})
        with tempfile.TemporaryDirectory() as td, \
                patch.dict(sys.modules, {"explore_weight_core": core, "focused_eval": evaluation}), \
                patch.object(ew, "_Episode", self.FakeEpisode):
            with self.assertRaisesRegex(ValueError, "bad map"):
                ew.run_episode("20220811", "1300", 42, Path(td), 0)
        self.assertTrue(self.FakeEpisode.instances[-1].closed)


if __name__ == "__main__":
    unittest.main()
