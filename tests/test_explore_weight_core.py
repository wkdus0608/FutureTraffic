"""WEIGHT1 planner and observation contracts; no SUMO process is launched."""
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import explore_pred as ep
import explore_weight_core as ew
from focused_rl import CROSSWALKS, SignalConstraints


def signal_model():
    guard = SignalConstraints()
    guard.phase_min = [15, 33, 15, 46]
    guard.serving = {tuple(CROSSWALKS)[0]: [3], tuple(CROSSWALKS)[1]: [1]}
    guard.base_yellow = 4
    guard.all_red = {(0, 2): 2, (1, 0): 1, (1, 2): 2, (1, 3): 1, (2, 0): 2, (3, 1): 1}
    return ep.SignalModel(served=[[(0, .5)], [(1, 1.5), (2, 1.0)], [(3, .5)], [(4, 1), (5, 1)]],
                          phase_min=guard.phase_min, max_green=120, step=5, guard=guard,
                          crosswalks=tuple(CROSSWALKS))


class WeightedPlanTest(unittest.TestCase):
    def test_mode_zero_delegates_original_without_numeric_changes(self):
        m = signal_model()
        st = ep.SignalState(2, 940.0, 4.0, (990.0, 990.0))
        q = [4, 12, 8, 7, 20, 13]
        arr = [[0.0] * 6 for _ in range(24)]
        expected = ep.plan(m, st, 1000.0, q, arr, [0, 1, 2, 3])
        with patch.object(ew.ep, "plan", wraps=ep.plan) as call:
            actual = ew.weighted_plan(m, st, 1000.0, q, arr, [0, 1, 2, 3], [1] * 6)
        self.assertEqual(actual, expected)
        call.assert_called_once()

    def test_weighted_running_and_terminal_cost(self):
        guard = SimpleNamespace(serving={}, intergreen=lambda *_: 0, _outcome=lambda *_: -1)
        m = ep.SignalModel(served=[[]], phase_min=[1], max_green=120, step=5, guard=guard, crosswalks=())
        a, info = ew.weighted_plan(m, ep.SignalState(0, 0, 0, ()), 10, [2, 3], [[0, 0]], [0], [2, 4])
        self.assertEqual(a, 0)
        self.assertEqual(info["cost"], (2 * 2 + 4 * 3) * (5 + 75))

    def test_positive_weights_pruned_and_exhaustive_match(self):
        rng, m = random.Random(20261025), signal_model()
        for trial in range(24):
            st = ep.SignalState(rng.randrange(4), 1000 - rng.choice([20, 40, 60, 90]), 4,
                                tuple(1000 - rng.uniform(0, 90) for _ in CROSSWALKS))
            q = [rng.uniform(0, 25) for _ in range(6)]
            arr = [[rng.uniform(0, 1.5) for _ in range(6)] for _ in range(9)]
            weights = [rng.choice([1.0, 1.5, 2.0]) for _ in q]
            allowed = m.actions(st, 1000.0)
            a, i = ew.weighted_plan(m, st, 1000.0, q, arr, allowed, weights)
            b, j = ew.weighted_plan(m, st, 1000.0, q, arr, allowed, weights, merge=False)
            self.assertEqual(a, b, trial)
            self.assertAlmostEqual(i["cost"], j["cost"], places=6, msg=str(trial))
            self.assertLessEqual(i["nodes"], j["nodes"])

    def test_nonpositive_nonfinite_or_misaligned_weights_rejected(self):
        m = signal_model()
        st = ep.SignalState(0, 940, 4, (990, 990))
        for w in ([1] * 5, [0] + [1] * 5, [-1] + [1] * 5,
                  [float("nan")] + [1] * 5, [float("inf")] + [1] * 5):
            with self.assertRaises(ValueError):
                ew.weighted_plan(m, st, 1000, [1] * 6, [[0] * 6], [0], w)

    def test_left_weights_follow_movement_identity_not_order(self):
        east, south = ew.LEFTS.values()
        moves = [("x", "y"), south, east]
        self.assertEqual(ew.movement_weights(0, moves), (1, 1, 1))
        self.assertEqual(ew.movement_weights(1, moves), (1, 1, 1.5))
        self.assertEqual(ew.movement_weights(2, moves), (1, 1.5, 1))
        self.assertEqual(ew.movement_weights(3, moves), (1, 1.5, 1.5))
        self.assertEqual(ew.movement_weights(4, moves), (1, 2, 2))


class ObservationTest(unittest.TestCase):
    def test_mode_aliases_and_forced_states_use_smallest_mode(self):
        self.assertEqual(ew.canonical_mode_mask([2, 2, 3, 1, 3]), [True, False, True, True, False])
        self.assertEqual(ew.canonical_mode_mask([3] * 5, forced=True), [True, False, False, False, False])
        with self.assertRaises(ValueError):
            ew.canonical_mode_mask([0, 1, 0, 0, 0], forced=True)

    def test_waiting_time_includes_road_served_pending_and_threshold_neighbours(self):
        moves = [ew.LEFTS["left_east"], ew.LEFTS["left_south"], ("a", "b"), ("c", "d")]
        movement_of = dict(zip(["below", "at", "pending", "over"], moves))
        depart = {"below": 10, "at": 30, "pending": 166, "over": 200}
        road_wait = {"below": 154, "at": 135, "over": 200}
        sumo = SimpleNamespace(
            edge=SimpleNamespace(getLastStepVehicleIDs=lambda e: ["below", "at", "over"]),
            vehicle=SimpleNamespace(getDepartDelay=lambda v: depart[v],
                                    getAccumulatedWaitingTime=lambda v: road_wait[v]),
            simulation=SimpleNamespace(getPendingVehicles=lambda: ["pending"]))
        # Repeated edge membership cannot double count a vehicle; served vehicles are retained.
        actual = ew.longest_wait_features(sumo, movement_of, moves, ["edge1", "edge2"])
        np.testing.assert_allclose(actual, [164 / 165, 1, 166 / 165, 2])

    def test_layout_cost_normalisation_and_mask_are_frozen(self):
        base = [0.] * 64
        actions = [2, 2, 3, 1, 3]
        mask = ew.canonical_mode_mask(actions)
        obs = ew.observation_vector(base, [.2] * 8, [1.1] * 8,
                                    [None, 100, 1100, 20100], actions, mask, .5)
        self.assertEqual(obs.shape, (110,))
        np.testing.assert_array_equal(obs[80:84], [0, 0, 1, 10])
        np.testing.assert_array_equal(obs[84:104].reshape(5, 4).argmax(axis=1), actions)
        np.testing.assert_array_equal(obs[ew.MODE_MASK_SLICE], mask)
        self.assertEqual(obs[-1], .5)
        low, high = ew.observation_bounds()
        self.assertTrue(np.all(obs >= low) and np.all(obs <= high))

    def test_snapshot_rejects_duplicate_mode_not_green_identifier(self):
        actions = [2, 2, 3, 1, 3]
        snap = ew.ModeSnapshot(np.zeros(110), [1, 2, 3], actions,
                               ew.canonical_mode_mask(actions), 2, [None, 1, 0, 3], True, 1000)
        self.assertEqual(snap.action(0), 2)
        self.assertEqual(snap.action(2), 3)
        self.assertEqual(snap.action(3), 1)
        for mode in (1, 4, 5, -1, 1.0, True):
            with self.assertRaises(ValueError):
                snap.action(mode)

    def test_observation_rejects_mixed_wrappers_and_nonfinite_cost(self):
        for base, cost in (([0] * 72, [0, 1, 2, 3]), ([0] * 64, [0, float("nan"), None, None])):
            with self.assertRaises(ValueError):
                ew.observation_vector(base, [0] * 8, [0] * 8, cost, [0] * 5,
                                      [True, False, False, False, False], 1)

    def test_complete_snapshot_matches_p0_caches_and_handles_forced_green(self):
        m = signal_model()
        moves = [*ew.LEFTS.values(), *[(f"approach{i}", f"exit{i}") for i in range(6)]]
        guard = m.guard
        guard.allowed = lambda env: env.allowed
        guard.reset(dict.fromkeys(m.crosswalks, 990.0))

        class BaseObservation:
            _capacity = dict.fromkeys((move[0] for move in moves), 20.0)
            calls = 0

            def __call__(self):
                self.calls += 1
                values = np.zeros(64, dtype=np.float32)
                values[0] = 1
                values[-4:] = [a in env.allowed for a in range(4)]
                return values

        observation = BaseObservation()
        ts = SimpleNamespace(green_phase=0, time_since_last_phase_change=60,
                             yellow_time=4, is_yellow=False, observation_fn=observation)
        sumo = SimpleNamespace(
            edge=SimpleNamespace(getLastStepVehicleIDs=lambda e: ["east_car"]),
            vehicle=SimpleNamespace(getDepartDelay=lambda v: 20,
                                    getAccumulatedWaitingTime=lambda v: 145),
            simulation=SimpleNamespace(getPendingVehicles=lambda: []))
        env = SimpleNamespace(sim_step=1000.0, delta_time=5,
                              traffic_signals={"204820": ts}, signal_constraints=guard,
                              sumo=sumo, allowed=[0, 1, 2, 3])
        q, arr = [2, 6, 8, 3, 12, 20, 4, 1], [[0.] * 8 for _ in range(24)]

        class FakeP0:
            def __init__(self, *args):
                self._model, self.movements = m, moves

            def queues_and_arrivals(self, env):
                return q, arr

        with patch.object(ew.ep, "controller_class", return_value=FakeP0):
            planner = ew.ModePlanner({"east_car": ew.LEFTS["left_east"]}, [""] * 4, 4600)
        snap = planner.snapshot(env)
        state = ep.SignalState(0, 940, 4, tuple(guard._served_now(ts, 1000)[c] for c in m.crosswalks))
        expected_action, expected_info = ep.plan(m, state, 1000, q, arr, env.allowed)
        self.assertEqual(snap.p0_action, expected_action)
        self.assertEqual(snap.p0_info, expected_info)
        self.assertEqual(snap.action(0), expected_action)
        self.assertEqual(snap.observation.shape, (110,))
        self.assertEqual(snap.observation[72], 1.0)
        self.assertIs(snap, planner.snapshot(env))
        self.assertEqual(observation.calls, 1)
        env.sim_step, env.allowed = 1005, [3]
        forced = planner.snapshot(env)
        self.assertEqual(forced.mode_actions, [3] * 5)
        self.assertEqual(forced.mode_mask, [True, False, False, False, False])
        self.assertEqual(forced.action(0), 3)
        self.assertFalse(forced.mode_disagrees)
        self.assertEqual(observation.calls, 2)


if __name__ == "__main__":
    unittest.main()
