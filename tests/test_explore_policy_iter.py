"""PI1의 표본 선택·확인 분리·한 차례 학습·중단 판정(실제 SUMO 없음)."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import explore_policy_iter as pi
import explore_combo as cb


class SamplingTest(unittest.TestCase):
    def test_spaced_endpoints_and_no_duplicate(self):
        self.assertEqual(pi.spaced_indices(0), [])
        self.assertEqual(pi.spaced_indices(3), [0, 1, 2])
        got = pi.spaced_indices(70)
        self.assertEqual(len(got), 16)
        self.assertEqual((got[0], got[-1]), (0, 69))
        self.assertEqual(got, sorted(set(got)))

    def test_ties_keep_parent_then_p0_then_number(self):
        costs = {0: [2, 2], 1: [2, 2], 2: [2, 2]}
        self.assertEqual(pi.choose(costs, 2, 1), 2)
        self.assertEqual(pi.choose(costs, 3, 1), 1)
        self.assertEqual(pi.choose(costs, 3, 3), 0)

    def test_independent_future_sets(self):
        self.assertFalse(set(pi.SELECT_SEEDS) & set(pi.CONFIRM_SEEDS))
        self.assertEqual((len(pi.SELECT_SEEDS), len(pi.CONFIRM_SEEDS)), (8, 8))

    def test_pilot_requires_changes_and_two_groups(self):
        rows = [{"group": f"{d}_{p}", "parent": 0, "chosen": 1, "confirmed_gain": 1.0}
                for d, p in pi.TRAIN_GROUPS for _ in range(2)]
        self.assertTrue(pi.pilot_reading(rows)["passed"])
        for r in rows:
            r["chosen"] = 0
        self.assertFalse(pi.pilot_reading(rows)["passed"])
        for i, r in enumerate(rows):
            r.update(chosen=1, confirmed_gain=10 if i < 2 else -1)
        self.assertFalse(pi.pilot_reading(rows)["passed"])


class BudgetTest(unittest.TestCase):
    def test_completed_work_not_counted_twice_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = pi.Budget(Path(tmp), 100)
            b.add("same", 30)
            b.add("same", 70)
            self.assertEqual(pi.Budget(Path(tmp), 100).used, 30)
            with self.assertRaises(pi.BudgetExceeded):
                b.check(reserve=71)

    def test_cache_rejects_changed_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            row = {"group": "g", "t0": 10}
            key = pi.rollout_key(row, 1, 3001, "parent")
            pi.write_json(out / "rollouts" / f"{key}.json", {"experiment": "same", "parent_hash": "old",
                          "candidate": 1, "seed": 3001, "continuation": "parent", "cpu_s": 5})
            fake = type("Parent", (), {"hash": "new"})()
            with patch.object(pi.pr, "FrozenPolicy", return_value=fake):
                with self.assertRaises(RuntimeError):
                    pi.run_jobs([(row, 1, 3001, "parent", "parent")], out, 1, pi.Budget(out, 100), "same")

    def test_cached_smoke_restores_cpu_without_reserving_new_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            pi.write_json(out / "check.json", {"experiment": "same", "parent_hash": "p", "passed": True, "cpu_s": 99})
            fake = type("Parent", (), {"hash": "p"})()
            budget = pi.Budget(out, 100)
            with patch.object(pi.pr, "FrozenPolicy", return_value=fake):
                result = pi.smoke_check([], out, "same", budget)
            self.assertTrue(result["passed"])
            self.assertEqual(budget.used, 99)

    def test_failed_restore_blocks_all_rollouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            fake = type("Parent", (), {"hash": "p"})()
            row = {"state_stem": "unused", "route_dir": "unused"}
            with patch.object(pi.pr, "FrozenPolicy", return_value=fake), \
                 patch.object(pi.pr, "verify_snapshot", return_value={"ok": False, "problems": ["obs"]}), \
                 patch.object(pi.pr, "measured") as measured:
                with self.assertRaisesRegex(RuntimeError, "복원 불일치"):
                    pi.smoke_check([row], out, "same", pi.Budget(out, 2000))
                measured.assert_not_called()


class LearningTest(unittest.TestCase):
    def test_json_cost_keys_learn_new_parent_target_and_do_not_change_parent(self):
        import torch as th
        th.set_num_threads(1)
        th.manual_seed(3)
        net = cb.make_net(96)
        with th.no_grad():
            for parameter in net.parameters():
                parameter.zero_()
            net[-1].bias[0] = -0.1
        initial = {k: v.detach().clone() for k, v in net.state_dict().items()}
        untouched = copy.deepcopy(initial)
        x = np.zeros(96, np.float32)
        x[-4:-2] = 1
        sample = json.loads(json.dumps({"obs": x.tolist(), "allowed": [0, 1], "p0": 0,
                                       "costs": {0: [3000.0] * 8, 1: [1000.0] * 8},
                                       "confirm_seeds": list(pi.CONFIRM_SEEDS), "confirmed_gain": -1e9}))
        states = pi.improve([initial], [sample], np.stack([x]), steps=1000)
        net.load_state_dict(states[0])
        with th.no_grad():
            prediction = net(th.as_tensor(x)).numpy()
        self.assertLess(prediction[1], prediction[0])
        for k in initial:
            self.assertTrue(th.equal(initial[k], untouched[k]))
        changed_confirmation = {**sample, "confirmed_gain": 1e9, "confirm_seeds": [999999]}
        other = pi.improve([initial], [changed_confirmation], np.stack([x]), steps=1000)
        for k in states[0]:
            self.assertTrue(th.equal(states[0][k], other[0][k]))


class ReadingTest(unittest.TestCase):
    def rows(self, gain=0.8, left_increase=0):
        return [{"group": f"g{g}", "seed": seed,
                 "parent": {"mean_delay_s": 40, "left_east": 100, "left_south": 100, "long_waits": 0},
                 "p0": {"mean_delay_s": 39.3, "left_east": 100, "left_south": 100, "long_waits": 0},
                 "child": {"mean_delay_s": 40 - gain, "left_east": 100 + left_increase,
                           "left_south": 100, "long_waits": 0}}
                for g in range(4) for seed in range(4)]

    def test_parent_improvement_and_p0_success_are_separate(self):
        rows = self.rows()
        self.assertTrue(pi.final_comparison(rows, "parent", 0.5)["passed"])
        self.assertFalse(pi.final_comparison(rows, "p0", 1.3)["passed"])

    def test_left_harm_or_long_wait_blocks_pass(self):
        rows = self.rows(gain=3, left_increase=6)
        self.assertFalse(pi.final_comparison(rows, "parent", 0.5)["passed"])
        rows = self.rows(gain=3)
        rows[0]["child"]["long_waits"] = 1
        self.assertFalse(pi.final_comparison(rows, "parent", 0.5)["passed"])

    def test_four_group_interval_does_not_treat_repeated_seeds_as_new_dates(self):
        rows = self.rows(gain=1)
        for r in rows:
            if r["group"] == "g0":
                r["child"]["mean_delay_s"] = 44
        comparison = pi.final_comparison(rows, "parent", 0.5)
        self.assertEqual(len(comparison["group_means"]), 4)
        self.assertFalse(comparison["passed"])


if __name__ == "__main__":
    unittest.main()
