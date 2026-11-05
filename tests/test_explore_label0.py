"""[추가 탐색] LABEL0 검사 — SUMO 실행 없음."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import explore_label0 as lb  # noqa: E402
import explore_weight as w  # noqa: E402

w.configure()
CFG = lb.config()


def point():
    sample = lambda delay, residual: {"delay": delay, "residual": residual, "cost": round(delay + 75.0 * residual, 3), "future_seed": 1}  # noqa: E731
    return {"state": "s", "rollouts": {"1": [sample(100.5, 2), None], "3": [sample(200.25, 0), sample(50.0, 4)]}}


class RelabelTest(unittest.TestCase):
    def test_weight_75_keeps_the_stored_cost(self):
        p = point()
        self.assertEqual(lb.relabel_point(p, 75.0), p)

    def test_weight_changes_only_cost(self):
        p = point()
        out = lb.relabel_point(p, 0.0)
        self.assertEqual(out["rollouts"]["1"][0]["cost"], 100.5)
        self.assertEqual(out["rollouts"]["3"][1]["cost"], 50.0)
        self.assertIsNone(out["rollouts"]["1"][1])
        self.assertEqual(out["rollouts"]["3"][1]["residual"], 4)
        out = lb.relabel_point(p, 25.0)
        self.assertEqual(out["rollouts"]["3"][1]["cost"], 150.0)
        self.assertEqual(p["rollouts"]["3"][1]["cost"], 350.0, "원본을 바꾸면 안 된다")

    def test_stops_when_stored_cost_is_not_the_formula(self):
        p = point()
        p["rollouts"]["1"][0]["cost"] += 1.0
        with self.assertRaises(ValueError):
            lb.relabel_point(p, 0.0)

    def test_labelled_restores_the_module(self):
        class Fake:
            @staticmethod
            def point_to_row(point, date, situation, cfg):
                return point["rollouts"]["3"][0]["cost"]
        before = Fake.point_to_row
        with lb.labelled(Fake, 0.0):
            self.assertEqual(Fake.point_to_row(point(), "d", "s", {}), 200.25)
        self.assertIs(Fake.point_to_row, before)


class MacRecordTest(unittest.TestCase):
    A = {"kind": "k", "lock_sha256": "l", "environment_fingerprint": "e", "files": {"a": "1"},
         "packages": {"numpy": "2.5.3", "torch": "2.14.0", "eclipse-sumo": "1.27.1"}}

    def test_only_numpy_and_torch_may_differ(self):
        b = {**self.A, "packages": {**self.A["packages"], "numpy": "2.1.3", "torch": "2.11.0+cpu"}}
        self.assertTrue(lb.differs_only_in_packages(self.A, b))

    def test_sumo_or_code_must_match(self):
        self.assertFalse(lb.differs_only_in_packages(self.A, {**self.A, "packages": {**self.A["packages"], "eclipse-sumo": "1.26"}}))
        self.assertFalse(lb.differs_only_in_packages(self.A, {**self.A, "files": {"a": "2"}}))
        self.assertFalse(lb.differs_only_in_packages(self.A, {**self.A, "environment_fingerprint": "x"}))


class ConfigTest(unittest.TestCase):
    def test_four_variants_in_two_worlds(self):
        self.assertEqual(set(CFG["variants"]), {f"{x}_{k}" for x in "PR" for k in CFG["weights"]})
        for name, spec in CFG["variants"].items():
            self.assertEqual(spec["weight"], CFG["weights"][name.split("_")[1]])
            self.assertEqual(spec["world"], name[0])
        self.assertNotIn(75.0, CFG["weights"].values())

    def test_worlds_use_their_own_training_settings(self):
        module, cfg, prefix = lb.world_parts("P")
        self.assertEqual((module.__name__, prefix), ("explore_platoon1_learn", "PT1"))
        module2, cfg2, prefix2 = lb.world_parts("R")
        self.assertEqual((module2.__name__, prefix2), ("arena3_pi_learn", "R1A"))
        for c in (cfg, cfg2):
            self.assertEqual(c["student"]["seeds"], [42, 43, 44])
            self.assertEqual(c["rollout"]["terminal_s"], lb.STORED_WEIGHT)
        self.assertEqual(cfg2["variants"]["A"]["name"], "R1A")


class ReadingTest(unittest.TestCase):
    RULES = CFG["rules"]

    def diff(self, mean, low, high, dates=7):
        return {"mean_s": mean, "t95_s": [low, high], "positive_dates": dates}

    def test_effect_needs_size_interval_and_dates(self):
        self.assertTrue(lb.effect_verdict(self.diff(0.3, 0.01, 0.6), self.RULES))
        self.assertFalse(lb.effect_verdict(self.diff(0.29, 0.01, 0.6), self.RULES))
        self.assertFalse(lb.effect_verdict(self.diff(1.0, -0.01, 2.0), self.RULES))
        self.assertFalse(lb.effect_verdict(self.diff(1.0, 0.1, 2.0, dates=4), self.RULES))

    def test_learning_reading(self):
        self.assertEqual(lb.read_learning(self.diff(1, 0.1, 2)), "출발 정책보다 짧다")
        self.assertEqual(lb.read_learning(self.diff(-1, -2, -0.1)), "출발 정책보다 길다")
        self.assertEqual(lb.read_learning(self.diff(0, -1, 1)), "구분 못 함")

    def test_overall_reading(self):
        weights = ["w0", "w25"]
        deltas = {"P": {"w0": 1.0, "w25": 2.0}, "R": {"w0": 1.0, "w25": 0.5}}
        both = {"P": {"w0": True, "w25": True}, "R": {"w0": True, "w25": True}}
        self.assertEqual(lb.overall_reading(both, deltas, weights)["weight"], "w25")      # 평균 1.25 > 1.0
        self.assertEqual(lb.overall_reading(both, deltas, weights)["reading"], "A")
        split = {"P": {"w0": True, "w25": False}, "R": {"w0": False, "w25": True}}
        self.assertEqual(lb.overall_reading(split, deltas, weights)["reading"], "B")
        none = {"P": {"w0": False, "w25": False}, "R": {"w0": False, "w25": False}}
        self.assertEqual(lb.overall_reading(none, deltas, weights)["reading"], "C")


if __name__ == "__main__":
    unittest.main()
