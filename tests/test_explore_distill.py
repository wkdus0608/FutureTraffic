"""[추가 탐색] 2단계 첫걸음(126절) 검사 — 가짜 탐색 기록으로 목표·규칙·평가·분할을 본다(SUMO 없음)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import explore_distill as ed  # noqa: E402


def _sample(group="g1", p0=0, search=1, costs=None, allowed=(0, 1), le=0, ls=0, les=False, lss=False, obs=None):
    costs = costs or {0: [100.0] * 8, 1: [90.0] * 8}
    return {"group": group, "seed": 42, "t0": 0.0, "obs": np.zeros(72, dtype=np.float32) if obs is None else obs,
            "allowed": list(allowed), "p0": p0, "search": search, "costs": costs,
            "left_east": le, "left_south": ls, "left_east_served": les, "left_south_served": lss}


class TargetTest(unittest.TestCase):
    def test_targets_relative_and_masked(self):
        y = ed.targets(_sample(costs={0: [1100.0] * 8, 2: [900.0] * 8}, allowed=(0, 2)))
        self.assertAlmostEqual(y[0], 0.1)
        self.assertAlmostEqual(y[2], -0.1)
        self.assertTrue(np.isnan(y[1]) and np.isnan(y[3]))

    def test_ambiguous(self):
        noisy = _sample(costs={0: [100, 300, 100, 300, 100, 300, 100, 300], 1: [110, 280, 120, 290, 90, 310, 100, 300]})
        self.assertTrue(ed.ambiguous(noisy))                                   # 차이가 흔들림보다 작음
        clear = _sample(costs={0: [100 + i for i in range(8)], 1: [50 + i for i in range(8)]})
        clear["costs"][1][0] = 51.0
        self.assertFalse(ed.ambiguous(clear))
        self.assertFalse(ed.ambiguous(_sample(search=0)))                      # 탐색이 P0와 같으면 아님


class RuleTest(unittest.TestCase):
    def test_rules(self):
        s = _sample(allowed=(0, 1, 3), p0=1)
        self.assertEqual(ed.rule_r(np.array([0.2, 0.1, 9, 0.0]), s), 3)        # P0보다 작은 것 가운데 가장 작은 것
        self.assertEqual(ed.rule_r(np.array([0.2, 0.1, -9, 0.3]), s), 1)       # 허용 안 된 녹색(2)은 고르지 않음
        self.assertEqual(ed.rule_f(np.array([0.1, 0.1, 9, 0.2]), s), 1)        # 같으면 P0

    def test_left_first(self):
        served_by = ed.left_greens()
        self.assertEqual(set(served_by), {"left_east", "left_south"})
        g_east, g_south = served_by["left_east"][0], served_by["left_south"][0]
        self.assertNotEqual(g_east, g_south)
        s = _sample(allowed=(0, 1, 2, 3), p0=1, le=9, ls=12)
        self.assertEqual(ed.left_first(s, served_by), g_south)                 # 둘 다면 기다리는 차가 많은 쪽
        s = _sample(allowed=tuple(g for g in range(4) if g != g_south), p0=1, le=9, ls=12)
        self.assertEqual(ed.left_first(s, served_by), g_east)                  # 많은 쪽이 허용 안 되면 다른 쪽
        self.assertEqual(ed.left_first(_sample(allowed=(0, 1, 2, 3), p0=1, le=7, ls=6), served_by), 1)   # 조건 미달 → P0
        self.assertEqual(ed.left_first(_sample(allowed=(0, 1, 2, 3), p0=1, le=9, les=True), served_by), 1)  # 지금 녹색이면 아님


class EvaluateTest(unittest.TestCase):
    def test_metrics(self):
        a = _sample(p0=0, search=1)                                            # 탐색이 다름(10 이득)
        b = _sample(p0=0, search=0, costs={0: [100.0] * 8, 1: [130.0] * 8})    # 탐색이 P0와 같음
        e = ed.evaluate([a, b], [1, 1])                                        # a는 맞힘, b는 불필요하게 바꿈(+30)
        self.assertEqual((e["gain"], e["search_gain"], e["share"]), (-20.0, 10.0, -2.0))
        self.assertEqual((e["same_as_search_when_differs"], e["needless_changes"], e["needless_cost"]), (1, 1, 30.0))
        self.assertEqual(e["wrong_all"]["n"], 1)
        self.assertEqual(e["wrong_all"]["max_regret"], 30.0)
        e0 = ed.evaluate([a, b], [0, 0])
        self.assertEqual((e0["gain"], e0["differs_from_p0_when_search_differs"]), (0.0, 0))


class CrossValidateTest(unittest.TestCase):
    def test_learns_simple_signal_and_holds_out_groups(self):
        rng = np.random.default_rng(0)
        samples = []
        for g in ("ga", "gb", "gc"):
            for _ in range(40):
                x = np.zeros(72, dtype=np.float32)
                flag = rng.random() < 0.5
                x[10] = 1.0 if flag else 0.0                                    # 이 칸이 1이면 녹색 1이 좋음
                costs = {0: [100.0] * 8, 1: [(60.0 if flag else 140.0)] * 8}
                samples.append(_sample(group=g, p0=0, search=1 if flag else 0, costs=costs, obs=x))
        saved = ed.MLP_SEEDS, ed.MLP_EPOCHS
        ed.MLP_SEEDS, ed.MLP_EPOCHS = (0, 1), 300
        try:
            cv = ed.cross_validate(samples)
        finally:
            ed.MLP_SEEDS, ed.MLP_EPOCHS = saved
        self.assertEqual(cv["groups"], ["ga", "gb", "gc"])
        self.assertEqual([cv["folds"][g]["test"] for g in cv["groups"]], [40, 40, 40])
        self.assertEqual([cv["folds"][g]["train"] for g in cv["groups"]], [80, 80, 80])
        self.assertGreater(cv["total"]["mlp_R"]["share"], 0.9)                 # 쉬운 신호는 거의 다 배움
        self.assertEqual(cv["total"]["p0"]["gain"], 0.0)
        r = ed.reading(cv["total"], cv["folds"])
        self.assertEqual(r["word"], "작은 신경망이 배울 수 있어 보인다")

    def test_load_groups_two_seeds_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            for seed in (42, 43):
                d = Path(tmp) / f"20220822_1300_s{seed}_search"
                d.mkdir()
                row = {"searched": True, "scores": {"0": [{"600": {"cost": 10.0}}] * 8, "1": [{"600": {"cost": 9.0}}] * 8},
                       "obs72": [0.0] * 72, "allowed": [0, 1], "p0_action": 0, "chosen": 1, "t0": 1.0,
                       "left_east": 9, "left_south": 0, "left_east_served": False, "left_south_served": False}
                (d / "decisions.jsonl").write_text(json.dumps(row) + "\n" + json.dumps({**row, "searched": False}) + "\n")
            s = ed.load(Path(tmp))
            self.assertEqual(len(s), 2)
            self.assertEqual({x["group"] for x in s}, {"20220822_1300"})
            self.assertEqual({x["seed"] for x in s}, {42, 43})


if __name__ == "__main__":
    unittest.main()
