"""[추가 탐색] 학생의 큰 손익 20곳 독립 재측정(137절) 검사 — SUMO 없음. 새 seed·대상 고르기·기록 대조·읽는 법을 본다."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import explore_online as eo  # noqa: E402
import explore_remeasure as rm  # noqa: E402


class SeedTest(unittest.TestCase):
    def test_fresh_seeds_are_new(self):
        old = {s for seed in eo.BASE_SEEDS for t0 in range(1800, 9001, 1) for s in eo.scenario_seeds(seed, float(t0))}
        old |= set(range(1001, 2009))
        new = [s for seed in eo.BASE_SEEDS for t0 in range(1800, 9001, 7) for s in rm.fresh_seeds(seed, float(t0))]
        self.assertEqual(len(new), len(set(new)))                              # 시점·seed끼리도 겹치지 않음
        self.assertFalse(set(new) & old)
        self.assertLess(max(new), 2 ** 31)
        self.assertEqual(len(rm.fresh_seeds(42, 3600.0)), 16)


def _c(g, se=1.0, group="ga", seed=42, t0=0.0):
    return {"group": group, "seed": seed, "t0": t0, "g": g, "se": se}


class ChooseTest(unittest.TestCase):
    def test_choose(self):
        changes = [_c(-50, t0=1), _c(-30, t0=2), _c(-1.5, t0=3), _c(-30, t0=0), _c(40, t0=4), _c(10, se=6, t0=5), _c(2.5, t0=6)]
        out = rm.choose(changes, n=2)
        self.assertEqual([(c["set"], c["rank"], c["t0"]) for c in out],
                         [("loss", 0, 1), ("loss", 1, 0), ("gain", 0, 4), ("gain", 1, 6)])   # 같은 g는 시각 순, 분명하지 않은 것은 뺌


class MismatchTest(unittest.TestCase):
    def test_mismatches(self):
        row = {"t0": 10.0, "allowed": [0, 1], "p0_action": 1, "obs64": [0.5, 0.25], "obs72": [0.5, 0.25, 1.0]}
        self.assertEqual(rm.mismatches(row, 10.0, [0, 1], 1, [0.5, 0.25], [0.5, 0.25, 1.0]), [])
        self.assertEqual(rm.mismatches(row, 10.0, [0, 1], 0, [0.5, 0.25], [0.5, 0.2500001, 1.0]), ["p0_action", "obs72"])


def _rows(orig, fresh, se, n=10):
    return [{"g": orig, "fresh_mean": fresh, "fresh_se": se}] * n


class ReadingTest(unittest.TestCase):
    def test_paired(self):
        mean, se = rm.paired([10.0, 12.0, 14.0], [9.0, 9.0, 9.0])
        self.assertEqual(mean, 3.0)
        self.assertAlmostEqual(se, 2.0 / 3 ** 0.5)

    def test_words(self):
        self.assertEqual(rm.set_reading(_rows(-3000, -2800, 300))["word"], "대부분 다시 나타난다")
        self.assertEqual(rm.set_reading(_rows(-3000, -800, 300))["word"], "일부만 다시 나타난다")
        self.assertEqual(rm.set_reading(_rows(-3000, 100, 300))["word"], "사라진다")
        self.assertEqual(rm.set_reading(_rows(2000, 1000, 1500))["word"], "불분명")              # 구간이 0.5를 걸침
        r = rm.set_reading(_rows(-3000, -1500, 0, n=4))
        self.assertEqual((r["retained"], r["retained_interval"]), (0.5, [0.5, 0.5]))
        self.assertEqual(r["word"], "대부분 다시 나타난다")
        self.assertEqual(rm.set_reading([])["word"], "불분명")

    def test_interval_math(self):
        r = rm.set_reading([{"g": 1000, "fresh_mean": 800, "fresh_se": 300}, {"g": 3000, "fresh_mean": 1200, "fresh_se": 400}])
        self.assertEqual((r["original_mean"], r["fresh_mean"], r["fresh_se"]), (2000.0, 1000.0, 250.0))
        self.assertEqual(r["fresh_interval"], [510.0, 1490.0])
        self.assertEqual(r["retained_interval"], [0.255, 0.745])


if __name__ == "__main__":
    unittest.main()
