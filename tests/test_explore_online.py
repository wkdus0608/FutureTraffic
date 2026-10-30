"""[추가 탐색] 제한된 1단계(123절) 검사 — SUMO 실행 없음. 조건·시나리오·고르기·지연 셈·판정·정리를 본다."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import explore_online as eo  # noqa: E402
import explore_rollout as er  # noqa: E402

EAST_LEFT = ["477714896_02", "477706392_02", "477706883_02"]
EAST_STRAIGHT = ["477714896_02", "477706392_02", "477707013_01"]


def _feat(le=0, ls=0, les=False, lss=False):
    return {"left_east": le, "left_south": ls, "left_east_served": les, "left_south_served": lss}


class PlanTest(unittest.TestCase):
    def test_values_match_plan(self):
        self.assertEqual(eo.EPISODES, [("20220822", "1300"), ("20220906", "1800"), ("20220907", "1300")])
        self.assertEqual(eo.BASE_SEEDS, (42, 43))
        self.assertEqual((eo.TAIL_S, eo.N_SCENARIOS, eo.MARK), (900, 8, "600"))
        self.assertEqual(eo.LEFT_CUT, {"left_east": 8, "left_south": 7})
        self.assertEqual(eo.ROLLOUT_CODE, "6cc31fa07d09")
        step0 = {(d, p) for d in ("20220811", "20220823", "20221030") for p in ("0730", "1230", "1730", "2130")}
        self.assertFalse(set(eo.EPISODES) & step0)                                # 0단계와 다른 에피소드
        self.assertFalse({d for d, _ in eo.EPISODES} & {d for d, _ in step0})


class RuleTest(unittest.TestCase):
    def test_left_point(self):
        self.assertTrue(eo.is_left_point(_feat(le=8)))
        self.assertFalse(eo.is_left_point(_feat(le=7)))
        self.assertFalse(eo.is_left_point(_feat(le=12, les=True)))                # 지금 녹색이면 아님
        self.assertTrue(eo.is_left_point(_feat(ls=7)))
        self.assertTrue(eo.is_left_point(_feat(le=12, les=True, ls=9)))

    def test_scenario_seeds(self):
        a = eo.scenario_seeds(42, 3600.0)
        self.assertEqual(len(set(a)), 8)
        self.assertFalse(set(a) & set(eo.scenario_seeds(43, 3600.0)))
        self.assertFalse(set(a) & set(eo.scenario_seeds(42, 3605.0)))
        self.assertFalse(set(a) & set(range(1001, 2009)))                         # 0단계 seed와 겹치지 않음

    def test_choose(self):
        best, means = eo.choose({0: [10, 12], 1: [8, 9], 2: [8, 9]}, 0)
        self.assertEqual((best, means[1]), (1, 8.5))                              # 같으면 작은 번호
        self.assertEqual(eo.choose({0: [5, 5], 1: [5, 5]}, 0)[0], 0)              # P0와 같으면 P0


class MetricsTest(unittest.TestCase):
    def test_episode_metrics(self):
        vehicles = {"a": (1000.0, EAST_LEFT, "passenger"), "b": (1500.0, EAST_STRAIGHT, "bus"), "c": (2000.0, EAST_LEFT, "truck"),
                    "early": (900.0, EAST_LEFT, "passenger"), "late": (4700.0, EAST_LEFT, "passenger"), "gone": (1200.0, EAST_LEFT, "passenger")}
        rows = {"a": {"timeLoss": "100", "departDelay": "5", "waitingTime": "250", "arrival": "1300"},
                "b": {"timeLoss": "20", "departDelay": "0", "waitingTime": "10", "arrival": "1600"},
                "c": {"timeLoss": "40", "departDelay": "1", "waitingTime": "20", "arrival": "-1"},
                "early": {"timeLoss": "999", "departDelay": "0", "waitingTime": "999", "arrival": "1200"}}
        m = eo.episode_metrics(rows, vehicles, 1000.0, 3600)
        self.assertEqual((m["vehicles"], m["measured"], m["missing"], m["unfinished"], m["long_waits"]), (4, 3, 1, 1, 0))
        self.assertAlmostEqual(m["mean_delay_s"], (105 + 20 + 41) / 3, places=3)
        self.assertEqual(m["left_east"], 73.0)                                    # (105 + 41) / 2
        rows["a"]["waitingTime"] = "296"
        self.assertEqual(eo.episode_metrics(rows, vehicles, 1000.0, 3600)["long_waits"], 1)   # 296 + 5 > 300


def _pair(delta, le=(100, 100), ls=(90, 90), long=(5, 5)):
    return {"delta": delta, "left_east_p0": le[0], "left_east_search": le[1], "left_south_p0": ls[0], "left_south_search": ls[1],
            "long_p0": long[0], "long_search": long[1]}


class ReadingTest(unittest.TestCase):
    def test_reading(self):
        self.assertEqual(eo.pair_reading([_pair(d) for d in (0.5, 0.3, 0.2, 0.4, 0.1, -0.1)])["word"], "이득이 쌓인다")
        self.assertEqual(eo.pair_reading([_pair(d) for d in (0.5, 0.3, 0.2, 0.4, 0.1, -0.1)][:5] + [_pair(-0.1, le=(100, 120))])["word"],
                         "불분명")                                                # 좌회전이 나빠지면 '쌓인다'가 아님
        self.assertEqual(eo.pair_reading([_pair(d) for d in (0.5, 0.3, 0.2, -0.4, -0.1, -0.1)])["word"], "이득이 없거나 지워진다")
        self.assertEqual(eo.pair_reading([_pair(d) for d in (0.1, 0.1, 0.1, 0.1, -0.1, -2.0)])["word"], "이득이 없거나 지워진다")
        self.assertEqual(eo.pair_reading([_pair(d) for d in (0.5, 0.3, 0.2, 0.4, -0.1, -0.2)])["word"], "불분명")


class SummaryTest(unittest.TestCase):
    def test_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for date, period in eo.EPISODES:
                for seed in eo.BASE_SEEDS:
                    for mode, mean, total in (("p0", 40.0, 4000.0), ("search", 39.5, 3950.0)):
                        d = out / "runs" / f"{date}_{period}_s{seed}_{mode}"
                        d.mkdir(parents=True)
                        m = {"date": date, "period": period, "seed": seed, "mode": mode, "code": "abc", "mean_delay_s": mean,
                             "total_delay": total, "left_east": 100.0, "left_south": 90.0, "long_waits": 3, "unfinished": 0,
                             "expected_gain_sum": 100.0 if mode == "search" else 0.0, "expected_gains": [60.0, 30.0, 10.0] if mode == "search" else [],
                             "searches": 20, "searched_differs": 3, "search_wall_s": 3600, "search_cpu_s": 28800,
                             "collisions_at_204820": 0, "collisions_unexplained": 0, "search_leaks": 0, "changed_by_rules": 0,
                             "search_changed_by_rules": 0}
                        (d / "metrics.json").write_text(json.dumps(m))
                        (d / "vehicle_delays.json").write_text(json.dumps({"a": total / 2, "b": total / 2}))
            eo.summary(SimpleNamespace(out=str(out)))
            r = json.loads((out / "summary.json").read_text())
            self.assertEqual(r["reading"]["word"], "이득이 쌓인다")
            p = r["pairs"][0]
            self.assertEqual((p["delta"], p["realized_total_gain"], p["realized_over_expected"], p["top2_share"]), (0.5, 50.0, 0.5, 0.9))
            (out / "runs" / "20220822_1300_s42_search" / "metrics.json").unlink()
            eo.summary(SimpleNamespace(out=str(out)))
            self.assertIsNone(json.loads((out / "summary.json").read_text())["reading"])   # 쌍이 다 모이지 않으면 판정 안 함


if __name__ == "__main__":
    unittest.main()
