"""[추가 탐색] 0단계 JOINT2 검사 — SUMO 실행 없음(가짜 모형·작은 XML)."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import explore_joint2 as ej  # noqa: E402
import explore_pred as ep  # noqa: E402
from explore_pred import SignalState  # noqa: E402

G = ["GGrr", "rGGr", "rrGG", "GrrG"]
Y = ["yyrr", "ryyr", "rryy", "yrry"]


def model(step=5.0):
    rates = [{0: 0.5}, {1: 0.5}, {2: 0.5}, {3: 0.5}]
    return ej.CyclicModel(rates, [10.0, 10.0, 10.0, 10.0], [30.0, 20.0, 20.0, 20.0], step)


class CyclicModelTest(unittest.TestCase):
    def test_min_max_and_order(self):
        m = model()
        st = SignalState(0, 0.0, 4.0, ())
        self.assertEqual(m.actions(st, 10.0), [0])            # 황색 4 + 최소 10초 전
        self.assertEqual(m.actions(st, 14.0), [0, 1])
        self.assertEqual(m.actions(st, 34.0), [1])            # 최대 30초면 다음 녹색만
        with self.assertRaises(ValueError):
            m.advance(st, 20.0, 2)                              # 순서를 건너뛸 수 없음
        st2, eff = m.advance(st, 20.0, 1)
        self.assertEqual((st2.phase, st2.switch_time), (1, 20.0))
        self.assertEqual(eff, 0.0)                               # 황색 4 + 출발 손실 2초 안

    def test_plan_matches_explore_pred_without_release(self):
        m = model()
        st = SignalState(0, -20.0, 4.0, ())
        arrivals = [[0.5, 0.2, 0.1, 0.3] for _ in range(24)]
        a = ej.plan_release(m, st, 0.0, [3, 5, 1, 2], arrivals, m.actions(st, 0.0))
        b = ep.plan(m, st, 0.0, [3, 5, 1, 2], arrivals, m.actions(st, 0.0))
        self.assertEqual(a[0], b[0])
        self.assertAlmostEqual(a[1]["cost"], b[1]["cost"])
        self.assertEqual(a[1]["plan"], b[1]["plan"])

    def test_release_penalty_adds_released_times_wait(self):
        m = model()
        st = SignalState(0, -20.0, 4.0, ())
        arrivals = [[0.0] * 4 for _ in range(2)]
        free = ej.plan_release(m, st, 0.0, [4, 0, 0, 0], arrivals, [0])
        paid = ej.plan_release(m, st, 0.0, [4, 0, 0, 0], arrivals, [0], [{0: 10.0}, {0: 10.0}])
        # 녹색 0이 두 단계 동안 0.5대/초 × 5초씩 4대를 모두 내보냄 → 4대 × 10초
        self.assertAlmostEqual(paid[1]["cost"] - free[1]["cost"], 40.0)


class HelperTest(unittest.TestCase):
    def test_wait_until(self):
        on = lambda t: 20.0 <= t < 30.0  # noqa: E731
        self.assertEqual(ej.wait_until(on, 5.0, 0.0, 120.0), 15.0)
        self.assertEqual(ej.wait_until(on, 25.0, 0.0, 120.0), 0.0)
        self.assertEqual(ej.wait_until(on, 130.0, 0.0, 120.0), 0.0)         # 모르는 범위
        self.assertEqual(ej.wait_until(lambda t: False, 100.0, 0.0, 120.0), 20.0 + ep.TERMINAL_S)

    def test_green_bounds(self):
        phases = []
        for k in range(40):
            for g, y, d in zip(G, Y, (50 + k % 5, 22, 45, 20)):
                phases += [(d, g), (4, y)]
        b = ej.green_bounds(phases)
        self.assertEqual(b["greens"], G)
        self.assertEqual(b["yellows"], Y)
        self.assertEqual(b["min"][0], 50.0)
        self.assertEqual(b["max"][0], 54.0)
        with self.assertRaises(ValueError):
            ej.green_bounds([p for p in phases if p[1] not in (G[3], Y[3])])

    def test_network_metrics(self):
        saved = ej._LAYOUT
        ej._LAYOUT = {"edge_pairs": [("a", "b")]}
        try:
            with tempfile.TemporaryDirectory() as d:
                routes, trips = Path(d) / "r.xml", Path(d) / "t.xml"
                routes.write_text('<routes><route id="r1" edges="x a b"/><route id="r2" edges="x y"/>'
                                  '<vehicle id="v1" route="r1" depart="100"/><vehicle id="v2" route="r2" depart="150"/>'
                                  '<vehicle id="v3" route="r2" depart="190"/><vehicle id="v4" route="r1" depart="10"/></routes>')
                trips.write_text('<tripinfos><tripinfo id="v1" timeLoss="30" departDelay="5" arrival="300"/>'
                                 '<tripinfo id="v2" timeLoss="12" departDelay="0" arrival="-1"/>'
                                 '<tripinfo id="v4" timeLoss="99" departDelay="0" arrival="50"/></tripinfos>')
                m = ej.network_metrics(routes, trips, 100.0, 200.0)
            self.assertEqual(m["network"]["vehicles"], 3)                   # v4는 제어 전 출발
            self.assertAlmostEqual(m["network"]["mean_delay_s"], (35 + 12 + 10) / 3)   # v3는 못 들어옴: 200 − 190
            self.assertEqual(m["network"]["not_inserted"], 1)
            self.assertEqual(m["network"]["unfinished"], 2)
            self.assertEqual(m["via_204822"]["vehicles"], 1)
        finally:
            ej._LAYOUT = saved


if __name__ == "__main__":
    unittest.main()
