"""12개 지도 수요·신호·지도 도구(network_demand, network_signals, network_map) 검사. SUMO 실행 없음."""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import network_demand as nd  # noqa: E402
import network_map as nm  # noqa: E402
import network_signals as ns  # noqa: E402


class YellowTest(unittest.TestCase):
    def test_direct_green_to_red_gets_yellow_from_ending_green(self):
        phases = [(30, "GGrr"), (20, "rrGG")]
        out, note = ns.add_yellows(phases)
        self.assertEqual(out, [(26, "GGrr"), (4, "yyrr"), (20, "rrGG")])
        self.assertEqual(sum(d for d, _ in out), 50)          # 하루 시간표는 그대로
        self.assertEqual(note["황색 넣음"], 1)

    def test_leftover_green_in_yellow_phase(self):
        phases = [(30, "GgGr"), (4, "ygyr"), (20, "rrrG")]
        out, note = ns.add_yellows(phases)
        self.assertEqual(out[1], (4, "yyyr"))                 # 황색 동안 g였다가 적색이 되는 연결(유턴)
        self.assertEqual(note["유턴 황색"], 1)

    def test_one_second_green_left_alone(self):
        out, note = ns.add_yellows([(1, "Gr"), (5, "rG")])
        self.assertEqual(out, [(1, "Gr"), (5, "rG")])
        self.assertEqual(note["황색 못 넣음(녹색 1초)"], 1)


class FitTest(unittest.TestCase):
    """경로 3개: r0 = 이동 a만, r1 = 이동 b만, r2 = a와 b를 함께 지나는 긴 경로."""
    A = csr_matrix(np.array([[1, 0, 1], [0, 1, 1]], dtype=float))
    w = np.ones(2)
    minutes = np.array([1.0, 1.0, 2.0])
    no_limits = (csr_matrix((0, 3)), np.zeros(0))

    def test_consistent_counts_use_fewer_vehicles(self):
        x, _ = nd.fit_period(self.A, np.array([10.0, 10.0]), self.w, self.minutes, self.no_limits)
        np.testing.assert_allclose(self.A @ x, [10, 10], atol=1e-6)
        np.testing.assert_allclose(x, [0, 0, 10], atol=1e-6)    # 짧은 두 대보다 긴 한 대

    def test_no_phantom_flow(self):
        # b 기록은 0인데 a만 있다: 긴 경로(r2)를 쓰면 b에 넘침이 생기므로 r0만 쓴다
        x, _ = nd.fit_period(self.A, np.array([10.0, 0.0]), self.w, self.minutes, self.no_limits)
        np.testing.assert_allclose(x, [10, 0, 0], atol=1e-6)

    def test_boundary_limit(self):
        limits = (csr_matrix(np.array([[1.0, 0.0, 0.0]])), np.array([4.0]))   # r0의 출발 도로는 15분 4대까지
        x, _ = nd.fit_period(self.A, np.array([10.0, 0.0]), self.w, self.minutes, limits)
        self.assertAlmostEqual(x[0], 4.0, places=6)
        self.assertAlmostEqual(x[2], 0.0, places=6)            # 모자라도 넘침(3배)을 만들지 않는다


class IncidenceTest(unittest.TestCase):
    def test_signature_occurrences(self):
        mvs = [nd.Movement(("a", "b"), ("a", "b"), "T", "직접 연결"),
               nd.Movement(("p", "s"), ("p", "q", "s"), "T", "분리 우회전")]
        routes = [("x", "a", "b", "y"), ("p", "q", "s"), ("p", "s"), ("a", "b", "z", "a", "b")]
        A = nd.incidence(routes, mvs).toarray()
        np.testing.assert_array_equal(A, [[1, 0, 0, 2], [0, 1, 0, 0]])


class MapTest(unittest.TestCase):
    def test_built_map_passes_and_detects_changed_signal_link(self):
        path = nm.network_12()
        report = nm.verify(nm.fd.NETWORK, path)
        self.assertEqual(set(report["removed_edges"]), nm.join_edges(nm.fd.NETWORK))
        self.assertEqual(report["tls_links"], 249)
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.net.xml"
            text = path.read_text(encoding="utf-8")
            i = text.index('linkIndex="0"')
            bad.write_text(text[:i] + 'linkIndex="99"' + text[i + len('linkIndex="0"'):], encoding="utf-8")
            with self.assertRaises(ValueError):
                nm.verify(nm.fd.NETWORK, bad)

    def test_special_signatures_connected_on_map(self):
        graph = nd.Graph(nm.network_12())
        for key, (sig, tls, kind) in nd.SPECIAL.items():
            for a, b in zip(sig, sig[1:]):
                self.assertIn(b, graph.succ.get(a, ()), f"{key}: {a} → {b}")



class JointFitTest(unittest.TestCase):
    def test_boundary_limit_is_shared_by_vehicle_types(self):
        A = csr_matrix(np.array([[1.0]]))
        limits = (csr_matrix(np.array([[1.0]])), np.array([6.0]))          # 경로 하나, 15분 6대까지
        X, _ = nd.fit_interval(A, np.array([[5.0], [5.0]]), np.ones(1), np.ones(1), limits)
        self.assertAlmostEqual(X.sum(), 6.0, places=6)                     # 두 차종을 더해 6대(차종마다 6대가 아님)


class DepartFixTest(unittest.TestCase):
    """출발 시각 보정표: 경로·시 평균을 경로 하루 평균 쪽으로 당기고(차 5대 몫), 0~900초로 자른다."""
    FIX = {"by_route": {"a": (40, 60.0), "neg": (10, -8.0), "big": (3, 2000.0)},
           "by_route_hour": {"a|8": (5, 100.0), "a|9": (1, 0.0), "neg|8": (10, -8.0), "big|8": (3, 2000.0)}}

    def test_hour_mean_shrinks_to_route_mean(self):
        self.assertAlmostEqual(nd.depart_shift(self.FIX, "a", 8), (5 * 100 + 5 * 60) / 10)
        self.assertAlmostEqual(nd.depart_shift(self.FIX, "a", 9), (1 * 0 + 5 * 60) / 6)
        self.assertAlmostEqual(nd.depart_shift(self.FIX, "a", 3), 60.0)        # 그 시에 잰 차가 없으면 하루 평균

    def test_clipped_and_unknown_route(self):
        self.assertEqual(nd.depart_shift(self.FIX, "neg", 8), 0.0)              # 늦게 출발시키지 않는다
        self.assertEqual(nd.depart_shift(self.FIX, "big", 8), nd.DEPART_FIX_MAX_S)
        self.assertEqual(nd.depart_shift(self.FIX, "none", 8), 0.0)             # 잰 적 없는 경로는 그대로

    def test_route_key_depends_on_edge_order(self):
        self.assertEqual(nd.route_key(("e1", "e2")), nd.route_key(["e1", "e2"]))
        self.assertNotEqual(nd.route_key(("e1", "e2")), nd.route_key(("e2", "e1")))


class ChainTest(unittest.TestCase):
    def test_chain_rule_matches_single_model_at_204820(self):
        from focused_scenario import DOWNSTREAM_LIMIT_M, UPSTREAM_LIMIT_M, FocusedNetwork
        self.assertEqual((nd.CHAIN_UP_M, nd.CHAIN_DOWN_M), (UPSTREAM_LIMIT_M, DOWNSTREAM_LIMIT_M))
        fn = FocusedNetwork()
        graph = nd.Graph(nm.network_12())
        for f, t in sorted(fn.movements):
            self.assertEqual(graph.chain(f, True, nd.CHAIN_UP_M), fn._chain(f, True, UPSTREAM_LIMIT_M), f)
            self.assertEqual(graph.chain(t, False, nd.CHAIN_DOWN_M), fn._chain(t, False, DOWNSTREAM_LIMIT_M), t)


if __name__ == "__main__":
    unittest.main()
