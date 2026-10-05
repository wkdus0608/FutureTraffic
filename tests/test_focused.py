"""석천사거리 중심 실험의 기본 정합성 검사."""
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from focused_data import COUNTS, find_network, load_counts  # noqa: E402
from focused_scenario import (FocusedNetwork, aihub_states, replay_phases,  # noqa: E402
                              write_dqn_template_network, write_routes)


@unittest.skipUnless(COUNTS.exists(), "counts_204820.csv가 없으면 건너뜀")
class FocusedScenarioChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fn = FocusedNetwork()
        cls.counts = load_counts(COUNTS)

    def test_vehicles_equal_recorded_counts(self):
        rows = self.counts[("20220810", "1800")]
        with tempfile.TemporaryDirectory() as tmp:
            info = write_routes(self.fn, rows, Path(tmp) / "r.rou.xml", seed=1)
            root = ET.parse(Path(tmp) / "r.rou.xml").getroot()
        self.assertEqual(info["vehicles"], sum(r["count"] for r in rows))
        self.assertEqual(len(root.findall("vehicle")), sum(r["count"] for r in rows))
        types = Counter(v.get("type") for v in root.findall("vehicle"))
        expected = Counter()
        for r in rows:
            expected[r["vehicle_type"]] += r["count"]
        self.assertEqual(types, expected)

    def test_routes_are_connected_and_cross_once(self):
        for (a, b) in self.fn.movements:
            route = self.fn.route_for(a, b)
            for x, y in zip(route, route[1:]):
                self.assertIn(y, self.fn.succ[x])
            crossings = sum(1 for x, y in zip(route, route[1:]) if (x, y) in self.fn.movements)
            self.assertEqual(crossings, 1)
            self.assertNotEqual(route[-1], a)  # 진입 도로에서 끝나는 경로 없음

    def test_entry_branch_split_keeps_records_and_follows_upstream_share(self):
        """길주로 서쪽 접근로(477707013_02) 차량을 두 갈래에 나눠도 이동별 대수는 기록과 같고, 4차로 가지 대수는 상류 기록 비율을 따른다."""
        from focused_scenario import ENTRY_BRANCHES, entry_branch_shares
        rows = self.counts[("20220811", "0830")]
        with tempfile.TemporaryDirectory() as tmp:
            info = write_routes(self.fn, rows, Path(tmp) / "r.rou.xml", seed=1)
            root = ET.parse(Path(tmp) / "r.rou.xml").getroot()
        routes = {r.get("id"): r.get("edges").split() for r in root.findall("route")}
        moves, first = Counter(), Counter()
        for v in root.findall("vehicle"):
            e = routes[v.get("route")]
            mv = next((a, b) for a, b in zip(e, e[1:]) if (a, b) in self.fn.movements)
            moves[mv] += 1
            if mv[0] == "477707013_02":
                first[e[0]] += 1
        expected = Counter()
        for r in rows:
            expected[(r["from_edge"], r["to_edge"])] += r["count"]
        self.assertEqual(moves, expected)
        share = entry_branch_shares(rows)["477707013_02"]
        self.assertAlmostEqual(info["entry_branch_shares"]["477707013_02"], share)
        spec = ENTRY_BRANCHES["477707013_02"]
        volume = sum(first.values())
        self.assertLessEqual(abs(first[spec["main"]] - share * volume), 2)
        self.assertEqual(first[spec["main"]] + first[spec["alternative"]], volume)

    def test_alternative_branch_route_is_connected(self):
        for b in ("477706392_01", "477706883_02", "477706911_02_02"):
            route = self.fn.route_for("477707013_02", b, "alternative")
            self.assertEqual(route[0], "477744390_01")
            for x, y in zip(route, route[1:]):
                self.assertIn(y, self.fn.succ[x])

    def test_template_network_changes_only_target_signal(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = write_dqn_template_network(Path(tmp) / "t.net.xml")
            a = ET.parse(find_network()).getroot()
            b = ET.parse(out).getroot()
        strip = lambda r: [ET.tostring(e) for e in r if not (e.tag == "tlLogic" and e.get("id") == "204820")]
        self.assertEqual(strip(a), strip(b))
        greens = [p.get("state") for p in b.find("tlLogic[@id='204820']").findall("phase") if "y" not in p.get("state")]
        self.assertEqual(len(greens), 4)

    def test_corrected_replay_uses_same_green_states(self):
        info = aihub_states()
        phases = replay_phases("0800")
        states = {s for _, s in phases if "y" not in s}
        self.assertTrue(states <= set(info["greens"]))
        self.assertGreaterEqual(sum(d for d, _ in phases), 1800)


if __name__ == "__main__":
    unittest.main()
