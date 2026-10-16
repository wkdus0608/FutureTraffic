"""전환 점검 도구(focused_intergreen, focused_conflicts)의 가벼운 검사. SUMO를 돌리지 않는다."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from focused_conflicts import Geometry, Track  # noqa: E402
from focused_intergreen import VEHICLES, conflict, is_real_order  # noqa: E402


class ConflictToolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geo = Geometry()

    def test_real_order(self):
        # 실제 순서: 3 석천로 직진 → 0 석천로 좌회전 → 1 길주로 직진 → 2 길주로 좌회전 → 3
        self.assertEqual({(i, j) for i in range(4) for j in range(4) if i != j and is_real_order(i, j)},
                         {(3, 0), (0, 1), (1, 2), (2, 3)})

    def test_changing_links_for_collision_transition(self):
        # 09-28 충돌 전환(녹색 2 → 0): 끝나는 연결에 버스의 10번, 시작하는 연결에 승용차의 15번
        ending, starting = self.geo.changing_links(2, 0)
        self.assertIn(10, ending)
        self.assertIn(15, starting)

    def test_collision_pair_overlaps(self):
        # 버스 10번 경로와 승용차 15번 경로는 겹치고, 승용차 쪽 겹치는 구역은 정지선 가까이에서 시작한다
        zone = conflict(self.geo.paths[10], self.geo.paths[15],
                        (VEHICLES["bus"]["width"] + VEHICLES["passenger"]["width"]) / 2)
        self.assertIsNotNone(zone)
        near_on_bus, far_on_bus, near_on_car, far_on_car = zone
        self.assertLess(near_on_bus, far_on_bus)
        self.assertLess(near_on_car, far_on_car)
        self.assertGreater(far_on_bus, 20.0)
        self.assertLess(near_on_car, 10.0)

    def test_track_cross_interpolates(self):
        tr = Track(link=10, cls="bus", length=12.0)
        tr.t, tr.d = [0, 1, 2], [-2.0, 4.0, 10.0]
        self.assertAlmostEqual(tr.cross(0.0), 1 / 3)
        self.assertAlmostEqual(tr.cross(7.0), 1.5)
        self.assertIsNone(tr.cross(20.0))


if __name__ == "__main__":
    unittest.main()
