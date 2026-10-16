"""12개 신호 칸 맞추기 도구(network_signal_columns)의 한 현시 밀림 보정과 계산 도우미 검사."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import network_signal_columns as nsc  # noqa: E402


def expand(cycle, cycles=4, green_s=10, yellow_s=3):
    """[상태, …] 한 주기를 1초마다의 목록으로 여러 번 잇는다."""
    out = []
    for _ in range(cycles):
        for s in cycle:
            out += [s] * (yellow_s if "y" in s else green_s)
    return out


class ShiftMapTest(unittest.TestCase):
    def test_regular_cycle(self):
        g0, y0, g1, y1, g2, y2 = "GGrrrr", "yyrrrr", "rrGGrr", "rryyrr", "rrrrGG", "rrrryy"
        shift = nsc.shift_map(expand([g0, y0, g1, y1, g2, y2]))
        self.assertEqual(shift[g1], g0)      # 녹색 → 앞 녹색
        self.assertEqual(shift[g0], g2)
        self.assertEqual(shift[y1], y0)      # 황색 → 앞 녹색의 황색
        self.assertEqual(shift[y0], y2)

    def test_missing_yellow_like_204920(self):
        # 204920처럼 녹색 g1 → g2가 황색 없이 바로 바뀐다.
        g0, y0, g3, y3, g1, g2, y2 = "GGrrrrrr", "yyrrrrrr", "rrGGrrrr", "rryyrrrr", "rrrrGGrr", "rrrrrrGG", "rrrrrryy"
        shift = nsc.shift_map(expand([g0, y0, g3, y3, g1, g2, y2]))
        self.assertEqual(shift[g0], g2)
        self.assertEqual(shift[g3], g0)
        self.assertEqual(shift[g1], g3)
        self.assertEqual(shift[g2], g1)      # 예전 규칙('두 단계 앞')은 여기서 황색 y3을 냈다
        self.assertEqual(shift[y0], y2)
        self.assertEqual(shift[y3], y0)
        self.assertEqual(shift[y2], "rrrryyrr")   # 기록에 없는 g1의 황색: g1에서 g2에 없는 칸만 황색


class HelperTest(unittest.TestCase):
    def test_detrend_matches_moving_average(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=200) + np.linspace(0, 10, 200)
        w = 15
        pad = np.pad(a, (w // 2, w // 2), mode="edge")
        expected = a - np.convolve(pad, np.ones(w) / w, mode="valid")
        np.testing.assert_allclose(nsc.detrend(a, w), expected, atol=1e-9)
        np.testing.assert_allclose(nsc.detrend(np.full(50, 7.0)), 0, atol=1e-12)
        m = rng.normal(size=(100, 3))
        np.testing.assert_allclose(nsc.detrend(m), np.stack([nsc.detrend(m[:, k]) for k in range(3)], 1))

    def test_column_map(self):
        self.assertEqual(nsc.column_map("x", 3, 5, [1, 3]), [0, 2, 4])
        self.assertEqual(nsc.column_map("201330", 22, 23)[10:13], [10, 12, 13])     # 원본 칸 11을 뺀다
        with self.assertRaises(ValueError):
            nsc.column_map("x", 4, 5, [1, 3])


if __name__ == "__main__":
    unittest.main()
