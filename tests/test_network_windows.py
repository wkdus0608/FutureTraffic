"""12개 지도 90분 조각 점검의 판정·빼기 규칙 검사(SUMO 실행 없음)."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import network_windows as nw  # noqa: E402

CFG = {"failed": {"20220823": [18]}, "screened_dates": ["20220823", "20220811"], "hours": list(range(1, 24))}


class WindowRuleTest(unittest.TestCase):
    def test_broken_means_frozen_road_or_large_backlog(self):
        self.assertFalse(nw.is_failed(128, []))                    # 붐빔(08-11 18시)은 무너짐이 아니다
        self.assertTrue(nw.is_failed(0, ["477706911_01"]))          # 굳은 도로가 있으면 무너짐
        self.assertTrue(nw.is_failed(nw.MAX_PENDING + 1, []))

    def test_training_scenarios_overlapping_a_broken_window_are_excluded(self):
        h = 3600
        self.assertTrue(nw.excluded("20220823", 18 * h, CFG))           # 같은 시각
        self.assertTrue(nw.excluded("20220823", 17 * h + 900, CFG))     # 17:15~18:15는 18시 조각과 겹침
        self.assertTrue(nw.excluded("20220823", 18 * h + 2700, CFG))    # 18:45~19:45도 겹침
        self.assertFalse(nw.excluded("20220823", 17 * h, CFG))          # 17:00~18:00은 겹치지 않음(끝이 18:00)
        self.assertFalse(nw.excluded("20220823", 19 * h, CFG))
        self.assertFalse(nw.excluded("20220811", 18 * h, CFG))          # 다른 날짜

    def test_evaluation_windows_skip_broken_and_need_screening(self):
        wins = nw.evaluation_windows(["20220823"], CFG)
        self.assertNotIn(("20220823", 18), wins)
        self.assertEqual(len(wins), 22)
        with self.assertRaises(SystemExit):
            nw.evaluation_windows(["20220928"], CFG)                    # 점검하지 않은 날짜는 평가하지 않는다


if __name__ == "__main__":
    unittest.main()
