"""AI Hub 신호를 하루 시간축으로 복원하는 코드의 검사 (docs/가상도로_정상화_20261004.md 참고)."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from focused_data import SIGNALS, read_signal_phases, signal_xml_files  # noqa: E402
from focused_scenario import (aihub_day_timeline, aihub_signal_program, aihub_unreliable_periods,  # noqa: E402
                              replay_phases)

SPLIT = json.loads((ROOT / "config/focused_split.json").read_text())
DATES = sorted(set(SPLIT["train"] + SPLIT["validation"] + SPLIT["test"]))


def per_second(phases):
    out = []
    for d, s in phases:
        out += [s] * d
    return out


@unittest.skipUnless((SIGNALS / "20220810").is_dir(), "19일 신호 자료(data/raw/signals)가 없으면 건너뜀")
class SignalTimelineChecks(unittest.TestCase):
    def test_every_day_is_continuous_and_exactly_one_day(self):
        for date in DATES:
            segs, _ = aihub_day_timeline(date)
            self.assertAlmostEqual(sum(d for _, d, _ in segs), 86400, msg=date)
            for (a0, ad, _), (b0, _, _) in zip(segs, segs[1:]):
                self.assertAlmostEqual(a0 + ad, b0, msg=date)
            yellows = [d for _, d, s in segs if "y" in s]
            self.assertGreaterEqual(min(yellows), 3, msg=date)
            self.assertLessEqual(max(yellows), 4, msg=date)

    def test_phase_running_at_file_end_continues_into_next_file(self):
        # 20220810 08:00 파일은 826초에서 끝나고 08:15 파일은 같은 현시 10초로 시작한다 → 84초 한 현시
        files = signal_xml_files("20220810")
        last = [(d, s) for d, s in read_signal_phases(files["0800"]) if d > 0]
        self.assertEqual(sum(d for d, _ in last), 826)
        first = read_signal_phases(files["0815"])[0]
        segs, _ = aihub_day_timeline("20220810")
        joined = [x for x in segs if x[0] == 8 * 3600 + 826]
        self.assertEqual(len(joined), 1)
        self.assertEqual(joined[0][2], first[1])
        self.assertEqual(joined[0][1], 74 + first[0])

    def test_no_green_to_red_without_yellow(self):
        for date in DATES:
            states = per_second(aihub_signal_program(date, 0, 86400))
            self.assertEqual(len(states), 86400)
            bad = sum(1 for a, b in zip(states, states[1:]) if any(x in "Gg" and y == "r" for x, y in zip(a, b)))
            self.assertEqual(bad, 0, msg=date)

    def test_recording_dropout_is_filled(self):
        # 20220810 15:17의 325초 녹색은 기록 누락이다(같은 시간 1분 통과 기록에 석천로 차량이 평소대로 지나감).
        segs, notes = aihub_day_timeline("20220810")
        window = [d for s0, d, s in segs if 15 * 3600 <= s0 < 16 * 3600 and "y" not in s]
        self.assertLess(max(window), 120)
        self.assertEqual(aihub_unreliable_periods("20220810"), ["1515", "1530"])

    def test_episode_program_covers_duration_and_beyond_midnight(self):
        phases = replay_phases("2345", date="20220810")
        self.assertEqual(sum(d for d, _ in phases), 1800)
        phases = replay_phases("2345", date="20220814")   # 다음 날 자료 없음 → 마지막 주기 반복
        self.assertEqual(sum(d for d, _ in phases), 1800)

    def test_window_program_is_exact_slice_of_day_program(self):
        """30분 실행의 신호는 하루 시간축의 같은 구간과 1초도 다르지 않다(파일 경계에서 밀리지 않음)."""
        day = per_second(aihub_signal_program("20220810", 0, 86400))
        for period in ("0800", "1745"):
            start = int(period[:2]) * 3600 + int(period[2:]) * 60
            window = per_second(replay_phases(period, date="20220810"))
            self.assertEqual(window, day[start:start + 1800], period)


if __name__ == "__main__":
    unittest.main()
