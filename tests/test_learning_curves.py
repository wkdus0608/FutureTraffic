"""학습곡선 도구 검사 — 가짜 학습 기록(SUMO·학습 없음)."""
import csv
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import learning_curves as lc  # noqa: E402


def fake_rows(slope_per_10k: float, seed: int, n: int = 138) -> list[dict]:
    """시각·날짜 효과가 큰 가짜 에피소드(무작위 시나리오) + 일정한 기울기."""
    rng = random.Random(seed)
    hour_eff = {h: -0.2 * abs(h - 3) for h in range(24)}
    date_eff = {f"d{i}": 0.3 * (i % 3) for i in range(13)}
    rows = []
    for k in range(n):
        h, d = rng.randrange(24), f"d{rng.randrange(13)}"
        s0 = k * lc.EPISODE_STEPS
        r = -3.5 + hour_eff[h] + date_eff[d] + slope_per_10k * (s0 / 10_000) + rng.gauss(0, 0.05)
        rows.append({"start": s0, "end": s0 + lc.EPISODE_STEPS, "reward": r, "date": d, "hour": h})
    return rows


class TrendTest(unittest.TestCase):
    def _judge(self, slope):
        raw = {s: fake_rows(slope, s) for s in lc.SEEDS}
        fit = lc.fit_difficulty([r for rows in raw.values() for r in rows])
        return lc.judge({s: lc.adjusted(rows, fit) for s, rows in raw.items()})

    def test_flat_is_flat(self):
        j = self._judge(0.0)
        self.assertTrue(j["ci95"][0] < 0 < j["ci95"][1], j)
        self.assertTrue(j["verdict"].startswith("평평"))

    def test_rising_and_falling_are_found_despite_scenario_effects(self):
        self.assertEqual(self._judge(0.05)["verdict"], "아직 나아지는 중")
        self.assertEqual(self._judge(-0.05)["verdict"], "나빠지는 중")
        self.assertAlmostEqual(self._judge(0.05)["slope_per_10k"], 0.05, delta=0.01)


class ResumedRunTest(unittest.TestCase):
    def test_drops_discarded_episodes_and_rebuilds_scenarios(self):
        scen = [("20220811", f"{h:02d}00") for h in range(1, 23)]
        original = lc._scenarios
        lc._scenarios = lambda kind: scen
        try:
            with tempfile.TemporaryDirectory() as tmp:
                d = Path(tmp)
                rng = random.Random(46)
                first = [rng.choice(scen) for _ in range(113)]                 # 끊긴 실행: 113개(81,360단계)까지 기록
                second = [random.Random(1046).choice(scen) for _ in range(1)]
                with open(d / "monitor.csv", "w", newline="") as f:
                    f.write('#{"t_start": 0}\n')
                    w = csv.writer(f)
                    w.writerow(["r", "l", "t"])
                    for k in range(113):
                        w.writerow([-3000 - k, 720, 30.0 * (k + 1)])
                    for k in range(27):                                        # 8만 단계에서 이어 학습, 27개
                        w.writerow([-2000 - k, 720, 30.0 * (k + 1)])
                json.dump({"resumed": True, "timesteps": 100_000, "scenarios_used_this_run": second * 28},
                          open(d / "training_config.json", "w"))
                rows = lc.episodes(d, "single", 46)
        finally:
            lc._scenarios = original
        self.assertEqual(len(rows), 111 + 27)                                  # 8만 단계 뒤에 끝난 끊긴 부분 2개를 뺌
        self.assertEqual(rows[110]["end"], 79_920)
        self.assertEqual(rows[111]["start"], 80_000)
        self.assertEqual((rows[0]["date"], rows[0]["hour"]), (first[0][0], int(first[0][1][:2])))
        self.assertEqual(rows[111]["reward"], -2000 / 720)


if __name__ == "__main__":
    unittest.main()
