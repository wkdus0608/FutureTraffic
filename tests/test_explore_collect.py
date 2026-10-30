"""[추가 탐색] 새 날짜 3개 수집과 학생 연속 제어 평가(141절) 검사 — SUMO 없음. 날짜 나누기·학생 자료·판정을 본다."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import explore_collect as ec  # noqa: E402
import explore_online as eo  # noqa: E402
import explore_rollout as er  # noqa: E402


class PlanTest(unittest.TestCase):
    def test_dates(self):
        split = json.loads((ROOT / "config/focused_split.json").read_text())
        new_dates = {d for d, _ in ec.NEW_GROUPS}
        eval_dates = {d for d, _ in ec.EVAL_GROUPS}
        old_dates = {d for d, _ in eo.EPISODES}
        step0_dates = {d for d, _ in er.EPISODES}
        self.assertEqual(ec.NEW_GROUPS, [("20220812", "1800"), ("20220819", "1300"), ("20220926", "1800")])
        self.assertTrue(new_dates | eval_dates <= set(split["train"]))                  # 검증·시험 날짜는 쓰지 않음
        self.assertFalse(new_dates & (old_dates | step0_dates | set(split["day_types"]["chuseok_holiday"])))
        self.assertTrue(new_dates <= set(split["day_types"]["weekday"]) and eval_dates <= set(split["day_types"]["weekday"]))
        self.assertFalse(eval_dates & (old_dates | new_dates))                          # 평가 날짜는 어떤 학습에도 안 들어감
        self.assertEqual(sorted(p for _, p in ec.NEW_GROUPS + eo.EPISODES), ["1300"] * 3 + ["1800"] * 3)
        self.assertEqual((ec.NEW_SEED, ec.EVAL_SEEDS, ec.EVAL_MODES), (42, (42, 43, 44, 45), ("p0", "s3", "s6", "s6h")))
        self.assertEqual(len(ec.eval_jobs("w", "o")), 64)


def _s(group, seed):
    return {"group": group, "seed": seed}


class StudentDataTest(unittest.TestCase):
    def test_student_samples(self):
        old = [_s(g, s) for g in ("a", "b", "c") for s in (42, 43)]
        new = [_s(g, 42) for g in ("d", "e", "f")]
        self.assertEqual(ec.student_samples("s3", old, new), old)
        self.assertEqual(len(ec.student_samples("s6", old, new)), 9)
        h = ec.student_samples("s6h", old, new)
        self.assertEqual(({x["group"] for x in h}, {x["seed"] for x in h}), (set("abcdef"), {42}))


def _m(delay, le=100.0, ls=90.0, long=5, changes=0):
    return {"mean_delay_s": delay, "left_east": le, "left_south": ls, "long_waits": long, "changes": changes}


def _rows(s3, s6, p0=40.0, s6_changes=10, s6_le=100.0):
    """평가 묶음 4개 × seed 4개. s3·s6는 묶음별 지연 목록(4개), 같은 묶음의 seed는 ±0.05씩 흔듦."""
    rows = []
    for gi, g in enumerate(("g1", "g2", "g3", "g4")):
        for k, seed in enumerate((42, 43, 44, 45)):
            j = (k - 1.5) * 0.05
            rows.append({"group": g, "seed": seed, "p0": _m(p0 + j), "s3": _m(s3[gi] + j, changes=10),
                         "s6": _m(s6[gi] + j, le=s6_le, changes=s6_changes), "s6h": _m(s3[gi] + j, changes=10)})
    return rows


class ReadingTest(unittest.TestCase):
    def test_improve(self):
        self.assertEqual(ec.improve_reading(_rows([41, 42, 41, 41], [40, 40.5, 40.2, 41.2]))["word"],
                         "새 날짜 자료로 학생 제어가 좋아졌다")                             # 묶음 3/4면 충분
        self.assertEqual(ec.improve_reading(_rows([41, 42, 41, 41], [40, 40.5, 41.3, 41.2]))["word"], "불분명")      # 평균 > 0, 묶음 2/4
        r = ec.improve_reading(_rows([41, 42, 41, 41], [40, 40.5, 40.2, 40.9]))
        self.assertEqual((r["word"], r["positive_groups"]), ("새 날짜 자료로 학생 제어가 좋아졌다", 4))
        self.assertEqual(ec.improve_reading(_rows([41, 42, 41, 41], [40, 40.5, 40.2, 40.9], s6_changes=4))["word"], "불분명")  # P0로 돌아가서만
        self.assertEqual(ec.improve_reading(_rows([41, 42, 41, 41], [40, 40.5, 40.2, 40.9], s6_le=120.0))["word"], "불분명")  # 좌회전 나빠짐
        self.assertEqual(ec.improve_reading(_rows([41, 41, 41, 41], [41.5, 41.2, 41, 41]))["word"], "변화 없음 또는 나빠짐")

    def test_beats_p0(self):
        rows = _rows([41, 42, 41, 41], [39, 39.5, 39.2, 39.9])
        self.assertEqual(ec.beats_p0(rows, "s6")["word"], "P0를 넘었다")
        self.assertEqual(ec.beats_p0(rows, "s3")["word"], "P0를 넘지 못했다")
        c = ec.compare(rows, "p0", "s6")
        self.assertEqual((c["episodes"], c["positive_episodes"], c["positive_groups"], c["changes"]), (16, 16, 4, {"s6": 160}))


if __name__ == "__main__":
    unittest.main()
