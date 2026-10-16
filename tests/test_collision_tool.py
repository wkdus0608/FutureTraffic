"""충돌 다시 돌려 보기 도구(focused_collision)의 가벼운 검사. SUMO를 돌리지 않는다."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from focused_collision import find_cases, junction_prefix  # noqa: E402


class CollisionToolTest(unittest.TestCase):
    def test_junction_prefix(self):
        self.assertEqual(junction_prefix(":204820_15_0"), ":204820_")
        self.assertEqual(junction_prefix(":SN20220622173835A_5_5"), ":SN20220622173835A_")
        self.assertEqual(junction_prefix(":cluster_204796_205202_3_1"), ":cluster_204796_205202_")
        self.assertIsNone(junction_prefix("477706883_01_0"))     # 교차로 안 차로가 아님

    def test_find_cases_reads_only_unexplained(self):
        with tempfile.TemporaryDirectory() as d:
            event = {"time": "70410.00", "lane": ":204820_15_0", "collider": "a", "victim": "b"}
            for run, unexplained, arrivals in (("greedy_eval42_platoon", [event], "platoon"), ("fixed_eval42", [], "random")):
                path = Path(d) / "20220928" / run
                path.mkdir(parents=True)
                (path / "metrics.json").write_text(json.dumps(
                    {"date": "20220928", "policy": run.split("_")[0], "arrivals": arrivals, "eval_seed": 42,
                     "collisions_unexplained": unexplained}))
            cases = find_cases(Path(d))
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["name"], "20220928_greedy_platoon_70410")
        self.assertEqual((cases[0]["time"], cases[0]["collider"], cases[0]["victim"]), (70410.0, "a", "b"))


if __name__ == "__main__":
    unittest.main()
