"""[추가 탐색] 예측 제어 충돌 재현 준비 검사 — SUMO 실행 없음(가짜 결과, 다시 돌리는 함수는 가짜로 바꿔 끼움)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import explore_pred as ep  # noqa: E402
import explore_pred_collision as epc  # noqa: E402
import focused_collision  # noqa: E402
import focused_eval  # noqa: E402

EVENT = {"time": "51396.00", "type": "side", "lane": ":204812_2_4", "collider": "1415_p455", "victim": "1415_p598"}


def fake_run(root: Path, **override) -> Path:
    run = root / "20220813" / "pred_p0_eval42_platoon"
    run.mkdir(parents=True)
    m = {"date": "20220813", "policy": "greedy", "arrivals": "platoon", "eval_seed": 42,
         "evaluation_fingerprint": focused_eval.evaluation_fingerprint(), "collisions_unexplained": [EVENT],
         "pred": {"variant": "p0", "horizon_s": 120, "code": ep.code_fingerprint()}}
    m.update(override)
    (run / "metrics.json").write_text(json.dumps(m))
    return run


class BuildCasesTest(unittest.TestCase):
    def test_case_from_pred_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            cases = epc.build_cases(fake_run(Path(tmp)))
        self.assertEqual(len(cases), 1)
        c = cases[0]
        self.assertEqual(c["name"], "20220813_pred_p0_platoon_51396")
        self.assertEqual((c["policy"], c["arrivals"], c["eval_seed"], c["time"]), ("greedy", "platoon", 42, 51396.0))
        self.assertEqual((c["pred_variant"], c["pred_horizon_s"]), ("p0", 120))
        self.assertEqual(focused_collision.junction_prefix(c["lane"]), ":204812_")   # 그 교차로 안 차를 기록

    def test_refuses_other_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                epc.build_cases(fake_run(Path(tmp) / "a", pred=None))                  # 예측 제어 실행이 아님
            with self.assertRaises(SystemExit):
                epc.build_cases(fake_run(Path(tmp) / "b", evaluation_fingerprint="000000000000"))
            self.assertEqual(epc.build_cases(fake_run(Path(tmp) / "c", collisions_unexplained=[])), [])


class RerunCaseTest(unittest.TestCase):
    def setUp(self):
        self.original_queue, self.original_rerun = focused_eval.QueueFirst, focused_collision.rerun

    def tearDown(self):
        focused_eval.QueueFirst, focused_collision.rerun = self.original_queue, self.original_rerun

    def test_installs_same_controller_then_reruns(self):
        seen = {}

        def fake_rerun(case, out):
            cls = focused_eval.QueueFirst
            seen.update(name=cls.__name__, out=out, case=case["name"])
            return {"ok": True}

        focused_collision.rerun = fake_rerun
        with tempfile.TemporaryDirectory() as tmp:
            case = epc.build_cases(fake_run(Path(tmp)))[0]
            self.assertEqual(epc.rerun_case(case, "/out"), {"ok": True})
        self.assertEqual(seen, {"name": "PredictiveControl_p0", "out": "/out", "case": "20220813_pred_p0_platoon_51396"})

    def test_refuses_different_controller_code(self):
        focused_collision.rerun = lambda case, out: self.fail("다른 코드로는 다시 돌리지 않아야 한다")
        with tempfile.TemporaryDirectory() as tmp:
            case = epc.build_cases(fake_run(Path(tmp)))[0]
        case["pred_code"] = "000000000000"
        with self.assertRaises(SystemExit):
            epc.rerun_case(case, "/out")
        self.assertIs(focused_eval.QueueFirst, self.original_queue)               # 끼우기 전에 멈춤


if __name__ == "__main__":
    unittest.main()
