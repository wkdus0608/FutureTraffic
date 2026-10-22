"""충돌 다시 돌려 보기 도구(focused_collision)의 가벼운 검사. SUMO를 돌리지 않는다(DQN 실행은 가짜 평가·가짜 연결)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import collision_fakes as fake  # noqa: E402
import focused_collision as fc  # noqa: E402
import focused_eval  # noqa: E402
import focused_sumo  # noqa: E402
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


EVENT = {"time": "100.00", "type": "side", "lane": ":J_1_0", "collider": "a", "victim": "b"}


def dqn_run(root: Path, **override) -> Path:
    run = root / "20220928" / "dqn_seed44_d289e75a30_eval42"
    run.mkdir(parents=True)
    m = {"date": "20220928", "policy": "dqn", "arrivals": "random", "eval_seed": 42, "train_seed": 44,
         "model": "/drive/models_single/pet1/seed_44", "model_sha256": "d289e75a30" + "0" * 54, "model_timesteps": 100000,
         "evaluation_fingerprint": focused_eval.evaluation_fingerprint(), "collisions_unexplained": [EVENT]}
    m.update(override)
    (run / "metrics.json").write_text(json.dumps(m))
    return run


class DqnCaseTest(unittest.TestCase):
    def test_dqn_case_carries_model_and_run_name(self):
        with tempfile.TemporaryDirectory() as d:
            cases = fc.cases_in_run(dqn_run(Path(d)))
        self.assertEqual(len(cases), 1)
        c = cases[0]
        self.assertEqual(c["name"], "20220928_dqn_seed44_d289e75a30_eval42_100")     # 모델마다 다른 이름
        self.assertEqual((c["model"], c["train_seed"], c["model_timesteps"]), ("/drive/models_single/pet1/seed_44", 44, 100000))
        self.assertEqual(c["model_sha256"][:10], "d289e75a30")

    def test_dqn_case_without_model_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit):
                fc.cases_in_run(dqn_run(Path(d), model_sha256=None))


class RerunTest(unittest.TestCase):
    def setUp(self):
        self.originals = focused_eval.run_day, focused_sumo.make_environment

    def tearDown(self):
        focused_eval.run_day, focused_sumo.make_environment = self.originals

    def test_dqn_rerun_passes_model_records_window_and_finds_same_collision(self):
        seen = {}
        envs = []

        def fake_make(**kwargs):
            envs.append(fake.FakeEnv())
            return envs[-1]

        def fake_run_day(job):
            seen.update(job)
            env = focused_sumo.make_environment(seed=job["eval_seed"])     # 평가처럼 실행할 때 이름을 가져다 씀
            fake.write_collisions(Path(job["run_dir"]), [EVENT])
            fake.run_until_stopped(env)

        focused_sumo.make_environment, focused_eval.run_day = fake_make, fake_run_day
        with tempfile.TemporaryDirectory() as d:
            case = fc.cases_in_run(dqn_run(Path(d) / "eval"))[0]
            r = fc.rerun(case, str(Path(d) / "out"))
            self.assertTrue((Path(d) / "out" / case["name"] / "case.json").exists())
            self.assertTrue((Path(d) / "out" / case["name"] / "trajectory.csv").exists())
        self.assertIs(focused_sumo.make_environment, fake_make)                       # 끝나면 되돌림
        self.assertEqual({k: seen[k] for k in fc.DQN_KEYS}, {k: case[k] for k in fc.DQN_KEYS})   # 같은 모델로
        self.assertEqual((seen["date"], seen["policy"], seen["eval_seed"], seen["arrivals"]), ("20220928", "dqn", 42, "random"))
        self.assertEqual(seen["fingerprint"], focused_eval.evaluation_fingerprint())
        self.assertEqual(envs[0].steps, 103)                                          # 충돌 + 3초에서 멈춤
        self.assertTrue(r["reproduced"])
        self.assertEqual(r["window_s"], [60.0, 103.0])
        s = r["summary"]
        self.assertEqual(s["victim"]["stopped_s_before_collision"], 41)               # 60~100초 내내 멈춤
        self.assertEqual(s["collider"]["stopped_s_before_collision"], 0)
        self.assertEqual(s["collider"]["entered_junction_s"], 95.0)
        self.assertEqual([a["edge"] for a in s["victim"]["route_ahead_at_collision"]], ["e2", "e3", "e4"])
        self.assertEqual(s["victim"]["route_ahead_at_collision"][0], {"edge": "e2", "vehicles": 4, "halting": 3, "length_m": 25.0})

    def test_not_reproduced_when_collision_differs(self):
        focused_sumo.make_environment = lambda **kw: fake.FakeEnv()

        def fake_run_day(job):
            env = focused_sumo.make_environment()
            fake.write_collisions(Path(job["run_dir"]), [{**EVENT, "victim": "c"}])
            fake.run_until_stopped(env)

        focused_eval.run_day = fake_run_day
        with tempfile.TemporaryDirectory() as d:
            r = fc.rerun(fc.cases_in_run(dqn_run(Path(d) / "eval"))[0], str(Path(d) / "out"))
        self.assertFalse(r["reproduced"])
        self.assertEqual(r["collisions_until_window_end"], 1)

    def test_refuses_other_evaluation_code(self):
        focused_eval.run_day = lambda job: self.fail("평가 지문이 다르면 다시 돌리지 않아야 한다")
        with tempfile.TemporaryDirectory() as d:
            case = fc.cases_in_run(dqn_run(Path(d), evaluation_fingerprint="000000000000"))[0]
            with self.assertRaises(SystemExit):
                fc.rerun(case, str(Path(d) / "out"))

    def test_reads_through_local_view_wrapper(self):
        conn = fake.fake_connection({"t": 100.0})
        self.assertIs(fc.raw_connection(fake.LocalView(conn)), conn)
        self.assertIs(fc.raw_connection(conn), conn)


if __name__ == "__main__":
    unittest.main()
