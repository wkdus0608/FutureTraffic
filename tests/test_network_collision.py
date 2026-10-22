"""12개 지도 조각 충돌 재현 도구(network_collision) 검사 — SUMO 실행 없음(가짜 조각 평가·가짜 연결)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import collision_fakes as fake  # noqa: E402
import focused_collision as fc  # noqa: E402
import focused_sumo  # noqa: E402
import network_collision as ncol  # noqa: E402
import network_eval  # noqa: E402

EVENT = {"time": "100.00", "type": "junction", "lane": ":J_1_0", "collider": "a", "victim": "b"}


def window_run(root: Path, **override) -> Path:
    run = root / "windows" / "20220928_08" / "dqn_seed42_52fefd83af_eval42"
    run.mkdir(parents=True)
    m = {"date": "20220928", "hour": 8, "policy": "dqn", "eval_seed": 42, "train_seed": 42,
         "model": "/drive/models_12map/pet1/seed_42/imitation", "model_sha256": "52fefd83af" + "0" * 54,
         "model_timesteps": 0, "handover_s": 30, "evaluation_fingerprint": network_eval.evaluation_fingerprint(),
         "collisions_in_204820": 1, "collisions_in_204820_events": [EVENT]}
    m.update(override)
    (run / "metrics.json").write_text(json.dumps(m))
    return run


class BuildCasesTest(unittest.TestCase):
    def test_case_from_window_run(self):
        with tempfile.TemporaryDirectory() as d:
            cases = ncol.build_cases(window_run(Path(d)))
        self.assertEqual(len(cases), 1)
        c = cases[0]
        self.assertEqual(c["name"], "20220928_08_dqn_seed42_52fefd83af_eval42_100")
        self.assertEqual((c["date"], c["hour"], c["policy"], c["eval_seed"], c["train_seed"]), ("20220928", 8, "dqn", 42, 42))
        self.assertEqual(c["model"], "/drive/models_12map/pet1/seed_42/imitation")
        self.assertEqual(c["model_timesteps"], 0)                                   # 흉내만 낸 모델(학습량 0)도 그대로
        self.assertEqual((c["time"], c["collider"], c["victim"]), (100.0, "a", "b"))

    def test_refuses_other_runs(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit):
                ncol.build_cases(window_run(Path(d) / "a", hour=None))                # 하루 연속 평가
            with self.assertRaises(SystemExit):
                ncol.build_cases(window_run(Path(d) / "b", evaluation_fingerprint="000000000000"))
            with self.assertRaises(SystemExit):
                ncol.build_cases(window_run(Path(d) / "c", model=None))
            self.assertEqual(ncol.build_cases(window_run(Path(d) / "e", collisions_in_204820_events=[])), [])


class RerunCaseTest(unittest.TestCase):
    def setUp(self):
        self.originals = network_eval.run_window, focused_sumo.make_environment

    def tearDown(self):
        network_eval.run_window, focused_sumo.make_environment = self.originals

    def test_reruns_same_window_and_model_through_wrapped_connection(self):
        seen, envs = {}, []

        def fake_make(**kwargs):
            envs.append(fake.FakeEnv(wrap=True))          # 12개 지도처럼 LocalView로 감싼 연결
            return envs[-1]

        def fake_run_window(job):
            seen.update(job)
            env = focused_sumo.make_environment(seed=job["eval_seed"])
            fake.write_collisions(Path(job["run_dir"]), [EVENT])
            fake.run_until_stopped(env)

        focused_sumo.make_environment, network_eval.run_window = fake_make, fake_run_window
        with tempfile.TemporaryDirectory() as d:
            case = ncol.build_cases(window_run(Path(d) / "eval"))[0]
            r = ncol.rerun_case(case, str(Path(d) / "out"))
            saved = json.loads((Path(d) / "out" / case["name"] / "case.json").read_text())
        self.assertIs(focused_sumo.make_environment, fake_make)
        self.assertEqual(set(seen), {"date", "hour", "policy", "eval_seed", "train_seed", "model", "model_sha256",
                                     "model_timesteps", "run_dir", "fingerprint"})
        self.assertEqual((seen["hour"], seen["model"], seen["model_timesteps"]), (8, case["model"], 0))
        self.assertEqual(seen["fingerprint"], network_eval.evaluation_fingerprint())
        self.assertEqual(envs[0].steps, 103)
        self.assertTrue(r["reproduced"])
        self.assertTrue(saved["reproduced"])
        self.assertEqual(r["summary"]["victim"]["stopped_s_before_collision"], 41)
        self.assertIn("재현됨", fc.report(r))

    def test_window_ended_before_collision_is_an_error(self):
        def fake_make(**kwargs):
            return fake.FakeEnv()

        focused_sumo.make_environment = fake_make

        def short_window(job):
            env = focused_sumo.make_environment()
            fake.write_collisions(Path(job["run_dir"]), [])
            fake.run_until_stopped(env, limit=50)          # 충돌(100초) 전에 끝남

        network_eval.run_window = short_window
        with tempfile.TemporaryDirectory() as d:
            case = ncol.build_cases(window_run(Path(d) / "eval"))[0]
            with self.assertRaises(RuntimeError):
                ncol.rerun_case(case, str(Path(d) / "out"))
        self.assertIs(focused_sumo.make_environment, fake_make)                      # 실패해도 되돌림


if __name__ == "__main__":
    unittest.main()
