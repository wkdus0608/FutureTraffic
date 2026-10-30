"""ARENA3 잠금·관측·B′ 결정·SMDP 전이·G1 학습 반복·심판 판정 검사(SUMO 없이)."""
import copy
import gzip
import statistics
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import arena3_bprime as bp  # noqa: E402
import arena3_common as ac  # noqa: E402
import arena3_g1 as g1  # noqa: E402
import arena3_referee as ref  # noqa: E402
import explore_weight as w  # noqa: E402

CFG = ac.config()


def base_obs(greens5, remaining=0.5):
    """WEIGHT1 110칸 모양(검사용): 모드 0~4의 첫 녹색 표시(84:104), 유효 표시(104:109), 남은 시간(109)."""
    obs = np.zeros(110, np.float32)
    for m, g in enumerate(greens5):
        obs[84 + 4 * m + g] = 1.0
    obs[104:109] = ac.canonical_mask6(list(greens5) + [greens5[0]])[:5]
    obs[109] = remaining
    return obs


def obs119(greens6, m4_costs=None):
    mask = ac.canonical_mask6(greens6)
    return ac.observation119(base_obs(greens6[:5]), greens6, mask, m4_costs)


class LockTest(unittest.TestCase):
    def test_lock_matches_design_and_split_rule(self):
        cfg = ac.validate_lock()
        self.assertEqual(cfg["dates"]["eval"], ["20220811", "20220819", "20220823", "20220906", "20220813", "20220820", "20220911"])
        self.assertEqual(len(cfg["dates"]["train"]), 8)
        self.assertFalse(set(cfg["dates"]["eval"]) & set(cfg["dates"]["train"]))
        self.assertFalse(set(cfg["dates"]["eval"]) & set(cfg["dates"]["not_used"]["test"]))
        self.assertEqual(cfg["judge"]["min_gain_s"], 0.3)
        self.assertAlmostEqual(cfg["judge"]["t975"], 2.4469, places=4)
        periods = {s["period"] for s in cfg["situations_eval"]}
        self.assertEqual(periods, {"0730", "1000", "1230", "1500", "1730", "2100"})
        for s in cfg["situations_eval"]:            # 날짜 하나에 12상황
            self.assertIn(s["tag"], ("E1", "E2"))
        self.assertEqual(statistics.mode(sum(1 for x in cfg["situations_eval"] if x["date"] == d)
                                         for d in cfg["dates"]["eval"]), 12)

    def test_route_files_reproduce_lock_and_previous_experiment(self):
        with tempfile.TemporaryDirectory() as d:
            s = CFG["situations_eval"][0]
            self.assertEqual(ac.route_file_sha(s, Path(d)), s["route_sha256"])
            case = CFG["reproduction"]["cases"][0]
            original = ac.make_situation(case["date"], case["period"], None, case["sumo_seed"])
            self.assertEqual(ac.route_file_sha(original, Path(d)), case["route_sha256"])

    def test_arrival_placement_restores_original_even_after_error(self):
        import focused_scenario as fs
        with self.assertRaises(ValueError):
            with ac.arrival_placement("E1"):
                self.assertIsNot(fs.route_seed, ac.ORIGINAL_ROUTE_SEED)
                with self.assertRaises(RuntimeError):
                    with ac.arrival_placement("E2"):
                        pass
                raise ValueError("boom")
        self.assertIs(fs.route_seed, ac.ORIGINAL_ROUTE_SEED)
        with ac.arrival_placement(None):
            self.assertIs(fs.route_seed, ac.ORIGINAL_ROUTE_SEED)

    def test_seeds_are_separate_streams(self):
        s = CFG["situations_eval"][0]
        self.assertNotEqual(ac.arrival_seed(s["tag"], s["date"], s["period"]), ac.sumo_seed(s["tag"], s["date"], s["period"]))
        self.assertLess(max(x["sumo_seed"] for x in CFG["situations_eval"]), 2 ** 31)


class ObservationTest(unittest.TestCase):
    def test_canonical_mask_and_layout(self):
        self.assertEqual(ac.canonical_mask6([1, 1, 2, 1, 2, 3]), [True, False, True, False, False, True])
        obs = obs119([1, 1, 2, 1, 2, 3], [100.0, 1100.0, None, 600.0])
        self.assertEqual(obs.shape, (119,))
        self.assertEqual(list(obs[110:114]), [0, 0, 0, 1])
        self.assertEqual(obs[114], 1.0)
        self.assertEqual(list(np.round(obs[115:119], 3)), [0.0, 1.0, 0.0, 0.5])
        quiet = obs119([2] * 6)
        self.assertEqual(list(quiet[115:119]), [0, 0, 0, 0])
        self.assertEqual(quiet[114], 0.0)

    def test_snapshot_candidates_and_mode_greens(self):
        snap = ac.ArenaSnapshot(base=None, greens=[1, 1, 2, 1, 2, 3], mask=ac.canonical_mask6([1, 1, 2, 1, 2, 3]),
                                choice=True, observation=obs119([1, 1, 2, 1, 2, 3]))
        self.assertEqual(snap.candidates, [1, 2, 3])
        self.assertEqual(snap.green(5), 3)
        with self.assertRaises(ValueError):
            snap.green(4)                              # M4는 모드 2와 같은 녹색이라 대표가 아니다
        obs = torch.from_numpy(snap.observation).view(1, -1)
        self.assertEqual(g1.mode_greens(obs)[0].tolist(), [1, 1, 2, 1, 2, 3])
        self.assertEqual(int(g1.anchor_mode(obs)[0]), 2)


class BPrimeRuleTest(unittest.TestCase):
    def test_no_or_weak_gain_keeps_m4(self):
        costs = {1: [100.0] * 8, 2: [100.0 + (5 if j % 2 else -5) for j in range(8)]}
        self.assertEqual(bp.choose(1, costs)["chosen"], 1)
        weak = {1: [100.0] * 8, 2: [99.0, 101.0, 98.0, 103.0, 99.5, 100.5, 97.0, 104.0]}
        self.assertEqual(bp.choose(1, weak)["chosen"], 1)

    def test_clear_gain_switches_and_picks_largest(self):
        costs = {1: [100.0 + j for j in range(8)], 2: [90.0 + j for j in range(8)], 3: [80.0 + j + (j % 2) for j in range(8)]}
        d = bp.choose(1, costs)
        self.assertEqual(d["chosen"], 3)
        self.assertAlmostEqual(d["gains"]["2"]["mean"], 10.0)

    def test_dropped_futures_are_dropped_for_all_candidates(self):
        costs = {1: [100.0] * 8, 2: [90.0] * 8}
        for j in range(5):
            costs[2][j] = None
        d = bp.choose(1, costs)
        self.assertEqual(d["valid_futures"], 3)
        self.assertEqual(d["chosen"], 1)            # 미래 4개 미만이면 바꾸지 않는다

    def test_trace_table_tolerance(self):
        a = {"time": 10.0, "tls": "GGrr", "pending": [], "vehicles": [("v", "l", 1.0, 2.0)]}
        b = copy.deepcopy(a)
        b["vehicles"] = [("v", "l", 1.0 + 5e-7, 2.0)]
        self.assertTrue(bp.same_table(a, b, 1e-6))
        b["vehicles"] = [("v", "l", 1.0 + 5e-6, 2.0)]
        self.assertFalse(bp.same_table(a, b, 1e-6))
        b = copy.deepcopy(a)
        b["pending"] = ["x"]
        self.assertFalse(bp.same_table(a, b, 1e-6))


class SmdpTest(unittest.TestCase):
    def test_choice_to_choice_rewards_and_discounts(self):
        rewards = [-1.0] * 10
        rewards[-1] = -5.0                              # 꼬리를 접은 마지막 단계
        ch = [{"step": 2, "observation": obs119([0, 0, 1, 0, 0, 0]), "mode": 0},
              {"step": 5, "observation": obs119([1, 1, 1, 1, 2, 2]), "mode": 4},
              {"step": 8, "observation": obs119([2, 2, 2, 3, 3, 3]), "mode": 3}]
        one = g1.smdp_transitions(ch, rewards, 0.5, 1)
        self.assertAlmostEqual(one[0]["reward"], -(1 + 0.5 + 0.25))
        self.assertAlmostEqual(one[0]["discount"], 0.125)
        self.assertFalse(one[0]["done"])
        self.assertTrue(np.array_equal(one[0]["next_obs"], ch[1]["observation"]))
        self.assertAlmostEqual(one[2]["reward"], -(1 + 0.5 * 5))
        self.assertTrue(one[2]["done"])
        two = g1.smdp_transitions(ch, rewards, 0.5, 2)
        self.assertAlmostEqual(two[0]["reward"], one[0]["reward"] + 0.125 * one[1]["reward"])
        self.assertAlmostEqual(two[0]["discount"], 0.125 * 0.125)
        self.assertTrue(np.array_equal(two[0]["next_obs"], ch[2]["observation"]))
        self.assertTrue(two[1]["done"])
        with self.assertRaises(ValueError):
            g1.smdp_transitions([ch[1], ch[0]], rewards, 0.5, 1)


def fake_episode(job, *, discard=False):
    """작업자 대신: 선택 지점 3개(모드 정책을 실제로 부름)와 720단계 보상."""
    net = g1._net_from_bytes(job["net"])
    pol = g1.EpsilonPolicy(net, job["base_choices"], job["rng_seed"], g1.gcfg())
    greens = [[0, 0, 1, 0, 0, 2], [1, 1, 1, 1, 3, 3], [2, 2, 2, 2, 2, 0]]
    for k, gr in enumerate(greens):
        snap = ac.ArenaSnapshot(base=None, greens=gr, mask=ac.canonical_mask6(gr), choice=True, observation=obs119(gr))
        pol.decide(snap, type("C", (), {"step": 100 * (k + 1)})())
    status = "discard" if discard else "ok"
    return {"ok": True, "status": status, "errors": [], "metrics": {"mean_delay_s": 40.0, "choice_points": 3},
            "choices": pol.taken, "rewards": [-0.3] * 720, "seconds": 60.0,
            "collision_details": {"x": 1} if discard else None}


class G1Test(unittest.TestCase):
    def test_untrained_greedy_is_m4_representative(self):
        net = g1.DuelingNet()
        for gr in ([0, 0, 1, 0, 0, 2], [1, 1, 1, 1, 3, 3], [3, 2, 2, 1, 1, 0]):
            obs = obs119(gr)
            mode = g1.greedy_mode(net, obs)
            self.assertEqual(gr[mode], gr[ac.M4])

    def test_epsilon_schedule_counts_choices(self):
        cfg = g1.gcfg()
        self.assertEqual(g1.epsilon_at(0, cfg), 0.5)
        self.assertAlmostEqual(g1.epsilon_at(1500, cfg), 0.3)
        self.assertAlmostEqual(g1.epsilon_at(10 ** 6, cfg), 0.1)

    def test_scenarios_use_train_dates_and_are_deterministic(self):
        train = set(CFG["dates"]["train"])
        for e in range(50):
            s = g1.scenario_for(42, e)
            self.assertIn(s["date"], train)
            self.assertIn(s["period"], ac.TRAIN_SLOTS)
            self.assertEqual(s, g1.scenario_for(42, e))
        self.assertNotEqual(g1.scenario_for(42, 0)["tag"], g1.scenario_for(43, 0)["tag"])

    def test_model_save_load_and_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            learner = g1.Learner(42)
            sha = g1.save_model(Path(d) / "final", learner, {"seed": 42})
            self.assertEqual(g1.model_identity(Path(d) / "final"), sha)
            net = g1.load_model(Path(d) / "final")
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(net.state_dict().values(), learner.online.state_dict().values())))
            (Path(d) / "final" / "model.pt").write_bytes(b"x")
            with self.assertRaises(ValueError):
                g1.model_identity(Path(d) / "final")

    def test_training_loop_learns_resumes_and_finishes(self):
        cfg = copy.deepcopy(CFG)
        cfg["g1"]["learning_starts"] = 4
        cfg["g1"]["updates_per_transition"] = 2
        cfg["g1"]["cpu_hours_per_seed"] = 5 * 2 * 60 / 3600     # 묶음 5개(에피소드 60초 × 2)
        with tempfile.TemporaryDirectory() as d, patch.object(ac, "config", return_value=cfg):
            out = Path(d)
            first = g1.train(42, out, out / "work", executor=ThreadPoolExecutor(2), worker=fake_episode, max_batches=2)
            self.assertFalse(first["finished"])
            self.assertEqual(first["episodes"], 4)
            self.assertGreater(first["updates"], 0)
            before = torch.load(out / "seed_42" / "checkpoint.pt", weights_only=False)["state"]["episodes"]
            self.assertEqual(before, 4)
            done = g1.train(42, out, out / "work", executor=ThreadPoolExecutor(2), worker=fake_episode)
            self.assertTrue(done["finished"])
            self.assertEqual(done["episodes"], 10)
            self.assertEqual(done["choices"], 30)
            self.assertTrue((out / "seed_42" / "final" / "model.pt").exists())
            g1.model_identity(out / "seed_42" / "final")

    def test_discarded_episodes_never_enter_replay(self):
        cfg = copy.deepcopy(CFG)
        cfg["g1"]["cpu_hours_per_seed"] = 2 * 60 / 3600
        with tempfile.TemporaryDirectory() as d, patch.object(ac, "config", return_value=cfg):
            r = g1.train(43, Path(d), Path(d) / "w", executor=ThreadPoolExecutor(2),
                         worker=lambda job: fake_episode(job, discard=True))
            self.assertEqual(r["discarded"], 2)
            self.assertEqual(r["choices"], 0)


# ----------------------------------------------------------------------------- 심판
def fake_run(s, delay, *, policy="M4", lefts=(100.0, 80.0), long_waits=0, events=(), quality_ok=True, n=100):
    sid = ac.situation_id(s)
    return {"situation": sid, "policy": policy, "vehicles": n, "route_sha256": s["route_sha256"], "cohort_sha256": "c" + sid,
            "actual_total_delay": delay * n, "mean_delay_s": delay, "long_waits": long_waits, "missing": 0, "unfinished": 0,
            "changed_by_rules": 0, "mode_action_errors": 0, "p0_mismatch": 0, "cost_check": {"passed": quality_ok},
            "collisions_at_204820": 0,
            "collision_details": {"collisions_unexplained": [{"time": t, "lane": "L", "collider": "a", "victim": "b"} for t in events]},
            "by_movement": {ref.LEFT_KEYS["left_east"]: {"vehicles": 10, "mean_delay_s": lefts[0]},
                            ref.LEFT_KEYS["left_south"]: {"vehicles": 10, "mean_delay_s": lefts[1]}}}


class JudgeTest(unittest.TestCase):
    sits = CFG["situations_eval"]

    def runs(self, gain_by_date=None, **kw):
        gain_by_date = gain_by_date or {}
        return [fake_run(s, 40.0 - gain_by_date.get(s["date"], 0.0), **kw) for s in self.sits]

    def test_clear_gain_passes(self):
        gains = {d: g for d, g in zip(CFG["dates"]["eval"], (0.6, 0.5, 0.7, 0.4, 0.6, 0.5, 0.55))}
        r = ref.judge([self.runs(gains)], self.runs(), self.runs(), CFG, self.sits)
        self.assertEqual(r["status"], "pass")
        self.assertAlmostEqual(r["gain_date_mean_s"], statistics.fmean(gains.values()))
        self.assertEqual(r["positive_dates"], 7)

    def test_small_or_noisy_gain_fails(self):
        small = {d: 0.2 for d in CFG["dates"]["eval"]}
        self.assertFalse(ref.judge([self.runs(small)], self.runs(), None, CFG, self.sits)["checks"]["gain_at_least_min"])
        noisy = {d: g for d, g in zip(CFG["dates"]["eval"], (3.0, -2.0, 2.5, -1.5, 2.0, -1.0, 0.5))}
        r = ref.judge([self.runs(noisy)], self.runs(), None, CFG, self.sits)
        self.assertFalse(r["checks"]["date_t95_lower_positive"])
        self.assertEqual(r["status"], "fail")

    def test_left_turn_and_long_wait_guards(self):
        gains = {d: 1.0 for d in CFG["dates"]["eval"]}
        r = ref.judge([self.runs(gains, lefts=(100.0, 86.0))], self.runs(), None, CFG, self.sits)
        self.assertFalse(r["checks"]["left_south_within_tolerance"])
        r = ref.judge([self.runs(gains, long_waits=1)], self.runs(), None, CFG, self.sits)
        self.assertFalse(r["checks"]["long_waits_not_more_than_m4"])

    def test_learner_models_are_averaged(self):
        a = {d: 1.0 for d in CFG["dates"]["eval"]}
        b = {d: 0.0 for d in CFG["dates"]["eval"]}
        r = ref.judge([self.runs(a), self.runs(b), self.runs(a)], self.runs(), None, CFG, self.sits)
        self.assertAlmostEqual(r["gain_date_mean_s"], 2 / 3)
        self.assertEqual(r["models"], 3)

    def test_collisions_shared_hold_and_specific_fail(self):
        gains = {d: 1.0 for d in CFG["dates"]["eval"]}
        cand = self.runs(gains)
        m4 = self.runs()
        cand[0] = fake_run(self.sits[0], 39.0, events=("100.00",))
        m4[0] = fake_run(self.sits[0], 40.0, events=("100.00",))
        r = ref.judge([cand], m4, None, CFG, self.sits)
        self.assertEqual(r["status"], "hold_shared_collision")
        cand[1] = fake_run(self.sits[1], 39.0, events=("200.00",))
        r = ref.judge([cand], m4, None, CFG, self.sits)
        self.assertFalse(r["checks"]["no_policy_specific_collision"])
        self.assertEqual(r["status"], "fail")

    def test_pairing_and_route_mismatch_raise(self):
        cand = self.runs()
        cand[3] = dict(cand[3], cohort_sha256="other")
        with self.assertRaises(ValueError):
            ref.judge([cand], self.runs(), None, CFG, self.sits)
        with self.assertRaises(ValueError):
            ref.judge([self.runs()[:-1]], self.runs(), None, CFG, self.sits)
        m4 = self.runs()
        m4[0] = dict(m4[0], route_sha256="x")
        with self.assertRaises(ValueError):
            ref.judge([[dict(r, route_sha256="x") if i == 0 else r for i, r in enumerate(self.runs())]], m4, None, CFG, self.sits)

    def test_quality_error_fails(self):
        gains = {d: 1.0 for d in CFG["dates"]["eval"]}
        cand = self.runs(gains)
        cand[5] = fake_run(self.sits[5], 39.0, quality_ok=False)
        r = ref.judge([cand], self.runs(), None, CFG, self.sits)
        self.assertFalse(r["checks"]["quality_clean"])


def write_fake(folder, policy, rows, fp="fp", evaluation="ev"):
    folder = Path(folder)
    w.write(folder / "experiment.json", {"fingerprint": fp, "evaluation_fingerprint": evaluation})
    for m in rows:
        target = folder / "runs" / policy / m["situation"]
        target.mkdir(parents=True, exist_ok=True)
        w.write(target / "metrics.json", m)
        with gzip.open(target / "decisions.jsonl.gz", "wt") as f:
            f.write("{}\n")
        w.write(target / "cache.json", {"fingerprint": fp, "model_identity": None,
                                        "metrics_sha256": w.file_hash(target / "metrics.json"),
                                        "decisions_sha256": w.file_hash(target / "decisions.jsonl.gz")})


class JudgeCommandTest(unittest.TestCase):
    def test_judge_reads_folders_skips_incomplete_and_checks_fingerprint(self):
        sits = CFG["situations_eval"]
        with tempfile.TemporaryDirectory() as d:
            base, g, b, other = (Path(d) / x for x in ("base", "g1", "bp", "other"))
            write_fake(base, "M4", [fake_run(s, 40.0) for s in sits])
            write_fake(base, "P0", [fake_run(s, 41.0, policy="P0") for s in sits])
            write_fake(base, "W4", [fake_run(s, 40.1, policy="W4") for s in sits])
            for i in range(3):
                write_fake(g, f"G1_m{i}", [fake_run(s, 39.0, policy=f"G1_m{i}") for s in sits])
            write_fake(b, "BPRIME", [fake_run(s, 39.5, policy="BPRIME") for s in sits[:10]])
            write_fake(other, "X", [fake_run(s, 30.0, policy="X") for s in sits], evaluation="different")
            args = type("A", (), {"baseline": base, "out": Path(d),
                                  "candidate": [f"W4={base}", f"G1={g}", f"BPRIME={b}", f"X={other}", f"NONE={Path(d) / 'none'}"]})()
            ref.cmd_judge(args)
            result = w.read(Path(d) / "judge.json")
            self.assertEqual(set(result["results"]), {"W4", "G1"})
            self.assertEqual(result["results"]["G1"]["status"], "pass")
            self.assertEqual(result["results"]["G1"]["models"], 3)
            self.assertEqual(result["results"]["W4"]["status"], "fail")
            self.assertAlmostEqual(-result["sensitivity_m4_vs_p0"]["gain_date_mean_s"], 1.0)
            self.assertIn("감도 있음", (Path(d) / "summary.md").read_text())
            # 학습기 모델이 하나만 끝났으면 판정하지 않는다
            import shutil
            shutil.rmtree(g / "runs" / "G1_m1")
            shutil.rmtree(g / "runs" / "G1_m2")
            ref.cmd_judge(args)
            self.assertNotIn("G1", w.read(Path(d) / "judge.json")["results"])


class RegisterTest(unittest.TestCase):
    def test_simultaneous_first_registration_and_code_change(self):
        with tempfile.TemporaryDirectory() as d:
            with ThreadPoolExecutor(4) as pool:
                fps = list(pool.map(lambda _: ref.register(Path(d), extra_files=("src/arena3_g1.py",)), range(4)))
            self.assertEqual(len(set(fps)), 1)
            self.assertEqual(ref.evaluation_fingerprint(Path(d)), w.read(Path(d) / "experiment.json")["evaluation_fingerprint"])
            with self.assertRaises(ValueError):                 # 같은 폴더를 다른 코드 묶음으로 이어 쓰지 않는다
                ref.register(Path(d), extra_files=("src/arena3_bprime.py",))
            other = Path(d) / "other"
            ref.register(other, extra_files=("src/arena3_bprime.py",))
            self.assertEqual(ref.evaluation_fingerprint(other), ref.evaluation_fingerprint(Path(d)))


class ColabHelperTest(unittest.TestCase):
    def test_launch_records_exit_codes_and_watch_reports_failure(self):
        import arena3_colab as colab
        with tempfile.TemporaryDirectory() as d:
            logs = Path(d) / "logs"
            pids = colab.launch({"good": ["-c", "print('안녕')"], "bad": ["-c", "import sys; sys.exit(3)"]}, Path(d), logs)
            code = colab.watch(pids, logs, every_s=1)
            self.assertEqual(code, 1)
            self.assertEqual((logs / "good.exit").read_text().strip(), "0")
            self.assertEqual((logs / "bad.exit").read_text().strip(), "3")
            self.assertIn("안녕", (logs / "good.log").read_text())
            pids = colab.launch({"good": ["-c", "print(1)"]}, Path(d), logs)
            self.assertEqual(colab.watch(pids, logs, every_s=1), 0)


class RunJobsTest(unittest.TestCase):
    def test_jobs_cached_failures_collected_and_others_continue(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            sits = CFG["situations_eval"][:4]
            jobs = ref.jobs_for(out, out / "work", "fp", "M4", sits)
            calls = []

            def worker(job):
                calls.append(job["sid"])
                if job["sid"] == jobs[1]["sid"]:
                    return {"ok": False, "seconds": 1.0, "name": job["sid"], "error": "boom"}
                target = Path(job["target"])
                target.mkdir(parents=True, exist_ok=True)
                w.write(target / "metrics.json", {"situation": job["sid"]})
                with gzip.open(target / "decisions.jsonl.gz", "wt") as f:
                    f.write("{}\n")
                w.write(target / "cache.json", {"fingerprint": "fp", "model_identity": None,
                                                "metrics_sha256": w.file_hash(target / "metrics.json"),
                                                "decisions_sha256": w.file_hash(target / "decisions.jsonl.gz")})
                return {"ok": True, "seconds": 2.0, "name": job["sid"]}
            status = ref.run_jobs(jobs, 2, ref.Ledger(out, "t"), worker=worker, executor=ThreadPoolExecutor(2))
            self.assertEqual(status["done"], 3)
            self.assertEqual(len(status["failed"]), 1)
            calls.clear()
            again = ref.run_jobs(jobs, 2, ref.Ledger(out, "t"), worker=worker, executor=ThreadPoolExecutor(2))
            self.assertEqual(calls, [jobs[1]["sid"]])
            self.assertEqual(again["done"], 0)
            self.assertAlmostEqual(ref.Ledger(out, "t").seconds, 2.0 * 3 + 1.0 + 1.0)
            # 해시가 바뀐 결과는 재사용하지 않고 멈춘다
            w.write(Path(jobs[0]["target"]) / "metrics.json", {"situation": "tampered"})
            with self.assertRaises(ValueError):
                ref.cached(jobs[0])


if __name__ == "__main__":
    unittest.main()
