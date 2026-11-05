"""ARENA3 PPO 학습·평가 연결 검사(가짜 SUMO, 실제 시뮬레이터 없음)."""
import copy
import json
import math
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import arena3_common as ac
import arena3_g1 as g1
import arena3_ppo as ppo


def make_obs(greens, rng=None):
    """WEIGHT1 110칸 + W4 + M4 비용 차이 배치에 맞춘 관측(나머지 칸은 0 또는 난수)."""
    mask = ac.canonical_mask6(greens)
    obs = np.zeros(119, np.float32)
    if rng is not None:
        obs[:84] = rng.random(84)
        obs[109] = rng.random()
        obs[115:119] = rng.random(4) * 3
    for m, g in enumerate(greens[:5]):
        obs[84 + m * 4 + g] = 1
    obs[104:109] = mask[:5]
    obs[110 + greens[5]] = 1
    obs[114] = mask[5]
    return obs


def snapshot(greens, rng=None):
    return ac.ArenaSnapshot(base=None, greens=list(greens), mask=ac.canonical_mask6(greens), choice=True,
                            observation=make_obs(greens, rng))


GREENS = ([0, 0, 1, 0, 1, 2], [0, 0, 0, 0, 1, 1], [1, 1, 1, 1, 0, 0])


def fake_worker(job):
    """선택 지점 세 개. M4와 다른 녹색을 고르면 그 뒤 보상이 조금 좋아지는 가짜 환경."""
    policy = ppo.SampledPolicy(ppo._actor_from_bytes(job["actor"], job["temperature"]), job["rng_seed"])
    for step, greens in zip((20, 40, 300), GREENS):
        policy.decide(snapshot(greens), SimpleNamespace(step=step))
    rewards = [-0.05] * 720
    off = ppo.non_anchor(policy.taken)
    for i in range(300, 360):
        rewards[i] += 0.002 * off
    return {"ok": True, "status": "ok", "errors": [], "metrics": {"mean_delay_s": 30.0 - off, "choice_points": 3},
            "choices": policy.taken, "rewards": rewards, "seconds": 1.0}


def small_config():
    cfg = copy.deepcopy(ppo.config())
    cfg.update(episodes_per_seed=6, episodes_per_update=2, parallel_episodes=1, critic_warmup_updates=1)
    cfg["ppo"].update(batch_size=4, n_epochs=2)
    return cfg


class PPOMathTest(unittest.TestCase):
    def test_gae_uses_time_discounts_and_stops_at_episode_end(self):
        r, d, c = [1.0, 2.0, 3.0], [0.9, 0.8, 0.7], [0.5, 0.4, 0.3]
        v, nv = [0.1, 0.2, 0.3], [0.2, 0.3, 9.0]          # 끝 전이의 다음 값(9.0)은 쓰지 않는다
        got = ppo.gae(r, d, c, [False, False, True], v, nv)
        a2 = 3.0 - 0.3
        a1 = (2.0 + 0.8 * 0.3 - 0.2) + 0.4 * a2
        a0 = (1.0 + 0.9 * 0.2 - 0.1) + 0.5 * a1
        np.testing.assert_allclose(got, [a0, a1, a2])

    def test_episode_batch_discounts_by_elapsed_steps(self):
        choices = [{"step": s, "observation": make_obs(GREENS[0]).tolist(), "mode": 0, "log_prob": -0.5}
                   for s in (10, 40, 100)]
        rewards = [-0.01] * 720
        critic = ppo.Critic()
        with torch.no_grad():
            for p in critic.parameters():
                p.zero_()                                    # V = 0 이면 λ=1의 이득 = 몬테카를로 반환값
        cfg = ppo.config()["ppo"]
        out = ppo.episode_batch({"choices": choices, "rewards": rewards}, critic, cfg, 1.0)
        g = cfg["gamma_per_5s"]
        seg = lambda a, b: sum(g ** (i - a) * -0.01 for i in range(a, b))
        mc2 = seg(100, 720)
        mc1 = seg(40, 100) + g ** 60 * mc2
        mc0 = seg(10, 40) + g ** 30 * mc1
        np.testing.assert_allclose(out["returns"], [mc0, mc1, mc2], rtol=1e-9)
        np.testing.assert_allclose(out["advantages"], out["returns"])

    def test_entropy_and_gradients_are_finite_with_blocked_modes(self):
        actor = ppo.Actor(0.25, math.log(3))
        obs = torch.from_numpy(np.stack([make_obs(g) for g in GREENS]))
        logits = actor.logits(obs)
        self.assertTrue(torch.isinf(logits).any())
        ent = ppo.entropy(logits).mean()
        ent.backward()
        self.assertTrue(torch.isfinite(ent))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None))


class PPOPolicyTest(unittest.TestCase):
    def test_p1_starts_at_m4_with_g1_exploration(self):
        torch.manual_seed(0)
        actor = ppo.Actor(0.25, math.log(3))
        two = make_obs([0, 0, 0, 0, 1, 1])                   # 후보 2개: 모드 0(녹색 0)·모드 4(녹색 1 = M4)
        probs = ppo.log_probs(actor.logits(torch.from_numpy(two).view(1, -1)))[0].exp().detach().numpy()
        self.assertAlmostEqual(float(probs[4]), 0.75, places=6)
        self.assertAlmostEqual(float(probs[0]), 0.25, places=6)
        self.assertEqual(probs[[1, 2, 3, 5]].sum(), 0.0)
        for greens in GREENS:
            obs = make_obs(greens)
            anchor = int(g1.anchor_mode(torch.from_numpy(obs).view(1, -1))[0])
            self.assertEqual(ppo.greedy_mode(actor, obs), anchor)

    def test_p2_initial_greedy_equals_s2a_pre_choice(self):
        torch.manual_seed(1)
        net = g1.DuelingNet()
        with torch.no_grad():                                 # 사전학습한 이득 머리, 값 머리는 0(S2A_pre와 같음)
            net.advantage.weight.normal_(0, 0.5)
            net.advantage.bias.normal_(0, 0.5)
        actor = ppo.Actor(0.25, 0.0, net=net)
        rng = np.random.default_rng(3)
        checked = 0
        while checked < 400:
            greens = [int(x) for x in rng.integers(0, 4, size=6)]
            if len(set(greens)) < 2:
                continue
            obs = make_obs(greens, rng)
            self.assertEqual(ppo.greedy_mode(actor, obs), g1.greedy_mode(net, obs))
            checked += 1

    def test_sampling_never_picks_blocked_modes_and_records_log_prob(self):
        torch.manual_seed(2)
        actor = ppo.Actor(0.25, math.log(3))
        policy = ppo.SampledPolicy(actor, 7)
        for k in range(300):
            greens = GREENS[k % 3]
            snap = snapshot(greens)
            green = policy.decide(snap, SimpleNamespace(step=k))
            taken = policy.taken[-1]
            self.assertTrue(snap.mask[taken["mode"]])
            self.assertEqual(green, greens[taken["mode"]])
            logp = ppo.log_probs(actor.logits(torch.from_numpy(snap.observation).view(1, -1)))[0]
            self.assertAlmostEqual(taken["log_prob"], float(logp[taken["mode"]]), places=6)
        two = [t for t in policy.taken if t["step"] % 3 == 1]
        share = np.mean([t["mode"] == 4 for t in two])
        self.assertGreater(share, 0.6)
        self.assertLess(share, 0.9)

    def test_ppo_update_moves_probability_toward_positive_advantage(self):
        torch.manual_seed(4)
        actor = ppo.Actor(0.25, math.log(3))
        critic = ppo.Critic()
        cfg = copy.deepcopy(ppo.config()["ppo"])
        learner = ppo.Learner(actor, critic, 42, cfg)
        obs = np.stack([make_obs([0, 0, 0, 0, 1, 1])] * 64)
        mode = np.asarray([0, 4] * 32, np.int64)
        with torch.no_grad():
            old = ppo.log_probs(actor.logits(torch.from_numpy(obs)))
        log_prob = old.gather(1, torch.from_numpy(mode).view(-1, 1)).squeeze(1).numpy().astype(np.float64)
        adv = np.where(mode == 0, 1.0, -1.0)
        batch = {"obs": obs, "mode": mode, "log_prob": log_prob, "advantages": adv, "returns": adv * 0 - 2.0,
                 "values": np.zeros(64)}
        before = float(old[0, 0].exp())
        info = ppo.ppo_update(learner, batch, cfg, update_actor=True)
        with torch.no_grad():
            after = float(ppo.log_probs(actor.logits(torch.from_numpy(obs[:1])))[0, 0].exp())
        self.assertGreater(after, before)
        self.assertLessEqual(after, 0.25 * 1.2 + 0.2)             # 클립 때문에 한 번에 크게 뛰지 않는다
        self.assertTrue(info["actor_updated"])
        warm = ppo.ppo_update(learner, batch, cfg, update_actor=False)
        self.assertFalse(warm["actor_updated"])
        self.assertIsNone(warm["policy_loss"])


class PPOTrainTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = small_config()
        self.patch = patch.object(ppo, "config", return_value=self.cfg)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def test_train_resumes_and_matches_uninterrupted_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            first = ppo.train("P1", 42, tmp / "a", tmp / "w", executor=ThreadPoolExecutor(1), worker=fake_worker,
                              max_updates=2)
            self.assertFalse(first["finished"])
            self.assertEqual((first["episodes"], first["updates"]), (4, 2))
            log = ppo.torch.load(tmp / "a/seed_42/checkpoint.pt", weights_only=False)["state"]["updates_log"]
            self.assertTrue(log[0]["warmup"])
            self.assertFalse(log[0]["actor_updated"])
            self.assertTrue(log[1]["actor_updated"])
            done = ppo.train("P1", 42, tmp / "a", tmp / "w", executor=ThreadPoolExecutor(1), worker=fake_worker)
            self.assertTrue(done["finished"])
            self.assertEqual((done["episodes"], done["updates"]), (6, 3))
            straight = ppo.train("P1", 42, tmp / "b", tmp / "w", executor=ThreadPoolExecutor(1), worker=fake_worker)
            a = ppo.load_actor(tmp / "a/seed_42/final")
            b = ppo.load_actor(tmp / "b/seed_42/final")
            for (ka, va), (kb, vb) in zip(a.state_dict().items(), b.state_dict().items()):
                self.assertEqual(ka, kb)
                self.assertTrue(torch.equal(va, vb), ka)
            self.assertEqual(straight["choices"], done["choices"])

    def test_situations_are_g1_training_situations(self):
        seen = []

        def recording_worker(job):
            seen.append(job["situation"])
            return fake_worker(job)

        with tempfile.TemporaryDirectory() as tmp:
            ppo.train("P1", 43, Path(tmp), Path(tmp) / "w", executor=ThreadPoolExecutor(1), worker=recording_worker)
        self.assertEqual(seen, [g1.scenario_for(43, i) for i in range(6)])
        self.assertTrue(all(s["date"] in ac.config()["dates"]["train"] for s in seen))

    def test_settings_change_refuses_old_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ppo.train("P1", 42, tmp, tmp / "w", executor=ThreadPoolExecutor(1), worker=fake_worker, max_updates=1)
            self.cfg["ppo"]["clip_range"] = 0.1
            with self.assertRaises(ValueError):
                ppo.train("P1", 42, tmp, tmp / "w", executor=ThreadPoolExecutor(1), worker=fake_worker)

    def test_declaration_and_evaluation_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for seed in (42, 43, 44):
                ppo.train("P1", seed, tmp / "train", tmp / "w", executor=ThreadPoolExecutor(1), worker=fake_worker)
            dirs = ppo.model_dirs(tmp / "train")
            decl = ppo.declaration("P1", dirs, "eval")
            self.assertEqual(decl["seeds"], [42, 43, 44])
            with self.assertRaises(ValueError):
                ppo.declaration("P2", dirs, "eval")
            with self.assertRaises(ValueError):
                ppo.declaration("P1", list(reversed(dirs)), "eval")
            captured = {}

            def fake_run_jobs(jobs, workers, ledger, worker=None, executor=None):
                captured["jobs"] = jobs
                return {"done": len(jobs), "failed": []}

            with patch.object(ppo.ref, "register", return_value="fake-fp"), \
                    patch.object(ppo.ref, "run_jobs", side_effect=fake_run_jobs):
                ppo.evaluate("P1", dirs, tmp / "eval", tmp / "ew", 2, worker=lambda job: None)
            jobs = captured["jobs"]
            self.assertEqual(len(jobs), 3 * 84)
            self.assertEqual({j["kind"] for j in jobs}, {"ppo"})
            self.assertEqual({j["policy"] for j in jobs}, {"P1_m0", "P1_m1", "P1_m2"})
            self.assertEqual(json.loads((tmp / "eval/models.json").read_text())["model_identities"], decl["model_identities"])

    def test_p2_starts_from_same_seed_s2a_pre(self):
        torch.manual_seed(5)
        net = g1.DuelingNet()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "seed_43" / "pretrained"
            folder.mkdir(parents=True)
            (folder / "manifest.json").write_text(json.dumps({"variant": "advantage", "stage": "pretrained", "seed": 43}))
            import arena3_stage2_train as s2
            with patch.object(s2, "model_identity", return_value="sha-of-s2a-pre"), \
                    patch.object(s2, "load_model", return_value=net):
                actor, init = ppo.initial_actor("P2", 43, self.cfg, Path(tmp))
                self.assertIs(actor.net, net)
                self.assertEqual(float(actor.anchor_bias), 0.0)
                self.assertEqual(init["sha256"], "sha-of-s2a-pre")
                wrong = Path(tmp) / "seed_42" / "pretrained"
                wrong.mkdir(parents=True)
                (wrong / "manifest.json").write_text(json.dumps({"variant": "advantage", "stage": "pretrained", "seed": 43}))
                with self.assertRaises(ValueError):
                    ppo.initial_actor("P2", 42, self.cfg, Path(tmp))   # 다른 난수의 사전학습 모델은 거부


class PPOJudgeTest(unittest.TestCase):
    def test_paired_difference_is_date_gain_difference(self):
        a = {"per_date_gain_s": {"d1": 1.0, "d2": 0.5, "d3": 0.2}}
        b = {"per_date_gain_s": {"d1": 0.4, "d2": 0.6, "d3": 0.0}}
        p = ppo.paired_difference(a, b, 4.302652730)
        self.assertAlmostEqual(p["mean_s"], (0.6 - 0.1 + 0.2) / 3)
        self.assertEqual(p["positive_dates"], 2)
        self.assertLess(p["t95_s"][0], p["mean_s"])

    def test_winner_only_among_passing_ppo(self):
        results = {"S2A_pre": {"status": "pass", "gain_date_mean_s": 9.0},
                   "P1": {"status": "fail", "gain_date_mean_s": 5.0},
                   "P2": {"status": "hold_shared_collision", "gain_date_mean_s": 1.0}}
        self.assertIsNone(ppo.choose_winner(results))
        results["P2"]["status"] = "pass"
        self.assertEqual(ppo.choose_winner(results), "P2")
        results["P1"]["status"] = "pass"
        self.assertEqual(ppo.choose_winner(results), "P1")

    def test_collect_reports_pairs_and_pending(self):
        dates = ["d1", "d2"]
        fake = lambda g: {"status": "pass", "gain_date_mean_s": g, "per_date_gain_s": {d: g for d in dates}}
        with patch.object(ppo.s2e, "verify_baseline", return_value={"evaluation_fingerprint": "e"}), \
                patch.object(ppo.s2e, "verified_runs", return_value=[]), \
                patch.object(ppo.ref, "quality_errors", return_value=[]), \
                patch.object(ppo, "_ppo_runs", side_effect=lambda spec, *a: (spec.split("=")[0], {"d": 1}, [[1]] * 3)
                             if spec.startswith("P2") else ("P1", None, None)), \
                patch.object(ppo.s2e, "situations_for", return_value=[{}]), \
                patch.object(ppo.s2e, "candidate_models", return_value=("S2A_pre", Path("x"), [[1]] * 3)), \
                patch.object(ppo.s2e, "folder_record", return_value={"evaluation_fingerprint": "e"}), \
                patch.object(ppo.ref, "judge", side_effect=[fake(0.7), fake(0.6), fake(0.0)]):
            report = ppo.collect(Path("base"), ["P2=x", "P1=y"], ["S2A_pre=z"])
        self.assertEqual(report["pending"], {"P1": "아직 평가 결과 없음"})
        self.assertAlmostEqual(report["paired"]["P2-S2A_pre"]["mean_s"], 0.1)
        self.assertNotIn("P1-G1", report["paired"])


if __name__ == "__main__":
    unittest.main()
