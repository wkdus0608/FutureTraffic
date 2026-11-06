"""[추가 탐색] TUNE1 검사 — SUMO 실행 없음."""
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import arena3_common as ac  # noqa: E402
import explore_nn3 as n3  # noqa: E402
import explore_nn3r2 as r2  # noqa: E402
import explore_nn3r3 as r3  # noqa: E402
import explore_tune as t  # noqa: E402
import explore_tune_package as pkg  # noqa: E402
import explore_weight as w  # noqa: E402

w.configure()
CFG = t.config()
SCALE = {"mean": [0.0] * t.INPUT_SIZE, "std": [1.0] * t.INPUT_SIZE}


class FakePool:
    """프로세스 없이 작업자를 부른다."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def map(self, fn, jobs):
        return [fn(j) for j in jobs]


def bandit_worker(job: dict) -> dict:
    """합성 문제: 1번 칸이 1이면 녹색 1, −1이면 녹색 2가 좋다. 녹색 0·3은 막혀 있고 나쁜 선택은 지연이 늘어난다. 앞 선택의 결과는 다음에 영향이 없다."""
    net = t.net_from_bytes(job["net"], job["network"], job["scale"])
    rng = random.Random(job["rng_seed"])
    allowed = [1, 2]
    choices, rewards, step, decisions = [], [0.0] * 720, 0, job["base_decisions"]
    for k in range(24):
        x = np.zeros(t.INPUT_SIZE, np.float32)
        sign = 1.0 if rng.random() < 0.5 else -1.0
        x[1] = sign
        got = t.pick(net, x, allowed, job["algo"], job["settings"], decisions + k, rng)
        step = k * 30
        choices.append({"step": step, "observation": x, "mode": got["action"], "mask": t.mask_array(allowed), **{a: b for a, b in got.items() if a != "action"}})
        good = 1 if sign > 0 else 2
        rewards[step] = 0.0 if got["action"] == good else -1.0
    return {"ok": True, "status": "ok", "errors": [], "metrics": {"mean_delay_s": -sum(rewards)}, "choices": choices, "rewards": rewards, "seconds": 0.01}


def greedy_score(net) -> float:
    right = 0
    for sign, good in ((1.0, 1), (-1.0, 2)):
        x = torch.zeros(1, t.INPUT_SIZE)
        x[0, 1] = sign
        scores = net(x)[0].masked_fill(~torch.from_numpy(t.mask_array([1, 2])), -torch.inf)
        right += int(scores.argmax()) == good
    return right / 2


def small_cfg() -> dict:
    return {**CFG, "input": {**CFG["input"], **SCALE}}


def any_situation(seed, episode):
    return ac.make_situation(n3.TRAIN_DATES[0], "0700", f"TEST{seed}e{episode}")


class ConfigTest(unittest.TestCase):
    def test_config_is_what_the_rule_makes(self):
        strip = lambda rows: [{k: s[k] for k in ("date", "period", "tag", "sumo_seed")} for s in rows]  # noqa: E731
        self.assertEqual(strip(CFG["situations_screen"]), strip(t.screen_situations()))
        self.assertEqual(CFG["curve_situations"], t.curve_ids())
        self.assertEqual(CFG["cells"], t.cells())
        self.assertEqual(CFG["training"], t.TRAINING)
        self.assertEqual(CFG["rules"], t.RULES)

    def test_six_cells_three_sizes(self):
        self.assertEqual(sorted(CFG["cells"]), ["DQN_256", "DQN_64", "DQN_64x3", "PPO_256", "PPO_64", "PPO_64x3"])
        self.assertEqual(CFG["cells"]["DQN_64x3"]["network"], [64, 64, 64])
        self.assertEqual(CFG["cells"]["PPO_256"]["network"], [256, 256])
        self.assertEqual(CFG["training"]["episodes"] % CFG["cells"]["PPO_64"]["settings"]["episodes_per_update"], 0)

    def test_sizes_and_new_routes(self):
        rows = [s["route_sha256"] for s in CFG["situations_screen"]]
        self.assertEqual(len(rows), 84)
        self.assertEqual(len(CFG["curve_situations"]), 12)
        self.assertEqual(len(set(rows)), 84)
        r3cfg = r3.config()
        used = r3.previous_routes(n3.config(), r2.config()) | {s["route_sha256"] for part in ("situations_round", "situations_screen", "situations_final", "situations_low") for s in r3cfg[part]}
        self.assertFalse(set(rows) & used)

    def test_curve_is_inside_screen(self):
        ids = {ac.situation_id(s) for s in CFG["situations_screen"]}
        self.assertTrue(set(CFG["curve_situations"]) <= ids)

    def test_dates_stay_apart(self):
        valid = {s["date"] for s in CFG["situations_screen"]}
        self.assertEqual(valid, set(n3.VALID_DATES))
        trained = {t.scenario_for(seed, e)["date"] for seed in t.SEEDS for e in range(600)}
        self.assertEqual(trained, set(n3.TRAIN_DATES))
        self.assertFalse(trained & valid)

    def test_scenarios_same_across_cells_and_differ_by_seed(self):
        self.assertEqual(t.scenario_for(42, 7), t.scenario_for(42, 7))
        self.assertNotEqual(t.scenario_for(42, 7)["tag"], t.scenario_for(43, 7)["tag"])

    def test_every_training_slot_has_demand(self):
        from focused_data import COUNTS, load_counts
        counts = load_counts(COUNTS)
        for date in n3.TRAIN_DATES:
            for slot in ac.TRAIN_SLOTS:
                self.assertIn((date, slot), counts)

    def test_input_scale(self):
        self.assertEqual(len(CFG["input"]["mean"]), 120)
        self.assertEqual(len(CFG["input"]["std"]), 120)
        self.assertTrue(all(x > 0 for x in CFG["input"]["std"]))


class NetworkTest(unittest.TestCase):
    def test_three_sizes(self):
        for hidden, params in (([64, 64], 120 * 64 + 64 + 64 * 64 + 64 + 64 * 4 + 4), ([256, 256], 120 * 256 + 256 + 256 * 256 + 256 + 256 * 4 + 4), ([64, 64, 64], 120 * 64 + 64 + 2 * (64 * 64 + 64) + 64 * 4 + 4)):
            net = t.TuneNet(hidden, CFG["input"]["mean"], CFG["input"]["std"])
            self.assertEqual(net(torch.zeros(3, 120)).shape, (3, 4))
            self.assertEqual(sum(p.numel() for p in net.parameters()), params)
            critic = t.TuneCritic(hidden, CFG["input"]["mean"], CFG["input"]["std"])
            self.assertEqual(critic(torch.zeros(3, 120)).shape, (3,))

    def test_save_load_same_output(self):
        with tempfile.TemporaryDirectory() as d:
            for hidden in t.NETWORKS.values():
                net = t.TuneNet(hidden, CFG["input"]["mean"], CFG["input"]["std"])
                folder = Path(d) / str(hidden)
                sha = t.save_net(folder, net, {"network": hidden})
                self.assertEqual(t.net_identity(folder), sha)
                x = torch.ones(2, 120)
                self.assertTrue(torch.equal(t.load_net(folder)(x), net(x)))

    def test_load_rejects_changed_file(self):
        with tempfile.TemporaryDirectory() as d:
            net = t.TuneNet([64, 64], CFG["input"]["mean"], CFG["input"]["std"])
            t.save_net(Path(d) / "m", net, {"network": [64, 64]})
            with (Path(d) / "m" / "model.pt").open("ab") as f:
                f.write(b"x")
            with self.assertRaises(ValueError):
                t.load_net(Path(d) / "m")

    def test_ppo_head_starts_nearly_uniform(self):
        learner = t.PPOLearner(42, [64, 64], SCALE, t.PPO)
        logp = t.masked_log_probs(learner.actor(torch.rand(5, 120)), torch.ones(5, 4, dtype=torch.bool)).exp()
        self.assertLess(float((logp.detach() - 0.25).abs().max()), 0.1)


class ChoiceTest(unittest.TestCase):
    def test_never_picks_a_blocked_green(self):
        net = t.TuneNet([64, 64], CFG["input"]["mean"], CFG["input"]["std"])
        rng = random.Random(1)
        x = np.zeros(120, np.float32)
        for algo, settings in (("DQN", t.DQN), ("PPO", t.PPO)):
            for allowed in ([0, 2], [1, 3], [0, 1, 2, 3], [3, 1]):
                for k in range(200):
                    self.assertIn(t.pick(net, x, allowed, algo, settings, k, rng)["action"], allowed)

    def test_epsilon_schedule(self):
        self.assertEqual(t.epsilon_at(0, t.DQN), 1.0)
        self.assertAlmostEqual(t.epsilon_at(15000, t.DQN), 0.525)
        self.assertAlmostEqual(t.epsilon_at(30000, t.DQN), 0.05)
        self.assertAlmostEqual(t.epsilon_at(90000, t.DQN), 0.05)

    def test_zero_epsilon_equals_evaluation_choice(self):
        import explore_nn1 as nn
        net = t.TuneNet([64, 64], CFG["input"]["mean"], CFG["input"]["std"])
        rng = random.Random(2)
        settings = {**t.DQN, "epsilon_start": 0.0, "epsilon_end": 0.0}
        for _ in range(100):
            x = np.random.default_rng(rng.randrange(10**6)).normal(size=120).astype(np.float32)
            allowed = rng.sample(range(4), rng.randint(2, 4))
            self.assertEqual(t.pick(net, x, allowed, "DQN", settings, 0, rng)["action"], nn.green_choice(net, x, allowed))


class TransitionTest(unittest.TestCase):
    def test_next_mask_and_discount(self):
        choices = [{"step": s, "observation": np.zeros(120, np.float32), "mode": m[0], "mask": t.mask_array(m)} for s, m in ((0, [1, 2]), (10, [0, 1]), (30, [2, 3]), (50, [1, 3]))]
        rewards = [-0.01] * 720
        rows = t.dqn_transitions(choices, rewards, {**t.DQN, "n_choice_steps": 2})
        self.assertEqual(rows[0]["next_mask"].tolist(), t.mask_array([2, 3]).tolist())
        self.assertAlmostEqual(rows[0]["discount"], 0.99 ** 30, places=9)
        self.assertEqual(rows[1]["next_mask"].tolist(), t.mask_array([1, 3]).tolist())
        self.assertTrue(rows[2]["done"])
        self.assertTrue(rows[3]["done"])
        self.assertEqual(rows[3]["next_mask"].tolist(), [True] * 4)

    def test_rejects_blocked_action(self):
        choices = [{"step": 0, "observation": np.zeros(120, np.float32), "mode": 0, "mask": t.mask_array([1, 2])}]
        with self.assertRaises(ValueError):
            t.dqn_transitions(choices, [0.0] * 720, t.DQN)

    def test_replay_state_keeps_only_filled_part(self):
        rep = t.Replay(10)
        rep.add([{"obs": np.ones(120, np.float32), "next_obs": np.ones(120, np.float32), "mode": 1, "reward": -1.0, "discount": 0.5, "done": False, "next_mask": t.mask_array([0, 1])}])
        s = rep.state()
        self.assertEqual(s["obs"].shape, (1, 120))
        other = t.Replay(10)
        other.load(s)
        self.assertEqual(other.size, 1)
        self.assertEqual(other.next_mask[0].tolist(), t.mask_array([0, 1]).tolist())


class LearningTest(unittest.TestCase):
    """합성 문제에서 두 알고리즘이 세 크기 모두 좋은 행동을 배우는지 본다."""

    def run_cell(self, cell, overrides, training, d):
        cfg = small_cfg()
        status = t.train(cell, 42, Path(d) / cell, Path(d) / "w", cfg, executor=FakePool(), worker=bandit_worker, overrides=overrides, training=training, situation_of=any_situation)
        self.assertTrue(status["finished"])
        return t.load_net(Path(d) / cell / "seed_42" / "final"), status

    def test_dqn_learns(self):
        over = {"learning_starts": 50, "updates_per_decision": 4, "target_every_updates": 100, "learning_rate": 0.003, "epsilon_decay_decisions": 400, "n_choice_steps": 1}
        with tempfile.TemporaryDirectory() as d:
            for cell in ("DQN_64", "DQN_256", "DQN_64x3"):
                net, _ = self.run_cell(cell, over, {"episodes": 40, "marks": [20]}, d)
                self.assertEqual(greedy_score(net), 1.0, cell)
                self.assertTrue((Path(d) / cell / "seed_42" / "marks" / "e20" / "manifest.json").exists())

    def test_ppo_learns(self):
        over = {"episodes_per_update": 4, "critic_warmup_updates": 1, "learning_rate": 0.003, "n_epochs": 4}
        with tempfile.TemporaryDirectory() as d:
            for cell in ("PPO_64", "PPO_256", "PPO_64x3"):
                net, status = self.run_cell(cell, over, {"episodes": 80, "marks": [40]}, d)
                self.assertEqual(greedy_score(net), 1.0, cell)
                self.assertTrue(status["last"]["actor_updated"])

    def test_resume_equals_one_run(self):
        for cell, over, cut in (("DQN_64", {"learning_starts": 10, "updates_per_decision": 1, "target_every_updates": 20, "checkpoint_every_episodes": 2}, 4), ("PPO_64", {"episodes_per_update": 2, "critic_warmup_updates": 1, "n_epochs": 2}, 4)):
            with tempfile.TemporaryDirectory() as d:
                whole, _ = self.run_cell(cell, over, {"episodes": 8, "marks": []}, Path(d) / "whole")
                cfg = small_cfg()
                # 앞 4판까지 돌린 체크포인트에서 이어 간다.
                seed_dir = Path(d) / "resume" / cell
                stop = StopAfter(cut)
                with self.assertRaises(StopIteration):
                    t.train(cell, 42, seed_dir, Path(d) / "w", cfg, executor=FakePool(), worker=stop, overrides=over, training={"episodes": 8, "marks": []}, situation_of=any_situation)
                saved = t.w.read(seed_dir / "seed_42" / "checkpoint.json")["episodes"]
                self.assertGreater(saved, 0)
                self.assertLess(saved, 8)
                status = t.train(cell, 42, seed_dir, Path(d) / "w", cfg, executor=FakePool(), worker=bandit_worker, overrides=over, training={"episodes": 8, "marks": []}, situation_of=any_situation)
                self.assertTrue(status["finished"])
                again = t.load_net(seed_dir / "seed_42" / "final")
                for a, b in zip(whole.state_dict().values(), again.state_dict().values()):
                    self.assertTrue(torch.equal(a, b), cell)

    def test_changed_settings_refuse_to_resume(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = small_cfg()
            over = {"episodes_per_update": 2, "critic_warmup_updates": 1, "n_epochs": 1}
            t.train("PPO_64", 42, Path(d), Path(d) / "w", cfg, executor=FakePool(), worker=bandit_worker, overrides=over, training={"episodes": 4, "marks": []}, situation_of=any_situation)
            with self.assertRaises(ValueError):
                t.train("PPO_64", 42, Path(d), Path(d) / "w", cfg, executor=FakePool(), worker=bandit_worker, overrides={**over, "n_epochs": 2}, training={"episodes": 8, "marks": []}, situation_of=any_situation)

    def test_discards_are_left_out_and_limited(self):
        def discard(job):
            return {"ok": True, "status": "discard", "errors": [], "metrics": {}, "choices": [], "rewards": [], "seconds": 0.01, "reason": "시험"}

        with tempfile.TemporaryDirectory() as d:
            cfg = small_cfg()
            status = t.train("DQN_64", 42, Path(d) / "a", Path(d) / "w", cfg, executor=FakePool(), worker=discard, training={"episodes": 4, "marks": [], "max_discarded": 10}, situation_of=any_situation)
            self.assertEqual((status["decisions"], status["discarded"]), (0, 4))
            with self.assertRaises(RuntimeError):
                t.train("DQN_64", 42, Path(d) / "b", Path(d) / "w", cfg, executor=FakePool(), worker=discard, training={"episodes": 4, "marks": [], "max_discarded": 1}, situation_of=any_situation)


class StopAfter:
    """cut판을 정상으로 돌리고 그다음 묶음에서 끊는다(런타임이 끊긴 경우)."""

    def __init__(self, cut):
        self.cut, self.seen = cut, 0

    def __call__(self, job):
        if self.seen >= self.cut:
            raise StopIteration
        self.seen += 1
        return bandit_worker(job)


class ReadingTest(unittest.TestCase):
    def gain(self, s, low):
        return {"gain_s": s, "t95": [low, s + 0.1], "n": 84, "positive": 50}

    def test_size_rule(self):
        rules = CFG["rules"]
        delays = {"64": 35.0, "256": 34.0, "64x3": 33.5}
        both = {"256": {"gain": self.gain(1.0, 0.2), "models": [1, 1, 1]}, "64x3": {"gain": self.gain(1.5, 0.2), "models": [1, 1, -1]}}
        self.assertEqual(t.size_reading(both, delays, rules), "64x3")
        small = {"256": {"gain": self.gain(0.29, 0.1), "models": [1, 1, 1]}, "64x3": {"gain": self.gain(1.0, -0.1), "models": [1, 1, 1]}}
        self.assertEqual(t.size_reading(small, delays, rules), "64")
        one_model = {"256": {"gain": self.gain(1.0, 0.2), "models": [2, -1, -1]}, "64x3": {"gain": self.gain(-1.0, -2.0), "models": [1, 1, 1]}}
        self.assertEqual(t.size_reading(one_model, delays, rules), "64")

    def test_beats(self):
        self.assertTrue(t.beats(self.gain(0.2, 0.01), 0.2))
        self.assertFalse(t.beats(self.gain(0.19, 0.01), 0.2))
        self.assertFalse(t.beats(self.gain(0.5, -0.01), 0.2))

    def test_mac_comparison(self):
        mac = {"executed_sha16": "a", "route_sha256": "r", "actual_total_delay": 250851.146}
        self.assertTrue(t.same_as_reference(mac, {**mac, "actual_total_delay": 250851.349}))
        self.assertFalse(t.same_as_reference(mac, {**mac, "actual_total_delay": 250861.0}))
        self.assertFalse(t.same_as_reference(mac, {**mac, "executed_sha16": "b"}))
        self.assertFalse(t.same_as_reference(mac, {**mac, "route_sha256": "x"}))

    def test_incomplete_report(self):
        with tempfile.TemporaryDirectory() as d:
            report = t.collect_report(Path(d), CFG)
            self.assertEqual(report["status"], "incomplete")
            self.assertIn("아직 끝나지 않았습니다", t.render(report))


class PackageTest(unittest.TestCase):
    def test_notebooks_are_per_cell(self):
        self.assertEqual(set(pkg.NOTEBOOKS), {*CFG["cells"], "baseline", "judge"})

    def test_notebooks_compile_and_use_cpu_runtime(self):
        for name in pkg.NOTEBOOKS:
            nb = pkg.notebook(name)
            self.assertEqual(nb["metadata"]["colab"]["machine_shape"], "hm")
            for cell in nb["cells"]:
                if cell["cell_type"] == "code":
                    compile("".join(cell["source"]), name, "exec")

    def test_tests_cell_uses_discover(self):
        self.assertIn('"discover"', pkg.TESTS_CELL)


if __name__ == "__main__":
    unittest.main()
