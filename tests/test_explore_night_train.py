"""WEIGHT1_NIGHT learner checks: no SUMO needed."""
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import explore_night_train as nt


def obs(step=0, actions=(2, 2, 2, 2, 2), mask=None):
    """A 110-value observation with the real layout: one-hots at 84:104, mode mask at 104:109."""
    o = np.zeros(110, np.float32)
    o[5] = step / 720
    for mode, green in enumerate(actions):
        o[84 + 4 * mode + green] = 1.0
    if mask is None:
        seen, mask = set(), []
        for green in actions:
            mask.append(green not in seen)
            seen.add(green)
    for mode, valid in enumerate(mask):
        o[104 + mode] = 1.0 if valid else 0.0
    o[-1] = max(0.0, 1 - step / 720)
    return o


CHOICE_ACTIONS = (0, 1, 2, 3, 0)       # M4 repeats mode 0's green -> M4 representative is mode 0


def stepped(step):
    """Every sixth control step is a decision point; the rest are forced."""
    return obs(step, CHOICE_ACTIONS) if step % 6 == 0 else obs(step)


def episode_rows(policy, reward_of):
    rows = []
    for step in range(720):
        o = stepped(step)
        mode = policy(o, None)
        rows.append({"obs": o, "next_obs": stepped(step + 1), "mode": mode, "executed_green": mode % 4,
                     "reward": reward_of(step, mode), "done": step == 719})
    return rows


def fake_episode(date, period, eval_seed, work, policy, **kwargs):
    rows = episode_rows(policy, lambda step, mode: (1.0 if mode == 3 else -1.0) if step % 6 == 0 else -0.1)
    return SimpleNamespace(metrics={"mean_delay_s": 1.0, "missing": 0, "unfinished": 0, "cost_check_passed": True,
                                    "changed_by_rules": 0, "collisions_unexplained": 0,
                                    "date": date, "period": period, "eval_seed": eval_seed},
                           transitions=rows, seconds=0.0)


class BaselineModeTest(unittest.TestCase):
    def test_baseline_is_smallest_valid_mode_with_the_m4_green(self):
        cases = [((1, 2, 1, 1, 2), 1),        # mode 4 shares mode 1's green
                 ((0, 1, 2, 3, 0), 0),        # mode 4 shares mode 0's green
                 ((0, 1, 2, 3, 3), 3),        # M4 repeats mode 3
                 ((2, 2, 2, 2, 2), 0),        # forced state
                 ((0, 1, 2, 3, 1), 1)]
        for actions, expected in cases:
            self.assertEqual(int(nt.baseline_mode(torch.from_numpy(obs(0, actions)))[0]), expected, actions)

    def test_batch_and_zero_placeholder(self):
        batch = torch.from_numpy(np.stack([obs(0, (1, 2, 1, 1, 2)), obs(1), np.zeros(110, np.float32)]))
        self.assertEqual(nt.baseline_mode(batch).tolist(), [1, 0, 0])

    def test_untrained_greedy_policy_is_m4_not_mode_zero(self):
        model = nt.build_model(5)
        self.assertEqual(nt.predict_mode(model, obs(0, (1, 2, 1, 1, 2))), 1)
        self.assertEqual(nt.predict_mode(model, obs(0, (0, 1, 2, 3, 3))), 3)
        self.assertEqual(nt.predict_mode(model, obs(0)), 0)

    def test_q_of_the_baseline_equals_the_value_head(self):
        net = nt.DuelingNet()
        with torch.no_grad():
            net.value.bias.fill_(5.0)
            net.advantage.weight.normal_()
            net.advantage.bias.normal_()
        x = torch.from_numpy(np.stack([obs(1, (1, 2, 1, 1, 2)), obs(2, CHOICE_ACTIONS), obs(3)]))
        q = net(x)
        anchor = nt.baseline_mode(x)
        for row, a in enumerate(anchor.tolist()):
            self.assertAlmostEqual(float(q[row, a].detach()), float(net.value(net.trunk(x))[row, 0].detach()), places=5)

    def test_invalid_modes_are_never_chosen(self):
        model = nt.build_model(1)
        with torch.no_grad():
            model.online.advantage.bias[:] = torch.tensor([0.0, 9.0, 0.0, 0.0, 0.0])
        o = obs(0, (0, 1, 2, 3, 0), mask=[True, False, True, True, False])
        self.assertIn(nt.predict_mode(model, o), (0, 2, 3))
        self.assertEqual(nt.predict_mode(model, o, epsilon=1.0, rng=random.Random(2)) in (0, 2, 3), True)

    def test_exploration_schedule(self):
        self.assertEqual(nt.exploration_at(0), 0.5)
        self.assertAlmostEqual(nt.exploration_at(10_000), 0.35)
        self.assertAlmostEqual(nt.exploration_at(20_000), 0.2)
        self.assertAlmostEqual(nt.exploration_at(100_000), 0.2)


class ReplayTest(unittest.TestCase):
    def rows(self, offset=0):
        return [{"obs": stepped(i), "next_obs": stepped(i + 1), "mode": 0, "executed_green": 0,
                 "reward": -0.1, "done": i == 11} for i in range(12)]

    def test_choice_flags_and_reserved_share(self):
        rb = nt.ChoiceReplay(100, choice_share=0.25)
        rb.add_episode(self.rows())
        self.assertEqual(int(rb.choice[:12].sum()), 2)           # steps 0 and 6
        rng = np.random.default_rng(0)
        for _ in range(20):
            data = rb.sample(64, rng)
            flags = (data["observations"][:, 104:109] > 0.5).sum(dim=1) > 1
            self.assertGreaterEqual(int(flags.sum()), 16)         # reserved 25% even though choices are 17% of rows

    def test_no_choice_points_falls_back_to_uniform(self):
        rb = nt.ChoiceReplay(100)
        rb.add_episode([{"obs": obs(i), "next_obs": obs(i + 1), "mode": 0, "executed_green": 0,
                         "reward": -0.1, "done": i == 4} for i in range(5)])
        self.assertEqual(rb.sample(8, np.random.default_rng(1))["observations"].shape, (8, 110))

    def test_ring_wrap_and_round_trip_recompute_flags(self):
        rb = nt.ChoiceReplay(10)
        rb.add_episode(self.rows())                              # 12 rows into capacity 10 -> wraps
        restored = nt.ChoiceReplay.from_state_dict(rb.state_dict())
        np.testing.assert_array_equal(restored.choice[:restored.size], rb.choice[:rb.size])
        manual = (rb.observations[:rb.size][:, 104:109] > 0.5).sum(axis=1) > 1
        np.testing.assert_array_equal(rb.choice[:rb.size], manual)


class LearningTest(unittest.TestCase):
    def test_advantage_head_learns_the_better_decision(self):
        model = nt.build_model(3)
        rng = random.Random(11)
        for _ in range(9):
            rows = episode_rows(lambda o, s: rng.choice([m for m in range(5) if o[104 + m] > 0.5]),
                                lambda step, mode: (1.0 if mode == 3 else -1.0) if step % 6 == 0 else -0.1)
            model.learn_episode(rows)
        self.assertGreater(model.n_updates, 1000)
        q = model.q_values(stepped(0))
        self.assertEqual(int(np.argmax(np.where(stepped(0)[104:109] > 0.5, q, -np.inf))), 3)
        self.assertEqual(nt.predict_mode(model, stepped(0)), 3)
        self.assertEqual(nt.predict_mode(model, stepped(1)), 0)  # forced state keeps the only legal mode


class CheckpointTest(unittest.TestCase):
    scenarios = [("20220812", "1300"), ("20220819", "1800")]

    def test_continuation_is_exact_at_episode_boundary(self):
        with tempfile.TemporaryDirectory() as d:
            straight, resumed = Path(d) / "straight", Path(d) / "resumed"
            a = nt.train_seed(straight, self.scenarios, 42, 5, fake_episode, fingerprint="fp")
            nt.train_seed(resumed, self.scenarios, 42, 2, fake_episode, fingerprint="fp")
            b = nt.train_seed(resumed, self.scenarios, 42, 5, fake_episode, fingerprint="fp")
            self.assertEqual((a["episodes"], a["control_steps"], a["updates"]),
                             (b["episodes"], b["control_steps"], b["updates"]))
            ma, sa, _ = nt._load_checkpoint(straight, "fp")
            mb, sb, _ = nt._load_checkpoint(resumed, "fp")
            for key, tensor in ma.online.state_dict().items():
                torch.testing.assert_close(tensor, mb.online.state_dict()[key], rtol=0, atol=0)
            for key in ("observations", "actions", "rewards"):
                np.testing.assert_array_equal(getattr(ma.replay, key)[:ma.replay.size],
                                              getattr(mb.replay, key)[:mb.replay.size])
            np.testing.assert_array_equal(ma.replay.choice[:ma.replay.size], mb.replay.choice[:mb.replay.size])
            self.assertEqual(sa["scenario_rng"], sb["scenario_rng"])
            self.assertTrue((resumed / "stages" / "episode_000002" / "model.pt").exists())
            self.assertTrue((resumed / "stages" / "episode_000005" / "model.pt").exists())

    def test_stage_loads_for_evaluation_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            nt.train_seed(out, self.scenarios, 42, 1, fake_episode, fingerprint="fp")
            stage = out / "stages" / "episode_000001"
            model = nt.load_model(stage, "fp")
            self.assertEqual(model.episodes, 1)
            self.assertIn(nt.predict_mode(model, stepped(0)), (0, 1, 2, 3))
            with self.assertRaises(ValueError):
                nt.load_model(stage, "other")
            with (Path(nt.validate_checkpoint(out, "fp")["path"]) / "model.pt").open("ab") as f:
                f.write(b"x")
            with self.assertRaises(ValueError):
                nt.load_model(out, "fp")

    def test_unexplained_collision_is_discarded_not_fatal_and_capped(self):
        calls = {"n": 0}

        def sometimes(date, period, eval_seed, work, policy, **kwargs):
            calls["n"] += 1
            result = fake_episode(date, period, eval_seed, work, policy, **kwargs)
            if calls["n"] == 2:
                result.metrics["collisions_unexplained"] = 1
            return result

        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            summary = nt.train_seed(out, self.scenarios, 42, 3, sometimes, fingerprint="fp")
            self.assertEqual((summary["episodes"], summary["discarded"], calls["n"]), (3, 1, 4))
            lines = (out / "discarded.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["reason"], "collisions_unexplained=1")
        def always(date, period, eval_seed, work, policy, **kwargs):
            result = fake_episode(date, period, eval_seed, work, policy, **kwargs)
            result.metrics["collisions_unexplained"] = 2
            return result
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "discarded"):
                nt.train_seed(Path(d), self.scenarios, 42, 3, always, fingerprint="fp")

    def test_other_quality_failures_and_protected_dates_still_stop(self):
        def bad(*args, **kwargs):
            result = fake_episode(*args, **kwargs)
            result.metrics["unfinished"] = 1
            return result
        def both(*args, **kwargs):
            result = fake_episode(*args, **kwargs)
            result.metrics["collisions_unexplained"] = 1
            result.metrics["missing"] = 1
            return result
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                nt.train_seed(Path(d) / "a", self.scenarios, 42, 1, bad, fingerprint="fp")
            with self.assertRaises(ValueError):
                nt.train_seed(Path(d) / "b", self.scenarios, 42, 1, both, fingerprint="fp")
            self.assertFalse((Path(d) / "a" / "latest.json").exists())
            for date in ("20220811", "20220823", "20220928", "20220813"):
                with self.assertRaises(ValueError):
                    nt.train_seed(Path(d) / date, [(date, "1300")], 42, 1, fake_episode, fingerprint="fp")


if __name__ == "__main__":
    unittest.main()
