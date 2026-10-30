"""WEIGHT1 learner checks: no SUMO or long training is needed."""
import copy
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import explore_weight_train as wt


def observation(step=0, modes=(0, 4)):
    o = np.zeros(110, np.float32)
    o[3] = 1.0                 # Existing green 3, deliberately unrelated to mode 4.
    o[63] = 1.0                # Existing green mask, not the mode mask.
    o[5] = step / 720
    o[104:109] = 0.0
    for mode in modes:
        o[104 + mode] = 1.0
    o[-1] = max(0.0, 1 - step / 720)
    return o


def transitions(rewards, action=4):
    return [{"obs": observation(i), "next_obs": observation(i + 1),
             "mode": action, "executed_green": 2, "reward": reward,
             "done": i == len(rewards) - 1} for i, reward in enumerate(rewards)]


def fake_episode(date, period, eval_seed, work, policy, **kwargs):
    rows = []
    for step in range(720):
        o = observation(step)
        action = policy(o, None)
        rows.append({"obs": o, "next_obs": observation(step + 1), "mode": action,
                     "executed_green": (action + 2) % 4,
                     "reward": -0.1 if action == 4 else -0.3,
                     "done": step == 719})
    return SimpleNamespace(metrics={"mean_delay_s": 1.0, "missing": 0, "unfinished": 0,
                                    "cost_check_passed": True, "changed_by_rules": 0,
                                    "date": date, "period": period, "eval_seed": eval_seed},
                           transitions=rows, seconds=0.0)


class ModeMaskTest(unittest.TestCase):
    def test_mode_mask_is_not_green_mask_or_remaining_time(self):
        o = observation(modes=(4,))
        model = wt.build_model(7)
        self.assertEqual(wt.predict_mode(model, o), 4)
        self.assertEqual(wt.predict_mode(model, o, epsilon=1.0, rng=random.Random(9)), 4)
        o[104:109] = 0
        with self.assertRaises(ValueError):
            wt.predict_mode(model, o)

    def test_double_target_selects_online_legal_then_uses_target(self):
        online = torch.tensor([[999.0, 1.0, 100.0, 80.0, 3.0]])
        target = torch.tensor([[999.0, 40.0, 20.0, 30.0, 7.0]])
        mask = torch.tensor([[False, True, False, False, True]])
        value = wt.double_targets(torch.tensor([-2.0]), torch.tensor([0.9]), torch.tensor([False]),
                                  online, target, mask)
        self.assertAlmostEqual(value.item(), -2 + 0.9 * 7, places=5)

    def test_terminal_never_bootstraps_even_without_next_mask(self):
        q = torch.full((1, 5), 1e6)
        empty = torch.zeros((1, 5), dtype=torch.bool)
        value = wt.double_targets(torch.tensor([-3.0]), torch.tensor([0.99]), torch.tensor([True]), q, q, empty)
        self.assertEqual(value.item(), -3.0)
        with self.assertRaises(ValueError):
            wt.double_targets(torch.tensor([-3.0]), torch.tensor([0.99]), torch.tensor([False]), q, q, empty)

    def test_exploration_uses_absolute_control_time(self):
        self.assertEqual(wt.exploration_at(0), 1.0)
        self.assertAlmostEqual(wt.exploration_at(10_000), 0.525)
        self.assertAlmostEqual(wt.exploration_at(20_000), 0.05)
        self.assertAlmostEqual(wt.exploration_at(30_240), 0.05)
        self.assertAlmostEqual(wt.exploration_at(100_080), 0.05)


class ReplayTest(unittest.TestCase):
    def test_mode_four_is_preserved_instead_of_executed_green(self):
        rb = wt.EpisodeReplay(30)
        rb.add_episode(transitions([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(rb.actions[:3], [4, 4, 4])
        self.assertAlmostEqual(rb.rewards[0], 1 + 0.99 * 2 + 0.99 ** 2 * 3, places=5)
        np.testing.assert_array_equal(rb.dones[:3], [True, True, True])
        self.assertAlmostEqual(rb.discounts[0], 0.99 ** 3, places=6)

    def test_n_step_never_crosses_episode_boundary(self):
        rb = wt.EpisodeReplay(30)
        rb.add_episode(transitions([1.0] * 3))
        rb.add_episode(transitions([100.0] * 3))
        self.assertAlmostEqual(rb.rewards[1], 1 + 0.99, places=5)
        self.assertAlmostEqual(rb.rewards[2], 1.0, places=5)
        self.assertAlmostEqual(rb.rewards[3], 100 * (1 + 0.99 + 0.99 ** 2), places=4)

    def test_incomplete_invalid_and_discontinuous_episode_rejected(self):
        rb = wt.EpisodeReplay(30)
        rows = transitions([1.0, 2.0])
        rows[-1]["done"] = False
        with self.assertRaises(ValueError):
            rb.add_episode(rows)
        rows = transitions([1.0, 2.0])
        rows[0]["mode"] = 3           # A green number that is not a valid mode here.
        with self.assertRaises(ValueError):
            rb.add_episode(rows)
        rows = transitions([1.0, 2.0])
        rows[1]["obs"][5] = 0.8
        with self.assertRaises(ValueError):
            rb.add_episode(rows)
        self.assertEqual(rb.size, 0)

    def test_ring_round_trip(self):
        rb = wt.EpisodeReplay(5)
        rb.add_episode(transitions([1.0] * 3))
        rb.add_episode(transitions([2.0] * 3))
        restored = wt.EpisodeReplay.from_state_dict(rb.state_dict())
        self.assertEqual((restored.position, restored.size), (1, 5))
        for key in ("observations", "next_observations", "actions", "rewards", "dones", "discounts"):
            np.testing.assert_array_equal(getattr(restored, key), getattr(rb, key))


class CheckpointTest(unittest.TestCase):
    def test_continuation_is_exact_at_episode_boundary(self):
        scenarios = [("20220812", "1300"), ("20220819", "1800")]
        with tempfile.TemporaryDirectory() as d:
            straight, resumed = Path(d) / "straight", Path(d) / "resumed"
            a = wt.train_seed(straight, scenarios, 42, 7, fake_episode, fingerprint="fp")
            wt.train_seed(resumed, scenarios, 42, 3, fake_episode, fingerprint="fp")
            b = wt.train_seed(resumed, scenarios, 42, 7, fake_episode, fingerprint="fp")
            ma, sa, _ = wt._load_checkpoint(straight, "fp")
            mb, sb, _ = wt._load_checkpoint(resumed, "fp")
            self.assertEqual((a["episodes"], a["control_steps"], a["updates"], a["target_syncs"]),
                             (7, 5040, 1010, 1))
            self.assertEqual((ma.num_timesteps, ma.n_updates, ma.target_syncs, ma.episodes),
                             (mb.num_timesteps, mb.n_updates, mb.target_syncs, mb.episodes))
            for key, tensor in ma.online.state_dict().items():
                torch.testing.assert_close(tensor, mb.online.state_dict()[key], rtol=0, atol=0)
            for key, tensor in ma.target.state_dict().items():
                torch.testing.assert_close(tensor, mb.target.state_dict()[key], rtol=0, atol=0)
            for key in ("observations", "next_observations", "actions", "rewards", "discounts", "dones"):
                np.testing.assert_array_equal(getattr(ma.replay, key)[:ma.replay.size],
                                              getattr(mb.replay, key)[:mb.replay.size])
            self.assertEqual(sa["scenario_rng"], sb["scenario_rng"])
            self.assertEqual(sa["exploration_rng"], sb["exploration_rng"])
            self.assertEqual(ma.sample_rng.bit_generator.state, mb.sample_rng.bit_generator.state)
            self.assertEqual([(r["date"], r["period"], r["sumo_seed"]) for r in sa["history"]],
                             [(r["date"], r["period"], r["sumo_seed"]) for r in sb["history"]])
            self.assertTrue((resumed / "stages" / "episode_000003" / "model.pt").exists())
            generations = list((resumed / "checkpoints").glob("ep_*"))
            self.assertEqual(len(generations), 2)

    def test_fingerprint_and_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            wt.train_seed(out, [("20220812", "1300")], 42, 1, fake_episode, fingerprint="fp")
            with self.assertRaises(ValueError):
                wt.load_model(out, "other")
            identity = wt.validate_checkpoint(out, "fp")
            with (Path(identity["path"]) / "model.pt").open("ab") as f:
                f.write(b"tampered")
            with self.assertRaises(ValueError):
                wt.load_model(out, "fp")

    def test_quality_failure_never_creates_checkpoint(self):
        def bad(*args, **kwargs):
            result = fake_episode(*args, **kwargs)
            result.metrics["unfinished"] = 1
            return result
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            with self.assertRaises(ValueError):
                wt.train_seed(out, [("20220812", "1300")], 42, 1, bad, fingerprint="fp")
            self.assertFalse((out / "latest.json").exists())

    def test_budget_before_collection_and_excluded_dates(self):
        def denied():
            raise RuntimeError("budget")
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError, "budget"):
                wt.train_seed(Path(d) / "budget", [("20220812", "1300")], 42, 1,
                              fake_episode, fingerprint="fp", budget_check=denied)
            with self.assertRaises(ValueError):
                wt.train_seed(Path(d) / "leak", [("20220811", "1300")], 42, 1,
                              fake_episode, fingerprint="fp")


if __name__ == "__main__":
    unittest.main()
