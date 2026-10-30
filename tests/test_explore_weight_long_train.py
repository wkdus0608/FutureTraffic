"""WEIGHT1-L orchestration tests: synthetic episodes only; no SUMO/Mac training."""
from __future__ import annotations

import hashlib
import json
import random
import sys
import tempfile
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import explore_weight_train as learner
import explore_weight_long_train as long

SCENARIOS = [("20220812", "1300"), ("20220819", "1800")]
FROZEN = {"config": {"test": "synthetic"}, "environment_fingerprint": "env",
          "files": {"original.py": "unchanged"}, "packages": {"test": "1"}}
PARENT_FP = long._digest(FROZEN)
STUDY_FP = "study-synthetic"


def observation(step=0):
    result = np.zeros(110, np.float32)
    result[3], result[63], result[104], result[108] = 1, 1, 1, 1
    result[5], result[-1] = step / 720, max(0, 1 - step / 720)
    return result


def fake_episode(date, period, eval_seed, work, policy, **kwargs):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    for name in ("tripinfo.xml", "collisions.xml"):
        (work / name).write_text("<synthetic/>")
    (work / "cost_ledger.json").write_text('{"synthetic":true}')
    rows = []
    for step in range(720):
        obs = observation(step)
        mode = policy(obs, None)
        rows.append({"obs": obs, "next_obs": observation(step + 1), "mode": mode,
                     "executed_green": (mode + 2) % 4, "reward": -0.01 if mode == 4 else -0.03,
                     "done": step == 719})
    return SimpleNamespace(metrics={"date": date, "period": period, "seed": eval_seed,
                                    "mean_delay_s": 1.0, "missing": 0, "unfinished": 0,
                                    "cost_check_passed": True},
                           transitions=rows, decisions=[{"mode": r["mode"]} for r in rows], seconds=0.0)


def cheap_gradient(model):
    """A deterministic state update exercises schedules/replay RNG, not learning."""
    sample = model.replay.sample(64, model.sample_rng)
    change = float(sample["rewards"].mean()) * 1e-6
    with torch.no_grad():
        model.online[-1].bias.add_(change)
    model.n_updates += 1
    return abs(change)


def parent_checkpoint(source: Path):
    model = learner.ModeDQN(42, replay_capacity=1024)  # Short synthetic fixture, not a real trained model.
    model.episodes, model.num_timesteps, model.n_updates, model.target_syncs = 42, 30_240, 7_310, 6
    sample = [{"obs": observation(i), "next_obs": observation(i + 1), "mode": 4,
               "executed_green": 2, "reward": -0.01, "done": i == 3} for i in range(4)]
    model.replay.add_episode(sample)
    scenario_rng, exploration_rng = random.Random(42), random.Random(200_042)
    for _ in range(42):
        scenario_rng.choice(SCENARIOS)
    for _ in range(30_240):
        exploration_rng.random()
    state = {"scenario_sha256": hashlib.sha256(json.dumps(SCENARIOS).encode()).hexdigest(),
             "scenarios": SCENARIOS, "scenario_rng": scenario_rng.getstate(),
             "exploration_rng": exploration_rng.getstate(),
             "history": [{"episode": i + 1, "episode_seconds": 0.0} for i in range(42)]}
    source.mkdir(parents=True)
    long._write(source / "experiment.json", {"fingerprint": PARENT_FP, **FROZEN})
    folder = source / "models" / "seed_42"
    folder.mkdir(parents=True)
    learner._save_checkpoint(folder, model, PARENT_FP, state)


def file_snapshot(folder):
    return {str(p.relative_to(folder)): long._sha(p) for p in Path(folder).rglob("*") if p.is_file()}


def job(root: Path, seed=42, episodes=(84,), cap=10_000, preserve=False):
    return {"seed": seed, "source_root": str(root / "parent"), "out": str(root / "long"),
            "work": str(root / "scratch"), "episodes": list(episodes),
            "study_fingerprint": STUDY_FP, "parent_fingerprint": PARENT_FP,
            "global_deadline_unix": time.time() + 10_000, "job_cap_s": cap,
            "preserve_milestone": preserve}


class CloneTest(unittest.TestCase):
    def test_validate_parent_uses_manifest_digest_and_validates_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "parent"
            parent_checkpoint(source)
            before = file_snapshot(source)
            with patch.object(long, "_current_parent_record", return_value=FROZEN):
                identity = long.validate_parent(source)
            parent_model = learner.validate_checkpoint(source / "models/seed_42", PARENT_FP)
            self.assertEqual(identity, {"fingerprint": PARENT_FP,
                "checkpoint_sha256": long._sha(Path(parent_model["path"]) / "manifest.json")})
            self.assertEqual(file_snapshot(source), before)
            with (Path(parent_model["path"]) / "replay.pkl").open("ab") as stream:
                stream.write(b"tampered")
            with patch.object(long, "_current_parent_record", return_value=FROZEN):
                with self.assertRaises(ValueError):
                    long.validate_parent(source, check_runtime=False)

    def test_validate_parent_can_skip_only_package_versions(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "parent"
            parent_checkpoint(source)
            changed_packages = {**FROZEN, "packages": {"test": "2"}}
            with patch.object(long, "_current_parent_record", return_value=changed_packages):
                with self.assertRaises(ValueError):
                    long.validate_parent(source, check_runtime=True)
                self.assertEqual(long.validate_parent(source, check_runtime=False)["fingerprint"], PARENT_FP)
            changed_source = {**changed_packages, "files": {"original.py": "changed"}}
            with patch.object(long, "_current_parent_record", return_value=changed_source):
                with self.assertRaises(ValueError):
                    long.validate_parent(source, check_runtime=False)
            changed_env = {**changed_packages, "environment_fingerprint": "changed"}
            with patch.object(long, "_current_parent_record", return_value=changed_env):
                with self.assertRaises(ValueError):
                    long.validate_parent(source, check_runtime=False)

    def test_parent_record_verification_is_read_only_and_checks_packages(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "parent"
            parent_checkpoint(source)
            before = file_snapshot(source)
            with patch.object(long, "_current_parent_record", return_value=FROZEN):
                self.assertEqual(long.verify_parent(source, PARENT_FP)["parent_fingerprint"], PARENT_FP)
            changed = {**FROZEN, "packages": {"test": "2"}}
            with patch.object(long, "_current_parent_record", return_value=changed):
                with self.assertRaises(ValueError):
                    long.verify_parent(source, PARENT_FP)
            self.assertEqual(file_snapshot(source), before)

    def test_clone_preserves_bytes_rng_and_parent_independence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, destination = root / "parent", root / "long/models/seed_42"
            parent_checkpoint(source)
            before = file_snapshot(source)
            lineage = long.clone_parent_checkpoint(source, destination,
                parent_fingerprint=PARENT_FP, study_fingerprint=STUDY_FP)
            original = learner.validate_checkpoint(source / "models/seed_42", PARENT_FP)
            cloned = learner.validate_checkpoint(destination, PARENT_FP)
            self.assertEqual(original["files"], cloned["files"])
            self.assertEqual(lineage, long.clone_parent_checkpoint(source, destination,
                parent_fingerprint=PARENT_FP, study_fingerprint=STUDY_FP))
            self.assertNotEqual((Path(original["path"]) / "model.pt").stat().st_ino,
                                (Path(cloned["path"]) / "model.pt").stat().st_ino)
            self.assertTrue((destination / "stages/episode_000042/manifest.json").exists())
            with (Path(cloned["path"]) / "model.pt").open("ab") as stream:
                stream.write(b"destination-only")
            self.assertEqual(file_snapshot(source), before)

    def test_wrong_parent_episode_or_overlapping_output_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent_checkpoint(root / "parent")
            with self.assertRaises(ValueError):
                long.clone_parent_checkpoint(root / "parent", root / "parent/models/seed_42",
                    parent_fingerprint=PARENT_FP, study_fingerprint=STUDY_FP)
            identity = learner.validate_checkpoint(root / "parent/models/seed_42", PARENT_FP)
            self.assertEqual(identity["episodes"], 42)
            invalid = job(root)
            invalid["out"] = invalid["source_root"]
            with self.assertRaises(ValueError):
                long._job_parameters(invalid)


class JobTest(unittest.TestCase):
    def patches(self, episode=fake_episode):
        stack = ExitStack()
        stack.enter_context(patch.object(long, "_runtime_guard"))
        stack.enter_context(patch.object(long, "_current_parent_record", return_value=FROZEN))
        stack.enter_context(patch.object(long, "_training_scenarios", return_value=SCENARIOS))
        stack.enter_context(patch.object(long, "_episode_runner", return_value=episode))
        stack.enter_context(patch.object(learner.ModeDQN, "gradient_step", cheap_gradient))
        return stack

    def test_mac_guard_blocks_real_runner(self):
        with patch.object(long.sys, "platform", "darwin"):
            with self.assertRaisesRegex(RuntimeError, "Colab"):
                long._runtime_guard()

    def test_time_budget_stops_before_episode_and_keeps_resume_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent_checkpoint(root / "parent")
            before = file_snapshot(root / "parent")
            with self.patches(), patch.object(long, "_episode_runner") as runner:
                result = long.run_training_job(job(root, episodes=(84,), cap=5))
            runner.assert_called_once()  # Factory is acquired, but no episode is executed.
            self.assertEqual(result["status"], "time_limit")
            self.assertEqual(result["completed_episodes"], 42)
            self.assertEqual(result["completed_milestones"], [42])
            self.assertEqual(file_snapshot(root / "parent"), before)
            self.assertTrue(result["parent_checkpoint_unchanged"])

    def test_chunk_resume_matches_uninterrupted_without_unofficial_snapshots(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent_checkpoint(root / "parent")
            parent_before = file_snapshot(root / "parent")
            with self.patches():
                first = long.run_training_job(job(root, episodes=(84,)))
                second = long.run_training_job(job(root, episodes=(126,)))
                direct_job = job(root, episodes=(126,))
                direct_job["out"] = str(root / "direct")
                direct = long.run_training_job(direct_job)
            self.assertEqual((first["status"], second["status"], direct["status"]), ("complete",) * 3)
            self.assertEqual((first["completed_episodes"], second["completed_episodes"]), (84, 126))
            self.assertEqual(second["completed_milestones"], [42])
            self.assertFalse((root / "long/models/seed_42/stages/episode_000084").exists())
            self.assertFalse((root / "long/models/seed_42/stages/episode_000126").exists())
            a, state_a, _ = learner._load_checkpoint(root / "long/models/seed_42", PARENT_FP)
            b, state_b, _ = learner._load_checkpoint(root / "direct/models/seed_42", PARENT_FP)
            for name, value in a.online.state_dict().items():
                torch.testing.assert_close(value, b.online.state_dict()[name], rtol=0, atol=0)
            for name, value in a.target.state_dict().items():
                torch.testing.assert_close(value, b.target.state_dict()[name], rtol=0, atol=0)
            for key in ("observations", "next_observations", "actions", "rewards", "discounts", "dones"):
                np.testing.assert_array_equal(getattr(a.replay, key), getattr(b.replay, key))
            self.assertEqual(state_a["scenario_rng"], state_b["scenario_rng"])
            self.assertEqual(state_a["exploration_rng"], state_b["exploration_rng"])
            self.assertEqual(a.sample_rng.bit_generator.state, b.sample_rng.bit_generator.state)
            self.assertEqual(file_snapshot(root / "parent"), parent_before)

    def test_failed_episode_does_not_advance_model_and_has_durable_status(self):
        def bad(*args, **kwargs):
            result = fake_episode(*args, **kwargs)
            result.metrics["unfinished"] = 1
            return result
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent_checkpoint(root / "parent")
            with self.patches(bad):
                result = long.run_training_job(job(root, episodes=(84,)))
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["completed_episodes"], 42)
            self.assertIn("unfinished", result["error"])
            durable = long._read(root / "long/training_status/seed_42.json")
            self.assertEqual(durable["status"], "failed")
            self.assertEqual(durable["study_fingerprint"], STUDY_FP)
            self.assertEqual(len(list((root / "long/training/seed_42/episode_000043").glob("attempt_*/metrics.json"))), 1)

    def test_fresh_seed_budget_resume_does_not_clone_parent_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent_checkpoint(root / "parent")
            before = file_snapshot(root / "parent")
            with self.patches():
                stopped = long.run_training_job(job(root, seed=43, episodes=(42,), cap=5, preserve=True))
            self.assertEqual((stopped["status"], stopped["completed_episodes"]), ("time_limit", 0))
            calls = 0
            def one_then_stop(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls > 1:
                    raise long.TrainingTimeLimit("synthetic interruption")
                return fake_episode(*args, **kwargs)
            with self.patches(one_then_stop):
                once = long.run_training_job(job(root, seed=43, episodes=(42,), preserve=True))
            self.assertEqual((once["status"], once["completed_episodes"], once["control_steps"]),
                             ("time_limit", 1, 720))
            original_hash = once["latest_model_sha256"]
            def stop_now(*args, **kwargs):
                raise long.TrainingTimeLimit("synthetic interruption")
            with self.patches(stop_now):
                resumed = long.run_training_job(job(root, seed=43, episodes=(42,), preserve=True))
            self.assertEqual(resumed["completed_episodes"], 1)
            self.assertEqual(resumed["latest_model_sha256"], original_hash)
            identity = learner.validate_checkpoint(root / "long/models/seed_43", PARENT_FP)
            self.assertEqual(identity["seed"], 43)
            self.assertEqual(file_snapshot(root / "parent"), before)


if __name__ == "__main__":
    unittest.main()
