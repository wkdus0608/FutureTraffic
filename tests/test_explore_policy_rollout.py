"""PI1 고정 정책·후속 운영의 순수 계산/모의 환경 검사. 실제 SUMO 검사가 아니다."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import explore_policy_rollout as pr


def _parent(directory, width=96, bias=None):
    import torch as th
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    weights = {}
    for seed in range(5):
        th.manual_seed(seed)
        net = pr.cb.make_net(width)
        if bias is not None:
            with th.no_grad():
                for p in net.parameters():
                    p.zero_()
                net[-1].bias.copy_(th.tensor(bias, dtype=th.float32))
        target = directory / f"seed{seed}.pt"
        th.save(net.state_dict(), target)
        weights[f"seed{seed}"] = hashlib.sha256(target.read_bytes()).hexdigest()[:12]
    policy = {"width": width, "method": "l0", "model_seeds": list(range(5)), "weights": weights}
    (directory / "policy.json").write_text(json.dumps(policy))
    return directory


def _features(left=True):
    return {"state": "test", "episode": "date_period", "t0": 100.0, "W": 10, "green": 0,
            "left_east": 8 if left else 0, "left_south": 0, "left_east_served": False, "left_south_served": False}


class FrozenPolicyTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_verifies_loads_once_and_freezes(self):
        import torch as th
        path = _parent(self.directory / "p")
        original = th.load
        with patch.object(th, "load", wraps=original) as loader:
            parent = pr.FrozenPolicy(path)
            for _ in range(3):
                parent.predict(np.zeros(96))
            self.assertEqual(loader.call_count, 5)
        self.assertTrue(all(not model.training for model in parent.models))
        self.assertTrue(all(not p.requires_grad for model in parent.models for p in model.parameters()))
        self.assertEqual(len(parent.hash), 64)
        self.assertEqual(parent.hash, pr.FrozenPolicy(path).hash)
        self.assertEqual(parent.manifest["parent_hash"], parent.hash)

    def test_tampered_weights_fail_before_inference(self):
        path = _parent(self.directory / "p")
        with (path / "seed0.pt").open("ab") as f:
            f.write(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA"):
            pr.FrozenPolicy(path)

    def test_width_and_model_count_rejected(self):
        path = _parent(self.directory / "p")
        policy = json.loads((path / "policy.json").read_text())
        policy["width"] = 72
        (path / "policy.json").write_text(json.dumps(policy))
        with self.assertRaises(ValueError):
            pr.FrozenPolicy(path)
        policy["width"], policy["model_seeds"] = 96, [0]
        (path / "policy.json").write_text(json.dumps(policy))
        with self.assertRaises(ValueError):
            pr.FrozenPolicy(path)

    def test_mask_and_ties(self):
        parent = pr.FrozenPolicy(_parent(self.directory / "p", bias=[0, 0, -100, 0]))
        obs = np.zeros(96)
        self.assertEqual(parent.choose(obs, [0, 1, 3], 3), 3)  # P0 tie priority; mask excludes 2
        self.assertEqual(parent.choose(obs, [0, 1, 2, 3], 3), 2)
        parent = pr.FrozenPolicy(_parent(self.directory / "q", bias=[-1, -1, -100, 0]))
        self.assertEqual(parent.choose(obs, [0, 1, 3], 3), 0)
        with self.assertRaises(ValueError):
            parent.choose(obs, [0, 1], 3)

    def test_matches_original_b_predictions(self):
        import torch as th
        path = _parent(self.directory / "p")
        parent = pr.FrozenPolicy(path)
        states = [th.load(path / f"seed{k}.pt", weights_only=True) for k in range(5)]
        X = np.random.default_rng(3).normal(size=(12, 96)).astype(np.float32)
        samples = [{"obs96": x, "allowed": [0, 1, 3], "p0": 1} for x in X]
        chosen, pred = pr.cb.picks(states, "b", samples)
        self.assertEqual([parent.choose(x, [0, 1, 3], 1) for x in X], chosen)
        np.testing.assert_allclose(np.stack([parent.predict(x) for x in X]), pred, atol=1e-7)

    def test_gate_forced_nonleft_and_control_end(self):
        parent = pr.FrozenPolicy(_parent(self.directory / "p", bias=[0, -1, 0, 0]))
        ep = SimpleNamespace(env=SimpleNamespace(sim_step=100), handover=0)
        with patch.object(pr, "_end_control", return_value=120), patch.object(pr.er, "_features", return_value=_features(False)), \
             patch.object(parent, "read_observation", return_value=[0.] * 96) as observation:
            self.assertEqual(parent.action(ep, [0], 0), 0)
            self.assertEqual(parent.action(ep, [0, 1], 0), 0)
            ep.env.sim_step = 120
            self.assertEqual(parent.action(ep, [0, 1], 0), 0)
            observation.assert_not_called()
        ep.env.sim_step = 100
        with patch.object(pr, "_end_control", return_value=120), patch.object(pr.er, "_features", return_value=_features()), \
             patch.object(parent, "read_observation", return_value=[0.] * 96):
            self.assertEqual(parent.action(ep, [0, 1], 0), 1)
            self.assertEqual(parent.action(ep, [0, 1], 0, end_control=100), 0)

    def test_sample_is_json_safe_and_keeps_mask(self):
        parent = pr.FrozenPolicy(_parent(self.directory / "p", bias=[0, -1, 0, 0]))
        ep = SimpleNamespace(env=SimpleNamespace(sim_step=100), handover=0)
        obs = [0.] * 92 + [1., 1., 0., 0.]
        with patch.object(pr, "_end_control", return_value=120), patch.object(pr.er, "_features", return_value=_features()), \
             patch.object(pr, "read_observation", return_value=obs):
            row = parent.sample_current(ep, [1, 0], 0)
        self.assertEqual(row["parent"], 1)
        self.assertEqual(row["parent_action"], 1)
        self.assertEqual(row["allowed"], [0, 1])
        self.assertEqual(row["obs"][-4:], obs[-4:])
        self.assertEqual(json.loads(json.dumps(row))["width"], 96)


class ObservationTest(unittest.TestCase):
    def test_order_and_safe_history_125(self):
        safe = {"past": (90., ["e"], "car")}
        ep = SimpleNamespace(env=SimpleNamespace(sim_step=100), vehicles={"future": (110., ["x"], "car")}, history_vehicles=safe)
        calls = []

        def hist(vehicles, t):
            calls.append((vehicles, t))
            return [0.2] * 29

        def combine(obs, h):
            return obs[:-4] + h + obs[-4:]

        module = SimpleNamespace(history_features=hist, observation_with_history=combine)
        with patch.object(pr.eo, "observation72", return_value=([0.] * 64, list(range(72)))), \
             patch.object(pr.cb, "arrivals24", return_value=[0.1] * 24), patch.dict(sys.modules, {"explore_history": module}):
            obs = pr.read_observation(ep, 125)
            base = pr.read_observation(ep, 96)
        self.assertEqual(len(obs), 125)
        self.assertEqual(obs[-4:], [68., 69., 70., 71.])
        self.assertEqual(obs[92:121], [0.2] * 29)
        self.assertEqual(calls, [(safe, 100.0)])
        self.assertEqual(len(base), 96)


class _Env:
    def __init__(self):
        self.sim_step, self.actions, self.closed = 100, [], False

    def _sumo_step(self):
        self.sim_step += 1

    def step(self, action):
        self.actions.append(action)
        for _ in range(5):
            self._sumo_step()

    def close(self):
        self.closed = True


class _Cons:
    def allowed(self, env):
        return [0, 1]

    def __call__(self, env, action):
        return action


class _Ctl:
    movement_of = {"past": ("e", "x")}

    def __init__(self):
        self.calls = []

    def __call__(self, env, allowed):
        self.calls.append(env.sim_step)
        return 0


class _Meter:
    def __init__(self, env, t0, vehicles, injected, marks):
        self.env, self.at, self.leaks = env, {}, set()
        self.marks, self.seed = marks, int(injected[0][0][3:])

    def step(self):
        pass

    def finish(self, trip):
        assert self.env.closed
        cost = sum(self.env.actions) * 10 + self.seed
        self.at = {h: {"cost": cost, "delay": cost, "terminal": 0} for h in self.marks}


class _Policy:
    hash, manifest = "h" * 64, {"test": True}
    seen = []

    def __init__(self, directory):
        pass

    def action(self, ep, allowed, p0_action):
        self.seen.append(dict(ep.history_vehicles))
        return 1 if ep.env.sim_step < 107 else 0


class MeasuredTest(unittest.TestCase):
    def _run(self, continuation):
        restored = []
        original = {"past": (90., ["e", "x"], "car"), "future": (106., ["future", "leak"], "car")}

        def restore(stem, routes, future, seed, horizon, tripinfo):
            self.assertEqual(future, "sampled")
            env, ctl = _Env(), _Ctl()
            restored.append((env, ctl))
            return env, _Cons(), object(), ctl, {"t0": 100., "handover": 0., "date": "d", "period": "p", "p0_action": 0}, \
                [(f"smp{seed}", ("e", "x"), "car", 106.)], original

        def safe_history(vehicles, t, injected):
            kept = {v: r for v, r in vehicles.items() if r[0] <= t}
            kept.update({v: (dep, list(edges), vt) for v, edges, vt, dep in injected})
            return kept

        _Policy.seen = []
        module = SimpleNamespace(history_vehicles=safe_history)
        with patch.object(pr, "FrozenPolicy", _Policy), patch.object(pr.er, "restore", side_effect=restore), \
             patch.object(pr.er, "DelayMeter", _Meter), patch.object(pr, "_end_control", return_value=107), \
             patch.dict(sys.modules, {"explore_history": module}):
            runs = [pr.measured(Path("state"), Path("routes"), 0, 3001, continuation, Path("parent"), horizon=10) for _ in range(2)]
        return runs, restored

    def test_first_candidate_then_parent_then_tail_p0(self):
        runs, restored = self._run("parent")
        first = runs[0]
        self.assertEqual(first["actions"], [0, 1])
        self.assertEqual(first["follow_actions"], [1, 0])
        self.assertEqual(first["follow_p0"], [0, 0])
        self.assertEqual(restored[0][1].calls, [105, 110])
        self.assertEqual(first["marks"], runs[1]["marks"])
        self.assertEqual(first["parent_hash"], "h" * 64)
        self.assertEqual(first["leaks"], 0)
        self.assertTrue(all("future" not in h and "smp3001" in h for h in _Policy.seen))
        self.assertTrue(all(e.closed for e, _ in restored))

    def test_p0_continuation_never_uses_parent_action(self):
        runs, restored = self._run("p0")
        self.assertEqual(runs[0]["actions"], [0, 0])
        self.assertEqual(_Policy.seen, [])
        self.assertEqual(restored[0][1].calls, [105, 110])

    def test_invalid_continuation_rejected_without_restore(self):
        with patch.object(pr.er, "restore") as restore:
            with self.assertRaises(ValueError):
                pr.measured(Path("state"), Path("routes"), 0, 1, "unknown", Path("parent"))
            restore.assert_not_called()


class SnapshotTest(unittest.TestCase):
    def test_checks_current_observation_and_action_without_advance(self):
        for mismatch in (False, True):
            env, ctl = _Env(), _Ctl()
            snap = {"date": "d", "period": "p", "t0": 100., "handover": 0., "allowed": [0, 1], "p0_action": 0}
            row = {"t0": 100., "allowed": [0, 1], "p0": 0, "parent": 1, "width": 96,
                   "left_point": True, "in_control": True, "obs": [0.] * 96, "obs96": [0.] * 96}
            actual = {**row, "obs": ([1.] + [0.] * 95) if mismatch else list(row["obs"])}
            policy = SimpleNamespace(width=96, hash="parent", sample_current=lambda ep, allowed, p0: actual)
            with patch.object(pr, "FrozenPolicy", return_value=policy), \
                 patch.object(pr.er, "restore", return_value=(env, _Cons(), object(), ctl, snap, [], {})) as restore:
                result = pr.verify_snapshot(Path("state"), Path("routes"), Path("parent"), row)
            self.assertEqual(result["ok"], not mismatch)
            self.assertEqual(result["problems"], ["obs"] if mismatch else [])
            self.assertEqual(ctl.calls, [100])
            self.assertEqual(env.sim_step, 100)
            self.assertEqual(env.actions, [])
            self.assertTrue(env.closed)
            self.assertEqual(restore.call_args.args[2], "real")


if __name__ == "__main__":
    unittest.main()
