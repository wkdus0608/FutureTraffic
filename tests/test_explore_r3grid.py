"""[추가 탐색] R3GRID 검사 — SUMO 실행 없음."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import explore_nn1 as nn  # noqa: E402
import explore_nn3 as n3  # noqa: E402
import explore_nn3r3 as r3  # noqa: E402
import explore_r3grid as g  # noqa: E402
import explore_r3grid_package as pkg  # noqa: E402
import explore_weight as w  # noqa: E402

w.configure()
CFG = g.config()


def synthetic(n=200, size=8, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, size)).astype(np.float32)
    base = rng.integers(0, 4, n)
    allowed = np.ones((n, 4), bool)
    gains = np.stack([x[:, 0], x[:, 1], -x[:, 0], -x[:, 1]], axis=1).astype(np.float32)
    gains[np.arange(n), base] = 0.0
    se = np.ones((n, 4), np.float32)
    return x, base, gains, se, allowed


class ConfigTest(unittest.TestCase):
    def test_grid_is_fifty_four(self):
        self.assertEqual(len(g.combos(CFG["grid"])), 54)
        self.assertEqual(len(g.candidates(CFG)), 53)
        self.assertEqual(len({g.combo_name(c) for c in g.combos(CFG["grid"])}), 54)
        self.assertNotIn(g.combo_name(CFG["default"]), g.candidates(CFG))
        self.assertTrue(all("_m" not in n for n in g.candidates(CFG)))

    def test_values_match_the_prereg(self):
        self.assertEqual(CFG["grid"], g.GRID)
        self.assertEqual(CFG["rules"], g.RULES)
        self.assertEqual(CFG["networks"]["64x3"], [64, 64, 64])
        self.assertEqual(CFG["share"], 0.10)
        self.assertEqual(CFG["recent"]["points"], 9199)

    def test_default_settings_are_round_three(self):
        r3cfg = r3.config()
        st = g.settings_of(CFG["default"], CFG["training"])
        for key in ("network", "learning_rate", "batch_size", "epochs_student", "weight_decay", "confidence_floor", "confidence_clip", "grad_clip", "seeds"):
            self.assertEqual(st[key], r3cfg["training"][key], key)
        self.assertEqual(CFG["label"], r3.LABEL)

    def test_situations(self):
        self.assertEqual(len(CFG["situations_screen"]), 24)
        self.assertEqual(len(CFG["situations_final"]), 84)
        self.assertEqual(CFG["situations_screen"], r3.config()["situations_screen"])
        self.assertEqual(CFG["situations_final"], w.read(g.TUNE1_CONFIG)["situations_screen"])
        self.assertEqual({s["tag"] for s in CFG["situations_final"]}, {"VU1", "VU2"})

    def test_screen_and_final_share_no_routes(self):
        a = {s["route_sha256"] for s in CFG["situations_screen"]}
        b = {s["route_sha256"] for s in CFG["situations_final"]}
        self.assertFalse(a & b)


class NetworkTest(unittest.TestCase):
    def test_two_layers_same_start_as_existing_student(self):
        torch.manual_seed(5)
        a = nn.GainNet(8, hidden=(64, 64))
        torch.manual_seed(5)
        b = g.StudentNet(8, hidden=(64, 64))
        self.assertEqual(list(a.state_dict()), list(b.state_dict()))
        for x, y in zip(a.state_dict().values(), b.state_dict().values()):
            self.assertTrue(torch.equal(x, y))

    def test_two_layer_training_same_as_existing(self):
        x, base, gains, se, allowed = synthetic()
        st = {**CFG["training"], "epochs_student": 3, "network": [64, 64]}
        old = nn.fit_gain(x, base, gains, se, allowed, 42, 3, st)
        new = g.fit_student(x, base, gains, se, allowed, 42, st)
        for a, b in zip(old.state_dict().values(), new.state_dict().values()):
            self.assertTrue(torch.equal(a, b))

    def test_three_layers_learn(self):
        x, base, gains, se, allowed = synthetic(n=600)
        st = g.settings_of({"network": "64x3", "learning_rate": "3e-3", "batch_size": 64, "epochs": 30}, CFG["training"])
        net = g.fit_student(x, base, gains, se, allowed, 42, st)
        self.assertEqual(len([m for m in net.trunk if isinstance(m, torch.nn.Linear)]), 3)
        best, predicted = nn.predicted_best(net, x, base, allowed)
        truth = gains.argmax(axis=1)
        self.assertGreater(float((best == truth).mean()), 0.8)

    def test_untrained_student_never_changes(self):
        net = g.StudentNet(120, hidden=(64, 64, 64)).eval()
        x = np.ones(120, np.float32)
        self.assertEqual(nn.student_choice(net, x, 2, [1, 2, 3], 0.5)[0], 2)

    def test_save_load_three_layers_and_old_student(self):
        with tempfile.TemporaryDirectory() as d:
            meta = {"kind": nn.KIND_GAIN, "world": "P", "seed": 42, "input_size": 8, "dataset_fingerprint": "t"}
            for hidden, cls in (((64, 64, 64), g.StudentNet), ((64, 64), nn.GainNet)):
                net = cls(8, hidden=hidden)
                folder = Path(d) / str(len(hidden))
                nn.save_model(folder, net, {**meta, "network": list(hidden)})
                loaded = g.load_student(folder)
                x, b = torch.ones(2, 8), torch.tensor([0, 1])
                self.assertTrue(torch.equal(loaded(x, b), net(x, b)))


class RuleTest(unittest.TestCase):
    def gain(self, s, low):
        return {"gain_s": s, "t95": [low, s + 0.1], "n": 84, "positive": 50}

    def test_finalists_order(self):
        self.assertEqual(g.finalists({"b": 30.0, "a": 30.0, "c": 29.0, "d": 31.0}, 3), ["c", "a", "b"])

    def test_adoption(self):
        rules = CFG["rules"]
        self.assertTrue(g.adoption(self.gain(0.2, 0.01), [1, 1, -1], rules))
        self.assertFalse(g.adoption(self.gain(0.19, 0.01), [1, 1, 1], rules))
        self.assertFalse(g.adoption(self.gain(0.5, -0.01), [1, 1, 1], rules))
        self.assertFalse(g.adoption(self.gain(0.5, 0.1), [1, -1, -1], rules))

    def test_factor_means(self):
        means = {g.combo_name(c): 30.0 for c in g.combos(CFG["grid"])}
        for name in means:
            if name.startswith("h256"):
                means[name] = 31.0
        got = g.factor_means(means, CFG)
        self.assertAlmostEqual(got["network"]["means"]["256"], 31.0)
        self.assertTrue(got["network"]["distinguished"])
        self.assertFalse(got["batch_size"]["distinguished"])

    def test_weight_difference(self):
        with tempfile.TemporaryDirectory() as d:
            meta = {"kind": nn.KIND_GAIN, "world": "P", "seed": 42, "input_size": 8, "dataset_fingerprint": "t", "network": [64, 64]}
            torch.manual_seed(1)
            net = g.StudentNet(8)
            nn.save_model(Path(d) / "a", net, {**meta, "seconds": 1})
            nn.save_model(Path(d) / "b", net, {**meta, "seconds": 2})
            self.assertEqual(g.weight_difference(Path(d) / "a", Path(d) / "b"), 0.0)
            with torch.no_grad():
                net.advantage.bias[0] += 0.5
            nn.save_model(Path(d) / "c", net, {**meta, "seconds": 3})
            self.assertAlmostEqual(g.weight_difference(Path(d) / "a", Path(d) / "c"), 0.5)

    def test_incomplete_report(self):
        with tempfile.TemporaryDirectory() as d:
            report = g.collect_report(Path(d), Path(d), CFG)
            self.assertEqual(report["status"], "before_students")
            self.assertIn("아직 끝나지 않았습니다", g.render(report))


class PackageTest(unittest.TestCase):
    def test_notebooks(self):
        self.assertEqual(set(pkg.NOTEBOOKS), {*(f"shard{s}" for s in range(4)), "judge"})
        for name in pkg.NOTEBOOKS:
            nb = pkg.notebook(name)
            self.assertEqual(nb["metadata"]["colab"]["machine_shape"], "hm")
            for cell in nb["cells"]:
                if cell["cell_type"] == "code":
                    compile("".join(cell["source"]), name, "exec")

    def test_tests_cell_uses_discover(self):
        self.assertIn('"discover"', pkg.TESTS_CELL)


class StudentPathTest(unittest.TestCase):
    def test_stack_choice_uses_student(self):
        net = g.StudentNet(n3.INPUT_SIZE, hidden=(64, 64, 64)).eval()
        with torch.no_grad():
            net.advantage.bias[:] = torch.tensor([0.0, 5.0, 0.0, 0.0])
        self.assertEqual(nn.student_choice(net, np.zeros(n3.INPUT_SIZE, np.float32), 2, [1, 2], 1.0)[0], 1)
        self.assertEqual(nn.student_choice(net, np.zeros(n3.INPUT_SIZE, np.float32), 2, [0, 2], 1.0)[0], 2)


if __name__ == "__main__":
    unittest.main()
