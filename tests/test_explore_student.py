"""[추가 탐색] 학생 재평가(131절) 검사 — 가짜 탐색 기록으로 혼합 정책·고르는 법·학습 길이 떼어 두기·분할을 본다(SUMO 없음)."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import explore_distill as ed  # noqa: E402
import explore_student as es  # noqa: E402


def _sample(group="g1", p0=0, search=1, costs=None, allowed=(0, 1), obs=None):
    costs = costs or {0: [100.0] * 8, 1: [90.0] * 8}
    return {"group": group, "seed": 42, "t0": 0.0, "obs": np.zeros(72, dtype=np.float32) if obs is None else obs,
            "allowed": list(allowed), "p0": p0, "search": search, "costs": costs,
            "left_east": 0, "left_south": 0, "left_east_served": False, "left_south_served": False}


def _easy(groups=("ga", "gb", "gc"), n=40, seed=0):
    """관측 10번 칸이 1이면 녹색 1이 40대·초 싸고, 0이면 40대·초 비싸다(P0는 늘 녹색 0)."""
    rng = np.random.default_rng(seed)
    out = []
    for g in groups:
        for _ in range(n):
            x = np.zeros(72, dtype=np.float32)
            flag = rng.random() < 0.5
            x[10] = 1.0 if flag else 0.0
            out.append(_sample(group=g, search=1 if flag else 0, costs={0: [100.0] * 8, 1: [(60.0 if flag else 140.0)] * 8}, obs=x))
    return out


class PolicyTest(unittest.TestCase):
    def test_hybrid_threshold(self):
        s = _sample(allowed=(0, 1, 3), p0=1)
        pred = np.array([0.2, 0.1, -9.0, -0.6])                               # 3이 0.7(=700대·초) 싸다고 예측, 2는 허용 안 됨
        self.assertEqual(es.hybrid(pred, s, 0.0), 3)
        self.assertEqual(es.hybrid(pred, s, 500.0), 3)
        self.assertEqual(es.hybrid(pred, s, 1000.0), 1)                       # 문턱을 못 넘으면 P0
        self.assertEqual(es.hybrid(np.array([0.1, 0.1, 0.0, 0.3]), s, 0.0), 1)  # 같으면 P0

    def test_tau0_is_rule_f(self):
        rng = np.random.default_rng(1)
        for _ in range(200):
            allowed = sorted(rng.choice(4, size=rng.integers(2, 5), replace=False).tolist())
            s = _sample(allowed=allowed, p0=int(rng.choice(allowed)))
            pred = rng.normal(size=4).round(1)
            self.assertEqual(es.hybrid(pred, s, 0.0), ed.rule_f(pred, s))

    def test_select(self):
        self.assertEqual(es.select({(500, 0.0): 10.0, (2000, 0.0): 30.0, (500, 500.0): 20.0}), (2000, 0.0))
        self.assertEqual(es.select({(500, 0.0): 30.0, (2000, 500.0): 30.0, (500, 500.0): 30.0}), (500, 500.0))   # 같으면 τ 큰 쪽, 짧은 쪽
        self.assertIsNone(es.select({(500, 0.0): -5.0, (500, 500.0): 0.0}))  # 0 이하뿐이면 P0


class TrainTest(unittest.TestCase):
    def test_snapshots_match_separate_training(self):
        rng = np.random.default_rng(2)
        X = rng.random((30, 72)).astype(np.float32)
        Y = rng.normal(size=(30, 4)).astype(np.float32)
        Y[::3, 2] = np.nan
        both = es.train(X, Y, seed=3, steps=(20, 60))
        short = es.train(X, Y, seed=3, steps=(20,))
        a, b = es.predict([both[20]], X), es.predict([short[20]], X)
        self.assertTrue(np.allclose(a, b))
        self.assertFalse(np.allclose(es.predict([both[60]], X), a))

    def test_predict_is_mean(self):
        rng = np.random.default_rng(4)
        X = rng.random((10, 72)).astype(np.float32)
        Y = rng.normal(size=(10, 4)).astype(np.float32)
        s0, s1 = es.train(X, Y, 0, (30,))[30], es.train(X, Y, 1, (30,))[30]
        self.assertTrue(np.allclose(es.predict([s0, s1], X), (es.predict([s0], X) + es.predict([s1], X)) / 2, atol=1e-6))


class CrossValidateTest(unittest.TestCase):
    def test_learns_easy_signal_without_seeing_test_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            cv = es.cross_validate(_easy(), Path(tmp), steps=(50, 300), thresholds=(0.0, 500.0), seeds=(0, 1))
            for g, f in cv["folds"].items():
                self.assertNotIn(g, f["train_groups"])
                self.assertEqual((f["train"], f["test"]), (80, 40))
                self.assertEqual(f["choice"]["tau"], 0.0)                       # 이득 40대·초 < 500이므로 문턱 500은 아무것도 안 바꿈
                self.assertTrue((Path(tmp) / "policies" / g / "seed1.pt").exists())
            self.assertGreater(cv["total"]["student"]["share"], 0.9)
            self.assertEqual(es.reading(cv["total"], cv["folds"])["word"], "배울 수 있어 보인다(혼합 제어 후보)")

    def test_falls_back_to_p0_when_changing_never_helps(self):
        samples = [_sample(group=g, search=0, costs={0: [100.0] * 8, 1: [150.0] * 8}) for g in ("ga", "gb", "gc") for _ in range(10)]
        cv = es.cross_validate(samples, None, steps=(50,), thresholds=(0.0,), seeds=(0,))
        self.assertTrue(all(f["choice"] is None for f in cv["folds"].values()))
        self.assertEqual(cv["total"]["student"]["gain"], 0.0)
        self.assertEqual(es.reading(cv["total"], cv["folds"])["word"], "이 자료로는 배우지 못했다")


if __name__ == "__main__":
    unittest.main()
