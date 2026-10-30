"""[추가 탐색] 같은 수요의 seed 전이 검사(134절) 검사 — 가짜 탐색 기록으로 나누기·변경 셈·읽는 법을 본다(SUMO 없음)."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import explore_seedsplit as ess  # noqa: E402


def _sample(group="ga", seed=42, p0=0, search=1, costs=None, allowed=(0, 1), obs=None):
    costs = costs or {0: [100.0] * 8, 1: [90.0] * 8}
    return {"group": group, "seed": seed, "t0": 0.0, "obs": np.zeros(72, dtype=np.float32) if obs is None else obs,
            "allowed": list(allowed), "p0": p0, "search": search, "costs": costs,
            "left_east": 0, "left_south": 0, "left_east_served": False, "left_south_served": False}


def _easy(n=20, seed=0):
    """관측 10번 칸이 1이면 녹색 1이 40대·초 싸고, 0이면 40대·초 비싸다(P0는 늘 녹색 0)."""
    rng = np.random.default_rng(seed)
    out = []
    for g in ("ga", "gb", "gc"):
        for s in (42, 43):
            for _ in range(n):
                x = np.zeros(72, dtype=np.float32)
                flag = rng.random() < 0.5
                x[10] = 1.0 if flag else 0.0
                out.append(_sample(group=g, seed=s, search=1 if flag else 0, obs=x,
                                   costs={0: [100.0] * 8, 1: [(60.0 if flag else 140.0)] * 8}))
    return out


class ChangesTest(unittest.TestCase):
    def test_changes(self):
        a = _sample(search=1)                                                  # 바꿔서 10 이득(탐색과 같음)
        b = _sample(search=0, costs={0: [100.0] * 8, 1: [130.0] * 8})          # 바꿔서 30 손해
        c = _sample(search=1)                                                  # P0 그대로
        r = ess.changes([a, b, c], [1, 1, 0])
        self.assertEqual((r["changes"], r["changes_same_as_search"]), (2, 1))
        self.assertEqual((r["good_changes"], r["good_gain"], r["bad_changes"], r["bad_loss"]), (1, 10.0, 1, 30.0))


class CheckTest(unittest.TestCase):
    def test_learns_easy_signal_across_seeds(self):
        r = ess.check(_easy(), steps=300, seeds=(0, 1))
        self.assertEqual([(d["train_seed"], d["test_seed"], d["train"], d["test"]) for d in r["directions"]],
                         [(42, 43, 60, 60), (43, 42, 60, 60)])
        self.assertEqual(set(r["total_per_episode"]), {"ga", "gb", "gc"})
        self.assertEqual(r["total"]["decisions"], 120)
        self.assertGreater(r["total"]["share"], 0.9)
        self.assertEqual(ess.reading(r)["word"], "같은 수요 안에서는 옮길 단서가 있다")

    def test_never_better_reads_as_failure(self):
        samples = [_sample(group=g, seed=s, search=0, costs={0: [100.0] * 8, 1: [150.0] * 8}, obs=np.full(72, 0.5, np.float32))
                   for g in ("ga", "gb", "gc") for s in (42, 43) for _ in range(5)]
        for s in samples[::4]:                                                 # 학습에는 바꾸면 좋아 보이는 잡음 하나씩
            s["costs"] = {0: [100.0] * 8, 1: [50.0] * 8}
            s["search"] = 1
        r = ess.check(samples, steps=300, seeds=(0,))                         # 관측이 모두 같으므로 늘 바꾸거나 늘 안 바꿈 → 이득 ≤ 0
        self.assertLessEqual(r["total"]["gain"], 0.0)
        self.assertEqual(ess.reading(r)["word"], "같은 수요 안에서도 옮기지 못했다")


if __name__ == "__main__":
    unittest.main()
