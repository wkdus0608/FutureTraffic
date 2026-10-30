"""[추가 탐색] 관측 정보 × 학습 방법 4조합(148절) 검사 — SUMO 없음. 새 24칸 셈·자리, 두 학습 방법, 실행 규칙, 자료 맞춤, 판정을 본다."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import explore_collect as ecl  # noqa: E402
import explore_combo as cb  # noqa: E402
import explore_distill as ed  # noqa: E402
import explore_student as es  # noqa: E402

M = [("e", "x"), ("e", "y"), ("s", "x")]


class PlanTest(unittest.TestCase):
    def test_constants(self):
        self.assertEqual((cb.ARRIVAL_EDGES_S, cb.ARRIVAL_SCALE), ((10.0, 20.0), 20.0))
        self.assertEqual((cb.N_ARRIVALS, cb.WIDTH), (24, {"o0": 72, "o1": 96}))
        self.assertEqual(cb.CELLS, {"a": ("o0", "l0"), "b": ("o1", "l0"), "c": ("o0", "l1"), "d": ("o1", "l1")})
        self.assertEqual((cb.STEPS, cb.SEEDS, cb.SAMPLES), (10_000, (0, 1, 2, 3, 4), 314))
        self.assertEqual((cb.EVAL_GROUPS, cb.EVAL_SEEDS), (ecl.EVAL_GROUPS, ecl.EVAL_SEEDS))  # 141절과 같은 평가
        self.assertEqual(len(cb.eval_jobs("w", "o")), 80)
        self.assertEqual(len(cb.OLD_RUNS) + len(cb.NEW_RUNS), 9)


class ArrivalTest(unittest.TestCase):
    def test_bins_and_counts(self):
        self.assertEqual([cb.arrival_bin(x) for x in (0.0, 9.99, 10.0, 19.99, 20.0, 80.0)], [0, 0, 1, 1, 2, 2])
        cars = [(("e", "x"), 50.0, 10.0, 8.0),      # 5초 → 칸 0
                (("e", "x"), 99.0, 10.0, 8.0),      # 9.9초 → 칸 0
                (("e", "x"), 100.0, 10.0, 8.0),     # 10초 → 칸 1
                (("e", "y"), 300.0, 10.0, 0.5),     # 30초 → 칸 2
                (("e", "y"), 30.0, 10.0, 0.05),     # 멈춘 차(0.1m/s 미만) → 안 셈
                (("s", "x"), -3.0, 10.0, 2.0),      # 정지선을 막 넘은 거리 → 0초
                (("w", "z"), 10.0, 10.0, 5.0),      # 보호 녹색 이동이 아님 → 안 셈
                (None, 10.0, 10.0, 5.0)]            # 경로를 모르는 차 → 안 셈
        f = cb.arrival_features(cars, M)
        self.assertEqual(len(f), 9)
        self.assertEqual(f, [0.1, 0.05, 0.0, 0.0, 0.0, 0.05, 0.05, 0.0, 0.0])        # 차 수 ÷ 20

    def test_clip(self):
        cars = [(("s", "x"), 1.0, 10.0, 5.0)] * 21
        self.assertEqual(cb.arrival_features(cars, M)[6], 1.0)

    def test_observation96_layout(self):
        obs72 = list(range(72))
        o = cb.observation96(obs72, [100.0 + i for i in range(24)])
        self.assertEqual(len(o), 96)
        self.assertEqual(o[:68], list(range(68)))                    # 72칸의 앞 68칸 그대로
        self.assertEqual(o[68:92], [100.0 + i for i in range(24)])   # 새 24칸
        self.assertEqual(o[92:], [68, 69, 70, 71])                   # 끝 4칸(그대로 실행될 행동 표시)은 맨 끝


def _sample(group="g1", seed=42, t0=0.0, p0=0, costs=None, allowed=(0, 1), obs=None):
    costs = costs or {0: [100.0] * 8, 1: [90.0] * 8}
    return {"group": group, "seed": seed, "t0": t0, "obs": np.zeros(72, dtype=np.float32) if obs is None else obs,
            "allowed": list(allowed), "p0": p0, "search": 1, "costs": costs,
            "left_east": 0, "left_south": 0, "left_east_served": False, "left_south_served": False}


def _easy(n=80, seed=0, column=10):
    """관측 column번 칸이 1이면 녹색 1이 4,000대·초 싸고, 0이면 4,000대·초 비싸다(P0는 늘 녹색 0, 실제 자료와 비슷한 크기). 녹색 2는 늘 허용되고 늘 비싸다."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        x = np.zeros(72, dtype=np.float32)
        flag = rng.random() < 0.5
        x[column] = 1.0 if flag else 0.0
        out.append(_sample(t0=float(i), obs=x, allowed=(0, 1, 2),
                           costs={0: [10_000.0] * 8, 1: [(6_000.0 if flag else 14_000.0)] * 8, 2: [18_000.0] * 8}))
    return out


class LearningTest(unittest.TestCase):
    def test_value_matches_student_training(self):
        """L0 학습은 72칸일 때 explore_student.train과 같은 가중치를 낸다(A = S6와 같은 방법)."""
        import torch as th
        data = _easy(n=30)
        X, Y = np.stack([s["obs"] for s in data]), np.stack([ed.targets(s) for s in data])
        mine = cb.train_value(X, Y, seed=3, steps=40)
        theirs = es.train(X, Y, seed=3, steps=(40,))[40]
        self.assertEqual(sorted(mine), sorted(theirs))
        self.assertTrue(all(th.equal(mine[k], theirs[k]) for k in mine))

    def test_cell_a_picks_like_s6_evaluation(self):
        """A 칸의 선택은 146절 평가(explore_collect.run_episode: es.predict → es.hybrid 문턱 0)와 같다."""
        data = _easy(n=40, seed=2)
        X, Y = np.stack([s["obs"] for s in data]), np.stack([ed.targets(s) for s in data])
        states = [es.train(X, Y, seed=k, steps=(60,))[60] for k in (0, 1, 2)]
        samples = [{**s, "obs72": s["obs"], "obs96": np.zeros(96, dtype=np.float32)} for s in data]
        mine, _ = cb.picks(states, "a", samples)
        theirs = [es.hybrid(es.predict(states, np.asarray([s["obs"]], dtype=np.float32))[0], {"allowed": s["allowed"], "p0": s["p0"]}, 0.0)
                  for s in data]
        self.assertEqual(mine, theirs)

    def test_policy_learns_easy_signal(self):
        data = [{**s, "obs72": s["obs"], "obs96": np.zeros(96, dtype=np.float32)} for s in _easy(n=120)]
        states = cb.fit(data, "c", seeds=(0, 1), steps=1_500)
        chosen, prob = cb.picks(states, "c", data)
        right = [c == (1 if s["obs"][10] else 0) for c, s in zip(chosen, data)]
        self.assertGreater(sum(right) / len(right), 0.9)
        self.assertTrue(np.allclose(prob.sum(axis=1), 1.0))
        self.assertTrue(all(c in s["allowed"] for c, s in zip(chosen, data)))

    def test_policy_uses_new_columns(self):
        """D(96칸)는 새 24칸에만 있는 신호도 배운다."""
        data = []
        for s in _easy(n=120, seed=1):
            o96 = np.zeros(96, dtype=np.float32)
            o96[70] = s["obs"][10]                                   # 신호를 새 24칸 자리(68~91)로 옮김
            data.append({**s, "obs72": np.zeros(72, dtype=np.float32), "obs96": o96})
        chosen, _ = cb.picks(cb.fit(data, "d", seeds=(0,), steps=1_500), "d", data)
        right = [c == (1 if s["obs96"][70] else 0) for c, s in zip(chosen, data)]
        self.assertGreater(sum(right) / len(right), 0.9)

    def test_policy_loss_ignores_disallowed(self):
        """허용 안 된 녹색은 확률 0이고, 비용이 nan이어도 손실이 nan이 되지 않는다."""
        data = [{**s, "obs72": s["obs"], "obs96": np.zeros(96, dtype=np.float32)}
                for s in _easy(n=20)] + [{**_sample(t0=99.0, allowed=(0, 3), costs={0: [10.0] * 8, 3: [50.0] * 8}),
                                          "obs72": np.zeros(72, dtype=np.float32), "obs96": np.zeros(96, dtype=np.float32)}]
        states = cb.fit(data, "c", seeds=(0,), steps=200)
        _, prob = cb.picks(states, "c", data)
        self.assertFalse(np.isnan(prob).any())
        self.assertEqual(prob[-1][1], 0.0)
        self.assertEqual(prob[-1][2], 0.0)


class RuleTest(unittest.TestCase):
    def test_pick_policy(self):
        s = {"allowed": [0, 1, 3], "p0": 1}
        self.assertEqual(cb.pick_policy(np.array([0.2, 0.3, 0.0, 0.5]), s), 3)
        self.assertEqual(cb.pick_policy(np.array([0.35, 0.3, 0.0, 0.35]), s), 0)   # 같으면 작은 번호
        self.assertEqual(cb.pick_policy(np.array([0.2, 0.4, 0.0, 0.4]), s), 1)     # P0가 같은 최댓값이면 P0
        self.assertEqual(cb.pick_policy(np.array([0.1, 0.2, 0.9, 0.3]), s), 3)     # 허용 안 된 녹색 2는 고르지 않음

    def test_attach(self):
        samples = [_sample(group="20220822_1300", seed=42, t0=1000.0), _sample(group="20220822_1300", seed=43, t0=1000.0)]
        feats = {("20220822_1300", 42, 1000.0): [0.5] * 24, ("20220822_1300", 43, 1000.0): [0.25] * 24}
        got = cb.attach(samples, feats)
        self.assertEqual((got[0]["obs96"].shape, float(got[0]["obs96"][70]), float(got[1]["obs96"][70])), ((96,), 0.5, 0.25))
        self.assertTrue(np.array_equal(got[0]["obs72"], samples[0]["obs"]))
        with self.assertRaises(SystemExit):
            cb.attach(samples, {("20220822_1300", 42, 1000.0): [0.5] * 24})


def _m(delay, le=100.0, ls=90.0, long=0, changes=10):
    return {"mean_delay_s": delay, "left_east": le, "left_south": ls, "long_waits": long, "changes": changes}


def _rows(a, b, c, d, p0=40.0, d_changes=10):
    """평가 묶음 4개 × seed 4개. 칸마다 묶음별 지연 목록(4개), 같은 묶음의 seed는 ±0.05씩 흔듦."""
    rows = []
    for gi, g in enumerate(("g1", "g2", "g3", "g4")):
        for k, seed in enumerate((42, 43, 44, 45)):
            j = (k - 1.5) * 0.05
            rows.append({"group": g, "seed": seed, "p0": _m(p0 + j, changes=0), "a": _m(a[gi] + j), "b": _m(b[gi] + j),
                         "c": _m(c[gi] + j), "d": _m(d[gi] + j, changes=d_changes)})
    return rows


class ReadingTest(unittest.TestCase):
    def test_effects(self):
        rows = _rows([41] * 4, [41] * 4, [41] * 4, [40] * 4)          # D만 1초 나음 → 관측·학습 각 0.5, 상호작용 1
        v = cb.effect_values(rows)
        self.assertTrue(np.allclose(v["observation"], 0.5) and np.allclose(v["learning"], 0.5) and np.allclose(v["interaction"], 1.0))
        r = cb.readings(rows)
        self.assertEqual(r["vs_a"]["d"]["word"], "기준보다 나아졌다")
        self.assertEqual((r["vs_a"]["b"]["word"], r["vs_a"]["c"]["word"]), ("변화 없음 또는 나빠짐", "변화 없음 또는 나빠짐"))
        self.assertEqual(r["effects"]["interaction"]["word"], "좋아지는 쪽 단서")
        self.assertTrue(r["together"])
        self.assertEqual(r["overall"], "기준보다 나아졌지만 아직 P0를 넘지 못함")    # D 40초 = P0 40초

    def test_overall_beats_p0(self):
        r = cb.readings(_rows([41] * 4, [39, 39.2, 39.1, 39.3], [41] * 4, [41] * 4))
        self.assertEqual(r["vs_a"]["b"]["word"], "기준보다 나아졌다")
        self.assertEqual(r["vs_p0"]["b"]["word"], "P0를 넘었다")
        self.assertEqual(r["overall"], "학습한 혼합 제어가 P0를 넘은 첫 단서(개발 평가)")
        self.assertFalse(r["together"])
        self.assertEqual(r["effects"]["observation"]["word"], "좋아지는 쪽 단서")
        self.assertEqual(r["effects"]["interaction"]["word"], "나빠지는 쪽 단서")   # 새 관측이 L1에서는 도움이 안 됨

    def test_none_and_unclear(self):
        r = cb.readings(_rows([41] * 4, [41.2] * 4, [41.5] * 4, [41.1] * 4))
        self.assertEqual(r["overall"], "이 4조합으로는 개선을 확인하지 못했다")
        self.assertEqual(r["effects"]["learning"]["word"], "나빠지는 쪽 단서")
        r = cb.readings(_rows([41] * 4, [41] * 4, [41] * 4, [40, 40, 41.3, 41.3]))    # D 평균 > 0, 묶음 2/4
        self.assertEqual(r["vs_a"]["d"]["word"], "불분명")
        r = cb.readings(_rows([41] * 4, [41] * 4, [41] * 4, [40] * 4, d_changes=4))  # 변경을 절반 넘게 줄여서만
        self.assertEqual(r["vs_a"]["d"]["word"], "불분명")


if __name__ == "__main__":
    unittest.main()
