"""[추가 학습] H1 이어 학습(162절) 검사 — SUMO 없음. 경계 표시, 탐험률 일정, 학습 기록, 판정을 본다."""
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import explore_h1long as hl  # noqa: E402


def _buffer(size=50, n_steps=10):
    from gymnasium import spaces
    from stable_baselines3.common.buffers import NStepReplayBuffer
    return NStepReplayBuffer(size, spaces.Box(-1e6, 1e6, (1,), np.float32), spaces.Discrete(2), device="cpu",
                             n_steps=n_steps, gamma=0.99)


def _add(rb, n, reward, start_obs):
    for k in range(n):
        o = np.array([[start_obs + k]], dtype=np.float32)
        rb.add(o, o + 1, np.array([[0]]), np.array([reward], dtype=np.float32), np.array([False]), [{}])


class PlanTest(unittest.TestCase):
    def test_constants(self):
        self.assertEqual((hl.SEEDS, hl.START, hl.STAGES), ((42, 43, 44, 45, 46), 100_000, (200_000, 300_000)))
        self.assertEqual(hl.H1_SHA10, {42: "0b12e12bd9", 43: "0763a1351b", 44: "97ee61162f", 45: "fa6dd9cedc", 46: "e4e73e9a2c"})
        self.assertEqual((hl.ENV_FP, hl.EVAL_FP, hl.FINAL_EPS), ("b20888514ea3", "2030250938d6", 0.05))


class BoundaryTest(unittest.TestCase):
    def _returns(self, rb, idx):
        s = rb._get_samples(np.array(idx))
        return s.rewards.numpy().ravel(), s.next_observations.numpy().ravel(), s.dones.numpy().ravel()

    def test_without_mark_rewards_cross_episodes(self):
        """표시 없이 이어 붙이면 끊긴 에피소드 끝 쪽 10단계 묶음에 새 에피소드 보상(100)이 섞인다(161절 문제)."""
        rb = _buffer()
        _add(rb, 15, 1.0, 0)                  # 끊긴 에피소드 A(끝 표시 없음)
        _add(rb, 12, 100.0, 1000)             # 이어 학습의 새 에피소드 B
        r, _, _ = self._returns(rb, [10])
        self.assertGreater(r[0], 100.0)

    def test_mark_stops_at_boundary(self):
        rb = _buffer()
        _add(rb, 15, 1.0, 0)
        info = hl.mark_boundary(rb)
        self.assertEqual((info["index"], info["marked"], info["before"]), (14, True, {"done": False, "timeout": False}))
        _add(rb, 12, 100.0, 1000)
        r, nxt, done = self._returns(rb, [10, 14, 15])
        g = 0.99
        self.assertAlmostEqual(r[0], sum(g ** k for k in range(5)), places=5)    # A의 10~14번 5개만
        self.assertAlmostEqual(r[1], 1.0, places=6)                              # 마지막 칸 하나
        self.assertEqual((nxt[0], nxt[1]), (15.0, 15.0))                         # A의 실제 다음 상태에서 가치를 잇는다
        self.assertEqual((done[0], done[1]), (0.0, 0.0))                         # 끝(종료)이 아니라 끊김 → 가치 잇기
        self.assertAlmostEqual(r[2], sum(100.0 * g ** k for k in range(10)), places=3)   # B 안에서는 그대로 10단계

    def test_mark_respects_real_end_and_wrap(self):
        rb = _buffer(size=20)
        _add(rb, 25, 1.0, 0)                  # 한 바퀴 넘게 채움(pos = 5, full)
        info = hl.mark_boundary(rb)
        self.assertEqual((info["index"], info["full"]), (4, True))
        rb2 = _buffer()
        o = np.array([[0.0]], dtype=np.float32)
        rb2.add(o, o, np.array([[0]]), np.array([1.0], dtype=np.float32), np.array([True]), [{}])
        self.assertFalse(hl.mark_boundary(rb2)["marked"])                       # 이미 에피소드 끝이면 그대로


class ScheduleTest(unittest.TestCase):
    def test_exploration(self):
        self.assertAlmostEqual(hl.exploration_at(20_000, 100_000), 0.05)        # H1 원래 학습: 2만 단계에서 0.05
        self.assertAlmostEqual(hl.exploration_at(100_000, 200_000), 0.05)       # 이어 학습 첫 호출
        self.assertAlmostEqual(hl.exploration_at(200_000, 300_000), 0.05)       # 두 번째 호출
        self.assertGreater(hl.exploration_at(30_000, 300_000), 0.5)             # 처음부터 30만이면 일정이 달라짐(쓰지 않는 길)


class RecordTest(unittest.TestCase):
    def test_zip_data_and_stage_config(self):
        with tempfile.TemporaryDirectory() as d:
            z = Path(d) / "m.zip"
            with zipfile.ZipFile(z, "w") as f:
                f.writestr("data", json.dumps({"num_timesteps": 100000, "_n_calls": 100000, "_n_updates": 24750, "exploration_rate": 0.05}))
            got = hl.zip_data(z)
            self.assertEqual((got["num_timesteps"], got["_n_updates"], got["exploration_rate"]), (100000, 24750, 0.05))
        cfg = {"seed": 42, "timesteps": 100000, "environment_fingerprint": "b20888514ea3", "environment": {"min_green": 15}}
        c = hl.stage_config(cfg, 200_000, {"restarts": 0})
        self.assertEqual((c["timesteps"], c["resumed"], c["seed"], c["environment"], c["h1long"]), (200_000, True, 42, {"min_green": 15}, {"restarts": 0}))
        self.assertFalse(hl.stage_config(cfg, 100_000, {})["resumed"])
        self.assertEqual(hl.eval_model_dir(Path("o"), 300_000, 46), Path("o/models/step_300000/seed_46"))

    def test_episode_windows(self):
        with tempfile.TemporaryDirectory() as d:
            m = Path(d) / "monitor.csv"
            m.write_text('#{"t_start": 0}\nr,l,t\n' + "".join(f"{-7.2 * (i + 1)},720,{i}\n" for i in range(140)))
            w = hl.episode_windows(m, 100_000)
            self.assertEqual((w[0]["from"], w[0]["episodes"]), (100_000, 69))      # 50,000 ÷ 720 → 69개가 첫 구간 안에서 끝남
            self.assertAlmostEqual(w[0]["reward_per_step"], -0.01 * 35, places=4)  # 1~69번 평균 = −0.01 × 35


def _s(delay, long=1.0, le=130.0, ls=90.0, col=0):
    return {"mean_delay_s": delay, "long_waits": long, "left_east": le, "left_south": ls, "collapses": col}


class ReadingTest(unittest.TestCase):
    def _r(self, d1, d2, d3, le3=130.0):
        s = {"step_100000": _s(d1), "step_200000": _s(d2), "step_300000": _s(d3, le=le3), "greedy": _s(39.69), "P0": _s(37.50),
             "fixedc": _s(47.31)}
        per = {f"step_{k}": {seed: v for seed in hl.SEEDS} for k, v in zip(hl.STEPS, (d1, d2, d3))}
        return hl.reading(s, per)

    def test_words(self):
        r = self._r(39.78, 39.5, 39.4)
        self.assertEqual((r["main"]["word"], r["main"]["sentence"]), ("차이 없음", "이 설정에서 추가 20만 단계의 효과를 확인하지 못했다"))
        self.assertEqual(r["step_300000_vs_P0"]["word"], "못하다")
        self.assertEqual(r["step_300000_vs_greedy"]["word"], "차이 없음")
        r = self._r(39.78, 38.9, 38.3)
        self.assertEqual((r["main"]["word"], r["200k_vs_100k"]["word"]), ("낫다", "차이 없음"))
        self.assertEqual(r["main"]["paired_seeds"]["shorter_seeds"], 5)
        self.assertEqual(self._r(39.78, 38.9, 38.3, le3=140.0)["main"]["word"], "차이 없음")   # 좌회전이 늘면 '낫다' 아님
        self.assertEqual(self._r(39.78, 40.5, 41.2)["main"]["sentence"], "이 설정에서 추가 학습 뒤 나빠졌다")
        self.assertEqual(self._r(39.78, 39.5, 39.4)["step_100000_vs_fixedc"]["percent"], -15.9)


if __name__ == "__main__":
    unittest.main()
