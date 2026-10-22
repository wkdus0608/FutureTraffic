"""[추가 탐색] 선생님 DQN(103절) 검사 — SUMO 실행 없음. 작은 가짜 환경과 가짜 시범으로 학습기 전체를 돌려 본다."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import explore_teacher as et  # noqa: E402
import focused_rl  # noqa: E402
from focused_data import environment_fingerprint  # noqa: E402

N = 4                       # 녹색 4개
OBS = N + 2 + N            # [지금 녹색 원-핫, 특징 2칸, 허용 표시]


def make_obs(rng, current: int, free: bool) -> np.ndarray:
    f = rng.uniform(0, 1, 2)
    mask = np.zeros(N)
    if free:
        mask[[1, 2]] = 1                                     # 고를 수 있는 판단: 녹색 1·2
    else:
        mask[current] = 1                                    # 규칙상 그대로
    return np.concatenate([np.eye(N)[current], f, mask]).astype(np.float32)


def teacher_rule(obs: np.ndarray) -> int:
    allowed = np.flatnonzero(obs[-N:] > 0.5)
    if len(allowed) == 1:
        return int(allowed[0])
    return 1 if obs[N] > obs[N + 1] else 2                   # 특징 0이 크면 녹색 1, 아니면 녹색 2


class FakeEnv(gym.Env):
    """가짜 신호 환경(20단계마다 시간 끝, 보상 = 선생님과 같으면 0, 다르면 −1)."""

    def __init__(self, seed=0):
        super().__init__()
        self.observation_space = gym.spaces.Box(0, 1, (OBS,), np.float32)
        self.action_space = gym.spaces.Discrete(N)
        self.rng = np.random.default_rng(seed)

    def reset(self, *, seed=None, options=None):
        self.t, self.cur = 0, 1
        self.obs = make_obs(self.rng, self.cur, True)
        return self.obs, {}

    def step(self, action):
        r = 0.0 if int(action) == teacher_rule(self.obs) else -1.0
        self.cur, self.t = int(action), self.t + 1
        self.obs = make_obs(self.rng, self.cur, bool(self.rng.random() < 0.6))
        return self.obs, r, False, self.t >= 20, {"executed_action": int(action)}

def write_demos(folder: Path, kind="single", episodes=6, seeds=et.COLLECT_SEEDS, steps=20, **meta_over):
    rng = np.random.default_rng(1)
    for s in seeds:
        rec = {k: [] for k in ("obs", "next_obs", "action", "reward", "truncated", "free", "episode")}
        for e in range(episodes):
            obs = make_obs(rng, 1, True)
            for t in range(steps):
                a = teacher_rule(obs)
                nxt = make_obs(rng, a, bool(rng.random() < 0.6))
                for k, v in (("obs", obs), ("next_obs", nxt), ("action", a), ("reward", 0.0), ("truncated", t == steps - 1),
                             ("free", int((obs[-N:] > 0.5).sum()) > 1), ("episode", e)):
                    rec[k].append(v)
                obs = nxt
        meta = {"map": kind, "seed": s, "teacher": {"code": et.TEACHER_CODE}, "environment_fingerprint": environment_fingerprint(),
                **meta_over}
        np.savez(et.demo_file(folder, kind, s), obs=np.array(rec["obs"], np.float32), next_obs=np.array(rec["next_obs"], np.float32),
                 action=np.array(rec["action"]), reward=np.array(rec["reward"], np.float32), truncated=np.array(rec["truncated"]),
                 free=np.array(rec["free"]), episode=np.array(rec["episode"]), meta=json.dumps(meta))


class MarginTest(unittest.TestCase):
    def test_margin_loss(self):
        import torch as th
        q = th.tensor([[1.0, 2.0, 1.5, 9.0], [0.0, 5.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]])
        actions = th.tensor([[1], [1], [0]])
        allowed = th.tensor([[True, True, True, False], [False, True, True, False], [True, False, False, False]])
        free = th.tensor([True, True, False])
        # 1줄: max(1+0.8, 2, 1.5+0.8) − 2 = 0.3 (녹색 3은 허용 밖이라 9가 들어가지 않음), 2줄: max(5, 0+0.8) − 5 = 0, 3줄: 고를 수 없어 뺌
        self.assertAlmostEqual(et.margin_loss(q, actions, allowed, free).item(), 0.15, places=6)
        self.assertEqual(et.margin_loss(q, actions, allowed, th.tensor([False] * 3)).item(), 0.0)


class DemoTest(unittest.TestCase):
    def test_split_and_refuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_demos(Path(tmp), episodes=16)
            files = [et.demo_file(Path(tmp), "single", s) for s in et.COLLECT_SEEDS]
            train, hold = et.load_demos(files, "single")
            self.assertEqual((len(train), len(hold)), (5 * 16 - 2, 2))         # seed 1005의 14·15번 에피소드가 확인용
            with self.assertRaises(SystemExit):
                et.load_demos(files, "network12")                              # 지도가 다른 시범
        with tempfile.TemporaryDirectory() as tmp:
            write_demos(Path(tmp), environment_fingerprint="000000000000")
            with self.assertRaises(SystemExit):
                et.load_demos([et.demo_file(Path(tmp), "single", s) for s in et.COLLECT_SEEDS], "single")


class LearnerTest(unittest.TestCase):
    def setUp(self):
        self.saved = (et.PRETRAIN_UPDATES, et.TARGET_EVERY, et.HOLDOUT_FROM)
        et.PRETRAIN_UPDATES, et.TARGET_EVERY, et.HOLDOUT_FROM = 1500, 500, 4

    def tearDown(self):
        et.PRETRAIN_UPDATES, et.TARGET_EVERY, et.HOLDOUT_FROM = self.saved

    def _model(self, tmp: Path):
        write_demos(tmp, episodes=6)
        cls = et.teacher_dqn_class()
        cls.demo_files = [str(et.demo_file(tmp, "single", s)) for s in et.COLLECT_SEEDS]
        cls.demo_map, cls.imitation_dir = "single", str(tmp / "imitation")
        return cls("MlpPolicy", FakeEnv(), seed=0, learning_rate=1e-3, buffer_size=2000, learning_starts=50, batch_size=32,
                   train_freq=4, target_update_interval=200, exploration_initial_eps=et.EPSILON, exploration_final_eps=et.EPSILON,
                   n_steps=3, verbose=0)

    def test_imitation_learns_teacher_then_fine_tunes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            model = self._model(tmp)
            model.learn(total_timesteps=200)
            rep = json.loads((tmp / "imitation" / "imitation.json").read_text())
            self.assertGreater(rep["holdout"]["agreement"], 0.9, rep)              # 확인용 시범에서 선생님을 따라 함
            self.assertTrue(model.teacher_pretrained)
            self.assertGreater(model._n_updates, 0)                                # 다듬기 단계의 경사 하강이 돌았음
            self.assertEqual(model._demo_buffer.size(), 4 * 6 * 20 + 4 * 20)       # 확인용 2개 에피소드는 버퍼에 없음
            self.assertTrue((tmp / "imitation" / "dqn_204820.zip").exists())

    def test_saved_model_loads_in_plain_learner_without_demos(self):
        import zipfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            model = self._model(tmp)
            model.learn(total_timesteps=60)
            model.save(tmp / "final")
            names = zipfile.ZipFile(tmp / "final.zip").read("data").decode()
            self.assertNotIn("_demo_buffer", names)
            plain = focused_rl.hold_aware_dqn_class().load(tmp / "final")
            obs = make_obs(np.random.default_rng(3), 1, True)
            self.assertIn(int(plain.predict(obs, deterministic=True)[0]), [1, 2])   # 허용 표시 안에서만 고름
            again = type(model).load(tmp / "final", env=FakeEnv())
            self.assertTrue(again.teacher_pretrained)                               # 이어 학습이면 흉내 내기를 다시 하지 않음

    def test_no_random_warmup(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self._model(Path(tmp))
            model._last_obs = make_obs(np.random.default_rng(5), 1, True)[None]
            model.exploration_rate = 0.0
            picks = {int(model._sample_action(1000)[0][0]) for _ in range(20)}
            want = int(model.predict(model._last_obs, deterministic=True)[0][0])
            self.assertEqual(picks, {want})                                         # 학습 시작 전에도 정책으로 고름(무작위 아님)


class SettingsTest(unittest.TestCase):
    def test_check_settings(self):
        saved = {k: os.environ.get(k) for k in ("FOCUSED_DQN_NSTEPS", "FOCUSED_DQN_DOUBLE")}
        try:
            os.environ["FOCUSED_DQN_NSTEPS"] = "10"
            os.environ.pop("FOCUSED_DQN_DOUBLE", None)
            et._check_settings("single")
            with self.assertRaises(SystemExit):
                et._check_settings("network12")                                     # 12개 지도는 n = 5
            os.environ["FOCUSED_DQN_DOUBLE"] = "1"
            with self.assertRaises(SystemExit):
                et._check_settings("single")
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


if __name__ == "__main__":
    unittest.main()


# ------------------------------------------------------------------ 판정(가짜 결과)
def _models(root: Path, name: str, agreement=0.8) -> tuple[list[str], list[str]]:
    teach, imit = [], []
    for s in [42, 43, 44, 45, 46]:
        d = root / name / f"seed_{s}"
        (d / "imitation").mkdir(parents=True)
        (d / "training_config.json").write_text(json.dumps({"seed": s, "timesteps": 100000}))
        (d / "dqn_204820.zip").write_bytes(f"t{name}{s}".encode())
        (d / "teacher.json").write_text(json.dumps({"stage": "teacher", "imitation": {"holdout": {"free_decisions": 100, "agreement": agreement}}}))
        (d / "imitation" / "training_config.json").write_text(json.dumps({"seed": s, "timesteps": 0, "stage": "imitation_only"}))
        (d / "imitation" / "dqn_204820.zip").write_bytes(f"i{name}{s}".encode())
        teach.append(str(d))
        imit.append(str(d / "imitation"))
    return teach, imit


def _plain(root: Path, name: str) -> list[str]:
    out = []
    for s in [42, 43, 44, 45, 46]:
        d = root / name / f"seed_{s}"
        d.mkdir(parents=True)
        (d / "training_config.json").write_text(json.dumps({"seed": s, "timesteps": 100000}))
        (d / "dqn_204820.zip").write_bytes(f"{name}{s}".encode())
        out.append(str(d))
    return out


class TeachDecisionTest(unittest.TestCase):
    def _single(self, folder, date, tag, delay, arrivals, seed=None):
        import gzip
        run = folder / date / f"{tag}_eval42{'' if arrivals == 'random' else '_' + arrivals}"
        run.mkdir(parents=True)
        by = {"477706392_02->477706883_02 (l)": {"vehicles": 50, "mean_delay_s": 140.0},
              "477706883_01->477707013_01 (l)": {"vehicles": 50, "mean_delay_s": 100.0}}
        (run / "metrics.json").write_text(json.dumps({"date": date, "train_seed": seed, "all": {
            "vehicles": 1000, "mean_delay_s": delay, "long_waits": 5, "not_arrived": 0, "by_movement": by},
            "collisions_unexplained": [], "pedestrian": {"x": {"waits_over_limit": 0}}}))
        with gzip.open(run / "events.json.gz", "wt") as f:
            json.dump([{"pet": {"clearing_first": {"pet": 2.0}}}], f)

    def test_single(self):
        import explore_decide as ed
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teach_m, imit_m = _models(root, "teach")
            h1_m = _plain(root, "h1")
            for arr in ("random", "platoon"):
                for d in ed.VALIDATION:
                    for m in teach_m:
                        self._single(root / "teach_eval", d, ed.dqn_tag(teach_m)(int(m[-2:])), 37.0, arr, int(m[-2:]))
                    for m in imit_m:
                        self._single(root / "teach_eval", d, ed.dqn_tag(imit_m)(int(Path(m).parent.name[-2:])), 38.0, arr,
                                     int(Path(m).parent.name[-2:]))
                    for m in h1_m:
                        self._single(root / ("h1_eval" if arr == "random" else "pred"), d, ed.dqn_tag(h1_m)(int(m[-2:])), 39.8, arr,
                                     int(m[-2:]))
                    self._single(root / "v6", d, "greedy", 39.7, arr)
                    self._single(root / "v6", d, "fixedc", 47.3, arr)
                    self._single(root / "pred", d, "pred_p0", 37.5, arr)
            from types import SimpleNamespace
            args = SimpleNamespace(teach_dir=str(root / "teach_eval"), teach_models=teach_m, imitation_models=imit_m,
                                   h1_dir=str(root / "h1_eval"), h1_models=h1_m, v6_dir=str(root / "v6"), pred_dir=str(root / "pred"))
            r = ed.decide_teach_single(args)
            c = r["validation_random"]["comparisons"]
            self.assertEqual(c["1_vs_same_dqn"]["word"], "낫다")                      # 37.0 대 39.8
            self.assertEqual(c["2_vs_greedy"]["word"], "낫다")                        # 37.0 대 39.7
            self.assertEqual(c["3_vs_teacher"]["teacher_word"], "선생님 수준")         # 0.5초 차이
            self.assertEqual(c["4_vs_imitation"]["word"], "차이 없음")                # 1.0초
            self.assertEqual(r["interpretation"], ["1. 선생님이 DQN을 개선했다.", "2. DQN이 단순 규칙을 넘었다.", "3. 선생님 수준.",
                                                   "4. 흉내 내기가 대부분이다."])
            self.assertEqual(r["agreement"][0]["agreement"], 0.8)
            self.assertEqual(r["safety_note"], "PET < 0 없음")
            args.imitation_models = list(reversed(imit_m))                           # 짝이 안 맞는 흉내 모델은 거부
            with self.assertRaises(SystemExit):
                ed.decide_teach_single(args)

    def test_network12(self):
        import explore_decide as ed
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teach_m, imit_m = _models(root, "teach")
            b12_m = _plain(root, "b12")

            def window(folder, w, tag, seed, delay):
                d = folder / "windows" / w / f"{tag}_eval{seed}"
                d.mkdir(parents=True)
                by = {"477706392_02->477706883_02": {"vehicles": 20, "mean_delay_s": 160.0},
                      "477706883_01->477707013_01": {"vehicles": 20, "mean_delay_s": 95.0}}
                (d / "metrics.json").write_text(json.dumps({"window_204820": {"vehicles": 1000, "mean_delay_s": delay + (seed - 44) * 0.1,
                                                                              "long_delays": 10, "by_movement": by},
                                                            "collisions_in_204820": 0}))
            for dt in ed.VALIDATION:
                for h in (8, 17):
                    w = f"{dt}_{h:02d}"
                    for s in [42, 43, 44, 45, 46]:
                        window(root / "teach_eval", w, ed.dqn_tag(teach_m)(s), s, 53.0)
                        window(root / "teach_eval", w, ed.dqn_tag(imit_m)(s), s, 52.0)
                        window(root / "b12", w, ed.dqn_tag(b12_m)(s), s, 53.8)
                        window(root / "ref", w, "greedy", s, 54.1)
                        window(root / "ref", w, "fixedc", s, 56.5)
                        window(root / "pred", w, "pred_p0", s, 51.2)
            from types import SimpleNamespace
            args = SimpleNamespace(teach_dir=str(root / "teach_eval"), teach_models=teach_m, imitation_models=imit_m,
                                   base_dir=str(root / "b12"), base_models=b12_m, ref_dir=str(root / "ref"), pred_dir=str(root / "pred"))
            r = ed.decide_teach_12(args)
            c = r["comparisons"]
            self.assertEqual(c["1_vs_same_dqn"]["word"], "차이 없음")                 # 0.8초 < 2.2초
            self.assertEqual(c["3_vs_teacher"]["teacher_word"], "선생님 수준")         # 1.8초 < 2.2초
            self.assertEqual(r["interpretation"][0], "1. 선생님의 효과가 보이지 않는다.")
            self.assertEqual(r["safety_note"], "석천사거리 안 충돌 없음")


class CollectTest(unittest.TestCase):
    """시범 모으기 반복을 가짜 학습 환경·가짜 선생님으로 끝까지(파일 → 불러오기 → 학습기 버퍼)."""

    def test_collect_writes_loadable_demos(self):
        from types import SimpleNamespace
        import explore_pred as ep

        class Sim:
            traffic_signals = {"204820": SimpleNamespace(green_phases=[SimpleNamespace(state=str(i)) for i in range(N)])}

        class Env:
            action_space = FakeEnv().action_space

            def __init__(self):
                self.inner, self.current, self.route, self.history = FakeEnv(seed=2), Sim(), "routes.xml", []

            def reset(self):
                self.history.append(("20220811", f"{len(self.history):02d}00"))
                return self.inner.reset()

            def step(self, a):
                return self.inner.step(a)

            def close(self):
                pass

        env = Env()
        constraints = SimpleNamespace(allowed=lambda sim: [i for i in range(N) if env.inner.obs[-N + i] > 0.5])

        class Teacher:
            movements = staticmethod(lambda route: {})

            def __init__(self, movement_of, greens):
                pass

            def __call__(self, sim, allowed):
                return teacher_rule(env.inner.obs)

        saved = (et._training_env, ep.controller_class, ep.code_fingerprint)
        et._training_env = lambda kind, seed, workdir: (env, constraints)
        ep.controller_class = lambda variant, horizon: Teacher
        ep.code_fingerprint = lambda: et.TEACHER_CODE
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out = et.demo_file(Path(tmp), "single", 1005)
                meta = et.collect("single", 1005, 3, out)
                self.assertEqual((meta["transitions"], meta["episodes"], meta["problems"]),
                                 (60, 3, {"changed_by_rules": 0, "mask_differs": 0}))
                self.assertEqual(meta["scenarios"], [["20220811", "0000"], ["20220811", "0100"], ["20220811", "0200"]])
                d = np.load(out)
                self.assertEqual(d["obs"].shape, (60, OBS))
                self.assertEqual(int(d["truncated"].sum()), 3)                     # 에피소드마다 마지막 판단에서 시간 끝
                self.assertTrue(all(int(a) == teacher_rule(o) for a, o in zip(d["action"], d["obs"])))
                saved_from = et.HOLDOUT_FROM
                et.HOLDOUT_FROM = 2
                try:
                    train, hold = et.load_demos([out], "single")
                finally:
                    et.HOLDOUT_FROM = saved_from
                self.assertEqual((len(train), len(hold)), (2, 1))
        finally:
            et._training_env, ep.controller_class, ep.code_fingerprint = saved
