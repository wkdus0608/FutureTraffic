"""[추가 탐색] 이동별 정보 DQN·고를 때만 묻는 DQN(114절) 검사 — SUMO 실행 없음. 가짜 환경·가짜 SUMO 연결·가짜 결과로 본다."""
import gzip
import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import explore_choice as ec  # noqa: E402
import focused_rl  # noqa: E402

N = 4                       # 녹색 4개
OBS = N + 2 + N            # [지금 녹색 원-핫, 특징 2칸, 허용 표시]


def make_obs(current: int, allowed, feat=(0.5, 0.5)) -> np.ndarray:
    mask = np.zeros(N)
    mask[list(allowed)] = 1
    return np.concatenate([np.eye(N)[current], feat, mask]).astype(np.float32)


class ScriptEnv(gym.Env):
    """allowed[t] = 시각 t 상태의 허용 녹색. step마다 보상 rewards[t], 마지막 상태에서 시간 끝."""

    def __init__(self, allowed, rewards):
        super().__init__()
        self.allowed, self.rewards = allowed, rewards
        self.observation_space = gym.spaces.Box(0, 1, (OBS,), np.float32)
        self.action_space = gym.spaces.Discrete(N)

    def reset(self, *, seed=None, options=None):
        self.t, self.cur = 0, self.allowed[0][0]
        return make_obs(self.cur, self.allowed[0]), {}

    def step(self, action):
        assert int(action) in self.allowed[self.t], (self.t, action)
        r = self.rewards[self.t]
        self.t, self.cur = self.t + 1, int(action)
        return make_obs(self.cur, self.allowed[self.t]), r, False, self.t >= len(self.allowed) - 1, {"executed_action": int(action)}


class RandomSignalEnv(gym.Env):
    """40단계(200초)마다 시간 끝. 허용 녹색이 하나뿐인 상태가 60%. 보상 = 특징 0이 크면 녹색 1, 아니면 녹색 2가 좋음."""

    def __init__(self, seed=0):
        super().__init__()
        self.observation_space = gym.spaces.Box(0, 1, (OBS,), np.float32)
        self.action_space = gym.spaces.Discrete(N)
        self.rng = np.random.default_rng(seed)

    def _obs(self):
        self.allowed = [self.cur] if self.rng.random() < 0.6 else [1, 2]
        self.obs = make_obs(self.cur, self.allowed, self.rng.uniform(0, 1, 2))
        return self.obs

    def reset(self, *, seed=None, options=None):
        self.t, self.cur = 0, 1
        return self._obs(), {}

    def step(self, action):
        a = int(action)
        assert a in self.allowed
        good = 1 if self.obs[N] > self.obs[N + 1] else 2
        r = 0.0 if len(self.allowed) == 1 or a == good else -1.0
        self.t, self.cur = self.t + 1, a
        return self._obs(), r, False, self.t >= 40, {"executed_action": a}


# ------------------------------------------------------------------ 설정
class ScheduleTest(unittest.TestCase):
    def test_matches_h1_settings(self):
        import focused_train
        s = focused_train.DQN_SETTINGS
        self.assertEqual(ec.BUDGET_S, 100_000 * 5)
        self.assertEqual(ec.WINDOW_S, 10 * 5)
        self.assertEqual(ec.RANDOM_S, s["learning_starts"] * 5)
        self.assertEqual(ec.EXPLORE_S, s["exploration_fraction"] * 100_000 * 5)
        self.assertEqual(ec.TRAIN_EVERY_S, s["train_freq"] * 5)
        self.assertEqual(ec.TARGET_EVERY_S, s["target_update_interval"] * 5)
        self.assertEqual((ec.EPS_START, ec.EPS_END), (s["exploration_initial_eps"], s["exploration_final_eps"]))
        self.assertEqual(ec.TOTAL_UPDATES, (100_000 - s["learning_starts"]) // s["train_freq"])    # H1의 경사 하강 24,750번
        self.assertEqual(ec.TOTAL_SYNCS, 100_000 // s["target_update_interval"])
        for k in ("learning_rate", "buffer_size", "batch_size"):
            self.assertEqual(ec.CHO_SETTINGS[k], s[k])

    def test_owed_counts(self):
        self.assertEqual(ec.owed_updates(ec.RANDOM_S), 0)
        self.assertEqual(ec.owed_updates(ec.RANDOM_S + 25), 1)
        self.assertEqual(ec.owed_updates(10 ** 9), ec.TOTAL_UPDATES)
        self.assertEqual(ec.owed_syncs(24_999), 0)
        self.assertEqual(ec.owed_syncs(25_000), 1)
        self.assertEqual(ec.owed_syncs(10 ** 9), ec.TOTAL_SYNCS)
        self.assertAlmostEqual(ec.exploration_at(0), 1.0)
        self.assertAlmostEqual(ec.exploration_at(ec.EXPLORE_S / 2), 0.525)
        self.assertAlmostEqual(ec.exploration_at(10 ** 9), 0.05)

    def test_check_env(self):
        names = (*ec.DQN_ENV_VARS, "FOCUSED_REWARD")
        saved = {k: os.environ.get(k) for k in names}
        try:
            for k in names:
                os.environ.pop(k, None)
            ec._check_env(ec.CHO_ENV)
            with self.assertRaises(SystemExit):
                ec._check_env(ec.MOV_ENV)                                          # B는 H1과 같게 n = 10 + Double
            os.environ.update(FOCUSED_DQN_NSTEPS="10", FOCUSED_DQN_DOUBLE="1")
            ec._check_env(ec.MOV_ENV)
            with self.assertRaises(SystemExit):
                ec._check_env(ec.CHO_ENV)                                          # C는 이 파일의 시간 일정만
            os.environ["FOCUSED_REWARD"] = "aging"
            with self.assertRaises(SystemExit):
                ec._check_env(ec.MOV_ENV)
            with self.assertRaises(SystemExit):                                    # B는 H1과 같은 10만 단계만
                ec.train_mov(["--seed", "42", "--timesteps", "5000", "--out", tempfile.mkdtemp()])
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_fingerprints(self):
        self.assertEqual(len(ec.observation_fingerprint()), 12)
        self.assertNotEqual(ec.observation_fingerprint(), ec.learner_fingerprint())
        self.assertEqual(ec.explore_record("MOV")["observation_code"], ec.explore_record("CHO")["observation_code"])


# ------------------------------------------------------------------ 저장소
def _add(buf, rng, done=False, trunc=False, units=None, reward=None):
    obs, nxt = rng.uniform(0, 1, (1, 3)).astype(np.float32), rng.uniform(0, 1, (1, 3)).astype(np.float32)
    info = {"TimeLimit.truncated": trunc}
    if units is not None:
        info["elapsed_units"] = units
    r = rng.normal() if reward is None else reward
    buf.add(obs, nxt, np.array([[rng.integers(0, 4)]]), np.array([r], dtype=np.float32), np.array([done or trunc]), [info])
    return nxt[0]


class BufferTest(unittest.TestCase):
    def _space(self):
        return gym.spaces.Box(0, 1, (3,), np.float32), gym.spaces.Discrete(4)

    def test_equals_sb3_nstep_when_units_are_one(self):
        from stable_baselines3.common.buffers import NStepReplayBuffer
        o, a = self._space()
        sb3 = NStepReplayBuffer(64, o, a, n_steps=10, gamma=0.99)
        ours = ec.TimedNStepBuffer(64, o, a, gamma=0.99, window_units=10)
        rng1, rng2 = np.random.default_rng(1), np.random.default_rng(1)
        flags = np.random.default_rng(2)
        for _ in range(150):                                                      # 64칸을 두 번 넘게 돌아 덮어씀
            done = bool(flags.random() < 0.03)
            trunc = (not done) and bool(flags.random() < 0.05)
            _add(sb3, rng1, done, trunc)
            _add(ours, rng2, done, trunc)
        pick = np.random.default_rng(3)
        for k in range(5):
            inds = pick.integers(0, 64, 32)
            np.random.seed(k)
            x = sb3._get_samples(inds)
            np.random.seed(k)
            y = ours._get_samples(inds)
            for f in x._fields:
                np.testing.assert_allclose(getattr(x, f).numpy(), getattr(y, f).numpy(), rtol=1e-5, atol=1e-6, err_msg=f)

    def test_variable_units(self):
        o, a = self._space()
        g = 0.99
        buf = ec.TimedNStepBuffer(32, o, a, gamma=g, window_units=10)
        rng = np.random.default_rng(0)
        units = [1, 9, 3, 2, 12, 1, 1]
        nxt = [_add(buf, rng, trunc=(i == 6), units=u, reward=float(i + 1)) for i, u in enumerate(units)]
        s = buf._get_samples(np.array([0, 1, 2, 4, 5]))
        want_r = [1 + g * 2, 2 + g ** 9 * 3, 3 + g ** 3 * 4 + g ** 5 * 5, 5, 6 + g * 7]
        want_d = [g ** 10, g ** 12, g ** 17, g ** 12, g ** 2]
        np.testing.assert_allclose(s.rewards.numpy()[:, 0], want_r, rtol=1e-5)
        np.testing.assert_allclose(s.discounts.numpy()[:, 0], want_d, rtol=1e-5)
        np.testing.assert_allclose(s.next_observations.numpy(), np.array([nxt[1], nxt[2], nxt[4], nxt[4], nxt[6]]), rtol=1e-6)
        np.testing.assert_array_equal(s.dones.numpy()[:, 0], [0, 0, 0, 0, 0])            # 시간 끝은 끝이 아니다(값을 이어 씀)
        stats = buf.window_stats()
        self.assertEqual(stats["transitions"], 7)
        self.assertEqual(stats["max_window_s"], 17 * 5)

    def test_episode_end_and_write_head(self):
        o, a = self._space()
        g = 0.99
        buf = ec.TimedNStepBuffer(32, o, a, gamma=g, window_units=10)
        rng = np.random.default_rng(0)
        _add(buf, rng, units=2, reward=1.0)
        _add(buf, rng, done=True, units=3, reward=2.0)                               # 진짜 끝(신호 환경에는 없지만 처리는 맞아야 함)
        _add(buf, rng, units=1, reward=4.0)
        _add(buf, rng, units=1, reward=8.0)                                          # 아직 끝나지 않은 마지막 저장 경험
        s = buf._get_samples(np.array([0, 2]))
        np.testing.assert_allclose(s.rewards.numpy()[:, 0], [1 + g ** 2 * 2, 4 + g * 8], rtol=1e-6)
        np.testing.assert_allclose(s.discounts.numpy()[:, 0], [g ** 5, g ** 2], rtol=1e-6)
        np.testing.assert_array_equal(s.dones.numpy()[:, 0], [1, 0])                 # 쓰는 자리 앞에서는 끝이 아니라 잠시 멈춤
        self.assertFalse(buf.timeouts[3, 0])                                         # 잠시 바꾼 표시는 되돌림


# ------------------------------------------------------------------ 감싸기
class ChoiceEnvTest(unittest.TestCase):
    def test_skips_forced_steps_and_discounts(self):
        allowed = [[1], [1], [1, 2], [2], [2], [0, 1, 2], [0], [1, 3], [3]]
        rewards = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0]
        g = 0.9
        env = ec.ChoiceOnlyEnv(ScriptEnv(allowed, rewards), g)
        obs, info = env.reset()
        self.assertEqual(info["reset_units"], 2)                                     # 처음 두 칸은 묻지 않고 진행
        self.assertEqual(ec.allowed_of(obs, N), [1, 2])
        obs, r, term, trunc, info = env.step(2)
        self.assertAlmostEqual(r, -3 + g * -4 + g ** 2 * -5)
        self.assertEqual((info["elapsed_units"], info["executed_action"]), (3, 2))
        self.assertEqual(ec.allowed_of(obs, N), [0, 1, 2])
        obs, r, term, trunc, info = env.step(1)
        self.assertAlmostEqual(r, -6 + g * -7)
        self.assertEqual(info["elapsed_units"], 2)
        obs, r, term, trunc, info = env.step(3)
        self.assertAlmostEqual(r, -8.0)
        self.assertEqual(info["elapsed_units"], 1)
        self.assertTrue(trunc)
        self.assertEqual(env.control_units, 8)                                       # 안쪽 5초 칸 수와 같음
        self.assertEqual(dict(env.stats), {"reset_forced": 2, "choices": 3, "forced": 3})

    def test_monitor_inside_counts_every_five_seconds(self):
        from stable_baselines3.common.monitor import Monitor
        env = ec.ChoiceOnlyEnv(Monitor(RandomSignalEnv(0)), ec.GAMMA)
        env.reset()
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(ec.allowed_of(env.unwrapped.obs, N)[0])
            done = term or trunc
        self.assertEqual(info["episode"]["l"], 40)                                   # 학습곡선 기록은 5초마다 원래 보상
        self.assertEqual(env.control_units, 40)


# ------------------------------------------------------------------ 이동별 관측
class FakeSumo:
    def __init__(self, on_edge, speed, pending):
        self.edge = SimpleNamespace(getLastStepVehicleIDs=lambda e: on_edge.get(e, []))
        self.vehicle = SimpleNamespace(getSpeed=lambda v: speed[v])
        self.simulation = SimpleNamespace(getPendingVehicles=lambda: pending)


class ObservationTest(unittest.TestCase):
    def test_protected_movements_on_real_signal(self):
        from focused_scenario import signal_greens
        moves = ec.protected_movements(signal_greens())
        self.assertEqual(len(moves), ec.N_MOVEMENTS)
        self.assertEqual(sorted({a for a, _ in moves}), focused_rl.APPROACHES)    # 진입로마다 둘(직진·좌회전)
        self.assertTrue(all(sum(m[0] == a for m in moves) == 2 for a in focused_rl.APPROACHES))

    def _world(self, tmp: Path):
        from focused_scenario import signal_greens
        greens = signal_greens()
        moves = ec.protected_movements(greens)
        m0, m1 = moves[0], moves[1]
        route = tmp / "r.rou.xml"
        route.write_text("<routes>"
                         f'<route id="a" edges="{m0[0]} {m0[1]}"/><route id="b" edges="{m1[0]} {m1[1]}"/>'
                         + "".join(f'<vehicle id="s{i}" route="a" depart="0"/>' for i in range(12))
                         + '<vehicle id="moving" route="a" depart="0"/><vehicle id="p1" route="b" depart="0"/>'
                         + '<vehicle id="q1" route="b" depart="0"/></routes>')
        zone = focused_rl.ZONE_EDGES
        on_edge = {zone[m0[0]][0]: [f"s{i}" for i in range(12)] + ["moving"], zone[m1[0]][-1]: ["q1"]}
        speed = {**{f"s{i}": 0.0 for i in range(12)}, "moving": 8.0, "q1": 0.05}
        sumo = FakeSumo(on_edge, speed, ["p1"])
        return greens, moves, route, sumo

    def test_features_count_like_simple_controller(self):
        from focused_eval import QueueFirst
        with tempfile.TemporaryDirectory() as tmp:
            greens, moves, route, sumo = self._world(Path(tmp))
            movement_of = QueueFirst.movements(route)
            counts = ec.waiting_by_movement(sumo, movement_of)
            self.assertEqual(counts, QueueFirst(movement_of, greens).waiting(sumo))   # 단순 제어와 같은 셈
            cap = {a: 10.0 for a in focused_rl.APPROACHES}
            feats = ec.movement_features(counts, moves, cap)
            self.assertEqual(feats[0], 1.0)                                          # 멈춘 12대 ÷ 10 → 최대 1(달리는 차는 안 셈)
            self.assertAlmostEqual(feats[1], 0.2)                                    # 멈춘 1대 + 못 들어온 1대
            self.assertEqual(feats[2:], [0.0] * 6)

    def test_observation_class_inserts_before_mask(self):
        from gymnasium import spaces
        with tempfile.TemporaryDirectory() as tmp:
            greens, moves, route, sumo = self._world(Path(tmp))
            base_obs = make_obs(1, [1, 2])

            class Base:
                def __init__(self, ts):
                    self.ts, self._capacity = ts, None

                def __call__(self):
                    self._capacity = {a: 10.0 for a in focused_rl.APPROACHES}
                    return base_obs.copy()

                def observation_space(self):
                    return spaces.Box(0, 1, (OBS,), np.float32)

            ts = SimpleNamespace(env=SimpleNamespace(_route=str(route)), sumo=sumo, num_green_phases=N,
                                 green_phases=[SimpleNamespace(state=g) for g in greens])
            fn = ec.movement_observation_class(Base)(ts)
            obs = fn()
            self.assertEqual(obs.shape, (OBS + ec.N_MOVEMENTS,))
            self.assertEqual(fn.observation_space().shape, (OBS + ec.N_MOVEMENTS,))
            np.testing.assert_array_equal(obs[:OBS - N], base_obs[:OBS - N])
            np.testing.assert_array_equal(obs[-N:], base_obs[-N:])                    # 행동 표시는 맨 끝 그대로
            np.testing.assert_allclose(obs[OBS - N:-N], [1.0, 0.2] + [0.0] * 6, rtol=1e-6)

    def test_install_once(self):
        saved = focused_rl.ZoneObservation
        try:
            ec.install_observation()
            once = focused_rl.ZoneObservation
            ec.install_observation()
            self.assertIs(focused_rl.ZoneObservation, once)
            self.assertTrue(once.movement_info)
        finally:
            focused_rl.ZoneObservation = saved


# ------------------------------------------------------------------ 학습기
class LearnerTest(unittest.TestCase):
    NAMES = ("BUDGET_S", "RANDOM_S", "EXPLORE_S", "TRAIN_EVERY_S", "TARGET_EVERY_S", "TOTAL_UPDATES", "TOTAL_SYNCS")

    def setUp(self):
        self.saved = {k: getattr(ec, k) for k in self.NAMES}
        ec.BUDGET_S, ec.RANDOM_S, ec.EXPLORE_S, ec.TRAIN_EVERY_S, ec.TARGET_EVERY_S = 3000, 200, 1000, 20, 500
        ec.TOTAL_UPDATES, ec.TOTAL_SYNCS = (3000 - 200) // 20, 3000 // 500

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(ec, k, v)

    def _env(self, seed=0):
        from stable_baselines3.common.monitor import Monitor
        return ec.ChoiceOnlyEnv(Monitor(RandomSignalEnv(seed)), ec.GAMMA)

    def _model(self, env):
        return ec.choice_dqn_class()("MlpPolicy", env, seed=0, learning_rate=1e-3, buffer_size=5000, batch_size=16,
                                     learning_starts=0, train_freq=1, gradient_steps=1, target_update_interval=10 ** 9,
                                     replay_buffer_class=ec.TimedNStepBuffer,
                                     replay_buffer_kwargs={"gamma": ec.GAMMA, "window_units": 10}, verbose=0)

    def test_time_schedule(self):
        env = self._env()
        model = self._model(env)
        model.learn(total_timesteps=10 ** 9, callback=[ec.budget_callback()])
        self.assertGreaterEqual(model.control_s, 3000)                               # 정한 제어 시간까지
        self.assertLess(model.control_s, 3000 + 5 * 40)                              # 넘어도 경험 하나 + 다음 에피소드 시작 진행 안
        self.assertEqual(model._n_updates, 140)                                      # (3000 − 200) ÷ 20
        self.assertEqual(model.target_syncs, 6)
        self.assertAlmostEqual(model.exploration_rate, ec.EPS_END)
        self.assertEqual(model.replay_buffer.size(), env.stats["choices"] - 1)       # 멈출 때 마지막 경험 하나는 저장 안 함
        self.assertLessEqual(model.control_s, 5 * env.control_units)
        stored = int(model.replay_buffer.units[:model.replay_buffer.size(), 0].sum())
        self.assertLessEqual(stored * 5, model.control_s)                            # 묻지 않은 칸도 제어 시간에 들어감
        lengths = env.get_wrapper_attr("get_episode_lengths")()
        self.assertTrue(lengths and all(n == 40 for n in lengths))

    def test_random_then_policy(self):
        model = self._model(self._env())
        model._last_obs = make_obs(1, [1, 2], (0.9, 0.1))[None]
        model.control_s = 0.0
        self.assertTrue({int(model._sample_action(0)[0][0]) for _ in range(30)} <= {1, 2})   # 무작위도 허용 안에서
        model.control_s, model.exploration_rate = 10 ** 6, 0.0
        want = int(model.predict(model._last_obs, deterministic=True)[0][0])
        self.assertEqual({int(model._sample_action(0)[0][0]) for _ in range(10)}, {want})

    def test_saved_model_loads_in_plain_learner_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            model = self._model(self._env())
            model.learn(total_timesteps=10 ** 9, callback=[ec.budget_callback()])
            model.save(tmp / "final")
            model.save_replay_buffer(tmp / "buffer")
            data = zipfile.ZipFile(tmp / "final.zip").read("data").decode()
            self.assertNotIn("replay_buffer_class", data)                            # 평가는 이 파일 없이 불러진다
            plain = focused_rl.hold_aware_dqn_class().load(tmp / "final")
            self.assertIn(int(plain.predict(make_obs(1, [1, 2]), deterministic=True)[0]), [1, 2])
            again = type(model).load(tmp / "final", env=self._env(1), replay_buffer_class=ec.TimedNStepBuffer,
                                     replay_buffer_kwargs={"gamma": ec.GAMMA, "window_units": 10})
            again.load_replay_buffer(tmp / "buffer")
            self.assertEqual(again.control_s, model.control_s)
            self.assertEqual(again.target_syncs, model.target_syncs)
            self.assertIsInstance(again.replay_buffer, ec.TimedNStepBuffer)
            n = model.replay_buffer.size()
            np.testing.assert_array_equal(again.replay_buffer.units[:n], model.replay_buffer.units[:n])


# ------------------------------------------------------------------ 판정(가짜 결과)
def _models(root: Path, name: str, experiment: str | None, code="obs1") -> list[str]:
    out = []
    for s in [42, 43, 44, 45, 46]:
        d = root / name / f"seed_{s}"
        d.mkdir(parents=True)
        (d / "training_config.json").write_text(json.dumps({"seed": s, "timesteps": 100000, "resumed": False}))
        (d / "dqn_204820.zip").write_bytes(f"{name}{s}".encode())
        if experiment:
            (d / "explore.json").write_text(json.dumps({"experiment": experiment, "observation_code": code}))
        out.append(str(d))
    return out


def _run(folder, date, tag, delay, arrivals, seed=None):
    run = folder / date / f"{tag}_eval42{'' if arrivals == 'random' else '_' + arrivals}"
    run.mkdir(parents=True)
    by = {"477706392_02->477706883_02 (l)": {"vehicles": 50, "mean_delay_s": 140.0},
          "477706883_01->477707013_01 (l)": {"vehicles": 50, "mean_delay_s": 100.0}}
    (run / "metrics.json").write_text(json.dumps({"date": date, "train_seed": seed, "all": {
        "vehicles": 1000, "mean_delay_s": delay, "long_waits": 5, "not_arrived": 0, "by_movement": by},
        "collisions_unexplained": [], "pedestrian": {"x": {"waits_over_limit": 0}}}))
    with gzip.open(run / "events.json.gz", "wt") as f:
        json.dump([{"pet": {"clearing_first": {"pet": 2.0}}}], f)


class ChoiceDecisionTest(unittest.TestCase):
    def _world(self, root: Path, b_code="obs1", c_code="obs1"):
        import explore_decide as ed
        h1, mov, cho = _models(root, "h1", None), _models(root, "mov", "MOV", b_code), _models(root, "cho", "CHO", c_code)
        for arr in ("random", "platoon"):
            for d in ed.VALIDATION:
                for models, folder, delay in ((h1, "h1_eval" if arr == "random" else "pred", 39.8), (mov, "choice", 38.0),
                                              (cho, "choice", 37.9)):
                    for m in models:
                        _run(root / folder, d, ed.dqn_tag(models)(int(m[-2:])), delay, arr, int(m[-2:]))
                _run(root / "v6", d, "greedy", 39.7, arr)
                _run(root / "v6", d, "fixedc", 47.3, arr)
                _run(root / "pred", d, "pred_p0", 37.5, arr)
        return SimpleNamespace(choice_dir=str(root / "choice"), mov_models=mov, cho_models=cho, h1_dir=str(root / "h1_eval"),
                               h1_models=h1, v6_dir=str(root / "v6"), pred_dir=str(root / "pred"))

    def test_choice(self):
        import explore_decide as ed
        with tempfile.TemporaryDirectory() as tmp:
            r = ed.decide_choice(self._world(Path(tmp)))
            c = r["validation_random"]["comparisons"]
            self.assertEqual(c["1_B_vs_A"]["word"], "낫다")                            # 38.0 대 39.8
            self.assertEqual(c["2_C_vs_B"]["word"], "차이 없음")                       # 0.1초
            self.assertEqual(r["interpretation"], [
                "1. 이동별 정보가 성능에 기여했다.", "2. 묶어 배우기의 추가 효과는 보이지 않는다.",
                "3. B: 같은 정보의 계산형 제어 수준이다. C: 같은 정보의 계산형 제어 수준이다.",
                "4. B: DQN이 단순 규칙을 넘었다. C: DQN이 단순 규칙을 넘었다."])
            self.assertEqual(r["base"], "B")
            self.assertEqual(r["validation_random"]["summary"]["B_mov"]["collapses"], 0)
            self.assertEqual(r["validation_random"]["per_seed_delay"]["C_cho"][42], 37.9)
            self.assertEqual(r["safety_note"], "PET < 0 없음; 확인 안 된 충돌 없음")

    def test_refuses_mixed_observation_or_wrong_models(self):
        import explore_decide as ed
        with tempfile.TemporaryDirectory() as tmp:
            args = self._world(Path(tmp), c_code="obs2")
            with self.assertRaises(SystemExit):
                ed.decide_choice(args)                                               # C는 B와 같은 관측이어야 함
            args.mov_models, args.cho_models = args.cho_models, args.mov_models
            with self.assertRaises(SystemExit):
                ed.decide_choice(args)                                               # 실험 이름이 바뀐 모델

    def test_collapse_blocks_better(self):
        import explore_decide as ed
        y = {"mean_delay_s": 40.0, "long_waits": 5, "left_east": 100, "left_south": 100, "collapses": 0}
        x = {**y, "mean_delay_s": 38.0}
        self.assertEqual(ed.verdict_choice(x, y)["word"], "낫다")
        self.assertEqual(ed.verdict_choice({**x, "collapses": 1}, y)["word"], "차이 없음")   # 무너진 실행이 늘면 '낫다' 아님
        self.assertEqual(ed.versus_same_info({**y, "mean_delay_s": 39.0}, {**y, "mean_delay_s": 37.5})["same_info_word"],
                         "같은 정보의 계산형 제어에 못 미친다")
        self.assertEqual(ed.collapses([{"date": "d", "all": {"mean_delay_s": 50.0}}, {"date": "d", "all": {"mean_delay_s": 40.0}}],
                                      [{"date": "d", "all": {"mean_delay_s": 47.3}}]), 1)


if __name__ == "__main__":
    unittest.main()
