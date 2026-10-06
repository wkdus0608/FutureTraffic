"""운영 규칙(보행 최소 녹색·보행 최대 대기·최대 녹색·차량 누적 대기)과 DQN 학습 장치 검사.

일부러 규칙을 어기려는 정책(한 현시만 고집, 매번 다음 현시, 무작위, 보행 현시만 번갈아)으로
붐비는 시간대(학습 날짜 08-11 18:00)를 돌리고, 매초 신호 상태에서 직접 잰다.
"""
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from focused_data import COUNTS, TARGET_TLS, load_counts  # noqa: E402
from focused_scenario import AIHUB_YELLOW_S, FocusedNetwork, dqn_template_network, signal_greens, write_routes  # noqa: E402

SECONDS = 900
CACHE = ROOT / "results/focused_204820/cache"
DATE, PERIOD = "20220811", "1800"


def green_runs(states):
    """(녹색 상태, 길이) 목록. 끝에서 잘린 마지막 녹색은 뺀다."""
    runs, cur, n = [], None, 0
    for s in states:
        if "y" in s:
            if cur is not None:
                runs.append((cur, n))
            cur, n = None, 0
        elif s == cur:
            n += 1
        else:
            if cur is not None:
                runs.append((cur, n))
            cur, n = s, 1
    return runs


def make_env(route: Path):
    from focused_rl import ZoneObservation, zone_queue_reward
    from focused_sumo import make_environment
    return make_environment(seed=1, net_file=dqn_template_network(CACHE), route_file=route,
                            simulation_duration_sec=SECONDS, observation_class=ZoneObservation,
                            reward_fn=zone_queue_reward, record_per_second_queue=True,
                            min_green=15, max_green=120, yellow_time=AIHUB_YELLOW_S)


@unittest.skipUnless(COUNTS.exists(), "counts_204820.csv가 없으면 건너뜀")
class ConstraintChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.route = Path(cls.tmp.name) / "r.rou.xml"
        write_routes(FocusedNetwork(), load_counts(COUNTS)[(DATE, PERIOD)], cls.route, seed=1)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_policy(self, choose, max_wait=None):
        from focused_rl import CROSSWALKS, SignalConstraints, pedestrian_service
        env = make_env(self.route)
        rules = SignalConstraints(max_wait=max_wait).bind(env)
        try:
            obs, _ = env.reset(seed=1)
            self.assertEqual(env.observation_space.shape, obs.shape)
            done = False
            while not done:
                ts = env.traffic_signals[TARGET_TLS]
                # 관측 끝 4칸(그대로 실행될 행동 표시)이 지금 규칙의 판단과 같아야 한다.
                self.assertEqual([i for i in range(4) if obs[-4 + i] > 0.5], rules.allowed(env))
                action = rules(env, choose(ts))
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
        finally:
            env.close()
        states = [r["tls_state"] for r in env.per_second_queue_records]
        return rules, states, pedestrian_service(states), CROSSWALKS

    def check(self, name: str, choose, max_wait=None):
        rules, states, ped, crosswalks = self.run_policy(choose, max_wait)
        order = signal_greens()
        for state, length in green_runs(states):
            with self.subTest(policy=name, phase=state):
                if state in order:
                    self.assertGreaterEqual(length, rules.phase_min[order.index(state)])
                self.assertLessEqual(length, 120 + 5)
        for crosswalk, m in ped.items():
            with self.subTest(policy=name, crosswalk=crosswalk):
                self.assertEqual(m["short_greens"], 0, m)
                self.assertEqual(m["waits_over_limit"], 0, m)
                self.assertLessEqual(m["max_wait_s"], crosswalks[crosswalk]["max_wait"], m)
                self.assertGreater(m["waits"], 0, m)
        return rules, states

    def test_adversarial_policies_cannot_break_rules(self):
        from focused_rl import phase_rules
        _, serving = phase_rules(signal_greens())
        no_ped = [i for i in range(4) if not any(i in idx for idx in serving.values())]
        rng = random.Random(7)
        policies = {
            "한 현시만 고집(0)": lambda ts: 0,
            "한 현시만 고집(3)": lambda ts: 3,
            "매번 다음 현시": lambda ts: (ts.green_phase + 1) % ts.num_green_phases,
            "무작위": lambda ts: rng.randrange(ts.num_green_phases),
            "보행자 없는 현시만": lambda ts: no_ped[0] if ts.green_phase != no_ped[0] else no_ped[-1],
        }
        for name, choose in policies.items():
            rules, _ = self.check(name, choose)
            # 일부러 어기는 정책이므로 제약이 실제로 끼어들었어야 한다.
            if name.startswith("한 현시") or name.startswith("보행자"):
                self.assertGreater(rules.interventions, 0, name)

    def test_vehicle_rule_serves_starved_left_turns(self):
        """보행 현시(직진 둘)만 번갈아 고르면 좌회전이 굶는다 → 누적 대기 규칙이 좌회전을 켜야 한다."""
        from focused_rl import phase_rules
        _, serving = phase_rules(signal_greens())
        ped_phases = sorted({i for idx in serving.values() for i in idx})
        choose = lambda ts: ped_phases[1] if ts.green_phase == ped_phases[0] else ped_phases[0]
        rules, states = self.check("보행 현시만 번갈아", choose, max_wait=120)
        self.assertGreater(rules.vehicle_interventions, 0)
        served = {s for s, _ in green_runs(states)}
        for i, g in enumerate(signal_greens()):
            if i not in ped_phases:
                self.assertIn(g, served, f"좌회전 현시 {i}가 한 번도 켜지지 않았다")


@unittest.skipUnless(COUNTS.exists(), "counts_204820.csv가 없으면 건너뜀")
class ActuatedPlanChecks(unittest.TestCase):
    def test_actuated_plan_respects_pedestrian_limits(self):
        """감응식도 DQN과 같은 보행자 최대 대기를 지킨다: 다른 세 현시가 모두 최대로 늘어도 한도 안."""
        from focused_eval import actuated_day_plan
        from focused_rl import CROSSWALKS, phase_rules
        for date in ("20220811", "20220906", "20220928"):
            greens, mins, maxes = actuated_day_plan(date)
            _, serving = phase_rules(greens)
            self.assertTrue(all(lo <= hi for lo, hi in zip(mins, maxes)), (date, mins, maxes))
            for name, phases in serving.items():
                worst = sum(maxes[i] for i in range(len(greens)) if i not in phases) + AIHUB_YELLOW_S * len(greens)
                self.assertLessEqual(worst, CROSSWALKS[name]["max_wait"], (date, name, maxes))


@unittest.skipUnless(COUNTS.exists(), "counts_204820.csv가 없으면 건너뜀")
class ZoneChecks(unittest.TestCase):
    def test_west_approach_zone_includes_both_branches(self):
        from focused_rl import ZONE_EDGES
        from focused_scenario import ENTRY_BRANCHES
        spec = ENTRY_BRANCHES["477707013_02"]
        self.assertIn(spec["alternative"], ZONE_EDGES["477707013_02"])
        self.assertIn(spec["main"], ZONE_EDGES["477707013_02"])
        fn = FocusedNetwork()
        # 모든 경로(두 갈래 포함)의 정지선 앞 도로가 감지 구간 안에 있다.
        for (a, b) in fn.movements:
            for branch in (["main", "alternative"] if a in ENTRY_BRANCHES else ["main"]):
                route = fn.route_for(a, b, branch)
                self.assertTrue(set(route[:route.index(a) + 1]) <= set(ZONE_EDGES[a]), (a, b, branch))


@unittest.skipUnless(COUNTS.exists(), "counts_204820.csv가 없으면 건너뜀")
class ArrivalChecks(unittest.TestCase):
    def test_arrivals_spread_over_period_and_volumes_kept(self):
        import xml.etree.ElementTree as ET
        from collections import Counter
        rows = load_counts(COUNTS)[("20220810", "0800")]
        fn = FocusedNetwork()
        with tempfile.TemporaryDirectory() as tmp:
            write_routes(fn, rows, Path(tmp) / "r.rou.xml", seed=1)
            root = ET.parse(Path(tmp) / "r.rou.xml").getroot()
        routes = {r.get("id"): r.get("edges").split() for r in root.findall("route")}
        moves = Counter()
        for v in root.findall("vehicle"):
            e = routes[v.get("route")]
            moves[next((a, b) for a, b in zip(e, e[1:]) if (a, b) in fn.movements)] += 1
        expected = Counter()
        for r in rows:
            expected[(r["from_edge"], r["to_edge"])] += r["count"]
        self.assertEqual(moves, expected)
        departs = [float(v.get("depart")) for v in root.findall("vehicle")]
        # 15분 동안 고르게 흩어져야 한다(각 5분 구간에 전체의 20~47%).
        thirds = Counter(min(2, int(d // 300)) for d in departs)
        for k in range(3):
            self.assertGreater(thirds[k] / len(departs), 0.20)


@unittest.skipUnless(COUNTS.exists(), "counts_204820.csv가 없으면 건너뜀")
class TrainingChecks(unittest.TestCase):
    def test_next_value_uses_only_executable_actions(self):
        import torch as th
        from focused_rl import hold_aware_dqn_class
        cls = hold_aware_dqn_class()
        fake = cls.__new__(cls)
        fake.action_space = type("A", (), {"n": 4})()
        next_q = th.tensor([[1.0, 5.0, 2.0, 3.0]] * 3)
        # 관측 = [현시 원-핫(4), ..., 실행 가능 표시(4)]
        next_obs = th.tensor([[0, 0, 1, 0, 0, 0.3, 0, 0, 1, 0],    # 최소 녹색 중: 지금 현시(2)만 → 2.0
                              [0, 0, 1, 0, 1, 0.3, 1, 0, 1, 1],    # 1번(5.0)은 규칙상 불가 → 남은 것 중 최대 3.0
                              [0, 0, 1, 0, 1, 0.3, 1, 1, 1, 1]],   # 모두 가능 → 5.0
                             dtype=th.float32)
        self.assertEqual(fake.allowed_next_values(next_q, next_obs).tolist(), [2.0, 3.0, 5.0])

    def test_warm_start_hands_over_at_a_real_green(self):
        """학습 에피소드는 실제 신호로 30분 돌린 뒤, 실제 녹색이 막 켜진 순간에 넘겨받는다."""
        from focused_rl import CROSSWALKS, SignalConstraints
        from focused_train import WARMUP_S, WarmStartTrainingEnv, warm_start
        counts = load_counts(COUNTS)
        with tempfile.TemporaryDirectory() as tmp:
            expected = warm_start(DATE, "0800", counts, Path(tmp) / "check.rou.xml")
            rules = SignalConstraints(max_wait=120)
            env = WarmStartTrainingEnv([(DATE, "0800")], counts, 3, Path(tmp), constraints=rules)
            obs, _ = env.reset()
            sim = env.current
            ts = sim.traffic_signals[TARGET_TLS]
            try:
                self.assertGreaterEqual(sim.sim_step, WARMUP_S)
                self.assertEqual(sim.sim_step, expected["handover_s"])
                self.assertEqual(sim.sumo.trafficlight.getRedYellowGreenState(TARGET_TLS), expected["handover_state"])
                self.assertEqual(ts.green_phases[ts.green_phase].state, expected["handover_state"])
                self.assertEqual(ts.time_since_last_phase_change, 0)
                self.assertGreater(sim.sumo.vehicle.getIDCount(), 50)          # 대기열이 있는 상태
                for name in CROSSWALKS:                                          # 보행 대기가 실제 신호와 이어진다
                    self.assertLessEqual(rules.ped_wait(name, ts), CROSSWALKS[name]["max_wait"])
                self.assertEqual(obs.shape, env.observation_space.shape)
            finally:
                env.close()

    def test_replay_buffer_stores_executed_action(self):
        """학습 기능 확인용 300단계 실행(본 학습 아님). 학습 전 무작위 탐색·탐험·Q값 선택 모두 실행 가능한 행동 안에서."""
        import numpy as np
        from focused_rl import SignalConstraints, hold_aware_dqn_class
        from focused_train import WarmStartTrainingEnv
        from stable_baselines3.common.monitor import Monitor
        counts = load_counts(COUNTS)
        rules = SignalConstraints(max_wait=120)
        with tempfile.TemporaryDirectory() as tmp:
            env = WarmStartTrainingEnv([(DATE, "0800")], counts, 3, Path(tmp), constraints=rules)
            model = hold_aware_dqn_class()("MlpPolicy", Monitor(env), learning_starts=50, buffer_size=1000,
                                           train_freq=4, seed=3, verbose=0)
            model.learn(total_timesteps=300)
            env.close()
        rb = model.replay_buffer
        n = rb.pos
        obs = rb.observations[:n, 0]
        actions = rb.actions[:n, 0, 0] if rb.actions.ndim == 3 else rb.actions[:n, 0]
        holding = obs[:, 4] < 0.5
        self.assertGreater(holding.sum(), 50)
        # 최소 녹색 진행 중에는 무엇을 골라도 현재 현시가 유지되므로 저장된 행동 = 현재 현시여야 한다.
        np.testing.assert_array_equal(actions[holding], obs[holding, :4].argmax(axis=1))
        # 그때 실행 가능 표시는 현재 현시 하나뿐이다.
        np.testing.assert_array_equal(obs[holding, -4:].sum(axis=1), 1)
        # DQN은 실행 가능 표시 안에서만 골랐으므로 규칙이 선택을 바꾼 적이 없고, 저장된 행동은 모두 표시 안에 있다.
        self.assertEqual(rules.interventions, 0)
        self.assertTrue((obs[np.arange(n), -4 + actions.astype(int)] > 0.5).all())
        self.assertEqual(rules.decisions, 300)
        self.assertGreater(rules.rule_decided["hold"], 50)

    def test_choice_is_the_best_executable_action(self):
        """평가·학습의 행동 선택 = 실행 가능 표시 안의 Q 최댓값. 탐험도 표시 안에서만."""
        import gymnasium as gym
        import numpy as np
        import torch as th
        from focused_rl import hold_aware_dqn_class

        class Dummy(gym.Env):
            observation_space = gym.spaces.Box(0, 1, (10,), np.float32)
            action_space = gym.spaces.Discrete(4)

            def reset(self, *, seed=None, options=None):
                return np.zeros(10, np.float32), {}

            def step(self, action):
                return np.zeros(10, np.float32), 0.0, True, False, {}

        model = hold_aware_dqn_class()("MlpPolicy", Dummy(), seed=1, verbose=0)
        model.q_net.forward = lambda obs: th.tensor([[1.0, 5.0, 2.0, 3.0]]).repeat(len(obs), 1)
        obs = np.array([0, 0, 1, 0, 1, 0.3, 1, 0, 1, 1], np.float32)     # 1번(Q 5.0)은 규칙상 실행 불가
        self.assertEqual(int(model.predict(obs, deterministic=True)[0]), 3)
        self.assertEqual(model.predict(np.stack([obs, obs]), deterministic=True)[0].tolist(), [3, 3])
        model.exploration_rate = 1.0                                      # 모두 무작위(탐험)여도 표시 안에서만
        picks = {int(model.predict(obs, deterministic=False)[0]) for _ in range(200)}
        self.assertEqual(picks, {0, 2, 3})


if __name__ == "__main__":
    unittest.main()
