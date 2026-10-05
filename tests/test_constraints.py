"""운영 제약(보행 최소 녹색·보행 최대 대기·최대 녹색)이 실제 SUMO-RL 실행에서 지켜지는지 검사.

일부러 규칙을 어기려는 정책(한 현시만 고집, 매번 다음 현시, 무작위, 보행자 없는 현시만)으로
가장 붐비는 시간대(18:00)를 돌리고, 매초 신호 상태에서 직접 잰다.
"""
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from focused_data import COUNTS, TARGET_TLS, load_counts  # noqa: E402
from focused_scenario import AIHUB_YELLOW_S, FocusedNetwork, write_routes  # noqa: E402

SECONDS = 900


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


@unittest.skipUnless(COUNTS.exists(), "counts_204820.csv가 없으면 건너뜀")
class ConstraintChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.route = Path(cls.tmp.name) / "r.rou.xml"
        write_routes(FocusedNetwork(), load_counts(COUNTS)[("20220810", "1800")], cls.route, seed=1)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_policy(self, choose):
        from focused_experiment import dqn_network
        from focused_rl import CROSSWALKS, SignalConstraints, ZoneObservation, pedestrian_service, zone_queue_reward
        from focused_sumo import make_environment
        env = make_environment(seed=1, net_file=dqn_network(), route_file=self.route,
                               simulation_duration_sec=SECONDS, observation_class=ZoneObservation,
                               reward_fn=zone_queue_reward, record_per_second_queue=True,
                               min_green=15, max_green=120, yellow_time=AIHUB_YELLOW_S)
        rules = SignalConstraints(max_wait=120).bind(env)
        try:
            obs, _ = env.reset(seed=1)
            self.assertEqual(env.observation_space.shape, obs.shape)
            done = False
            while not done:
                ts = env.traffic_signals[TARGET_TLS]
                action = rules(env, choose(ts))
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
        finally:
            env.close()
        states = [r["tls_state"] for r in env.per_second_queue_records]
        return rules, states, pedestrian_service(states), CROSSWALKS

    def check(self, name: str, choose):
        from focused_experiment import signal_greens
        rules, states, ped, crosswalks = self.run_policy(choose)
        order = signal_greens()
        for state, length in green_runs(states):
            with self.subTest(policy=name, phase=state):
                # 녹색 상태 문자열로 현시 번호를 찾는다.
                if state in order:
                    self.assertGreaterEqual(length, rules.phase_min[order.index(state)])
                self.assertLessEqual(length, 120 + 5)
        for crosswalk, m in ped.items():
            with self.subTest(policy=name, crosswalk=crosswalk):
                self.assertEqual(m["short_greens"], 0, m)
                self.assertEqual(m["waits_over_limit"], 0, m)
                self.assertLessEqual(m["max_wait_s"], crosswalks[crosswalk]["max_wait"], m)
                self.assertGreater(m["waits"], 0, m)
        return rules

    def test_adversarial_policies_cannot_break_rules(self):
        from focused_experiment import signal_greens
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
            rules = self.check(name, choose)
            # 일부러 어기는 정책이므로 제약이 실제로 끼어들었어야 한다.
            if name.startswith("한 현시") or name.startswith("보행자"):
                self.assertGreater(rules.interventions, 0, name)


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
class HoldAwareDQNChecks(unittest.TestCase):
    def test_next_value_uses_current_phase_while_holding(self):
        import torch as th
        from focused_rl import hold_aware_dqn_class
        cls = hold_aware_dqn_class()
        fake = cls.__new__(cls)
        fake.action_space = type("A", (), {"n": 4})()
        next_q = th.tensor([[1.0, 5.0, 2.0, 3.0], [1.0, 5.0, 2.0, 3.0]])
        # 첫 줄: 현시 2, 최소녹색 진행 중(0) → 현재 현시 값 2.0 / 둘째 줄: 바꿀 수 있음(1) → 최대 5.0
        next_obs = th.tensor([[0, 0, 1, 0, 0, 0.3], [0, 0, 1, 0, 1, 0.3]], dtype=th.float32)
        values = fake.allowed_next_values(next_q, next_obs)
        self.assertEqual(values.tolist(), [2.0, 5.0])

    def test_replay_buffer_stores_executed_action(self):
        """학습 기능 확인용 300단계 실행(본 학습 아님)."""
        import numpy as np
        from focused_experiment import RandomScenarioEnv
        from focused_rl import SignalConstraints, hold_aware_dqn_class
        from stable_baselines3.common.monitor import Monitor
        counts = load_counts(COUNTS)
        with tempfile.TemporaryDirectory() as tmp:
            env = RandomScenarioEnv([("20220810", "0800")], counts, 3, Path(tmp),
                                    min_green=15, max_green=120, yellow=4,
                                    constraints=SignalConstraints(max_wait=120))
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


if __name__ == "__main__":
    unittest.main()
