"""v6 환경 변형(전환 규칙의 전적색, 북쪽 진입로 연장) 검사."""
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import focused_data as fd  # noqa: E402
from focused_scenario import AIHUB_YELLOW_S, all_red_state, apply_all_red, signal_greens  # noqa: E402

CACHE = ROOT / "results/focused_204820/cache"


class Env:
    """환경 변수를 잠시 바꾼다."""

    def __init__(self, **values):
        self.values, self.old = values, {}

    def __enter__(self):
        for k, v in self.values.items():
            self.old[k] = os.environ.get(k)
            os.environ[k] = v

    def __exit__(self, *exc):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def yellow_between(a: str, b: str) -> str:
    return "".join("y" if x in "Gg" and y in "rs" else x for x, y in zip(a, b))


class VariantTests(unittest.TestCase):
    def test_rule_sets_match_measured_requirement(self):
        """규칙 묶음이 측정한 가장 작은 PET에서 ceil(기준 − PET)로 나온 값과 같다."""
        import json
        import math
        cfg = json.loads(fd.TRANSITION_RULES.read_text())
        pet = {tuple(map(int, k.split(">"))): v for k, v in cfg["evidence"]["smallest_pet_s"].items()}
        for name, th in (("pet1", 1.0), ("pet1_5", 1.5)):
            rules = {tuple(map(int, k.split(">"))): v for k, v in cfg["rule_sets"][name].items()}
            for pair, p in pet.items():
                self.assertEqual(rules.get(pair, 0), max(0, math.ceil(th - p)), (name, pair))
        self.assertEqual(cfg["rule_sets"]["none"], {})

    def test_variant_changes_fingerprint(self):
        with Env(FOCUSED_RULESET="none", FOCUSED_NORTH_EXT="240"):
            a = fd.environment_fingerprint()
        with Env(FOCUSED_RULESET="pet1", FOCUSED_NORTH_EXT="240"):
            b = fd.environment_fingerprint()
        with Env(FOCUSED_RULESET="none", FOCUSED_NORTH_EXT="0"):
            c = fd.environment_fingerprint()
        self.assertEqual(len({a, b, c}), 3)

    def test_unknown_rule_set_stops(self):
        with Env(FOCUSED_RULESET="없는 묶음"):
            with self.assertRaises(SystemExit):
                fd.variant()

    def test_north_extension_moves_only_the_first_point(self):
        text = fd.NETWORK.read_text(encoding="utf-8")
        out = fd._extended_network_text(text, "477706911_01_01", 240.0)
        import re
        old = re.findall(r'<lane id="477706911_01_01_\d"[^>]*?length="([\d.]+)"[^>]*?shape="([^"]+)"', text)
        new = re.findall(r'<lane id="477706911_01_01_\d"[^>]*?length="([\d.]+)"[^>]*?shape="([^"]+)"', out)
        self.assertEqual(len(old), 3)
        for (l0, s0), (l1, s1) in zip(old, new):
            self.assertAlmostEqual(float(l1) - float(l0), 240.0, places=1)
            p0, p1 = s0.split(), s1.split()
            self.assertEqual(p0[-1], p1[-1])                 # 정지선 쪽 끝점은 그대로
            x0, y0 = map(float, p0[0].split(","))
            x1, y1 = map(float, p1[0].split(","))
            self.assertAlmostEqual(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5, 240.0, delta=0.1)
        self.assertEqual(text.replace("\r", "").count("<lane"), out.count("<lane"))

    def test_apply_all_red_keeps_the_timetable(self):
        g = signal_greens()
        y01, y12 = yellow_between(g[3], g[0]), yellow_between(g[0], g[1])
        y12b = yellow_between(g[1], g[2])
        program = [(30, g[3]), (4, y01), (21, g[0]), (4, y12), (60, g[1]), (4, y12b), (25, g[2])]
        rules = {(1, 2): 2, (0, 1): 1}
        out, log = apply_all_red(program, rules)
        self.assertEqual(sum(d for d, _ in out), sum(d for d, _ in program))         # 하루 시간표가 같다
        self.assertEqual(log["with_all_red"], 2)
        self.assertEqual(log["inserted_s"], 3)
        # 0→1: 녹색 0이 1초 줄고 황색 뒤에 전적색 1초, 1→2: 녹색 1이 2초 줄고 전적색 2초
        self.assertEqual([d for d, _ in out], [30, 4, 20, 4, 1, 58, 4, 2, 25])
        self.assertEqual(out[4][1], all_red_state(y12))
        self.assertNotIn("y", out[4][1])
        # 규칙에 없는 전환(3→0, 2→...)은 그대로다
        self.assertEqual(out[0], (30, g[3]))
        self.assertEqual(out[1], (4, y01))

    def test_apply_all_red_never_removes_a_whole_green(self):
        g = signal_greens()
        program = [(2, g[1]), (4, yellow_between(g[1], g[2])), (30, g[2])]
        out, log = apply_all_red(program, {(1, 2): 3})
        self.assertEqual(out[0][0], 1)           # 1초는 남긴다
        self.assertEqual(log["shortened_below_rule"], 1)
        self.assertEqual(sum(d for d, _ in out), sum(d for d, _ in program))


class RuleSignalTimingTests(unittest.TestCase):
    """SUMO-RL 신호가 전환마다 황색 4초 뒤 전적색을 정확히 둔다(판단 간격 5초 사이에서 끊기지 않는다)."""

    @classmethod
    def setUpClass(cls):
        from focused_data import COUNTS, load_counts
        from focused_scenario import FocusedNetwork, write_routes
        if not COUNTS.exists():
            raise unittest.SkipTest("counts_204820.csv가 없으면 건너뜀")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.route = Path(cls.tmp.name) / "r.rou.xml"
        write_routes(FocusedNetwork(), load_counts(COUNTS)[("20220811", "1800")], cls.route, seed=1)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_cycle(self, rule_set: str, seconds: int = 900) -> list[str]:
        from focused_rl import ZoneObservation, zone_queue_reward
        from focused_scenario import dqn_template_network
        from focused_sumo import make_environment
        with Env(FOCUSED_RULESET=rule_set):
            env = make_environment(seed=1, net_file=dqn_template_network(CACHE), route_file=self.route,
                                   simulation_duration_sec=seconds, observation_class=ZoneObservation,
                                   reward_fn=zone_queue_reward, record_per_second_queue=True,
                                   min_green=15, max_green=120, yellow_time=AIHUB_YELLOW_S)
            try:
                env.reset(seed=1)
                ts = env.traffic_signals["204820"]
                done = False
                while not done:
                    # 일부러 늘 다음 현시를 고른다. 전환 중에 판단 시각이 와도 전환이 끊기면 안 된다.
                    _, _, term, trunc, _ = env.step((ts.green_phase + 1) % ts.num_green_phases)
                    done = term or trunc
                return [r["tls_state"] for r in env.per_second_queue_records]
            finally:
                env.close()

    def intergreens(self, states: list[str]) -> list[tuple[int, int, int, int]]:
        """(앞 녹색, 뒤 녹색, 황색 초, 전적색 초) 목록."""
        greens = signal_greens()
        out, cur, y, red = [], None, 0, 0
        for s in states:
            if s in greens:
                if cur is not None and (y or red) and greens.index(s) != cur:
                    out.append((cur, greens.index(s), y, red))
                cur, y, red = greens.index(s), 0, 0
            elif "y" in s:
                y += 1
            else:
                red += 1
        return out

    def test_no_rule_means_yellow_then_green(self):
        got = self.intergreens(self.run_cycle("none"))
        self.assertGreater(len(got), 10)
        for i, j, y, red in got:
            self.assertEqual((y, red), (AIHUB_YELLOW_S, 0), (i, j))

    def test_all_red_follows_the_rule_exactly(self):
        import json
        rules = {tuple(map(int, k.split(">"))): v
                 for k, v in json.loads(fd.TRANSITION_RULES.read_text())["rule_sets"]["pet1"].items()}
        got = self.intergreens(self.run_cycle("pet1"))
        self.assertGreater(len(got), 10)
        seen_red = False
        for i, j, y, red in got:
            self.assertEqual(y, AIHUB_YELLOW_S, (i, j))
            self.assertEqual(red, rules.get((i, j), 0), (i, j))
            seen_red = seen_red or red > 0
        self.assertTrue(seen_red, "전적색이 필요한 전환(1→2)이 한 번도 안 나왔다")


if __name__ == "__main__":
    unittest.main()
