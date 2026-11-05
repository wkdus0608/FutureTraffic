"""[추가 탐색] RELAX0 검사 — SUMO 실행 없음."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import explore_platoon0 as p0  # noqa: E402
import explore_relax0 as rx  # noqa: E402
import explore_weight as w  # noqa: E402

w.configure()
import focused_rl as fr  # noqa: E402

CFG = p0.config()


def greens() -> list[str]:
    """신호 연결 24개짜리 가짜 녹색 4개. 1번은 석천로 횡단을, 3번은 길주로 횡단을 함께 연다."""
    out = []
    for walk in ((), fr.CROSSWALKS["석천로 횡단"]["links"], (), fr.CROSSWALKS["길주로 횡단"]["links"]):
        out.append("".join("G" if i in walk else "r" for i in range(24)))
    return out


class RuleTest(unittest.TestCase):
    def tearDown(self):
        rx.apply_rule(rx.RULES["base"])

    def test_base_is_the_locked_rule(self):
        rx.apply_rule(rx.RULES["base"])
        phase_min, _ = fr.phase_rules(greens())
        self.assertEqual(phase_min, [15, 33, 15, 46])
        self.assertEqual(fr.CROSSWALKS["길주로 횡단"]["max_wait"], 155)
        self.assertEqual(fr.CROSSWALKS["석천로 횡단"]["max_wait"], 120)

    def test_each_variant_changes_only_its_rule(self):
        rx.apply_rule(rx.RULES["A"])
        self.assertEqual(fr.phase_rules(greens())[0], [10, 33, 10, 46])
        self.assertEqual(fr.CROSSWALKS["길주로 횡단"]["max_wait"], 155)
        rx.apply_rule(rx.RULES["B"])
        self.assertEqual(fr.phase_rules(greens())[0], [15, 33, 15, 46])
        self.assertEqual(fr.CROSSWALKS["길주로 횡단"]["max_wait"], 200)
        self.assertEqual(fr.CROSSWALKS["석천로 횡단"]["max_wait"], 155)
        rx.apply_rule(rx.RULES["C"])
        self.assertEqual(fr.phase_rules(greens())[0], [10, 33, 10, 46])
        self.assertEqual(fr.CROSSWALKS["길주로 횡단"]["max_wait"], 200)
        rx.apply_rule(rx.RULES["D"])
        self.assertEqual(fr.phase_rules(greens())[0], [10, 10, 10, 10])

    def test_changes_do_not_carry_over_between_runs(self):
        rx.apply_rule(rx.RULES["D"])
        rx.apply_rule(rx.RULES["base"])
        self.assertEqual(fr.phase_rules(greens())[0], [15, 33, 15, 46])
        self.assertEqual(fr.CROSSWALKS["석천로 횡단"]["max_wait"], 120)

    def test_vehicle_wait_rule_stays(self):
        """차량 누적 대기 165초는 계획기 보호 장치(explore_weight_core)에 묶여 있어 어떤 변형도 바꾸지 않는다."""
        for rule in rx.RULES.values():
            self.assertEqual(set(rule) - {"veh_min", "ped_wait", "walk_min"}, set())
        rx.apply_rule(rx.RULES["D"])
        self.assertEqual(fr.SignalConstraints().max_wait, 165)

    def test_guard_sees_the_new_limits(self):
        rx.apply_rule(rx.RULES["B"])
        guard = fr.SignalConstraints()
        guard.phase_min, guard.serving = fr.phase_rules(greens())
        self.assertEqual({c["max_wait"] for c in fr.CROSSWALKS.values()}, {200, 155})
        self.assertEqual(guard.serving["길주로 횡단"], [3])
        self.assertEqual(guard.serving["석천로 횡단"], [1])


class StatsTest(unittest.TestCase):
    def test_green_stats(self):
        executed = [1, 1, 0, 0, 0, 3, 3, 3, 3, 2, 2, 1, 1]
        s = rx.green_stats(executed)
        self.assertEqual(s["switches"], 4)
        self.assertEqual(s["median_green"], {"0": 15, "2": 10, "3": 20})
        self.assertEqual(s["min_green"]["2"], 10)
        self.assertEqual(s["ped_gap_max"]["길주로 횡단"], 25)      # 앞 5칸, 뒤 4칸
        self.assertEqual(s["ped_gap_max"]["석천로 횡단"], 45)
        self.assertAlmostEqual(s["ped_gap_mean"]["길주로 횡단"], 22.5)
        self.assertAlmostEqual(s["ped_wait_estimate"]["석천로 횡단"], 45 * 45 / (2 * 65))

    def test_no_green_at_all_counts_as_one_long_gap(self):
        s = rx.green_stats([0, 0, 2, 2])
        self.assertEqual(s["ped_gap_max"]["길주로 횡단"], 20)
        self.assertAlmostEqual(s["ped_wait_estimate"]["길주로 횡단"], 10.0)


class JobsTest(unittest.TestCase):
    def test_every_variant_has_one_folder(self):
        pairs = [p for name in rx.FOLDERS for p in rx.PAIRS[name]]
        self.assertEqual(len(pairs), 8)
        self.assertEqual(set(pairs), {f"{world}/{v}" for world in rx.POLICY for v in rx.VARIANTS})
        for name in rx.FOLDERS:
            for pair in rx.PAIRS[name]:
                world, variant = pair.split("/")
                self.assertEqual(rx.variant_folder(Path("/o"), world, variant), Path("/o") / name / "runs" / world / variant)

    def test_jobs_use_the_pinned_routes(self):
        jobs = rx.jobs_for(CFG, Path("/o"), Path("/k"), "fp", rx.PAIRS["R"])
        self.assertEqual(len(jobs), 84 * 4)
        self.assertEqual(len({j["target"] for j in jobs}), len(jobs))
        s = {x["date"] + x["period"] + x["tag"]: x for x in CFG["situations"]}
        j = jobs[0]
        self.assertEqual(j["route_sha256"], s[j["situation"]["date"] + j["situation"]["period"] + j["situation"]["tag"]]["route_sha256"]["R"])
        self.assertEqual(j["model_identity"], "R/A")


if __name__ == "__main__":
    unittest.main()
