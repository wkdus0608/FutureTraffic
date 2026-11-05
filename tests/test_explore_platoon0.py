"""[추가 탐색] 0단계 PLATOON0 검사 — SUMO 실행 없음(실제 창 차량 파일·가짜 실행 기록)."""
import math
import statistics
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import arena3_common as ac  # noqa: E402
import arena3_referee as ref  # noqa: E402
import explore_platoon0 as p0  # noqa: E402
import explore_platoon0_report as rp  # noqa: E402
import explore_weight as w  # noqa: E402
import focused_scenario as fs  # noqa: E402
import focused_train as ft  # noqa: E402

CFG = p0.config()
LOCK = ac.config()


class TimetableRateTest(unittest.TestCase):
    def test_rates_per_state_and_future_arrivals(self):
        # 204822 상태: 0~99초 'G', 100초부터 'r'(구간 진입 시각 기준, 204822→구간 시간은 state_at 안에서 이미 뺐다고 둠)
        state_at = lambda t: "G" if t < 100 else "r"  # noqa: E731
        log = [(10.0, 0, "G"), (20.0, 0, "G"), (30.0, 1, "G"), (95.0, 0, "G"), (150.0, 1, "r")]
        rates = p0.state_rates(log, 0.0, 200.0, state_at, 60.0)
        self.assertAlmostEqual(rates["G"][0], 3 / 100)          # 'G'였던 100초 동안 이동 0이 3대
        self.assertAlmostEqual(rates["G"][1], 1 / 100)
        self.assertAlmostEqual(rates["r"][1], 1 / 100)          # 'r'도 100초
        short = p0.state_rates([(5.0, 0, "G")], 0.0, 30.0, state_at, 60.0)
        self.assertAlmostEqual(short["G"][0], 1 / 60)           # 분모는 60초 이상
        # 앞으로: 구간 통과 12초, 지금 80초 → 1단계(5초)부터 넣는다. 정지선 τ의 진입 = τ − 12초
        arrivals = [[0.0, 0.0] for _ in range(24)]
        p0.add_timetable_arrivals(arrivals, rates, 80.0, 5.0, 12.0, state_at)
        self.assertEqual(arrivals[0], [0.0, 0.0])
        self.assertEqual(arrivals[2], [0.0, 0.0])               # 진입 전 차는 12초 뒤부터(2단계 = 10~15초는 일부)
        k_g = 3                                                   # 정지선 95~100초 → 진입 83~88초('G')
        self.assertAlmostEqual(arrivals[k_g][0], 5 * 3 / 100)
        k_r = 10                                                  # 정지선 130~135초 → 진입 118~123초('r')
        self.assertAlmostEqual(arrivals[k_r][1], 5 * 1 / 100)
        self.assertAlmostEqual(arrivals[k_r][0], 0.0)

    def test_window_bounds(self):
        state_at = lambda t: "G"  # noqa: E731
        rates = p0.state_rates([(5.0, 0, "G"), (50.0, 0, "G")], 10.0, 70.0, state_at, 1.0)
        self.assertAlmostEqual(rates["G"][0], 1 / 60)            # 5초 진입은 창 밖


class WorldTest(unittest.TestCase):
    def window(self, s, world, d):
        from explore_weight_episode import CONTROL_S, TAIL_S
        from focused_data import COUNTS, load_counts
        path = Path(d) / f"{world}.rou.xml"
        origin = fs.period_start_s(s["period"]) - ft.WARMUP_S
        with p0.world_context(world, s["tag"]) as stats, ac.arrival_placement(s["tag"]):
            fs.window_route_file(s["date"], origin, origin + ft.WARMUP_S + CONTROL_S + TAIL_S, path, origin_s=origin,
                                 counts=load_counts(COUNTS))
        root = ET.parse(path).getroot()
        edges = {r.get("id"): r.get("edges").split() for r in root.iter("route")}
        rows = {v.get("id"): (v.get("type"), edges[v.get("route")], float(v.get("depart"))) for v in root.iter("vehicle")}
        return rows, stats, w.file_hash(path), origin

    def test_only_east_departures_move_within_their_quarter(self):
        s = CFG["situations"][0]
        with tempfile.TemporaryDirectory() as d:
            r, _, sha_r, origin = self.window(s, "R", d)
            p, stats, sha_p, _ = self.window(s, "P", d)
        self.assertEqual(sha_r, s["route_sha256"]["R"])
        self.assertEqual(sha_p, s["route_sha256"]["P"])
        self.assertEqual(set(r), set(p))
        fn, start, link_tt = p0.east_geometry()
        moved = [v for v in r if r[v][2] != p[v][2]]
        self.assertTrue(moved)
        self.assertTrue(all(r[v][1][0] == start for v in moved))
        self.assertTrue(all(r[v][:2] == p[v][:2] for v in r))
        self.assertGreater(stats["platoon"], 0.8 * sum(stats.values()))
        # 정지선 도착(출발 + 빈 도로 시간)이 원래 15분 시간대 안에 든다(파일 시작 이전으로 당겨진 차는 0초).
        # 기존 규칙처럼 녹색 '초'를 고른 뒤 그 1초 안에서 고르므로 끝은 1초까지 넘을 수 있다.
        for v in moved:
            period = v.split("_")[0]
            lead = fn.travel_time_to_stopline(p[v][1], p0.fp.EAST_APPROACH)
            arrival = origin + p[v][2] + lead
            p_start = fs.period_start_s(period)
            if p[v][2] > 0:
                self.assertGreaterEqual(arrival, p_start - 0.02)
                self.assertLess(arrival, p_start + 900 + 1.0 + 0.02)

    def test_context_restores_and_refuses_nesting(self):
        original = fs.window_route_file
        with p0.platoon_world("E1"):
            self.assertTrue(getattr(fs.window_route_file, "platoon0", False))
            with self.assertRaises(RuntimeError):
                with p0.platoon_world("E1"):
                    pass
        self.assertIs(fs.window_route_file, original)


class ConfigAndJobsTest(unittest.TestCase):
    def test_config_matches_arena3_eval_situations(self):
        lock = {ac.situation_id(s): s for s in LOCK["situations_eval"]}
        self.assertEqual(len(CFG["situations"]), 84)
        for s in CFG["situations"]:
            l = lock[ac.situation_id(s)]
            self.assertEqual({k: s[k] for k in ("date", "period", "tag", "sumo_seed")},
                             {k: l[k] for k in ("date", "period", "tag", "sumo_seed")})
            self.assertEqual(s["route_sha256"]["R"], l["route_sha256"])
            self.assertNotEqual(s["route_sha256"]["P"], l["route_sha256"])

    def test_jobs(self):
        jobs = p0.jobs_for(CFG, Path("/o"), Path("/w"), "f")
        self.assertEqual(len(jobs), 84 * 6)
        self.assertEqual(len({j["target"] for j in jobs}), len(jobs))
        self.assertEqual({(j["world"], j["policy"]) for j in jobs},
                         {(wd, n) for wd, names in p0.RUNS.items() for n in names})
        for j in jobs:
            s = next(x for x in CFG["situations"] if ac.situation_id(x) == j["sid"])
            self.assertEqual(j["route_sha256"], s["route_sha256"][j["world"]])
            self.assertEqual(j["model_identity"], f"{j['world']}/{j['policy']}")

    def test_policy_names(self):
        self.assertEqual(p0.build_policy("M4").name, "M4")
        for name in ("P0M4", "P1M4", "TM4", "OM4"):
            self.assertEqual(p0.build_policy(name).name, name)
            self.assertIsInstance(p0.build_policy(name), ac.P0Policy)    # 실행기가 모든 단계에서 부르는 종류
        with self.assertRaises(ValueError):
            p0.InfoM4("X")


def fake(s, world, delay, policy, n=100, lefts=(100.0, 80.0)):
    sid = ac.situation_id(s)
    return {"situation": sid, "policy": policy, "vehicles": n, "route_sha256": s["route_sha256"][world],
            "cohort_sha256": f"{world}{sid}", "actual_total_delay": delay * n, "mean_delay_s": delay, "long_waits": 0,
            "missing": 0, "unfinished": 0, "changed_by_rules": 0, "mode_action_errors": 0, "p0_mismatch": 0,
            "cost_check": {"passed": True}, "collisions_at_204820": 0, "collision_details": {"collisions_unexplained": []},
            "by_movement": {ref.LEFT_KEYS["left_east"]: {"vehicles": 10, "mean_delay_s": lefts[0]},
                            ref.LEFT_KEYS["left_south"]: {"vehicles": 10, "mean_delay_s": lefts[1]}}}


class ReportTest(unittest.TestCase):
    def rows(self, world, policy, gain_by_date):
        return {ac.situation_id(s): fake(s, world, 40.0 - gain_by_date.get(s["date"], 0.0), policy) for s in CFG["situations"]}

    def test_verdicts_cause_and_share(self):
        dates = LOCK["dates"]["eval"]
        sits = {wd: rp.situations_for(CFG, wd) for wd in ("P", "R")}
        m4 = self.rows("P", "M4", {})
        om4 = self.rows("P", "OM4", {d: g for d, g in zip(dates, (2.6, 2.4, 2.8, 2.2, 2.5, 2.7, 2.3))})
        tm4 = self.rows("P", "TM4", {d: 1.5 for d in dates})
        g_o = rp.compare(om4, m4, LOCK, sits["P"])
        self.assertAlmostEqual(g_o["gain_date_mean_s"], 2.5)
        self.assertEqual(rp.main_verdict(g_o, CFG["verdict"]), "room")
        g_t = rp.compare(tm4, m4, LOCK, sits["P"])
        self.assertAlmostEqual(g_t["gain_date_mean_s"] / g_o["gain_date_mean_s"], 0.6)
        r_m4, r_om4 = self.rows("R", "M4", {}), self.rows("R", "OM4", {d: 0.1 for d in dates})
        g_or = rp.compare(r_om4, r_m4, LOCK, sits["R"])
        cause = rp.paired_dates(g_o["per_date_gain_s"], g_or["per_date_gain_s"], LOCK["judge"]["t975"])
        self.assertAlmostEqual(cause["mean_s"], 2.4)
        self.assertGreater(cause["t95_s"][0], 0)
        small = rp.compare(self.rows("P", "OM4", {d: 0.5 for d in dates}), m4, LOCK, sits["P"])
        self.assertEqual(rp.main_verdict(small, CFG["verdict"]), "no_room")
        noisy = rp.compare(self.rows("P", "OM4", {d: g for d, g in zip(dates, (6.0, -2.0, 5.0, -1.0, 4.0, -0.5, 3.0))}),
                           m4, LOCK, sits["P"])
        self.assertGreaterEqual(noisy["gain_date_mean_s"], 2.0)
        self.assertEqual(rp.main_verdict(noisy, CFG["verdict"]), "unclear")      # 평균은 넘어도 하한 ≤ 0

    def test_world_mismatch_is_refused(self):
        sits = rp.situations_for(CFG, "P")
        with self.assertRaises(ValueError):
            rp.compare(self.rows("R", "OM4", {}), self.rows("P", "M4", {}), LOCK, sits)


if __name__ == "__main__":
    unittest.main()
