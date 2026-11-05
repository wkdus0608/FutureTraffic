"""[추가 탐색] 1단계 PLATOON1 검사 — SUMO 실행 없음(실제 204822 시간표·가짜 지점·가짜 실행 기록)."""
import bisect
import math
import sys
import tempfile
import unittest
from collections import Counter, deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import arena3_common as ac  # noqa: E402
import arena3_referee as ref  # noqa: E402
import explore_platoon0 as p0  # noqa: E402
import explore_platoon1 as pt  # noqa: E402
import explore_platoon1_learn as pl  # noqa: E402
import explore_weight as w  # noqa: E402

w.configure()
CFG = pt.config()
LOCK = ac.config()


def obs167(allowed=(0, 1), tm4=0, m4=None, extra_seed=0):
    """허용·모드·TM4가 맞는 가짜 관측 167칸(모드 여섯 개는 모두 m4 녹색)."""
    obs = np.zeros(pt.OBS, np.float32)
    for a in allowed:
        obs[60 + a] = 1.0
    m4 = tm4 if m4 is None else m4
    for mode in range(5):
        obs[84 + 4 * mode + m4] = 1.0
    obs[110 + m4] = 1.0
    obs[119 + tm4] = 1.0
    rng = np.random.default_rng(extra_seed)
    obs[127:167] = rng.random(40).astype(np.float32)
    return obs


class ConfigTest(unittest.TestCase):
    def test_situations_and_world(self):
        self.assertEqual((len(CFG["situations_data"]), len(CFG["situations_eval"]), len(CFG["situations_confirmation"])),
                         (64, 84, 84))
        p0cfg = p0.config()
        self.assertEqual([s["route_sha256"] for s in CFG["situations_eval"]], [s["route_sha256"]["P"] for s in p0cfg["situations"]])
        pi = w.read(pt.PI_R1_CONFIG)
        self.assertEqual([ac.situation_id(s) for s in CFG["situations_data"]], [ac.situation_id(s) for s in pi["situations_data"]])
        random_shas = {s["route_sha256"] for s in pi["situations_data"]}
        self.assertFalse(random_shas & {s["route_sha256"] for s in CFG["situations_data"]})
        self.assertFalse({s["date"] for s in CFG["situations_data"]} & set(LOCK["dates"]["eval"]))
        self.assertEqual({s["tag"] for s in CFG["situations_confirmation"]}, {"C1", "C2"})
        self.assertEqual(CFG["observation"]["size"], pt.OBS)
        self.assertEqual(CFG["observation"]["release_states"], list(pt.release_states()))

    def test_point_sampling_is_a_fixed_quarter(self):
        hits = [pt.point_sampled(CFG, "20220812_0730_R1T1", k) for k in range(4000)]
        self.assertAlmostEqual(sum(hits) / len(hits), 0.25, delta=0.03)
        self.assertEqual(hits, [pt.point_sampled(CFG, "20220812_0730_R1T1", k) for k in range(4000)])


class FutureTest(unittest.TestCase):
    def test_east_sampled_vehicles_move_onto_204822_greens(self):
        date = CFG["situations_data"][0]["date"]
        origin = pt.origin_of(CFG["situations_data"][0])
        fn, start, link_tt = p0.east_geometry()
        import focused_platoon as fp
        import focused_scenario as fs
        chain = fn._chain(fp.EAST_APPROACH, True, fs.UPSTREAM_LIMIT_M)
        self.assertEqual(chain[0], start)
        east = (*chain, fp.EAST_APPROACH, "477707013_01")           # 동쪽 시작 도로에서 출발하는 직진 경로
        hist = [(east, "car")] * 30 + [(("477707011_02", "x"), "car")] * 30
        import explore_rollout as er
        t0, horizon = 2400.0, 300
        base = er.sample_future(hist, t0, 12345, horizon)
        out, kinds = pt.retime_sampled(base, t0, horizon, date, origin, 12345)
        self.assertEqual(sorted(v[0] for v in out), sorted(v[0] for v in base))
        before = {v[0]: v for v in base}
        windows = p0._green_windows(date)
        states = set(pt.release_states())
        for vid, edges, vt, dep in out:
            self.assertTrue(t0 < dep <= t0 + horizon)
            if edges[0] != start:
                self.assertEqual(dep, before[vid][3])
            elif dep != before[vid][3]:
                u = math.floor(dep + origin - link_tt - 1e-9)
                self.assertTrue(any(u in set(windows[s]) or u - 1 in set(windows[s]) for s in states))
        self.assertGreater(kinds["east_platoon"], 0)
        self.assertEqual(sum(kinds.values()), sum(1 for v in base if v[1][0] == start))
        again, _ = pt.retime_sampled(base, t0, horizon, date, origin, 12345)
        self.assertEqual(out, again)

    def test_patch_restores_and_refuses_nesting(self):
        import explore_rollout as er
        original = er.sample_future
        with pt.platoon_futures("20220812", 0.0):
            self.assertTrue(getattr(er.sample_future, "platoon1", False))
            with self.assertRaises(RuntimeError):
                with pt.platoon_futures("20220812", 0.0):
                    pass
        self.assertIs(er.sample_future, original)


class ViewTest(unittest.TestCase):
    def test_forecast_timetable_and_cells(self):
        moves = pt.east_moves()
        index = {m: i for i, m in enumerate(moves)}
        arrivals = [[0.0] * 6 for _ in range(24)]
        arrivals[0][0], arrivals[7][1], arrivals[23][3] = 5.0, 20.0, 200.0
        f = pt.east_forecast(arrivals, index)
        self.assertEqual(len(f), 16)
        self.assertAlmostEqual(f[0], 0.5)
        self.assertAlmostEqual(f[4 + 1], 2.0)
        self.assertAlmostEqual(f[12 + 3], 10.0)                    # 0~10으로 자름
        main, side = pt.release_states()
        state_at = lambda t: main if (t % 100) < 50 else side  # noqa: E731
        table = pt.timetable_view(state_at, 1000.0, 20.0)
        self.assertEqual(len(table), 24)
        self.assertTrue(all(abs(a + b - 1.0) < 1e-9 for a, b in zip(table[:12], table[12:])))
        cells = pt.extra_cells(1, [5000.0, 3000.0, None, 4500.0], f, table)
        self.assertEqual(cells[:4], [0.0, 1.0, 0.0, 0.0])
        self.assertEqual(cells[4:8], [2.0, 0.0, 0.0, 1.5])
        with self.assertRaises(ValueError):
            pt.extra_cells(2, [5000.0, 3000.0, None, 4500.0], f, table)   # TM4 녹색 비용 없음
        self.assertEqual(pt.observation167(np.zeros(119), cells).shape, (167,))

    def test_state_export_import_roundtrip(self):
        ctl = SimpleNamespace(_seen={"a", "b"}, _entries=deque([(5.0, "e", 1, None)]), _first_time=0.0,
                              _east_log=deque([(5.0, 1, "G")]))
        core = pt.TM4Core()
        core.ctl = ctl
        state = core.export_state()
        other = pt.TM4Core()
        other.ctl = SimpleNamespace(_seen=set(), _entries=deque(), _first_time=None, _east_log=deque())
        other.import_state(state)
        self.assertEqual(other.ctl._seen, {"a", "b"})
        self.assertEqual(list(other.ctl._entries), [(5.0, "e", 1, None)])
        self.assertEqual(list(other.ctl._east_log), [(5.0, 1, "G")])
        self.assertEqual(other.ctl._first_time, 0.0)


class NetTest(unittest.TestCase):
    def test_untrained_is_tm4_and_mask(self):
        net = pt.PlatoonNet().eval()
        for tm4, allowed in ((0, (0, 1)), (2, (1, 2, 3)), (3, (3,))):
            self.assertEqual(pt.greedy_green(net, obs167(allowed, tm4)), tm4)
        with self.assertRaises(ValueError):
            pt.greedy_green(net, obs167((0, 1), tm4=2))           # TM4 녹색이 허용 밖
        x = torch.from_numpy(np.stack([obs167((0, 1, 2), 1), obs167((0, 3), 3)]))
        out = net(x)
        self.assertTrue(torch.all(out == 0))


def fake_point(state, allowed, tm4, costs, *, ok=True, m4=None):
    obs = obs167(allowed, tm4, m4)
    rollouts = {str(a): [{"cost": c, "future_seed": j, "collisions_unexplained": 0, "collisions_at_204820": 0}
                         for j, c in enumerate(costs[a])] for a in allowed}
    return {"state": state, "allowed": list(allowed), "greens": pl.mode_greens(obs).tolist(), "m4": int(m4 if m4 is not None else tm4),
            "tm4": tm4, "executed": tm4, "observation": obs.tolist(), "rollouts": rollouts,
            "restore_check": {"ok": ok}}


class LearnTest(unittest.TestCase):
    def test_point_to_row_gains_relative_to_tm4(self):
        costs = {0: [1000.0 + 10 * j for j in range(8)], 1: [400.0 + 10 * j for j in range(8)]}
        row, why = pl.point_to_row(fake_point("s_p000", (0, 1), 0, costs), "20220812", "s", CFG)
        self.assertIsNone(why)
        self.assertAlmostEqual(float(row["gains"][1]), 0.6)        # (1000 − 400)/1000
        self.assertEqual(float(row["gains"][0]), 0.0)
        self.assertEqual(row["anchor"], 0)
        _, why = pl.point_to_row(fake_point("s_p001", (0, 1), 0, costs, ok=False), "20220812", "s", CFG)
        self.assertEqual(why, "restore_mismatch")
        bad = fake_point("s_p002", (0, 1), 0, costs)
        for j in range(5):
            bad["rollouts"]["1"][j] = None
        _, why = pl.point_to_row(bad, "20220812", "s", CFG)
        self.assertEqual(why, "insufficient_valid_futures")
        wrong = fake_point("s_p003", (0, 1), 0, costs)
        wrong["executed"] = 1
        with self.assertRaises(ValueError):
            pl.point_to_row(wrong, "20220812", "s", CFG)

    def test_student_learns_when_to_leave_tm4(self):
        rows = []
        rng = np.random.default_rng(0)
        for k in range(240):
            better = bool(k % 2)
            obs = obs167((0, 1), 0, extra_seed=k)
            obs[130] = 1.0 if better else 0.0                       # 시간표 칸 하나가 답을 알려 준다
            gain = 0.8 if better else -0.8
            rows.append({"observations": obs, "allowed": pt.allowed_mask(obs), "anchor": 0, "m4": 0,
                         "gains": np.asarray([0.0, gain, 0, 0], np.float32), "gain_se": np.full(4, 0.05, np.float32),
                         "fit_gains": np.asarray([0.0, gain, 0, 0], np.float32), "fit_gain_se": np.full(4, 0.05, np.float32),
                         "score_gains": np.asarray([0.0, gain + rng.normal(0, 0.05), 0, 0], np.float32),
                         "dates": ("20220812", "20220814")[k % 2 == 0 and k % 4 == 0], "state_ids": f"x{k}",
                         "situations": f"s{k % 8}", "valid_futures": 8})
        dataset = {key: np.asarray([r[key] for r in rows]) for key in pl.ARRAY_KEYS}
        dataset["fingerprint"] = "fake"
        net, stats = pl.train_student(dataset, CFG, 42, epochs=60)
        greens = pl.predict(net, dataset["observations"])
        want = np.asarray([1 if k % 2 else 0 for k in range(240)])
        self.assertGreater(float((greens == want).mean()), 0.95)
        self.assertLess(stats["final_full_loss"], stats["initial_full_loss"])
        s = pl.score(dataset, greens, CFG)
        self.assertGreater(s["mean"], 0)
        self.assertEqual(pl.teacher_greens(dataset).tolist(), want.tolist())


def fake_run(s, delay, policy, n=100, lefts=(100.0, 80.0)):
    sid = ac.situation_id(s)
    return {"situation": sid, "policy": policy, "vehicles": n, "route_sha256": s["route_sha256"], "cohort_sha256": f"P{sid}",
            "actual_total_delay": delay * n, "mean_delay_s": delay, "long_waits": 0, "missing": 0, "unfinished": 0,
            "changed_by_rules": 0, "mode_action_errors": 0, "p0_mismatch": 0, "cost_check": {"passed": True},
            "collisions_at_204820": 0, "collision_details": {"collisions_unexplained": []},
            "by_movement": {ref.LEFT_KEYS["left_east"]: {"vehicles": 10, "mean_delay_s": lefts[0]},
                            ref.LEFT_KEYS["left_south"]: {"vehicles": 10, "mean_delay_s": lefts[1]}}}


class JudgeTest(unittest.TestCase):
    def rows(self, policy, gain_by_date):
        return [fake_run(s, 40.0 - gain_by_date.get(s["date"], 0.0), policy) for s in CFG["situations_eval"]]

    def test_main_verdict(self):
        dates = LOCK["dates"]["eval"]
        sits = CFG["situations_eval"]
        m4 = self.rows("M4", {})
        tm4 = ref.judge([self.rows("TM4", {d: 3.8 for d in dates})], m4, None, LOCK, sits)
        good = ref.judge([self.rows(f"PT1_m{i}", {d: g for d, g in zip(dates, (4.5, 4.4, 4.6, 4.2, 4.5, 4.7, 4.3))})
                          for i in range(3)], m4, None, LOCK, sits)
        learning = pl.paired_difference(good, tm4, LOCK["judge"]["t975"])
        self.assertAlmostEqual(learning["mean_s"], 0.657142857, places=6)
        v = pl.main_verdict(good, learning, CFG["rules"])
        self.assertTrue(v["passed"])
        self.assertEqual(v["learning_reading"], "TM4보다 짧다(학습의 몫)")
        tiny = ref.judge([self.rows("PT1", {d: 3.9 for d in dates})], m4, None, LOCK, sits)
        v = pl.main_verdict(tiny, pl.paired_difference(tiny, tm4, LOCK["judge"]["t975"]), CFG["rules"])
        self.assertFalse(v["passed"])                                # 하한은 양수지만 평균 0.1초 < 0.3초
        worse = ref.judge([self.rows("PT1", {d: g for d, g in zip(dates, (3.0, 3.2, 2.9, 3.1, 3.0, 3.3, 2.8))})],
                          m4, None, LOCK, sits)
        v = pl.main_verdict(worse, pl.paired_difference(worse, tm4, LOCK["judge"]["t975"]), CFG["rules"])
        self.assertEqual((v["passed"], v["learning_reading"]), (False, "TM4보다 길다"))

    def test_jobs_and_world_mismatch(self):
        jobs = ref.jobs_for(Path("/o"), Path("/w"), "f", "PT1_m0", CFG["situations_eval"], model_dir="/m", model_identity="x",
                            kind="student")
        self.assertEqual(len(jobs), 84)
        self.assertTrue(all(j["situation"]["route_sha256"] == s["route_sha256"] for j, s in zip(jobs, CFG["situations_eval"])))
        m4 = self.rows("M4", {})
        r_world = [dict(r, route_sha256="R") for r in self.rows("PT1", {})]
        with self.assertRaises(ValueError):
            ref.judge([r_world], m4, None, LOCK, CFG["situations_eval"])


if __name__ == "__main__":
    unittest.main()
