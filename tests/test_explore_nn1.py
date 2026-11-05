"""[추가 탐색] NN1 검사 — SUMO 실행 없음."""
import math
import statistics
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import arena3_common as ac  # noqa: E402
import explore_nn1 as nn1  # noqa: E402
import explore_platoon1 as pt  # noqa: E402
import explore_platoon1_learn as pl  # noqa: E402
import explore_platoon1c as pc  # noqa: E402
import explore_pred as ep  # noqa: E402
import explore_weight as w  # noqa: E402
import explore_weight_core as wc  # noqa: E402

w.configure()
CFG = nn1.config()
ST = CFG["training"]


class FakeScores:
    """녹색 점수를 그대로 돌려주는 신경망."""

    def __init__(self, values):
        self.values = values

    def __call__(self, x, base=None):
        return torch.tensor([self.values], dtype=torch.float32)


def small_green(size=6, seed=1):
    torch.manual_seed(seed)
    return nn1.GreenNet(size, np.zeros(size), np.ones(size), (8, 8))


class ConfigTest(unittest.TestCase):
    def test_dates_are_split_before_models(self):
        lock = ac.config()
        self.assertEqual(set(CFG["dates"]["train"]) | set(CFG["dates"]["validation"]), set(lock["dates"]["train"]))
        self.assertFalse(set(CFG["dates"]["train"]) & set(CFG["dates"]["validation"]))
        self.assertFalse(set(CFG["dates"]["train"]) & set(CFG["dates"]["eval"]))
        self.assertEqual(len(CFG["dates"]["train"]), 6)
        self.assertEqual(CFG["dates"]["eval"], lock["dates"]["eval"])

    def test_candidates_match_the_plan(self):
        self.assertEqual(set(CFG["candidates"]["P"]), {"N1", "N2", "N4", "N5"})
        self.assertEqual(set(CFG["candidates"]["R"]), {"Rn", "RnS", "Rn2"})
        self.assertEqual(sum(len(v) for v in CFG["candidates"].values()) * 3 * 84, 1764)
        self.assertEqual(CFG["reference_planner"], {"P": "TM4", "R": "M4"})

    def test_input_columns_skip_planner_cells(self):
        self.assertEqual(len(nn1.COLUMNS["P"]), 120)
        self.assertEqual(len(nn1.COLUMNS["R"]), 80)
        planner = set(range(80, 127))
        self.assertFalse(planner & set(nn1.COLUMNS["P"].tolist()))
        self.assertFalse(planner & set(nn1.COLUMNS["R"].tolist()))
        self.assertEqual((pt.FORECAST.start, pt.TIMETABLE.stop), (127, 167))
        self.assertNotIn(109, nn1.COLUMNS["P"].tolist())

    def test_same_settings_for_all_candidates(self):
        self.assertEqual(ST["epochs_imitation"], 30)
        self.assertEqual(ST["epochs_student"], 30)
        self.assertEqual(ST["network"], [64, 64])
        self.assertEqual(ST["seeds"], [42, 43, 44])

    def test_teachers_are_pinned(self):
        pccfg = pc.config()
        self.assertEqual(CFG["teachers"]["P"]["model_sha256"], pccfg["students"]["model_sha256"][0])
        self.assertEqual(CFG["teachers"]["P"]["tau"], pccfg["candidates"]["t20"]["tau"])
        self.assertEqual(CFG["teachers"]["R"]["model_sha256"], w.read(ROOT / "config/arena3_pi_r1.json")["base_policy"]["model_sha256"])

    def test_check_situations_are_in_evaluation(self):
        for world in ("P", "R"):
            ids = {ac.situation_id(s) for s in nn1.eval_situations(world)}
            self.assertEqual(len(ids), 84)
            self.assertTrue(set(CFG["check_situations"]) <= ids)
        self.assertEqual(len(CFG["check_situations"]), 12)


class SituationTest(unittest.TestCase):
    def test_rounds_dates_and_bands(self):
        rows = CFG["situations_aug"]
        self.assertEqual(len(rows), 4 * 6 * 8)
        for r in range(4):
            mine = [s for s in rows if s["round"] == r]
            self.assertEqual(len(mine), 48)
            for date in CFG["dates"]["train"]:
                bands = Counter(s["band"] for s in mine if s["date"] == date)
                self.assertEqual(bands, Counter({"am_peak": 2, "pm_peak": 2, "morning": 1, "noon": 1, "afternoon": 1, "evening": 1}))
        for s in rows:
            self.assertIn(s["period"], nn1.AUG_BANDS[s["band"]][1])
            self.assertEqual(s["sumo_seed"], ac.sumo_seed(s["tag"], s["date"], s["period"]))
        self.assertEqual(len({ac.situation_id(s) for s in rows}), len(rows))

    def test_config_is_what_the_rule_makes(self):
        made = nn1.aug_situations(ST["rounds"])
        self.assertEqual([ac.situation_id(s) for s in made], [ac.situation_id(s) for s in CFG["situations_aug"]])
        self.assertEqual([ac.situation_id(s) for s in nn1.validation_situations()], [ac.situation_id(s) for s in CFG["situations_validation"]])

    def test_validation_uses_validation_dates_only(self):
        rows = CFG["situations_validation"]
        self.assertEqual(len(rows), 24)
        self.assertEqual({s["date"] for s in rows}, set(CFG["dates"]["validation"]))
        self.assertEqual({s["period"] for s in rows}, set(nn1.EVAL_PERIODS))

    def test_named_situations_pick_the_world_route(self):
        p, r = nn1.named_situations(CFG, "situations_validation", "P"), nn1.named_situations(CFG, "situations_validation", "R")
        self.assertEqual(p[0]["route_sha256"], CFG["situations_validation"][0]["route_sha256"]["P"])
        self.assertNotEqual(p[0]["route_sha256"], r[0]["route_sha256"])
        self.assertEqual(len(nn1.named_situations(CFG, "situations_aug", "P", 2)), 48)

    def test_routes_are_new_and_distinct(self):
        shas = [s["route_sha256"][x] for s in [*CFG["situations_aug"], *CFG["situations_validation"]] for x in ("P", "R")]
        self.assertEqual(len(shas), len(set(shas)))
        used = {s["route_sha256"] for s in nn1.eval_situations("P")} | {s["route_sha256"] for s in nn1.eval_situations("R")}
        self.assertFalse(set(shas) & used)


class NetworkTest(unittest.TestCase):
    def test_scale_handles_constant_cells(self):
        x = np.array([[1.0, 5.0], [3.0, 5.0]])
        mean, std = nn1.scale_of(x)
        self.assertEqual(mean.tolist(), [2.0, 5.0])
        self.assertEqual(std.tolist(), [1.0, 1.0])

    def test_green_net_applies_stored_scale(self):
        net = nn1.GreenNet(2, [10.0, 20.0], [2.0, 4.0], (4, 4))
        a = net(torch.tensor([[12.0, 24.0]]))
        b = net.net(torch.tensor([[1.0, 1.0]]))
        self.assertTrue(torch.equal(a, b))

    def test_scale_is_saved_with_the_model(self):
        net = nn1.GreenNet(80, np.full(80, 3.0), np.full(80, 2.0), (64, 64))
        meta = {"kind": nn1.KIND_GREEN, "world": "R", "candidate": "Rn", "seed": 42, "input_size": 80, "network": [64, 64], "dataset_fingerprint": "d"}
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d) / "m"
            sha = nn1.save_model(folder, net, meta)
            self.assertEqual(nn1.model_identity(folder), sha)
            loaded, manifest = nn1.load_model(folder)
        self.assertEqual(manifest["seed"], 42)
        x = torch.ones(1, 80)
        self.assertTrue(torch.equal(net(x), loaded(x)))
        self.assertEqual(float(loaded.mean[0]), 3.0)

    def test_model_file_change_is_caught(self):
        net = nn1.GreenNet(80, hidden=(64, 64))
        meta = {"kind": nn1.KIND_GREEN, "world": "R", "candidate": "Rn", "seed": 42, "input_size": 80, "network": [64, 64], "dataset_fingerprint": "d"}
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d) / "m"
            nn1.save_model(folder, net, meta)
            with (folder / "model.pt").open("ab") as f:
                f.write(b"x")
            with self.assertRaises(ValueError):
                nn1.model_identity(folder)

    def test_wrong_world_size_is_caught(self):
        net = nn1.GreenNet(80, hidden=(64, 64))
        meta = {"kind": nn1.KIND_GREEN, "world": "P", "candidate": "N1", "seed": 42, "input_size": 80, "network": [64, 64], "dataset_fingerprint": "d"}
        with tempfile.TemporaryDirectory() as d:
            nn1.save_model(Path(d) / "m", net, meta)
            with self.assertRaises(ValueError):
                nn1.load_model(Path(d) / "m")

    def test_gain_net_is_zero_before_training_and_relative_to_base(self):
        net = nn1.GainNet(5, hidden=(8, 8))
        rel = net(torch.zeros(2, 5), torch.tensor([1, 3]))
        self.assertTrue(torch.equal(rel, torch.zeros(2, 4)))
        torch.nn.init.ones_(net.advantage.weight)
        rel = net(torch.ones(2, 5), torch.tensor([1, 3]))
        self.assertEqual(float(rel[0, 1]), 0.0)
        self.assertEqual(float(rel[1, 3]), 0.0)


class ChoiceTest(unittest.TestCase):
    def test_green_choice_stays_in_allowed(self):
        net = FakeScores([9.0, 1.0, 5.0, 2.0])
        self.assertEqual(nn1.green_choice(net, np.zeros(3), [1, 2]), 2)
        self.assertEqual(nn1.green_choice(net, np.zeros(3), [0, 1]), 0)

    def test_student_changes_only_above_threshold(self):
        self.assertEqual(nn1.student_choice(FakeScores([0.0, 1.5, 0.5, 9.0]), np.zeros(3), 0, [0, 1, 2], 1.0), (1, 1.5))
        self.assertEqual(nn1.student_choice(FakeScores([0.0, 1.5, 0.5, 9.0]), np.zeros(3), 0, [0, 1, 2], 2.0), (0, 1.5))

    def test_student_never_picks_base_or_disallowed(self):
        green, gain = nn1.student_choice(FakeScores([0.0, -1.0, -2.0, 9.0]), np.zeros(3), 0, [0, 1, 2], -5.0)
        self.assertEqual((green, gain), (1, -1.0))
        with self.assertRaises(ValueError):
            nn1.student_choice(FakeScores([0.0] * 4), np.zeros(3), 3, [0, 1], 1.0)

    def test_nan_scores_stop_the_run(self):
        with self.assertRaises(ValueError):
            nn1.green_choice(FakeScores([float("nan"), 0.0, 0.0, 0.0]), np.zeros(3), [0, 1])


class FakeEnv:
    sim_step = 100.0


class PolicyTest(unittest.TestCase):
    def policy(self, student=None, tau=None, keep=False):
        return nn1.NNPolicy("R", small_green(80), "NN", student, tau, keep_inputs=keep)

    def test_forced_step_does_not_call_the_network(self):
        p = self.policy()
        p.net = None
        self.assertEqual(p.act(FakeEnv(), np.zeros(80, np.float32), [2]), 2)
        self.assertEqual(p.stats["forced"], 1)

    def test_free_step_counts_and_keeps_inputs(self):
        p = self.policy(keep=True)
        a = p.act(FakeEnv(), np.zeros(80, np.float32), [0, 1])
        self.assertIn(a, (0, 1))
        self.assertEqual(p.stats["planned"], 1)
        self.assertEqual(len(p.rows), 1)
        self.assertEqual(p.rows[0][1].shape, (80,))

    def test_student_requires_threshold(self):
        with self.assertRaises(ValueError):
            nn1.NNPolicy("R", small_green(80), "NN", nn1.GainNet(80, hidden=(8, 8)))

    def test_untrained_student_never_changes(self):
        p = self.policy(nn1.GainNet(80, hidden=(8, 8)), 0.5)
        p.act(FakeEnv(), np.zeros(80, np.float32), [0, 1, 2])
        self.assertEqual(p.stats["student_changed"], 0)

    def test_z5_cells_are_counted(self):
        p = nn1.NNPolicy("R", nn1.GreenNet(80, np.zeros(80), np.ones(80), (8, 8)), "NN")
        x = np.zeros(80, np.float32)
        x[3] = 6.0
        p.act(FakeEnv(), x, [0, 1])
        self.assertEqual(p.stats["z5_cells"], 1)

    def test_world_is_checked(self):
        with self.assertRaises(ValueError):
            nn1.NNPolicy("X", small_green(80), "NN")


class PlannerGuardTest(unittest.TestCase):
    def test_planner_calls_stop_the_run(self):
        before = ep.plan, wc.weighted_plan, ac.make_planner
        with nn1.planner_free(object()):
            with self.assertRaises(RuntimeError):
                ep.plan()
            with self.assertRaises(RuntimeError):
                wc.weighted_plan()
        self.assertEqual((ep.plan, wc.weighted_plan, ac.make_planner), before)

    def test_guard_restores_after_errors(self):
        before = ep.plan, wc.weighted_plan, ac.make_planner
        with self.assertRaises(KeyError):
            with nn1.planner_free(object()):
                raise KeyError("x")
        self.assertEqual((ep.plan, wc.weighted_plan, ac.make_planner), before)

    def test_p0_tail_allows_only_after_control(self):
        real = ep.plan
        ep.plan = lambda *a, **k: "planned"
        try:
            with nn1.planner_free(object(), "P0") as guard:
                guard.update(env=SimpleNamespace(sim_step=99.0), control_end=100.0)
                with self.assertRaises(RuntimeError):
                    ep.plan()
                self.assertEqual(guard["tail_calls"], 0)
                guard["env"].sim_step = 100.0
                self.assertEqual(ep.plan(), "planned")
                self.assertEqual(guard["tail_calls"], 1)
                with self.assertRaises(RuntimeError):
                    wc.weighted_plan()
        finally:
            ep.plan = real

    def test_unknown_tail_is_refused(self):
        with self.assertRaises(ValueError):
            with nn1.planner_free(object(), "M4"):
                pass


def point(costs_by_green):
    rollouts = {str(g): [None if c is None else {"cost": c} for c in row] for g, row in costs_by_green.items()}
    return {"rollouts": rollouts}


class GainsTest(unittest.TestCase):
    COSTS = {0: [1000.0, 2000.0, 3000.0, 4000.0], 2: [1500.0, 1000.0, 5000.0, 2000.0]}

    def matrix(self):
        row = {"fit_future_indices": [0, 1], "score_future_indices": [2, 3]}
        return nn1.cost_matrix(point(self.COSTS), row, 4)

    def test_cost_matrix_marks_unused_as_nan(self):
        m = self.matrix()
        self.assertEqual(m.shape, (4, 4))
        self.assertEqual(m[0].tolist(), [1.0, 2.0, 3.0, 4.0])
        self.assertTrue(np.isnan(m[1]).all())
        row = {"fit_future_indices": [0], "score_future_indices": [2]}
        m = nn1.cost_matrix(point(self.COSTS), row, 4)
        self.assertTrue(np.isnan(m[0, 1]))

    def test_gains_match_the_stored_formula(self):
        m = self.matrix()[None]
        allowed = np.array([[True, False, True, False]])
        gains, se = nn1.gains_against(m, np.array([0]), allowed)
        got, got_se = pl._gain_rows({0: [{"cost": c} for c in self.COSTS[0]], 2: [{"cost": c} for c in self.COSTS[2]]}, [0, 1, 2, 3], [0, 2], 0)
        np.testing.assert_allclose(gains[0], got, rtol=1e-6)
        np.testing.assert_allclose(se[0], got_se, rtol=1e-6)
        self.assertEqual(gains[0, 0], 0.0)
        self.assertEqual(gains[0, 1], 0.0)

    def test_changing_base_recomputes_standard_error(self):
        """기존 rebased의 문제: 기준을 바꿔도 표준오차가 옛 기준 그대로였다."""
        m = np.array([[[1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 2.0], [np.nan] * 4, [1.0, 2.0, 3.0, 5.0]]])
        allowed = np.array([[True, True, False, True]])
        _, se0 = nn1.gains_against(m, np.array([0]), allowed)
        g3, se3 = nn1.gains_against(m, np.array([3]), allowed)
        self.assertAlmostEqual(g3[0, 0], 0.25)
        self.assertAlmostEqual(se3[0, 0], statistics.stdev([0.0, 0.0, 0.0, 1.0]) / 2.0, places=6)
        self.assertNotAlmostEqual(se3[0, 1], se0[0, 1])
        self.assertEqual(se3[0, 3], 0.0)

    def test_half_costs_split_only_valid_futures(self):
        m = np.array([[[1.0, np.nan, 3.0, 4.0, 5.0]] * 4])
        front, back = nn1.half_costs(m)
        self.assertEqual(np.flatnonzero(np.isfinite(front[0][0])).tolist(), [0, 2])
        self.assertEqual(np.flatnonzero(np.isfinite(back[0][0])).tolist(), [3, 4])

    def test_margin_weights_clip_and_use_two_lowest(self):
        costs = np.array([[1.0, 3.0, np.nan, 9.0], [0.0, 100.0, np.nan, np.nan], [5.0, 5.0, 7.0, np.nan], [2.0, 2.5, np.nan, np.nan]])
        weights, median = nn1.margin_weights(costs, [0.25, 4.0])
        self.assertEqual(median, statistics.median([2.0, 100.0, 0.0, 0.5]))
        np.testing.assert_allclose(weights, [2.0 / median, 4.0, 0.25, 0.5 / median])

    def test_margin_needs_two_greens(self):
        with self.assertRaises(ValueError):
            nn1.margin_weights(np.array([[1.0, np.nan, np.nan, np.nan]]), [0.25, 4.0])


class TeacherLabelTest(unittest.TestCase):
    def test_batch_labels_match_the_single_rule(self):
        torch.manual_seed(3)
        net = pt.PlatoonNet((16, 16))
        torch.nn.init.normal_(net.advantage.weight, std=3.0)
        rng = np.random.default_rng(0)
        rows = []
        for _ in range(40):
            obs = rng.random(pt.OBS).astype(np.float32)
            obs[pt.ALLOWED] = 0.0
            allowed = rng.choice(4, size=rng.integers(2, 5), replace=False)
            obs[pt.ALLOWED.start + allowed] = 1.0
            obs[pt.TM4_ONEHOT] = 0.0
            obs[pt.TM4_ONEHOT.start + allowed[0]] = 1.0
            rows.append(obs)
        rows = np.stack(rows)
        batch = nn1.t20_labels(net, rows, 0.5)
        single = [pc.sure_green(net, o, 0.5)[0] for o in rows]
        self.assertEqual(batch.tolist(), single)
        self.assertTrue((batch != pt.anchor_green(rows)).any())
        self.assertTrue((batch == pt.anchor_green(rows)).any())


class TrainingTest(unittest.TestCase):
    def data(self, n=400):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(n, 6)).astype(np.float32) * np.array([1, 10, 100, 1, 1, 1], np.float32)
        y = (x[:, 0] > 0).astype(np.int64)
        allowed = np.zeros((n, 4), bool)
        allowed[:, :2] = True
        return x, y, allowed

    def test_imitation_learns_and_stays_in_allowed(self):
        x, y, allowed = self.data()
        net = nn1.fit_green(x, y, allowed, 1, 30, {**ST, "network": [16, 16]})
        pred = nn1.greens_of(net, x, allowed)
        self.assertGreater((pred == y).mean(), 0.95)
        self.assertTrue(set(pred.tolist()) <= {0, 1})

    def test_imitation_is_repeatable(self):
        x, y, allowed = self.data(100)
        a = nn1.fit_green(x, y, allowed, 5, 3, {**ST, "network": [8, 8]})
        b = nn1.fit_green(x, y, allowed, 5, 3, {**ST, "network": [8, 8]})
        for p, q in zip(a.parameters(), b.parameters()):
            self.assertTrue(torch.equal(p, q))

    def test_weights_change_the_loss(self):
        x, y, allowed = self.data(100)
        a = nn1.fit_green(x, y, allowed, 5, 3, {**ST, "network": [8, 8]})
        b = nn1.fit_green(x, y, allowed, 5, 3, {**ST, "network": [8, 8]}, weights=np.where(x[:, 1] > 0, 4.0, 0.25))
        self.assertFalse(all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters())))

    def test_heldout_greens_do_not_see_their_date(self):
        x, y, allowed = self.data(120)
        dates = np.array(["a", "b", "c"] * 40)
        seen = []
        real = nn1.fit_green

        def spy(x_, y_, a_, seed, epochs, st, **kw):
            seen.append(len(x_))
            return real(x_, y_, a_, seed, epochs, st, **kw)

        nn1.fit_green = spy
        try:
            out = nn1.heldout_greens(x, y, allowed, dates, 1, 1, {**ST, "network": [8, 8]})
        finally:
            nn1.fit_green = real
        self.assertEqual(seen, [80, 80, 80])
        self.assertEqual(len(out), 120)

    def test_gain_loss_ignores_base_and_disallowed(self):
        net = nn1.GainNet(3, hidden=(4, 4))
        x, base = torch.zeros(1, 3), torch.tensor([1])
        allowed = torch.tensor([[True, True, False, False]])
        se = torch.zeros(1, 4)
        a = nn1.gain_loss(net, x, base, torch.tensor([[2.0, 0.0, 0.0, 0.0]]), se, allowed, ST)
        b = nn1.gain_loss(net, x, base, torch.tensor([[2.0, 99.0, 99.0, 99.0]]), se, allowed, ST)
        self.assertEqual(float(a), float(b))

    def test_student_threshold_keeps_the_share(self):
        x, y, allowed = self.data(300)
        costs = np.full((300, 4, 4), np.nan)
        rng = np.random.default_rng(2)
        costs[:, :2, :] = rng.normal(size=(300, 2, 4))
        costs[:, 1, :] -= (x[:, 0:1] > 0) * 2.0
        dates = np.array(["a", "b", "c"] * 100)
        net, info = nn1.build_student(x, y, costs, allowed, dates, 1, {**ST, "network": [16, 16], "epochs_imitation": 3, "epochs_student": 5, "student_change_share": 0.10})
        self.assertAlmostEqual(info["change_share_heldout"], 0.10, places=1)
        self.assertTrue(math.isfinite(info["tau"]))


class ReadingTest(unittest.TestCase):
    def table(self, **delays):
        fixed = 47.27
        return {k: {"mean_delay_s": v, "reduction_vs_fixc_pct": 100 * (fixed - v) / fixed} for k, v in delays.items()}

    def test_shortest_model_starts(self):
        t = self.table(N1=36.0, N2=35.0, N4=35.5, N5=34.0)
        r = nn1.world_reading(t, {"mean_delay_s": 34.7}, "P", CFG)
        self.assertEqual(r["start_model"], "N5")
        self.assertEqual(r["action"], "go")
        self.assertTrue(r["target_met"])

    def test_tie_goes_to_fewer_parts(self):
        t = self.table(N1=35.2, N2=35.0, N4=35.1, N5=35.0)
        r = nn1.world_reading(t, {"mean_delay_s": 34.7}, "P", CFG)
        self.assertEqual(r["start_model"], "N1")
        self.assertIn(r["shortest"], ("N2", "N5"))

    def test_behind_planner_needs_more_labels(self):
        t = self.table(N1=35.3, N2=36.0, N4=36.0, N5=36.0)
        r = nn1.world_reading(t, {"mean_delay_s": 34.7}, "P", CFG)
        self.assertAlmostEqual(r["behind_planner_s"], 0.6)
        self.assertEqual(r["action"], "more_labels")
        self.assertFalse(r["target_met"])

    def test_below_22_percent_stops(self):
        t = self.table(N1=37.0, N2=37.5, N4=37.5, N5=37.5)
        self.assertEqual(nn1.world_reading(t, {"mean_delay_s": 34.7}, "P", CFG)["action"], "stop")

    def test_round_selection_prefers_later_within_tolerance(self):
        self.assertEqual(nn1.select_round({0: 36.0, 1: 35.0, 2: 35.15, 3: 35.5}, 0.2), 2)
        self.assertEqual(nn1.select_round({0: 35.0, 1: 35.5, 2: 35.5, 3: 35.5}, 0.2), 0)
        self.assertEqual(nn1.select_round({0: 35.0, 1: 35.1, 2: 35.2, 3: 35.2}, 0.2), 3)


class DeclarationTest(unittest.TestCase):
    def saved(self, folder, kind, world, seed, dataset="d", **more):
        cls = nn1.GreenNet if kind == nn1.KIND_GREEN else nn1.GainNet
        size = len(nn1.COLUMNS[world])
        return nn1.save_model(Path(folder), cls(size, hidden=(64, 64)), {"kind": kind, "world": world, "candidate": "x", "seed": seed, "input_size": size,
                                                                         "network": [64, 64], "dataset_fingerprint": dataset, **more})

    def test_student_candidate_binds_base_and_threshold(self):
        with tempfile.TemporaryDirectory() as d:
            for seed in ST["seeds"]:
                self.saved(Path(d) / "Rn" / f"seed_{seed}", nn1.KIND_GREEN, "R", seed)
                self.saved(Path(d) / "RnS" / f"seed_{seed}", nn1.KIND_GAIN, "R", seed, tau=0.5)
            decl = nn1.declaration("R", "RnS", Path(d), CFG)
            self.assertEqual(len(decl["models"]), 3)
            self.assertTrue(decl["models"][0]["identity"].endswith("|tau=0.500000"))
            self.assertIn("/Rn/seed_42", decl["models"][0]["base_dir"])
            self.assertIn("/RnS/seed_42", decl["models"][0]["student_dir"])
            jobs = nn1.eval_jobs("R", "RnS", decl, Path(d) / "out", Path(d) / "work", "fp", nn1.eval_situations("R")[:15], CFG)
            self.assertEqual(len(jobs), 45)
            self.assertEqual({j["mode"] for j in jobs}, {"free"})
            self.assertEqual({j["tail"] for j in jobs}, {"NN"})
            self.assertTrue(any(j["save_inputs"] for j in jobs))
            self.assertFalse(any(j["save_inputs"] for j in jobs if j["policy"] != "RnS_m0"))

    def test_aug_candidate_reads_the_round_folder(self):
        with tempfile.TemporaryDirectory() as d:
            for seed in ST["seeds"]:
                self.saved(Path(d) / f"seed_{seed}", nn1.KIND_GREEN, "P", seed)
            decl = nn1.declaration("P", "N5", Path(d), CFG)
            self.assertEqual(decl["kind"], "aug")
            self.assertIsNone(decl["models"][2]["student_dir"])

    def test_mismatched_student_data_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            for seed in ST["seeds"]:
                self.saved(Path(d) / "Rn" / f"seed_{seed}", nn1.KIND_GREEN, "R", seed, dataset="a")
                self.saved(Path(d) / "RnS" / f"seed_{seed}", nn1.KIND_GAIN, "R", seed, dataset="b", tau=0.5)
            with self.assertRaises(ValueError):
                nn1.declaration("R", "RnS", Path(d), CFG)

    def test_unknown_candidate_is_refused(self):
        with self.assertRaises(ValueError):
            nn1.declaration("R", "N1", Path("."), CFG)


class NotebookTest(unittest.TestCase):
    def test_parts_cover_all_candidates(self):
        import explore_nn1_package as pkg
        for world, spec in pkg.WORLD.items():
            names = spec["base"].split(",") + [spec["aug"]]
            self.assertEqual(set(names), set(CFG["candidates"][world]))
            self.assertEqual(CFG["candidates"][world][spec["aug"]]["kind"], "aug")
            self.assertTrue(all(CFG["candidates"][world][c]["kind"] != "aug" for c in spec["base"].split(",")))

    def test_notebooks_compile_and_point_to_their_world(self):
        import explore_nn1_package as pkg
        for name in pkg.NOTEBOOKS:
            nb = pkg.notebook(name)
            text = "".join("".join(c["source"]) for c in nb["cells"])
            for i, c in enumerate(nb["cells"]):
                if c["cell_type"] == "code":
                    compile("".join(c["source"]), f"{name}[{i}]", "exec")
            self.assertIn("NN1_V1", text)
            self.assertEqual(nb["metadata"]["colab"]["machine_shape"], "hm")
            if name != "judge":
                world, kind = name.split("_")
                self.assertIn(f'WORLD, PART = "{world}", "{kind}"', text)
                self.assertIn("check-paths" if kind == "base" else "check-logs", text)
                self.assertIn("evaluate" if kind == "base" else "augment", text)


if __name__ == "__main__":
    unittest.main()
