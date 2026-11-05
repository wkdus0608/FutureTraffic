"""[추가 탐색] NN3 검사 — SUMO 실행 없음."""
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import arena3_common as ac  # noqa: E402
import arena3_referee as ref  # noqa: E402
import explore_nn1 as nn  # noqa: E402
import explore_nn3 as n3  # noqa: E402
import explore_nn3_package as pkg  # noqa: E402
import explore_platoon1 as pt  # noqa: E402
import explore_weight as w  # noqa: E402

w.configure()
CFG = n3.config()


def strip(rows):
    return [{k: s[k] for k in ("date", "period", "tag", "sumo_seed")} for s in rows]


class ConfigTest(unittest.TestCase):
    def test_config_is_what_the_rule_makes(self):
        self.assertEqual(strip(CFG["situations_low_logs"]), strip(n3.low_log_situations()))
        self.assertEqual(strip(CFG["situations_low_validation"]), strip(n3.low_validation_situations()))
        self.assertEqual(strip(CFG["situations_round"]), strip(n3.round_situations()))
        self.assertEqual(strip(CFG["situations_final"]), strip(n3.final_validation_situations()))

    def test_sizes_match_the_prereg(self):
        self.assertEqual(len(CFG["situations_low_logs"]), 48)
        self.assertEqual(len(CFG["situations_low_validation"]), 12)
        self.assertEqual(len(CFG["situations_round"]), 60)
        self.assertEqual(len(CFG["situations_final"]), 84)
        self.assertEqual(len(n3.FINAL_PERIODS), 21)
        self.assertEqual((n3.FINAL_PERIODS[0], n3.FINAL_PERIODS[-1]), ("0600", "2100"))

    def test_dates_stay_apart(self):
        for part in ("situations_low_logs", "situations_round"):
            self.assertEqual({s["date"] for s in CFG[part]}, set(n3.TRAIN_DATES))
        for part in ("situations_low_validation", "situations_final"):
            self.assertEqual({s["date"] for s in CFG[part]}, set(n3.VALID_DATES))
        self.assertFalse(set(n3.TRAIN_DATES) & set(n3.VALID_DATES))
        self.assertEqual({s["date"] for s in n3.day_validation(CFG)}, set(n3.VALID_DATES))
        self.assertEqual(len(n3.day_validation(CFG)), 24)

    def test_every_date_has_the_same_bands(self):
        by_date = {}
        for s in CFG["situations_round"]:
            by_date.setdefault(s["date"], Counter())[s["band"]] += 1
        want = Counter({"am_peak": 2, "pm_peak": 2, "morning": 1, "noon": 1, "afternoon": 1, "evening": 1, "dawn": 1, "late": 1})
        self.assertEqual(len(by_date), 6)
        for counts in by_date.values():
            self.assertEqual(counts, want)
        logs = Counter((s["round"], s["date"], s["band"]) for s in CFG["situations_low_logs"])
        self.assertEqual({v for k, v in logs.items() if k[2] == "dawn"}, {3})
        self.assertEqual({v for k, v in logs.items() if k[2] == "late"}, {1})

    def test_low_slots_stay_in_the_day_and_outside_trained_hours(self):
        all_low = {*n3.LOW_SLOTS["dawn"], *n3.LOW_SLOTS["late"]}
        self.assertFalse(all_low & set(ac.TRAIN_SLOTS))
        self.assertEqual(max(int(p[:2]) * 60 + int(p[2:]) for p in all_low) + 75, 24 * 60)      # 제어 60분 + 꼬리 15분이 자정 안에 끝난다
        self.assertGreaterEqual(min(int(p[:2]) * 60 + int(p[2:]) for p in all_low) - 30, 0)     # 준비 30분도 같은 날

    def test_routes_are_distinct(self):
        rows = [s["route_sha256"] for part in ("situations_low_logs", "situations_low_validation", "situations_round", "situations_final") for s in CFG[part]]
        self.assertEqual(len(rows), len(set(rows)))
        self.assertTrue(all(isinstance(x, str) and len(x) == 64 for x in rows))

    def test_settings_are_pinned(self):
        self.assertEqual(CFG["n5"]["seeds"], [42, 43, 44])
        self.assertEqual(CFG["teachers"]["P"]["name"], "t20")
        self.assertEqual(CFG["training"]["network"], [64, 64])
        self.assertEqual(CFG["rollout"]["futures"], 8)
        self.assertEqual(CFG["points"]["fraction"], 0.25)
        self.assertEqual(CFG["rules"], {"n6_low_gain_s": 0.5, "n6_day_tolerance_s": 0.2, "adopt_gain_s": 0.2})
        self.assertEqual(n3.INPUT_SIZE, 120)
        self.assertEqual(CFG["world"], "P")


class ChoiceTest(unittest.TestCase):
    @staticmethod
    def student(values):
        net = nn.GainNet(n3.INPUT_SIZE)
        with torch.no_grad():
            net.advantage.bias[:] = torch.tensor(values, dtype=torch.float32)
        return net.eval()

    @staticmethod
    def green_net(values):
        net = nn.GreenNet(n3.INPUT_SIZE)
        with torch.no_grad():
            for p in net.parameters():
                p.zero_()
            net.net[-1].bias[:] = torch.tensor(values, dtype=torch.float32)
        return net.eval()

    def setUp(self):
        self.x = np.zeros(n3.INPUT_SIZE, np.float32)

    def test_no_students_is_the_network(self):
        net = self.green_net([0.0, 2.0, 1.0, 0.0])
        self.assertEqual(n3.choose(net, [], self.x, [0, 1, 2]), (1, []))
        self.assertEqual(n3.choose(net, [], self.x, [0, 2])[0], 2)

    def test_untrained_student_never_changes(self):
        net = self.green_net([0.0, 2.0, 1.0, 0.0])
        self.assertEqual(n3.choose(net, [(nn.GainNet(n3.INPUT_SIZE).eval(), 0.5)], self.x, [0, 1, 2]), (1, []))

    def test_student_changes_only_at_or_above_threshold(self):
        net = self.green_net([0.0, 2.0, 1.0, 0.0])
        student = self.student([0.0, 0.0, 3.0, 0.0])
        self.assertEqual(n3.choose(net, [(student, 3.5)], self.x, [0, 1, 2]), (1, []))
        self.assertEqual(n3.choose(net, [(student, 2.5)], self.x, [0, 1, 2]), (2, [(1, 2)]))

    def test_student_stays_in_allowed(self):
        net = self.green_net([0.0, 2.0, 1.0, 0.0])
        student = self.student([0.0, 0.0, 3.0, 9.0])
        self.assertEqual(n3.choose(net, [(student, 0.1)], self.x, [0, 1, 2])[0], 2)

    def test_stack_applies_in_order(self):
        net = self.green_net([0.0, 2.0, 1.0, 0.0])
        first, second = self.student([0.0, 0.0, 3.0, 0.0]), self.student([5.0, 0.0, 0.0, 0.0])
        action, changes = n3.choose(net, [(first, 1.0), (second, 1.0)], self.x, [0, 1, 2])
        self.assertEqual(action, 0)
        self.assertEqual(changes, [(1, 2), (2, 0)])


class CandidateTest(unittest.TestCase):
    def test_names_round_trip(self):
        for h in CFG["learn"]["label_marks"]:
            for a in CFG["learn"]["label_weights"]:
                for s in CFG["learn"]["shares"]:
                    self.assertEqual(n3.parse_candidate(n3.candidate_name(h, a, s)), (h, a, s))

    def test_screen_has_ten_labels_at_ten_percent(self):
        names = n3.screen_candidates(CFG)
        self.assertEqual(len(names), 10)
        self.assertTrue(all(n.endswith("s10") for n in names))
        self.assertIn("h30w25s10", names)
        self.assertIn("h600w75s10", names)

    def test_thresholds_take_top_three_labels_with_other_shares(self):
        means = {n: 50.0 + i for i, n in enumerate(n3.screen_candidates(CFG))}
        got = n3.threshold_candidates(means, CFG)
        top = sorted(means, key=means.get)[:3]
        self.assertEqual(len(got), 6)
        for name in top:
            h, a, _ = n3.parse_candidate(name)
            self.assertIn(n3.candidate_name(h, a, 0.05), got)
            self.assertIn(n3.candidate_name(h, a, 0.20), got)

    def test_ties_prefer_small_share_then_long_horizon(self):
        self.assertEqual(n3.finalists({"h300w25s20": 10.0, "h300w25s05": 10.0, "h600w25s05": 10.0, "h30w25s05": 9.0}, CFG), ["h30w25s05", "h600w25s05"])

    def test_n6_rule(self):
        rules = CFG["rules"]
        self.assertEqual(n3.n6_decision(0.5, -0.2, rules)["start"], "N6")
        self.assertEqual(n3.n6_decision(0.49, 1.0, rules)["start"], "N5")
        self.assertEqual(n3.n6_decision(3.0, -0.21, rules)["start"], "N5")

    def test_adoption_rule(self):
        rules = CFG["rules"]
        self.assertTrue(n3.adoption({"gain_s": 0.2, "t95": [0.01, 0.4]}, rules))
        self.assertFalse(n3.adoption({"gain_s": 0.19, "t95": [0.1, 0.3]}, rules))
        self.assertFalse(n3.adoption({"gain_s": 0.5, "t95": [-0.01, 1.0]}, rules))

    def test_paired_gain_is_vehicle_weighted(self):
        row = lambda total, vehicles: {"total": total, "vehicles": vehicles}  # noqa: E731
        base = {"a": [row(1000.0, 10)], "b": [row(100.0, 100)], "c": [row(200.0, 100)]}
        cand = {"a": [row(900.0, 10)], "b": [row(100.0, 100)], "c": [row(100.0, 100)]}
        got = n3.paired_gain(base, cand)
        self.assertAlmostEqual(got["gain_s"], (1300.0 - 1100.0) / 210)
        self.assertEqual((got["n"], got["positive"]), (3, 2))
        self.assertLess(got["t95"][0], got["gain_s"])
        with self.assertRaises(ValueError):
            n3.paired_gain(base, {"a": cand["a"]})

    def test_shards_cover_every_job_once(self):
        jobs = [{"i": i} for i in range(11)]
        parts = [n3.shard_jobs(jobs, s, 3) for s in range(3)]
        self.assertEqual(sorted(j["i"] for p in parts for j in p), list(range(11)))

    def test_wait_for_returns_when_files_appear(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.json"
            sleeps = []

            def sleep(_):
                sleeps.append(1)
                path.write_text("{}")

            n3.wait_for([path], "시험", 0.0, sleep=sleep, stamp=lambda *_: None)
            self.assertEqual(len(sleeps), 1)


def point(futures=8, mark_cost=None, restore=True, drop=(), collide=()):
    """녹색 0·2를 허용하고 실행은 0인 가짜 지점."""
    marks = CFG["learn"]["label_marks"]
    rollouts = {}
    for g in (0, 2):
        runs = []
        for j in range(futures):
            if j in drop and g == 2:
                runs.append(None)
                continue
            cell = {str(h): {"delay": float(100 * h + g * 10 + j), "residual": float(g + 1), "depart_wait": 0.0} for h in marks}
            runs.append({"future_seed": 1000 + j, "collisions_unexplained": int(j in collide), "collisions_at_204820": 0, "marks": cell})
        rollouts[str(g)] = runs
    return {"state": "s_p000", "x": [0.0] * n3.INPUT_SIZE, "allowed": [0, 2], "executed": 0, "restore_check": {"ok": restore}, "rollouts": rollouts}


class LabelTest(unittest.TestCase):
    def test_good_point_makes_a_row(self):
        row, reason = n3.point_row(point(), CFG)
        self.assertIsNone(reason)
        self.assertEqual(row["base"], 0)
        self.assertEqual(row["delay"].shape, (5, 4, 8))
        self.assertTrue(np.isnan(row["delay"][0, 1, 0]))
        self.assertEqual(row["valid_futures"], 8)

    def test_restore_mismatch_and_few_futures_are_dropped(self):
        self.assertEqual(n3.point_row(point(restore=False), CFG), (None, "restore_mismatch"))
        self.assertEqual(n3.point_row(point(drop=range(1, 8)), CFG), (None, "insufficient_valid_futures"))
        row, _ = n3.point_row(point(collide=(0, 1)), CFG)
        self.assertEqual(row["valid_futures"], 6)
        self.assertTrue(np.isnan(row["delay"][:, :, 0]).all())

    def test_wrong_shape_stops(self):
        bad = point()
        bad["executed"] = 1
        with self.assertRaises(ValueError):
            n3.point_row(bad, CFG)
        bad = point()
        bad["rollouts"]["2"] = bad["rollouts"]["2"][:3]
        with self.assertRaises(ValueError):
            n3.point_row(bad, CFG)

    def test_label_cost_formula(self):
        row, _ = n3.point_row(point(), CFG)
        costs = n3.label_costs(row["delay"][None], row["residual"][None], 0, 25.0)
        self.assertAlmostEqual(costs[0, 2, 3], (100 * 30 + 20 + 3 + 25.0 * 3) / n3.SCALE)
        self.assertTrue(np.isnan(costs[0, 1, 3]))

    def test_back_half_score_counts_only_chosen(self):
        back = np.full((3, 4, 2), np.nan)
        back[:, 0, :] = 10.0
        back[0, 1, :] = 7.0                 # 고른 바꿈: 3 이득
        back[1, 1, :] = 12.0                # 고른 바꿈: 2 손해
        back[2, 1, :] = 0.0                 # 안 고름
        got = n3.back_half_score(np.array([1, 1, 1]), np.array([5.0, 5.0, 1.0]), np.array([0, 0, 0]), back, 2.0)
        self.assertEqual(got["changes"], 2)
        self.assertAlmostEqual(got["gain_per_change"], 0.5)
        self.assertAlmostEqual(got["gain_per_decision"], 1.0 / 3)
        self.assertAlmostEqual(got["win_rate"], 0.5)

    def test_long_points_are_about_ten_percent(self):
        share = np.mean([n3.long_point(CFG, f"s_p{i:05d}") for i in range(5000)])
        self.assertAlmostEqual(share, 0.10, delta=0.02)


def fake_metrics(policy: str, sid: str, delay: float, fingerprint: str) -> dict:
    return {"vehicles": 100, "actual_total_delay": delay * 100, "mean_delay_s": delay, "long_waits": 0, "left_east": 80.0, "left_south": 70.0, "collisions_unexplained": 0, "collisions_at_204820": 0, "changed_by_rules": 0,
            "cost_check": {"passed": True}, "missing": 0, "unfinished": 0, "mode_action_errors": 0, "p0_mismatch": 0, "job_seconds": 1.0, "information": {"planned": 100, "student_changed": 5, "z5_cells": 0, "pairs": {"0->2": 5}},
            "fingerprint": fingerprint, "policy": policy, "situation": sid}


class ReportTest(unittest.TestCase):
    def test_whole_report_on_fake_results(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            gain = lambda s: {"gain_s": s, "t95": [s - 0.1, s + 0.1], "n": 12, "positive": 8}  # noqa: E731
            w.write(root / "prep" / "start.json", {"start": "N6", "means": {"N5_day": 33.0, "N6_day": 33.1, "N5_low": 30.0, "N6_low": 28.0, "TM4_low": 27.0}, "day_gain": gain(-0.1), "low_gain": gain(2.0)})
            scores = {f"{s:.2f}": {"gain_per_decision": 0.01, "gain_per_change": 0.1, "changes": 10, "share": s, "win_rate": 0.6} for s in CFG["learn"]["shares"]}
            students = [{"seed": seed, "label": f"h{h}w{int(a)}", "scores": scores} for h in CFG["learn"]["label_marks"] for a in CFG["learn"]["label_weights"] for seed in CFG["training"]["seeds"]]
            w.write(root / "r1" / "students" / "done.json", {"points": 2000, "skipped": {}, "agreement": {}, "runs": [{"cpu_seconds": 3600.0}] * 60, "students": students})
            evaluation = root / "r1" / "eval"
            w.write(evaluation / "experiment.json", {"fingerprint": "fp"})
            names = n3.screen_candidates(CFG)
            day = n3.day_validation(CFG)
            value = {n3.BASE: 40.0, **{n: 39.0 + i * 0.1 for i, n in enumerate(names)}}
            for n in names[:3]:
                h, a, _ = n3.parse_candidate(n)
                value[n3.candidate_name(h, a, 0.05)] = 39.5
                value[n3.candidate_name(h, a, 0.20)] = 38.0 if n == names[0] else 39.6

            def write(name, sits, delay):
                for i in range(3):
                    for s in sits:
                        sid = ac.situation_id(s)
                        w.write(evaluation / "runs" / f"{name}_m{i}" / sid / "metrics.json", fake_metrics(f"{name}_m{i}", sid, delay, "fp"))

            for name, v in value.items():
                write(name, day, v)
            first = n3.candidate_name(*n3.parse_candidate(names[0])[:2], 0.20)
            for name, v in ((n3.BASE, 40.0), (first, 39.5), (names[0], 39.9)):
                write(name, [*CFG["situations_final"], *CFG["situations_low_validation"]], v)
            with self.assertRaises(ValueError):                     # 지문이 다른 결과는 읽지 않는다
                w.write(evaluation / "experiment.json", {"fingerprint": "other"})
                n3.read_stage_runs(evaluation, [n3.BASE], day)
            w.write(evaluation / "experiment.json", {"fingerprint": "fp"})
            report = n3.collect_report(root, CFG)
            self.assertEqual(report["status"], "done")
            self.assertEqual(report["finalists"], [first, names[0]])
            self.assertEqual(report["reading"]["winner"], first)
            self.assertAlmostEqual(report["reading"]["gain"]["gain_s"], 0.5)
            self.assertFalse(report["reading"]["adopted"] is None)
            text = n3.render(report)
            self.assertIn("채택", text)
            self.assertIn(first, text)

    def test_incomplete_report_says_what_is_missing(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            gain = {"gain_s": 0.0, "t95": [0.0, 0.0], "n": 12, "positive": 6}
            w.write(root / "prep" / "start.json", {"start": "N5", "means": {"N5_day": 33.0, "N6_day": 33.1, "N5_low": 30.0, "N6_low": 30.0, "TM4_low": 27.0}, "day_gain": gain, "low_gain": gain})
            report = n3.collect_report(root, CFG)
            self.assertEqual(report["status"], "prep_only")
            self.assertIn("아직", n3.render(report))


class SelectionTest(unittest.TestCase):
    def test_selection_reads_status_summaries_only(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            sids = [ac.situation_id(s) for s in n3.day_validation(CFG)]
            names = [n3.BASE, *n3.screen_candidates(CFG)]
            rows = {f"{n}_m{i}": {s: {"vehicles": 100, "total": (40.0 - j * 0.1) * 100} for s in sids} for j, n in enumerate(names) for i in range(3)}
            for shard in range(3):
                w.write(n3._stage_status(out, "screen", shard), {"jobs": 1, "rows": rows if shard == 0 else {}})
            got = n3.selection(out, 3, CFG)
            self.assertEqual(len(got["thresholds"]), 6)
            self.assertNotIn("finalists", got)
            more = {f"{n}_m{i}": {s: {"vehicles": 100, "total": 3000.0} for s in sids} for n in got["thresholds"] for i in range(3)}
            for shard in range(3):
                w.write(n3._stage_status(out, "thresholds", shard), {"jobs": 1, "rows": more if shard == 1 else {}})
            got = n3.selection(out, 3, CFG)
            self.assertEqual(len(got["finalists"]), 2)
            self.assertTrue(all(n in got["thresholds"] for n in got["finalists"]))

    def test_stack_identity_changes_with_threshold(self):
        with tempfile.TemporaryDirectory() as d:
            net = nn.GreenNet(n3.INPUT_SIZE)
            meta = {"kind": nn.KIND_GREEN, "world": "P", "seed": 42, "input_size": n3.INPUT_SIZE, "network": [64, 64], "dataset_fingerprint": "x"}
            model = Path(d) / "m"
            nn.save_model(model, net, meta)
            student = Path(d) / "s"
            nn.save_model(student, nn.GainNet(n3.INPUT_SIZE), {**meta, "kind": nn.KIND_GAIN})
            sits = CFG["situations_final"][:1]
            a = n3.nn_jobs(Path(d) / "o", Path(d) / "w", "fp", "X", {42: model}, sits, stacks={42: [[str(student), 1.0]]})
            b = n3.nn_jobs(Path(d) / "o", Path(d) / "w", "fp", "X", {42: model}, sits, stacks={42: [[str(student), 2.0]]})
            c = n3.nn_jobs(Path(d) / "o", Path(d) / "w", "fp", n3.BASE, {42: model}, sits)
            self.assertNotEqual(a[0]["model_identity"], b[0]["model_identity"])
            self.assertEqual(c[0]["stack"], [])
            self.assertTrue(a[0]["policy"].endswith("_m0"))


class PackageTest(unittest.TestCase):
    def test_notebooks_compile_and_point_at_this_package(self):
        for name in pkg.NOTEBOOKS:
            nb = pkg.notebook(name)
            text = "".join("".join(c["source"]) for c in nb["cells"])
            for c in nb["cells"]:
                if c["cell_type"] == "code":
                    compile("".join(c["source"]), name, "exec")
            self.assertIn("FutureTraffic_NN3.zip", text)
            self.assertIn("test_explore_nn3.py", text)

    def test_shard_numbers_are_set(self):
        self.assertEqual(pkg.SHARDS, 4)
        for s in range(pkg.SHARDS):
            text = "".join("".join(c["source"]) for c in pkg.notebook(f"R1_shard{s}")["cells"])
            self.assertIn(f"SHARD = {s}", text)
            self.assertIn(f'"--shards", "{pkg.SHARDS}"', text)

    def test_old_packages_are_protected(self):
        self.assertIn("FutureTraffic_NN12_fix1.zip", pkg.PROTECTED_NAMES)
        self.assertIn("FutureTraffic_NN1.zip", pkg.PROTECTED_NAMES)
        self.assertNotIn("FutureTraffic_NN3.zip", pkg.PROTECTED_NAMES)


if __name__ == "__main__":
    unittest.main()
