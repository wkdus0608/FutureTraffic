"""[추가 탐색] NN3R2 검사 — SUMO 실행 없음."""
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import arena3_common as ac  # noqa: E402
import explore_nn1 as nn  # noqa: E402
import explore_nn3 as n3  # noqa: E402
import explore_nn3r2 as r2  # noqa: E402
import explore_nn3r2_package as pkg  # noqa: E402
import explore_weight as w  # noqa: E402

w.configure()
CFG = r2.config()
N3 = n3.config()


def strip(rows):
    return [{k: s[k] for k in ("date", "period", "tag", "sumo_seed")} for s in rows]


class ConfigTest(unittest.TestCase):
    def test_config_is_what_the_rule_makes(self):
        self.assertEqual(strip(CFG["situations_round"]), strip(r2.round_situations()))
        self.assertEqual(strip(CFG["situations_screen"]), strip(r2.screen_situations(N3)))
        self.assertEqual(strip(CFG["situations_final"]), strip(r2.final_situations()))
        self.assertEqual(strip(CFG["situations_low"]), strip(r2.low_situations()))

    def test_sizes_match_the_prereg(self):
        self.assertEqual(len(CFG["situations_round"]), 120)
        self.assertEqual(len(CFG["situations_screen"]), 24)
        self.assertEqual(len(CFG["situations_final"]), 84)
        self.assertEqual(len(CFG["situations_low"]), 12)
        self.assertEqual(len(r2.screen_candidates(CFG)), 7)

    def test_dates_stay_apart(self):
        train = {s["date"] for s in CFG["situations_round"]}
        valid = {s["date"] for part in ("situations_screen", "situations_final", "situations_low") for s in CFG[part]}
        self.assertEqual(train, set(n3.TRAIN_DATES))
        self.assertEqual(valid, set(n3.VALID_DATES))
        self.assertFalse(train & valid)

    def test_every_date_has_twenty_slots_in_doubled_bands(self):
        for date in n3.TRAIN_DATES:
            bands = Counter(s["band"] for s in CFG["situations_round"] if s["date"] == date)
            self.assertEqual(dict(bands), r2.BAND_COUNTS)

    def test_routes_are_new(self):
        rows = [s["route_sha256"] for part in ("situations_round", "situations_screen", "situations_final", "situations_low") for s in CFG[part]]
        self.assertEqual(len(rows), len(set(rows)))
        old = {s["route_sha256"] for part in ("situations_low_logs", "situations_low_validation", "situations_round", "situations_final") for s in N3[part]}
        self.assertFalse(set(rows) & old)

    def test_settings_are_pinned(self):
        self.assertEqual(CFG["rollout"]["horizon_s"], 300)
        self.assertEqual(CFG["rollout"]["marks"][-1], 300)
        self.assertEqual(CFG["rollout"]["long_fraction"], 0.0)
        self.assertEqual(CFG["learn"]["label_marks"], [300])
        self.assertEqual(CFG["learn"]["label_weights"], [25.0])
        self.assertEqual(CFG["points"]["fraction"], 0.25)
        self.assertNotEqual(CFG["points"]["salt"], N3["points"]["salt"])
        self.assertNotEqual(CFG["rollout"]["future_salt"], N3["rollout"]["future_salt"])
        self.assertEqual(CFG["rules"]["adopt_gain_s"], 0.2)
        self.assertEqual(set(CFG["base"]["frozen"]), {"42", "43", "44"})

    def test_half_is_balanced(self):
        ids = set(r2.half_ids(CFG))
        self.assertEqual(len(ids), 60)
        rows = [(i, s) for i, s in enumerate(CFG["situations_round"]) if ac.situation_id(s) in ids]
        for date in n3.TRAIN_DATES:
            bands = Counter(s["band"] for _, s in rows if s["date"] == date)
            self.assertEqual(dict(bands), {k: v // 2 for k, v in r2.BAND_COUNTS.items()})
        self.assertEqual(Counter(i % 3 for i, _ in rows), Counter({0: 20, 1: 20, 2: 20}))

    def test_long_points_are_off(self):
        self.assertFalse(any(n3.long_point(CFG, f"s{i}") for i in range(500)))


class CandidateTest(unittest.TestCase):
    def test_names_round_trip(self):
        self.assertEqual(r2.parse_candidate(r2.candidate_name("pool", 0.03)), ("pool", 0.03))

    def test_half_cannot_be_a_finalist(self):
        means = {"half_s05": 1.0, "new_s05": 2.0, "pool_s05": 3.0, "new_s03": 4.0}
        self.assertEqual(r2.finalists(means, CFG), ["new_s05", "pool_s05"])

    def test_ties_prefer_small_share_then_more_data(self):
        self.assertEqual(r2.finalists({"new_s10": 1.0, "new_s03": 1.0, "pool_s03": 1.0}, CFG), ["pool_s03", "new_s03"])

    def test_final_names_are_stable_and_cover_the_curve(self):
        names = r2.final_names(["pool_s10", "new_s05"], CFG)
        self.assertEqual(names[:2], [r2.BASE, r2.N5])
        self.assertEqual(set(names[2:]), {"pool_s10", "new_s05", "half_s05", "pool_s05"})
        self.assertEqual(names, r2.final_names(["new_s05", "pool_s10"], CFG))
        self.assertEqual(len(names), len(set(names)))

    def test_percent(self):
        self.assertAlmostEqual(r2.percent(30.0, 40.0), -25.0)


def fake_points(n, drivers, sids=None, base=0):
    allowed = np.zeros((n, 4), bool)
    allowed[:, [0, 2]] = True
    return {"x": np.zeros((n, n3.INPUT_SIZE), np.float32), "allowed": allowed, "base": np.full(n, base, np.int64), "dates": np.asarray(["20220812"] * n), "delay": np.zeros((n, 1, 4, 8)), "residual": np.zeros((n, 1, 4, 8)),
            "drivers": np.asarray(drivers), "sids": np.asarray(sids if sids is not None else [f"x{i}" for i in range(n)]), "fingerprint": "fp", "skipped": {}, "runs": []}


class DataTest(unittest.TestCase):
    def test_rebase_uses_the_points_own_driver(self):
        d = fake_points(3, [42, 43, 44])
        got = r2.rebased_points(d, lambda seed, x, allowed: 2 if seed == 43 else 0)
        self.assertEqual(got.tolist(), [0, 2, 0])

    def test_rebase_stops_on_a_green_outside_allowed(self):
        with self.assertRaises(ValueError):
            r2.rebased_points(fake_points(1, [42]), lambda seed, x, allowed: 1)

    def test_datasets_have_the_registered_sizes(self):
        half = r2.half_ids(CFG)
        sids = [half[0], "other", half[1], "other2"]
        new, old = fake_points(4, [42] * 4, sids), fake_points(2, [42, 43], base=0)
        sets = r2.build_datasets(new, old, lambda seed, x, allowed: 2, CFG)
        self.assertEqual(len(sets["half"]["x"]), 2)
        self.assertEqual(len(sets["new"]["x"]), 4)
        self.assertEqual(len(sets["pool"]["x"]), 6)
        self.assertEqual(sets["pool"]["base"].tolist(), [0, 0, 0, 0, 2, 2])
        self.assertEqual(old["base"].tolist(), [0, 0], "라운드 1 표를 고치면 안 된다")
        self.assertEqual(len({sets[k]["fingerprint"] for k in sets}), 3)

    def test_student_learns_a_rebased_gain_from_the_same_rollouts(self):
        """같은 미래의 후보 비용에서 기준만 바꾸면 이득의 부호가 뒤집힌다(다시 굴릴 필요가 없다)."""
        costs = np.full((1, 4, 8), np.nan)
        costs[0, 0] = 3.0
        costs[0, 2] = 1.0
        allowed = np.asarray([[True, False, True, False]])
        at0, _ = nn.gains_against(costs, np.asarray([0]), allowed)
        at2, _ = nn.gains_against(costs, np.asarray([2]), allowed)
        self.assertAlmostEqual(float(at0[0, 2]), 2.0)
        self.assertAlmostEqual(float(at2[0, 0]), -2.0)


def row(total, vehicles=100, **more):
    return {"vehicles": vehicles, "total": total, "planned": 100, "changed": 7, "long": 0, "left_east": 80.0, "left_south": 80.0, "collisions": 0, "at_204820": 0, "rules": 0, "dirty": [], "seconds": 1.0, "pairs": {"0->1": 3}, "z5": 0, **more}


class ReportTest(unittest.TestCase):
    def _report(self, tmp: Path, gain: float):
        """모든 정책·상황의 가짜 결과를 파일로 쓰고 판정을 돌린다. new_s05가 BASE보다 gain초 짧다."""
        r = tmp / "r2"
        evaluation = r / "eval"
        evaluation.mkdir(parents=True)
        w.write(evaluation / "experiment.json", {"fingerprint": "fp"})
        (r / "students").mkdir()
        w.write(r / "students" / "done.json", {"points": {"half": 1, "new": 2, "pool": 3}, "skipped_new": {}, "skipped_old": {}, "old_base_changed_share": 0.05, "runs": [{"cpu_seconds": 3600.0}], "students": [{"data": "new", "seed": 42, "scores": {}}]})
        delays = {r2.BASE: 33.0, r2.N5: 33.7, "new_s05": 33.0 - gain, "new_s03": 33.2, "new_s10": 33.5, "pool_s05": 33.3, "pool_s03": 33.4, "pool_s10": 33.6, "half_s05": 33.25}
        sits = [*CFG["situations_screen"], *CFG["situations_final"], *CFG["situations_low"]]
        for k, s in enumerate(sits):
            sid = ac.situation_id(s)
            jitter = 0.05 * (k % 3 - 1)
            for name, value in delays.items():
                for i in range(3):
                    folder = evaluation / "runs" / f"{name}_m{i}" / sid
                    folder.mkdir(parents=True)
                    w.write(folder / "metrics.json", self._metrics(value + jitter if name == "new_s05" else value))
            folder = evaluation / "runs" / r2.FIXC / sid
            folder.mkdir(parents=True)
            w.write(folder / "metrics.json", self._metrics(47.0))
        return r2.collect_report(r, CFG)

    @staticmethod
    def _metrics(delay):
        return {"fingerprint": "fp", "vehicles": 100, "actual_total_delay": delay * 100, "long_waits": 0, "left_east": 80.0, "left_south": 80.0, "collisions_unexplained": 0, "collisions_at_204820": 0, "changed_by_rules": 0, "job_seconds": 1.0, "information": {"planned": 100, "student_changed": 7, "pairs": {}}, "cost_check_passed": True}

    def test_whole_report_on_fake_results(self):
        with tempfile.TemporaryDirectory() as d:
            report = self._report(Path(d), 0.5)
            self.assertEqual(report["status"], "done")
            self.assertEqual(report["reading"]["winner"], "new_s05")
            self.assertTrue(report["reading"]["adopted"])
            self.assertAlmostEqual(report["reading"]["gain"]["gain_s"], 0.5, places=1)
            self.assertAlmostEqual(report["reading"]["round1_replication"]["gain_s"], 0.7, places=6)
            self.assertAlmostEqual(report["reading"]["cumulative"]["gain_s"], 1.2, places=1)
            self.assertAlmostEqual(report["reading"]["percent_vs_fixed"][r2.BASE], 100 * (33 - 47) / 47, places=6)
            self.assertAlmostEqual(report["data_curve"]["pool_minus_new"]["gain_s"], -0.8, places=1)
            text = r2.render(report)
            self.assertIn("채택", text)
            self.assertIn("고정신호 대비", text)
            self.assertIn("자료 곡선", text)

    def test_small_gain_is_not_adopted(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(self._report(Path(d), 0.1)["reading"]["adopted"])

    def test_incomplete_report_says_what_is_missing(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.assertEqual(r2.collect_report(root, CFG)["status"], "before_students")
            (root / "eval").mkdir()
            w.write(root / "eval" / "experiment.json", {"fingerprint": "fp"})
            (root / "students").mkdir()
            w.write(root / "students" / "done.json", {"points": {}, "skipped_new": {}, "skipped_old": {}, "old_base_changed_share": 0.0, "runs": [], "students": []})
            report = r2.collect_report(root, CFG)
            self.assertEqual(report["status"], "incomplete")
            self.assertIn("아직", r2.render(report))


class JobTest(unittest.TestCase):
    def test_fixed_jobs_carry_what_the_worker_needs(self):
        with tempfile.TemporaryDirectory() as d:
            s = CFG["situations_final"][0]
            job = r2.fixed_jobs(Path(d), Path(d) / "w", "fp", [s])[0]
            self.assertEqual(job["mode"], "fixed")
            self.assertEqual(job["world"], r2.WORLD)
            self.assertEqual(job["route_sha256"], s["route_sha256"])
            self.assertEqual(set(job["situation"]), {"date", "period", "tag", "sumo_seed"})
            self.assertEqual(job["policy"], r2.FIXC)

    def test_data_identity_changes_with_threshold(self):
        with tempfile.TemporaryDirectory() as d:
            meta = {"kind": nn.KIND_GREEN, "world": "P", "seed": 42, "input_size": n3.INPUT_SIZE, "network": [64, 64], "dataset_fingerprint": "x"}
            model, student = Path(d) / "m", Path(d) / "s"
            nn.save_model(model, nn.GreenNet(n3.INPUT_SIZE), meta)
            nn.save_model(student, nn.GainNet(n3.INPUT_SIZE), {**meta, "kind": nn.KIND_GAIN})
            self.assertNotEqual(r2.data_identity(model, [[str(student), 1.0]]), r2.data_identity(model, [[str(student), 2.0]]))
            self.assertNotEqual(r2.data_identity(model, []), r2.data_identity(model, [[str(student), 1.0]]))


class PackageTest(unittest.TestCase):
    def test_notebooks_compile_and_point_at_this_package(self):
        for name in pkg.NOTEBOOKS:
            nb = pkg.notebook(name)
            text = "".join("".join(c["source"]) for c in nb["cells"])
            for c in nb["cells"]:
                if c["cell_type"] == "code":
                    compile("".join(c["source"]), name, "exec")
            self.assertIn("FutureTraffic_NN3R2.zip", text)
            self.assertIn("test_explore_nn3r2.py", text)

    def test_shard_numbers_are_set(self):
        self.assertEqual(pkg.SHARDS, 4)
        for s in range(pkg.SHARDS):
            text = "".join("".join(c["source"]) for c in pkg.notebook(f"R2_shard{s}")["cells"])
            self.assertIn(f"SHARD = {s}", text)
            self.assertIn(f'"--shards", "{pkg.SHARDS}"', text)

    def test_old_packages_are_protected(self):
        self.assertIn("FutureTraffic_NN3.zip", pkg.PROTECTED_NAMES)
        self.assertNotIn("FutureTraffic_NN3R2.zip", pkg.PROTECTED_NAMES)


if __name__ == "__main__":
    unittest.main()
