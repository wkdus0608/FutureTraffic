"""[추가 탐색] NN3R3 검사 — SUMO 실행 없음."""
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import arena3_common as ac  # noqa: E402
import explore_nn3 as n3  # noqa: E402
import explore_nn3r2 as r2  # noqa: E402
import explore_nn3r3 as r3  # noqa: E402
import explore_nn3r3_package as pkg  # noqa: E402
import explore_weight as w  # noqa: E402

w.configure()
CFG = r3.config()
N3 = n3.config()
R2 = r2.config()


def strip(rows):
    return [{k: s[k] for k in ("date", "period", "tag", "sumo_seed")} for s in rows]


class ConfigTest(unittest.TestCase):
    def test_config_is_what_the_rule_makes(self):
        self.assertEqual(strip(CFG["situations_round"]), strip(r3.round_situations()))
        self.assertEqual(strip(CFG["situations_screen"]), strip(r3.screen_situations(N3)))
        self.assertEqual(strip(CFG["situations_final"]), strip(r3.final_situations()))
        self.assertEqual(strip(CFG["situations_low"]), strip(r3.low_situations()))

    def test_sizes_match_the_prereg(self):
        self.assertEqual(len(CFG["situations_round"]), 120)
        self.assertEqual(len(CFG["situations_screen"]), 24)
        self.assertEqual(len(CFG["situations_final"]), 84)
        self.assertEqual(len(CFG["situations_low"]), 12)
        self.assertEqual(len(r3.screen_candidates(CFG)), 7)

    def test_dates_stay_apart(self):
        train = {s["date"] for s in CFG["situations_round"]}
        valid = {s["date"] for part in ("situations_screen", "situations_final", "situations_low") for s in CFG[part]}
        self.assertEqual(train, set(n3.TRAIN_DATES))
        self.assertEqual(valid, set(n3.VALID_DATES))
        self.assertFalse(train & valid)

    def test_every_date_has_twenty_slots(self):
        for date in n3.TRAIN_DATES:
            self.assertEqual(dict(Counter(s["band"] for s in CFG["situations_round"] if s["date"] == date)), r3.BAND_COUNTS)

    def test_routes_are_new(self):
        rows = [s["route_sha256"] for part in ("situations_round", "situations_screen", "situations_final", "situations_low") for s in CFG[part]]
        self.assertEqual(len(rows), len(set(rows)))
        self.assertFalse(set(rows) & r3.previous_routes(N3, R2))

    def test_settings_are_pinned(self):
        self.assertEqual(CFG["rollout"]["horizon_s"], 300)
        self.assertEqual(CFG["rollout"]["long_fraction"], 0.0)
        self.assertEqual(CFG["learn"]["label_marks"], [300])
        self.assertEqual(CFG["learn"]["label_weights"], [25.0])
        self.assertEqual(CFG["points"]["fraction"], 0.25)
        self.assertNotEqual(CFG["points"]["salt"], R2["points"]["salt"])
        self.assertNotEqual(CFG["rollout"]["future_salt"], R2["rollout"]["future_salt"])
        self.assertEqual(CFG["rules"]["adopt_gain_s"], 0.2)

    def test_base_is_round_two_winner_stacked_on_round_one(self):
        frozen = CFG["base"]["frozen"]
        self.assertEqual(set(frozen), {"42", "43", "44"})
        self.assertEqual(CFG["base"]["round2"]["winner"], "pool_s03")
        for key, row in frozen.items():
            self.assertEqual([s["round"] for s in row["students"]], [1, 2])
            self.assertEqual(row["students"][0]["student_sha256"], R2["base"]["frozen"][key]["student_sha256"])
            self.assertEqual(row["n5_sha256"], R2["base"]["frozen"][key]["n5_sha256"])

    def test_folder_points_at_the_right_round(self):
        row = {"round": 2, "folder": "r2/students/pool/seed_42"}
        self.assertEqual(r3.student_folder(row, Path("/a"), Path("/b")), Path("/b/pool/seed_42"))
        self.assertEqual(r3.student_folder({"round": 1, "folder": "r1/students/h300w25/seed_42"}, Path("/a"), Path("/b")), Path("/a/h300w25/seed_42"))


class CandidateTest(unittest.TestCase):
    def test_new_cannot_be_a_finalist(self):
        means = {"new_s05": 1.0, "recent_s05": 2.0, "all_s05": 3.0, "all_s03": 4.0}
        self.assertEqual(r3.finalists(means, CFG), ["recent_s05", "all_s05"])

    def test_ties_prefer_small_share_then_more_data(self):
        self.assertEqual(r3.finalists({"recent_s10": 1.0, "recent_s03": 1.0, "all_s03": 1.0}, CFG), ["all_s03", "recent_s03"])

    def test_final_names_are_stable_and_cover_the_curve(self):
        names = r3.final_names(["all_s10", "recent_s05"], CFG)
        self.assertEqual(names[:2], [r3.BASE, r3.N5])
        self.assertEqual(set(names[2:]), {"all_s10", "recent_s05", "new_s05", "all_s05"})
        self.assertEqual(names, r3.final_names(["recent_s05", "all_s10"], CFG))
        self.assertEqual(len(names), len(set(names)))


def fake_points(n, drivers, base=0):
    allowed = np.zeros((n, 4), bool)
    allowed[:, [0, 2]] = True
    return {"x": np.zeros((n, n3.INPUT_SIZE), np.float32), "allowed": allowed, "base": np.full(n, base, np.int64), "dates": np.asarray(["20220812"] * n), "delay": np.zeros((n, 1, 4, 8)), "residual": np.zeros((n, 1, 4, 8)), "drivers": np.asarray(drivers), "sids": np.asarray([f"x{i}" for i in range(n)]), "fingerprint": f"fp{n}", "skipped": {}, "runs": []}


class DataTest(unittest.TestCase):
    def test_datasets_have_the_registered_sizes_and_rebased_old_points(self):
        new, two, one = fake_points(4, [42] * 4), fake_points(3, [42, 43, 44]), fake_points(2, [42, 43])
        sets = r3.build_datasets(new, two, one, lambda seed, x, allowed: 2)
        changed = sets.pop("_changed")
        self.assertEqual({k: len(v["x"]) for k, v in sets.items()}, {"new": 4, "recent": 7, "all": 9})
        self.assertEqual(sets["all"]["base"].tolist(), [0] * 4 + [2] * 5)
        self.assertEqual(two["base"].tolist(), [0, 0, 0], "앞 라운드 표를 고치면 안 된다")
        self.assertEqual(one["base"].tolist(), [0, 0], "앞 라운드 표를 고치면 안 된다")
        self.assertEqual(changed, {"round2": 1.0, "round1": 1.0})
        self.assertEqual(len({sets[k]["fingerprint"] for k in sets}), 3)

    def test_rebase_uses_each_points_own_driver(self):
        sets = r3.build_datasets(fake_points(1, [42]), fake_points(2, [42, 43]), fake_points(1, [44]), lambda seed, x, allowed: 2 if seed == 43 else 0)
        self.assertEqual(sets["all"]["base"].tolist(), [0, 0, 2, 0])

    def test_collect_writes_this_round_and_gives_the_number_back(self):
        seen = []
        original = r2.collect
        r2.collect = lambda *a, **k: seen.append(r2.ROUND) or 0
        try:
            self.assertEqual(r3.collect({}, {}, CFG, Path("o"), Path("w"), 0, 1, 1, "fp"), 0)
        finally:
            r2.collect = original
        self.assertEqual(seen, [3])
        self.assertEqual(r2.ROUND, 2)


class ReportTest(unittest.TestCase):
    def _report(self, tmp: Path, gain: float):
        """모든 정책·상황의 가짜 결과를 파일로 쓰고 판정을 돌린다. all_s05가 BASE보다 gain초 짧다."""
        root = tmp / "r3"
        evaluation = root / "eval"
        evaluation.mkdir(parents=True)
        w.write(evaluation / "experiment.json", {"fingerprint": "fp"})
        (root / "students").mkdir()
        w.write(root / "students" / "done.json", {"points": {"new": 1, "recent": 2, "all": 3}, "skipped_new": {}, "skipped_round2": {}, "skipped_round1": {}, "old_base_changed_share": {"round2": 0.05, "round1": 0.1}, "runs": [{"cpu_seconds": 3600.0}], "students": [{"data": "new", "seed": 42, "scores": {}}]})
        delays = {r3.BASE: 32.3, r3.N5: 33.3, "all_s05": 32.3 - gain, "all_s03": 32.5, "all_s10": 32.8, "recent_s05": 32.6, "recent_s03": 32.7, "recent_s10": 32.9, "new_s05": 32.55}
        for k, s in enumerate([*CFG["situations_screen"], *CFG["situations_final"], *CFG["situations_low"]]):
            sid = ac.situation_id(s)
            jitter = 0.05 * (k % 3 - 1)
            for name, value in delays.items():
                for i in range(3):
                    folder = evaluation / "runs" / f"{name}_m{i}" / sid
                    folder.mkdir(parents=True)
                    w.write(folder / "metrics.json", self._metrics(value + jitter if name == "all_s05" else value))
            folder = evaluation / "runs" / r3.FIXC / sid
            folder.mkdir(parents=True)
            w.write(folder / "metrics.json", self._metrics(47.0))
        return r3.collect_report(root, CFG)

    @staticmethod
    def _metrics(delay):
        return {"fingerprint": "fp", "vehicles": 100, "actual_total_delay": delay * 100, "long_waits": 0, "left_east": 80.0, "left_south": 80.0, "collisions_unexplained": 0, "collisions_at_204820": 0, "changed_by_rules": 0, "job_seconds": 1.0, "information": {"planned": 100, "student_changed": 7, "pairs": {}}, "cost_check_passed": True}

    def test_whole_report_on_fake_results(self):
        with tempfile.TemporaryDirectory() as d:
            report = self._report(Path(d), 0.5)
            self.assertEqual(report["status"], "done")
            self.assertEqual(report["reading"]["winner"], "all_s05")
            self.assertTrue(report["reading"]["adopted"])
            self.assertAlmostEqual(report["reading"]["gain"]["gain_s"], 0.5, places=1)
            self.assertAlmostEqual(report["reading"]["round2_replication"]["gain_s"], 1.0, places=6)
            self.assertAlmostEqual(report["reading"]["cumulative"]["gain_s"], 1.5, places=1)
            self.assertAlmostEqual(report["reading"]["percent_vs_fixed"][r3.BASE], 100 * (32.3 - 47) / 47, places=6)
            self.assertAlmostEqual(report["data_curve"]["recent_minus_new"]["gain_s"], -0.05, places=6)
            text = r3.render(report)
            self.assertIn("채택", text)
            self.assertIn("고정신호 대비", text)
            self.assertIn("자료 곡선", text)

    def test_small_gain_is_not_adopted(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(self._report(Path(d), 0.1)["reading"]["adopted"])

    def test_incomplete_report_says_what_is_missing(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.assertEqual(r3.collect_report(root, CFG)["status"], "before_students")
            (root / "eval").mkdir()
            w.write(root / "eval" / "experiment.json", {"fingerprint": "fp"})
            (root / "students").mkdir()
            w.write(root / "students" / "done.json", {"points": {}, "skipped_new": {}, "skipped_round2": {}, "skipped_round1": {}, "old_base_changed_share": {}, "runs": [], "students": []})
            report = r3.collect_report(root, CFG)
            self.assertEqual(report["status"], "incomplete")
            self.assertIn("아직", r3.render(report))


class PackageTest(unittest.TestCase):
    def test_notebooks_compile_and_point_at_this_package(self):
        for name in pkg.NOTEBOOKS:
            nb = pkg.notebook(name)
            text = "".join("".join(c["source"]) for c in nb["cells"])
            for c in nb["cells"]:
                if c["cell_type"] == "code":
                    compile("".join(c["source"]), name, "exec")
            self.assertIn("FutureTraffic_NN3R3.zip", text)

    def test_shard_numbers_are_set(self):
        self.assertEqual(pkg.SHARDS, 4)
        for s in range(pkg.SHARDS):
            text = "".join("".join(c["source"]) for c in pkg.notebook(f"R3_shard{s}")["cells"])
            self.assertIn(f"SHARD = {s}", text)
            self.assertIn(f'"--shards", "{pkg.SHARDS}"', text)
            self.assertIn("test_explore_nn3r3.py", text)

    def test_map_notebook_reads_network_data_and_runs_its_tests(self):
        text = "".join("".join(c["source"]) for c in pkg.notebook("map")["cells"])
        self.assertIn("FutureTraffic_network12.zip", text)
        self.assertIn("test_explore_nn3map.py", text)
        self.assertIn("--stored-n5", text)

    def test_old_packages_are_protected(self):
        self.assertIn("FutureTraffic_NN3R2.zip", pkg.PROTECTED_NAMES)
        self.assertNotIn("FutureTraffic_NN3R3.zip", pkg.PROTECTED_NAMES)


if __name__ == "__main__":
    unittest.main()
