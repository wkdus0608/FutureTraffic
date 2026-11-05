"""[추가 탐색] NN3MAP 검사 — SUMO 실행 없음."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import explore_nn1 as nn  # noqa: E402
import explore_nn12 as n12  # noqa: E402
import explore_nn3 as n3  # noqa: E402
import explore_nn3map as nm  # noqa: E402
import explore_nn3r3 as r3  # noqa: E402
import explore_weight as w  # noqa: E402

w.configure()
CFG = r3.config()
N12 = n12.config()
TAGS = [(f"{d}_{int(h):02d}", s) for d, h in N12["windows"] for s in N12["seeds"]]


def make_models(root: Path) -> tuple[dict, dict]:
    """NN12 설정의 해시와 맞을 수 없으므로 model_identity를 시험 안에서만 맞춘다."""
    meta = {"kind": nn.KIND_GREEN, "world": "P", "seed": 42, "input_size": n3.INPUT_SIZE, "network": [64, 64], "dataset_fingerprint": "x"}
    models, stacks = {}, {}
    for seed in (42, 43, 44):
        model, student = root / f"m{seed}", root / f"s{seed}"
        nn.save_model(model, nn.GreenNet(n3.INPUT_SIZE), {**meta, "seed": seed})
        nn.save_model(student, nn.GainNet(n3.INPUT_SIZE), {**meta, "kind": nn.KIND_GAIN, "seed": seed})
        models[seed], stacks[seed] = model, [[str(student), 0.5]]
    return models, stacks


class PlanTest(unittest.TestCase):
    def test_config_keeps_the_nn12_fingerprints(self):
        self.assertEqual(N12["evaluation_fingerprint"], "9d3b7504064b")
        self.assertEqual(N12["network12_fingerprint"], "711875f6c939")

    def test_n5_in_round_three_is_the_n5_in_nn12(self):
        self.assertEqual({k: v["n5_sha256"] for k, v in CFG["base"]["frozen"].items()}, N12["models"]["N5"]["identities"])

    def test_jobs_pair_with_n5_window_by_window(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            models, stacks = make_models(root)
            cfg = {**N12, "models": {**N12["models"], "N5": {**N12["models"]["N5"], "identities": {str(s): nn.model_identity(models[s]) for s in models}}}}
            jobs = nm.plan_jobs("R2W", models, stacks, cfg, root / "o", root / "w")
            n5 = n12.plan_jobs("N5", cfg, root / "o", root / "w", models=root)
            self.assertEqual(len(jobs), 230)
            self.assertEqual([(j["tag"], j["model_seed"]) for j in jobs], [(j["tag"], j["model_seed"]) for j in n5])
            self.assertTrue(all(not j["save_inputs"] for j in jobs))
            self.assertTrue(all(j["name"] == "R2W" and len(j["stack"]) == 1 for j in jobs))
            self.assertNotEqual(jobs[0]["model_identity"], jobs[0]["n5_identity"])

    def test_wrong_model_stops(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            models, stacks = make_models(root)
            with self.assertRaises(ValueError):
                nm.plan_jobs("R2W", models, stacks, N12, root / "o", root / "w")

    def test_r3w_needs_an_adopted_round(self):
        adopted = {"experiment": r3.EXPERIMENT, "status": "done", "reading": {"winner": "all_s05", "adopted": True}}
        self.assertEqual(nm.winner_of(adopted), "all_s05")
        self.assertIsNone(nm.winner_of({**adopted, "reading": {"winner": "all_s05", "adopted": False}}))
        self.assertIsNone(nm.winner_of({**adopted, "status": "incomplete"}))
        self.assertIsNone(nm.winner_of({}))


class WordsTest(unittest.TestCase):
    def test_words(self):
        self.assertEqual(nm.word_of({"t95": [0.1, 1.0]}), "남음")
        self.assertEqual(nm.word_of({"t95": [-0.1, 1.0]}), "구분 안 됨")
        self.assertEqual(nm.word_of({"t95": [-1.0, -0.1]}), "나빠짐")

    def test_reading_follows_the_prereg_table(self):
        self.assertEqual(nm.reading_text("남음", "남음"), nm.READING[("남음", "남음")])
        self.assertEqual(nm.reading_text("남음", "구분 안 됨"), nm.READING["partial"])
        self.assertEqual(nm.reading_text("남음", "나빠짐"), nm.READING["partial"])
        self.assertEqual(nm.reading_text("구분 안 됨", "남음"), nm.READING["none"])
        self.assertEqual(nm.reading_text("나빠짐", "구분 안 됨"), nm.READING["none"])


def metrics(delay, network, code="c", name="R2W", model=42, vehicles=100):
    return {"evaluation_fingerprint": n12.EVALUATION_FP, "network12_fingerprint": n12.NETWORK_FP, "window_204820": {"mean_delay_s": delay, "vehicles": vehicles, "long_delays": 0, "unfinished": 0, "by_movement": {}}, "network": {"mean_delay_s": network, "vehicles": vehicles, "long_delays": 0, "not_inserted": 0}, "via_204822": {"mean_delay_s": network + 10}, "guard": {"changed_by_rules": {}},
            "nn12": {"name": name, "code": code, "job_seconds": 100.0, "model_seed": model, "stats": {"planned": 100, "student_changed": 10, "timetable_mismatch": 0}}}


class ReportTest(unittest.TestCase):
    def _root(self, tmp: Path, code: str, *, skip_one=False, wrong_model=False):
        root, nn12_root, joint2 = tmp / "map", tmp / "nn12", tmp / "joint2"
        values = {"R2W": (47.0, 92.5), "N5": (48.0, 93.0), "TM4": (48.2, 92.5)}
        for name, (seg, net) in values.items():
            base = root if name == "R2W" else nn12_root
            for k, (window, seed) in enumerate(TAGS):
                if skip_one and name == "R2W" and k == 0:
                    continue
                jitter = 0.1 * (k % 3 - 1)
                model = 43 if wrong_model and name == "R2W" and k == 0 else 42
                folder = base / name / "runs" / f"{window}_eval{seed}"
                folder.mkdir(parents=True)
                w.write(folder / "metrics.json", metrics(seg + (jitter if name == "R2W" else 0), net + (jitter if name == "R2W" else 0), code=code, name=name, model=model))
        for name, folder_name, seg, net in (("FIXC", "fixedc", 56.5, 94.0), ("P2", "p2", 48.7, 93.1)):
            for window, seed in TAGS:
                folder = joint2 / n12.EVALUATION_FP / "windows" / window / f"{folder_name}_eval{seed}"
                folder.mkdir(parents=True)
                w.write(folder / "metrics.json", metrics(seg, net, code=code, name=name))
        return root, nn12_root, joint2

    def test_whole_report_reads_the_table(self):
        with tempfile.TemporaryDirectory() as d:
            root, nn12_root, joint2 = self._root(Path(d), nm.code_fingerprint())
            report = nm.collect_report(root, None, nn12_root, joint2, N12)
            self.assertEqual(report["status"], "done")
            self.assertEqual(report["policies"], ["R2W"])
            self.assertEqual(report["top"], "R2W")
            self.assertAlmostEqual(report["pairs"]["R2W-N5"]["segment"]["mean"], 1.0, places=1)
            self.assertAlmostEqual(report["pairs"]["R2W-N5"]["network"]["mean"], 0.5, places=1)
            self.assertEqual(report["reading"]["segment"], "남음")
            self.assertEqual(report["reading"]["network"], "남음")
            text = nm.render(report)
            self.assertIn("R2W − N5", text)
            self.assertIn("미채택", text)

    def test_incomplete_report_says_what_is_missing(self):
        with tempfile.TemporaryDirectory() as d:
            root, nn12_root, joint2 = self._root(Path(d), nm.code_fingerprint(), skip_one=True)
            report = nm.collect_report(root, None, nn12_root, joint2, N12)
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["missing"], {"R2W": 1})
            self.assertIn("아직", nm.render(report))

    def test_other_code_stops(self):
        with tempfile.TemporaryDirectory() as d:
            root, nn12_root, joint2 = self._root(Path(d), "다른 코드")
            with self.assertRaises(ValueError):
                nm.collect_report(root, None, nn12_root, joint2, N12)

    def test_unpaired_model_stops(self):
        with tempfile.TemporaryDirectory() as d:
            root, nn12_root, joint2 = self._root(Path(d), nm.code_fingerprint(), wrong_model=True)
            with self.assertRaises(ValueError):
                nm.collect_report(root, None, nn12_root, joint2, N12)


if __name__ == "__main__":
    unittest.main()
