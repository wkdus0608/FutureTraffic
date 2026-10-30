"""Referee adaptation and confirmation controls; no real SUMO calls."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import arena3_stage2_eval as ev


def record_for(folder):
    evaluation = {"lock_sha256": ev.w.file_hash(ev.ac.CONFIG), "kind": "fake-evaluation"}
    saved = {"evaluation": evaluation, "evaluation_fingerprint": ev.w.digest(evaluation), "extra_files": {}}
    saved["fingerprint"] = ev.w.digest(saved)
    ev.w.write(Path(folder) / "experiment.json", saved)
    return saved["fingerprint"]


def make_models(root, name, stage="final"):
    out = []
    for seed in (42, 43, 44):
        folder = Path(root) / name / f"seed_{seed}" / stage
        folder.mkdir(parents=True, exist_ok=True)
        raw = f"{name}|{seed}|{stage}".encode()
        (folder / "model.pt").write_bytes(raw)
        ev.w.write(folder / "manifest.json", {"seed": seed, "variant": ev.VARIANTS[name], "stage": stage,
                                               "dataset_fingerprint": "same-teacher-data",
                                               "model_sha256": hashlib.sha256(raw).hexdigest()})
        out.append(folder)
    return out


def identity(folder):
    manifest = ev.w.read(Path(folder) / "manifest.json")
    value = hashlib.sha256((Path(folder) / "model.pt").read_bytes()).hexdigest()
    if value != manifest["model_sha256"]:
        raise ValueError("model changed")
    return value


class Stage2EvaluationTest(unittest.TestCase):
    def test_winner_excludes_pretraining_and_requires_pass(self):
        results = {"S2A_pre": {"status": "pass", "gain_date_mean_s": 9},
                   "S2A": {"status": "hold_shared_collision", "gain_date_mean_s": 2},
                   "S2B": {"status": "fail", "gain_date_mean_s": 4}}
        self.assertIsNone(ev.choose_winner(results))
        results["S2A"]["status"] = "pass"
        results["S2B"]["status"] = "pass"
        self.assertEqual(ev.choose_winner(results), "S2B")
        results["S2B"]["gain_date_mean_s"] = 2
        self.assertEqual(ev.choose_winner(results), "S2A")

    def test_confirmation_has_new_route_ids(self):
        development = ev.situations_for("eval")
        confirmation = ev.situations_for("confirmation")
        self.assertEqual(len(confirmation), 84)
        self.assertEqual({s["tag"] for s in confirmation}, {"C1", "C2"})
        self.assertFalse({s["route_sha256"] for s in development} &
                         {s["route_sha256"] for s in confirmation})

    def test_pending_does_not_overwrite_complete_report(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / "judge.json").write_text("complete-result")
            (out / "summary.md").write_text("complete-summary")
            self.assertFalse(ev.save_judgement(out, {"pending": {"S2B": "waiting"}}))
            self.assertEqual((out / "judge.json").read_text(), "complete-result")
            self.assertEqual((out / "summary.md").read_text(), "complete-summary")
            self.assertTrue((out / "pending.json").exists())

    def test_selection_cannot_change_after_confirmation_started(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a = make_models(root, "S2A")
            b = make_models(root, "S2B")
            with patch.object(ev, "training", return_value=SimpleNamespace(model_identity=identity)):
                report = {"pending": {}, "evaluation_fingerprint": "same-referee", "results": {
                    "S2A": {"status": "pass", "gain_date_mean_s": 0.6},
                    "S2B": {"status": "pass", "gain_date_mean_s": 0.4}},
                    "models": {"S2A": ev.model_declaration("S2A", a, "eval"),
                               "S2B": ev.model_declaration("S2B", b, "eval")}}
                first = ev.freeze_selection(root / "selection.json", report,
                                            {"S2A": root / "S2A", "S2B": root / "S2B"})
                report["results"]["S2B"]["gain_date_mean_s"] = 10
                resumed = ev.freeze_selection(root / "selection.json", report,
                                              {"S2A": root / "S2A", "S2B": root / "S2B"})
                self.assertEqual(first, resumed)
                self.assertEqual(resumed["winner"], "S2A")
                (a[0] / "model.pt").write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    ev.freeze_selection(root / "selection.json", report, {})

    def test_model_declaration_rejects_wrong_stage_seed_and_dataset(self):
        with tempfile.TemporaryDirectory() as d:
            paths = make_models(d, "S2A")
            with patch.object(ev, "training", return_value=SimpleNamespace(model_identity=identity)):
                with self.assertRaises(ValueError):
                    ev.model_declaration("S2A_pre", paths, "eval")
                with self.assertRaises(ValueError):
                    ev.model_declaration("S2A", paths[::-1], "eval")
                metadata = ev.w.read(paths[0] / "manifest.json")
                metadata["dataset_fingerprint"] = "different-data"
                ev.w.write(paths[0] / "manifest.json", metadata)
                with self.assertRaises(ValueError):
                    ev.model_declaration("S2A", paths, "eval")

    def test_confirmation_executes_only_c1_c2_and_freezes_first(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            baseline = root / "original_baseline"
            record_for(baseline)
            ev.w.write(baseline / "reproduction.json", {"passed": True})
            a, b = make_models(root, "S2A"), make_models(root, "S2B")
            out = root / "confirm"
            seen = []

            def run(jobs, workers, ledger, **kwargs):
                self.assertTrue((out / "selection.json").exists())
                for job in jobs:
                    s = job["situation"]
                    self.assertIn(s["tag"], ("C1", "C2"))
                    seen.append((job["policy"], s["tag"], s["route_sha256"]))
                    value = 98 if job["policy"].startswith("S2A") else (101 if job["policy"] == "P0" else 100)
                    movements = {key: {"vehicles": 1, "mean_delay_s": value} for key in ev.ref.LEFT_KEYS.values()}
                    m = {"situation": job["sid"], "date": s["date"], "vehicles": 2,
                         "mean_delay_s": value, "actual_total_delay": value * 2,
                         "route_sha256": s["route_sha256"], "cohort_sha256": "same-cohort",
                         "cost_check": {"passed": True}, "by_movement": movements,
                         "long_waits": 0, "collisions_at_204820": 0,
                         "collision_details": {"collisions_unexplained": []}}
                    result = SimpleNamespace(metrics=m, decisions=[])
                    ev.ac.write_run(Path(job["target"]), result, {"fingerprint": job["fingerprint"],
                                            "model_identity": job.get("model_identity")})
                return {"done": len(jobs), "failed": []}

            def registration(folder, extra_files=()):
                return record_for(folder)

            original_collect = ev.collect_judgement
            with patch.object(ev, "training", return_value=SimpleNamespace(model_identity=identity)):
                report = {"pending": {}, "evaluation_fingerprint": ev.folder_record(baseline)["evaluation_fingerprint"],
                    "results": {"S2A": {"status": "pass", "gain_date_mean_s": 0.6},
                                "S2B": {"status": "fail", "gain_date_mean_s": 0}},
                    "models": {"S2A": ev.model_declaration("S2A", a, "eval")}}

                def collect(base, candidates, **kwargs):
                    if kwargs.get("part") == "confirmation":
                        return original_collect(base, candidates, **kwargs)
                    return copy.deepcopy(report)

                with patch.object(ev.ref, "register", side_effect=registration), \
                        patch.object(ev.ref, "run_jobs", side_effect=run), \
                        patch.object(ev, "collect_judgement", side_effect=collect):
                    result = ev.confirm(baseline, ["S2A=unused", "S2B=unused"],
                                        {"S2A": root / "S2A", "S2B": root / "S2B"},
                                        out, root / "work", 2)
            self.assertEqual(result["status"], "pass")
            self.assertEqual(len(seen), 504)
            self.assertEqual({t for _, t, _ in seen}, {"C1", "C2"})
            self.assertTrue((out / "judge.json").exists())

    def test_actual_worker_is_blocked_on_mac_before_simulation(self):
        job = {"work": "/tmp/test_arena3_stage2_eval_no_sumo", "policy": "M4", "sid": "fake"}
        with patch.object(ev.platform, "system", return_value="Darwin"), \
                patch.object(ev.ac, "run_arena_episode") as simulator:
            result = ev.eval_worker(job)
            self.assertFalse(result["ok"])
            self.assertIn("Colab", result["error"])
            simulator.assert_not_called()


if __name__ == "__main__":
    unittest.main()
