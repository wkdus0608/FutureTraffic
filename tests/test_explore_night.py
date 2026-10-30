"""WEIGHT1_NIGHT driver checks: no SUMO and no long training."""
import json
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import explore_night as en
import explore_night_train as nt
import explore_weight as ew
import explore_weight_episode as epi

EAST, SOUTH = "477706392_02->477706883_02", "477706883_01->477707013_01"


def put(root, part, tag, date, period, seed, mean, vehicles=100, folder_tag=None):
    """Write one finished run exactly where the driver looks for it."""
    name = f"{folder_tag or tag}_{date}_{period}_s{seed}"
    folder = Path(root) / name if part is None else Path(root) / "runs" / part / name
    folder.mkdir(parents=True, exist_ok=True)
    metrics = {"date": date, "period": period, "seed": seed, "eval_seed": seed, "policy": tag, "vehicles": vehicles,
               "total_delay": mean * vehicles, "actual_total_delay": mean * vehicles, "mean_delay_s": mean,
               "long_waits": 0, "missing": 0, "unfinished": 0, "changed_by_rules": 0, "mode_action_errors": 0,
               "collisions_at_204820": 0, "collisions_unexplained": 0, "cost_check": {"passed": True},
               "mode_disagreement_points": 15,
               "by_movement": {EAST: {"vehicles": 10, "mean_delay_s": 100.0}, SOUTH: {"vehicles": 10, "mean_delay_s": 80.0}}}
    (folder / "metrics.json").write_text(json.dumps(metrics))
    return metrics


def write_parent(parent, p0=40.0, m4=39.0):
    (Path(parent)).mkdir(parents=True, exist_ok=True)
    (Path(parent) / "budget.json").write_text(json.dumps({"used_s": 228_756.0, "limit_s": 345_600}))
    for date, period, seed in en.dev_conditions():
        put(Path(parent) / "runs" / "development_references", None, "P0", date, period, seed, p0, folder_tag="P0")
        put(Path(parent) / "runs" / "development_references", None, "M4", date, period, seed, m4, folder_tag="M4")


def make_ledger(out, prior=1000.0):
    (Path(out) / "ledger").mkdir(parents=True, exist_ok=True)
    (Path(out) / "ledger" / "prior.json").write_text(json.dumps({"prior_used_s": prior}))


def small_config(**budget):
    cfg = json.loads((ROOT / "config/explore_night_v1.json").read_text())
    cfg["budget"].update(budget)
    return cfg


class LedgerTest(unittest.TestCase):
    def test_roles_add_up_and_caps_stop_before_overrun(self):
        with tempfile.TemporaryDirectory() as d:
            make_ledger(d, prior=340_000.0)
            with patch.object(en, "config", return_value=small_config(night_cap_s=5000, total_cap_s=345_600, stop_margin_s=900)):
                a, b = en.Ledger(d, "checks"), en.Ledger(d, "train_42")
                a.add(1000)
                b.add(2000)
                self.assertEqual(en.Ledger(d, "checks").seconds, 1000)       # persisted across restarts
                self.assertEqual(a.night_s(), 3000)
                a.check(reserve=1000)                                        # 3000 + 1000 < 5000 and below the 96h cap
                with self.assertRaises(ew.BudgetExceeded):
                    a.check(reserve=2100)                                    # night cap
                a.add(1500)
                with self.assertRaises(ew.BudgetExceeded):
                    a.check(reserve=1000)                                    # 340000 + 4500 + 1000 > 345600 - 900

    def test_prepare_must_run_first(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                en.Ledger(d, "checks")


class PrepareAndRegisterTest(unittest.TestCase):
    def test_prepare_records_prior_usage_and_checks_references(self):
        with tempfile.TemporaryDirectory() as d:
            parent, out = Path(d) / "explore_weight_resume_v1", Path(d) / "night"
            write_parent(parent)
            args = SimpleNamespace(out=str(out), parent=str(parent))
            self.assertEqual(en.cmd_prepare(args), 0)
            self.assertEqual(json.loads((out / "ledger" / "prior.json").read_text())["prior_used_s"], 228_756.0)
            (parent / "budget.json").write_text(json.dumps({"used_s": 999.0}))
            en.cmd_prepare(args)                                              # keeps the first record
            self.assertEqual(json.loads((out / "ledger" / "prior.json").read_text())["prior_used_s"], 228_756.0)
            next(iter((parent / "runs" / "development_references").glob("P0_*"))).joinpath("metrics.json").unlink()
            with self.assertRaises(ValueError):
                en.cmd_prepare(args)

    def test_register_is_idempotent_and_protects_existing_results(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "night"
            fp = en.register(out)
            self.assertEqual(en.register(out), fp)
            record = json.loads((out / "experiment.json").read_text())
            record["files"]["src/explore_night.py"] = "0" * 64
            (out / "experiment.json").write_text(json.dumps(record))
            fp2 = en.register(out)                                            # nothing produced yet -> replaced
            self.assertEqual(fp2, fp)
            (out / "runs").mkdir()
            (out / "experiment.json").write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                en.register(out)                                              # results exist -> protected


class JobsTest(unittest.TestCase):
    def test_check_jobs_are_the_preregistered_set(self):
        jobs = en.check_jobs(Path("/o"), Path("/w"), "fp")
        self.assertEqual(len(jobs), 96)
        tags = {}
        for j in jobs:
            tags.setdefault((j["part"], j["tag"]), []).append(j)
        self.assertEqual({k: len(v) for k, v in tags.items() if k[0] == "n1"}, {("n1", "p0"): 6, ("n1", "m4"): 6})
        self.assertEqual(len(tags[("repro", "p0")]) + len(tags[("repro", "m4")]), 4)
        sweep = {k[1]: v for k, v in tags.items() if k[0] == "n2"}
        self.assertEqual(sorted(sweep), ["w2.5_2.5", "w2_3", "w3_2", "w3_3", "w4_4"])
        self.assertTrue(all(len(v) == 16 for v in sweep.values()))
        self.assertEqual(sweep["w3_2"][0]["factors"], [3.0, 2.0])
        self.assertIsNone(sweep["w3_2"][0]["mode"])
        self.assertEqual({j["mode"] for j in tags[("n1", "m4")]}, {4})
        self.assertIsNone(tags[("n1", "p0")][0]["mode"])
        self.assertTrue(all(" " not in j["work"] for j in jobs))
        # P0/M4 runs are on exactly the paired search conditions
        self.assertEqual({(j["date"], j["period"], j["seed"]) for j in tags[("n1", "p0")]},
                         {tuple(p) for p in en.config()["n1_search_pairs"]["pairs"]})


class RunJobsTest(unittest.TestCase):
    def fake_worker(self, job):
        target = Path(job["target"])
        put(target.parent, None, job["tag"], job["date"], job["period"], job["seed"], 40.0, folder_tag=job["tag"])
        ew.write(target / "cache.json", {"fingerprint": job["fingerprint"], "model_identity": job.get("model_identity"),
                                         "metrics_sha256": ew.file_hash(target / "metrics.json")})
        return {"ok": True, "seconds": getattr(self, "seconds", 2.0), "name": target.name}

    def jobs(self, out, n=6):
        return en.eval_jobs(out, Path(out) / "w", "fp", "n1", "p0", en.dev_conditions()[:n], mode=None)

    def test_runs_everything_then_reuses_and_charges_the_ledger(self):
        with tempfile.TemporaryDirectory() as d:
            make_ledger(d)
            ledger = en.Ledger(d, "checks")
            first = en.run_jobs(self.jobs(d), 3, ledger, time.time() + 60, executor_factory=ThreadPoolExecutor, worker=self.fake_worker)
            self.assertEqual((first["done"], first["failed"], first["stopped"]), (6, [], None))
            self.assertEqual(ledger.seconds, 12.0)
            again = en.run_jobs(self.jobs(d), 3, ledger, time.time() + 60, executor_factory=ThreadPoolExecutor, worker=self.fake_worker)
            self.assertEqual(again["done"], 0)
            self.assertEqual(ledger.seconds, 12.0)                              # cached runs cost nothing

    def test_failure_is_recorded_without_stopping_the_others(self):
        def flaky(job):
            if job["seed"] == 43:
                return {"ok": False, "seconds": 1.0, "name": Path(job["target"]).name, "error": "boom"}
            return self.fake_worker(job)
        with tempfile.TemporaryDirectory() as d:
            make_ledger(d)
            ledger = en.Ledger(d, "checks")
            result = en.run_jobs(self.jobs(d), 2, ledger, time.time() + 60, executor_factory=ThreadPoolExecutor, worker=flaky)
            self.assertEqual(result["done"], 4)
            self.assertEqual([f["error"] for f in result["failed"]], ["boom", "boom"])

    def test_budget_and_wall_limits_stop_new_work(self):
        with tempfile.TemporaryDirectory() as d:
            make_ledger(d)
            self.seconds = 450.0
            with patch.object(en, "config", return_value=small_config(night_cap_s=1000)):
                ledger = en.Ledger(d, "checks")
                result = en.run_jobs(self.jobs(d), 1, ledger, time.time() + 60, executor_factory=ThreadPoolExecutor, worker=self.fake_worker)
            self.assertTrue(result["stopped"].startswith("budget"))
            self.assertEqual((result["done"], result["unsubmitted"]), (1, 5))   # 450 used + 600 reserve > 1000 -> no second job
            self.seconds = 2.0
        with tempfile.TemporaryDirectory() as d:
            make_ledger(d)
            ledger = en.Ledger(d, "checks")
            result = en.run_jobs(self.jobs(d), 1, ledger, time.time() - 1, executor_factory=ThreadPoolExecutor, worker=self.fake_worker)
            self.assertEqual((result["stopped"], result["done"], result["unsubmitted"]), ("wall_time", 0, 6))


def fake_run_episode(date, period, eval_seed, work, policy, collect=False, train_seed=None, trace=True, max_seconds=None):
    from test_explore_night_train import episode_rows
    rows = episode_rows(policy, lambda step, mode: (1.0 if mode == 3 else -1.0) if step % 6 == 0 else -0.1)
    metrics = {"mean_delay_s": 1.0, "missing": 0, "unfinished": 0, "cost_check_passed": True, "changed_by_rules": 0,
               "collisions_unexplained": 0, "date": date, "period": period, "eval_seed": eval_seed}
    return SimpleNamespace(metrics=metrics, transitions=rows, decisions=[], seconds=0.25)


class CommandTest(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(ROOT / "tests"))

    def cfg(self, **budget):
        cfg = small_config(**budget)
        cfg["n3_learner"]["stages"] = [1, 2]
        return cfg

    def test_train_makes_both_stages_and_stops_cleanly_at_the_budget(self):
        scenarios = [("20220812", "1300"), ("20220819", "1800")]
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "night"
            make_ledger(out)
            args = SimpleNamespace(out=str(out), work=str(Path(d) / "w"), seed="42", wall_hours="1")
            with patch.object(en, "register", return_value="fp"), patch.object(en, "config", return_value=self.cfg()), \
                    patch.object(ew, "training_scenarios", return_value=scenarios), patch.object(epi, "run_episode", fake_run_episode):
                self.assertEqual(en.cmd_train(args), 0)
            model = out / "models" / "seed_42"
            self.assertTrue((model / "stages" / "episode_000001" / "model.pt").exists())
            self.assertTrue((model / "stages" / "episode_000002" / "model.pt").exists())
            status = json.loads((out / "status" / "train_42.json").read_text())
            self.assertEqual((status["state"], status["completed"]), ("complete", 2))
            self.assertGreaterEqual(en.Ledger(out, "train_42").seconds, 0.0)
            self.assertFalse(list((Path(d) / "w").rglob("episode_*")))           # scratch removed
            # the saved stage is exactly what evaluate will load
            self.assertEqual(nt.load_model(model / "stages" / "episode_000002", "fp").episodes, 2)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "night"
            make_ledger(out)
            args = SimpleNamespace(out=str(out), work=str(Path(d) / "w"), seed="43", wall_hours="1")
            with patch.object(en, "register", return_value="fp"), patch.object(en, "config", return_value=self.cfg(night_cap_s=500)), \
                    patch.object(ew, "training_scenarios", return_value=scenarios), patch.object(epi, "run_episode", fake_run_episode):
                self.assertEqual(en.cmd_train(args), en.EXIT_LIMIT)
            self.assertEqual(json.loads((out / "status" / "train_43.json").read_text())["state"], "limit")
            self.assertFalse((out / "models" / "seed_43" / "latest.json").exists())

    def test_train_rejects_unplanned_seed(self):
        with tempfile.TemporaryDirectory() as d:
            make_ledger(d)
            args = SimpleNamespace(out=d, work=d, seed="45", wall_hours="1")
            with patch.object(en, "register", return_value="fp"):
                with self.assertRaises(ValueError):
                    en.cmd_train(args)

    def test_evaluate_builds_sixteen_condition_jobs_for_each_saved_stage(self):
        scenarios = [("20220812", "1300"), ("20220819", "1800")]
        captured = []
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "night"
            make_ledger(out)
            with patch.object(en, "register", return_value="fp"), patch.object(en, "config", return_value=self.cfg()), \
                    patch.object(ew, "training_scenarios", return_value=scenarios), patch.object(epi, "run_episode", fake_run_episode):
                en.cmd_train(SimpleNamespace(out=str(out), work=str(Path(d) / "w"), seed="42", wall_hours="1"))
                def fake_run(jobs, workers, ledger, deadline, **kw):
                    captured.extend(jobs)
                    return {"done": 0, "failed": [], "stopped": None, "unsubmitted": 0}
                with patch.object(en, "run_jobs", fake_run):
                    en.cmd_evaluate(SimpleNamespace(out=str(out), work=str(Path(d) / "w"), workers="2", wall_hours="1"))
            self.assertEqual(len(captured), 32)                                   # 2 stages x 16 conditions
            self.assertEqual({j["tag"] for j in captured}, {"w2l_s42_e1", "w2l_s42_e2"})
            self.assertTrue(all(j["model_dir"] and j["model_identity"] for j in captured))


class SummaryTest(unittest.TestCase):
    def build(self, d, sweep_means=None, n3_means=None, new_p0_shift=0.0):
        parent, out = Path(d) / "parent", Path(d) / "night"
        write_parent(parent, p0=40.0, m4=39.0)
        make_ledger(out, prior=228_756.0)
        en.ledger_seconds = None
        (out / "experiment.json").write_text(json.dumps({"fingerprint": "abc"}))
        cfg = en.config()
        for date, period, seed in map(tuple, cfg["n1_search_pairs"]["pairs"]):
            stored = cfg["n1_search_pairs"]["stored"]["p0"][f"{date}_{period}_s{seed}"]
            put(out, "n1", "p0", date, period, seed, stored + new_p0_shift)
            put(out, "n1", "m4", date, period, seed, stored + new_p0_shift - 0.5)
        for date, period, seed in map(tuple, cfg["n2_static_sweep"]["reproduction_checks"]):
            put(out, "repro", "p0", date, period, seed, 40.0)
            put(out, "repro", "m4", date, period, seed, 39.0)
        for tag, mean in (sweep_means or {}).items():
            for date, period, seed in en.dev_conditions():
                put(out, "n2", tag, date, period, seed, mean)
        for (seed_, stage), mean in (n3_means or {}).items():
            for date, period, seed in en.dev_conditions():
                put(out, "n3", f"w2l_s{seed_}_e{stage}", date, period, seed, mean)
        return out, parent

    def test_n1_n2_verdicts_and_reproduction(self):
        with tempfile.TemporaryDirectory() as d:
            out, parent = self.build(d, sweep_means={"w3_3": 38.6, "w4_4": 39.5, "w2.5_2.5": 38.9})
            data = en.summary_data(out, parent)
            self.assertTrue(data["reproduction_ok"])
            n1 = data["n1"]
            self.assertEqual(n1["complete"], 6)
            self.assertAlmostEqual(n1["mean_m4_gain"], 0.5, places=9)
            self.assertAlmostEqual(n1["mean_search_gain"], 1.1915, places=4)
            self.assertAlmostEqual(n1["mean_search_minus_m4"], 0.6915, places=4)
            self.assertTrue(n1["p0_reproduced_all"])
            self.assertIn("여지 단서", n1["verdict"])
            by = {tuple(c["factors"]): c for c in data["n2"]["configs"]}
            self.assertAlmostEqual(by[(3.0, 3.0)]["gain_vs_m4"], 0.4, places=9)
            self.assertIn("해 볼 만한", by[(3.0, 3.0)]["verdict"])
            self.assertIn("비슷하거나 나쁨", by[(4.0, 4.0)]["verdict"])
            self.assertIn("비슷하거나 나쁨", by[(2.5, 2.5)]["verdict"])               # only 0.1 s better than M4
            self.assertEqual(by[(3.0, 2.0)]["runs"], 0)
            text = en.render(data)
            self.assertIn("N1", text)
            self.assertIn("3, 3", text)

    def test_n1_says_no_headroom_when_m4_already_matches_search(self):
        with tempfile.TemporaryDirectory() as d:
            out, parent = self.build(d)
            for path in (out / "runs" / "n1").glob("m4_*/metrics.json"):
                m = json.loads(path.read_text())
                stored = m["mean_delay_s"] + 0.5
                m["mean_delay_s"] = m["actual_total_delay"] = stored - 1.4
                m["total_delay"] = m["actual_total_delay"] = (stored - 1.4) * m["vehicles"]
                path.write_text(json.dumps(m))
            data = en.summary_data(out, parent)
            self.assertIn("근거를 못 얻음", data["n1"]["verdict"])

    def test_reproduction_mismatch_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            out, parent = self.build(d, new_p0_shift=0.5)
            for path in (out / "runs" / "repro").glob("m4_*/metrics.json"):
                m = json.loads(path.read_text())
                m["mean_delay_s"] += 0.01
                path.write_text(json.dumps(m))
            data = en.summary_data(out, parent)
            self.assertFalse(data["reproduction_ok"])
            self.assertFalse(data["n1"]["p0_reproduced_all"])
            self.assertIn("재현 불일치", en.render(data))

    def test_n3_verdicts_need_all_three_seeds_and_margin(self):
        with tempfile.TemporaryDirectory() as d:
            out, parent = self.build(d, n3_means={(42, 139): 38.0, (43, 139): 38.1, (44, 139): 38.9, (42, 42): 38.8})
            entry = en.summary_data(out, parent)["n3"]["stages"]["139"]
            self.assertAlmostEqual(entry["gain_vs_m4"], 39.0 - (38.0 + 38.1 + 38.9) / 3, places=9)
            self.assertEqual((entry["seeds_better_than_m4"], entry["seeds"]), (3, 3))
            self.assertEqual(entry["verdict"], "M4 초과 단서")
            self.assertEqual(entry["per_seed"][42]["matched_8"], 38.0)
        with tempfile.TemporaryDirectory() as d:
            out, parent = self.build(d, n3_means={(42, 139): 39.1, (43, 139): 39.0, (44, 139): 38.9})
            self.assertTrue(en.summary_data(out, parent)["n3"]["stages"]["139"]["verdict"].startswith("M4 수준"))
        with tempfile.TemporaryDirectory() as d:
            out, parent = self.build(d, n3_means={(42, 139): 39.6, (43, 139): 39.5, (44, 139): 39.7})
            self.assertEqual(en.summary_data(out, parent)["n3"]["stages"]["139"]["verdict"], "M4보다 나쁨")
        with tempfile.TemporaryDirectory() as d:
            out, parent = self.build(d, n3_means={(42, 139): 38.0, (43, 139): 38.0})
            self.assertIn("모두 끝나지 않아", en.summary_data(out, parent)["n3"]["stages"]["139"]["verdict"])

    def test_empty_output_still_renders(self):
        with tempfile.TemporaryDirectory() as d:
            parent, out = Path(d) / "parent", Path(d) / "night"
            write_parent(parent)
            make_ledger(out)
            args = SimpleNamespace(out=str(out), parent=str(parent))
            self.assertEqual(en.cmd_summary(args), 0)
            self.assertTrue((out / "summary.md").read_text().startswith("# WEIGHT1_NIGHT"))


if __name__ == "__main__":
    unittest.main()
