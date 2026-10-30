"""WEIGHT1 사전 판정·짝 비교·재표집·예산 경계. SUMO 실행 없음."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import explore_weight as w
from explore_weight_core import LEFTS


def rows(part, gain=0.0):
    result = []
    for i, (d, p) in enumerate(w.groups(part)):
        for seed in w.config()[part]["eval_seeds"]:
            n = 100 + i
            delay = 30 + i + seed % 3 - gain
            result.append({"date": d, "period": p, "seed": seed, "vehicles": n,
                           "total_delay": delay * n, "actual_total_delay": delay * n, "mean_delay_s": delay,
                           "long_waits": 0, "missing": 0, "unfinished": 0, "changed_by_rules": 0,
                           "collisions_unexplained": 0, "cost_check": {"passed": True},
                           "route_sha256": f"route-{d}-{p}", "cohort_sha256": f"cohort-{d}-{p}",
                           "by_movement": {"->".join(m): {"vehicles": 10, "mean_delay_s": 90 - gain} for m in LEFTS.values()}})
    return result


class DecisionTest(unittest.TestCase):
    def test_config_matches_the_implemented_algorithm(self):
        w.validate_implementation(w.config())
        changed = copy.deepcopy(w.config())
        changed["learning"]["gamma"] = .9
        with self.assertRaises(ValueError):
            w.validate_implementation(changed)

    def test_small_positive_pilot_allowed_and_zero_is_not(self):
        ref = rows("development")
        self.assertTrue(w.pilot_reading(rows("development", 0.05), {"P0": ref, "F": ref})["passed"])
        self.assertFalse(w.pilot_reading(ref, {"P0": ref, "F": ref})["passed"])

    def test_pilot_needs_three_distinct_groups(self):
        candidate = rows("development", -0.01)
        for m in candidate[:8]:
            m["total_delay"] = m["actual_total_delay"] = 0
        self.assertFalse(w.pilot_reading(candidate, {"P0": rows("development")})["passed"])

    def test_pairs_not_matching_by_list_order(self):
        a, b = rows("development"), list(reversed(rows("development")))
        w.verify_pairs(a, b)
        b[0]["route_sha256"] = "changed"
        with self.assertRaises(ValueError):
            w.verify_pairs(a, b)
        with self.assertRaises(ValueError):
            w.verify_pairs(a, a + a[:1])

    def test_guards_use_both_lefts_and_long_rate(self):
        candidate = rows("development", 1)
        key = "->".join(LEFTS["left_south"])
        for m in candidate:
            m["by_movement"][key]["mean_delay_s"] = 96
        result = w.pilot_reading(candidate, {"P0": rows("development")})
        self.assertFalse(result["passed"])
        self.assertFalse(result["comparisons"]["P0"]["checks"]["left_south"])

    def test_normal_forcing_is_not_a_quality_error(self):
        row = rows("development")[0]
        row["rule_decided"] = {"vehicle": 10, "hold": 400}
        self.assertEqual(w.valid_metrics(row), [])
        row["mode_action_errors"] = 1
        self.assertIn("mode_action_errors", w.valid_metrics(row))

    def test_confirmation_small_consistent_gain_and_reproducible_ci(self):
        refs = {"P0": rows("confirmation"), "F": rows("confirmation", .01)}
        candidates = {s: rows("confirmation", .1 + (s - 42) * .001) for s in range(42, 47)}
        first = w.confirmation_reading(candidates, refs, repeats=200)
        self.assertTrue(first["passed"])
        self.assertEqual(first, w.confirmation_reading(candidates, refs, repeats=200))
        self.assertLess(first["comparisons"]["P0"]["gain_s"], 1.3)

    def test_confirmation_does_not_select_only_best_models(self):
        candidates = {s: rows("confirmation", .1) for s in range(42, 47)}
        candidates[45], candidates[46] = rows("confirmation", -.001), rows("confirmation", -.001)
        result = w.confirmation_reading(candidates, {"P0": rows("confirmation")}, repeats=100)
        self.assertFalse(result["comparisons"]["P0"]["checks"]["four_models_positive"])
        with self.assertRaises(ValueError):
            w.confirmation_reading({42: candidates[42]}, {"P0": rows("confirmation")})

    def test_confirmation_rejects_missing_runs(self):
        candidates = {s: rows("confirmation", .1) for s in range(42, 47)}
        candidates[42].pop()
        with self.assertRaises(ValueError):
            w.confirmation_reading(candidates, {"P0": rows("confirmation")})


class BudgetTest(unittest.TestCase):
    def test_failed_jobs_and_resume_are_charged(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = w.Budget(Path(tmp), initial=True)
            b.charge("failed", 100, "failed")
            b.charge("ok", 200)
            resumed = w.Budget(Path(tmp), initial=True)
            self.assertEqual(resumed.used, 300)
            self.assertEqual(resumed.initial_used, 300)
            resumed.reserved = resumed.remaining
            with self.assertRaises(w.BudgetExceeded):
                resumed.check(1)

    def test_initial_and_total_limits_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = w.Budget(Path(tmp), initial=True)
            b.charge("initial", 21600)
            with self.assertRaises(w.BudgetExceeded):
                b.check()
            b.initial = False
            self.assertEqual(b.remaining, 172800 - 21600)

    def test_interrupted_job_cannot_escape_budget_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = w.Budget(Path(tmp), initial=True)
            b.begin("completed_cache_but_parent_died", 600)
            resumed = w.Budget(Path(tmp), initial=False)
            self.assertEqual(resumed.used, 600)
            self.assertEqual(resumed.initial_used, 600)
            self.assertEqual(resumed.entries[0]["status"], "interrupted_conservative_cap")
            self.assertFalse(resumed.pending)

    def test_parallel_reservations_and_atomic_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = w.Budget(Path(tmp), initial=True)
            a, c = b.begin("a", 100), b.begin("c", 200)
            self.assertEqual(b.reserved, 300)
            b.finish(a, 50)
            self.assertEqual(b.used, 50)
            self.assertEqual(b.reserved, 200)
            b.finish(c, 60, "failed")
            self.assertEqual(w.Budget(Path(tmp)).used, 110)

    def test_cannot_train_before_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            w.write(out / "check.json", {"passed": False})
            with self.assertRaises(ValueError), patch.object(w, "train") as train:
                w.pilot(out, out, "fp", w.Budget(out), 1)
            train.assert_not_called()

    def test_expansion_forecast_uses_all_models_and_evaluations(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = w.Budget(Path(tmp))
            b.charge("train42/episode1", 500)
            b.charge("development/M0", 100)
            result = w.forecast(Path(tmp), b)
            self.assertFalse(result["passed"])
            self.assertEqual(result["training_episodes"], 5 * 139 - 42)
            self.assertEqual(result["max_confirmation_runs"], 448)


if __name__ == "__main__":
    unittest.main()
