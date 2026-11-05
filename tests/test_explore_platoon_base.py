"""[추가 탐색] PBASE 검사 — SUMO 실행 없음(실제 신호 자료·가짜 실행 기록)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import arena3_common as ac  # noqa: E402
import explore_platoon0 as p0  # noqa: E402
import explore_platoon_base as pb  # noqa: E402
import explore_weight as w  # noqa: E402
import focused_scenario as fs  # noqa: E402
import focused_train as ft  # noqa: E402
from explore_weight_episode import CONTROL_S, TAIL_S  # noqa: E402
from focused_data import COUNTS, load_counts, variant  # noqa: E402

w.configure()
CFG = p0.config()


class FixedProgramTest(unittest.TestCase):
    def test_hands_over_on_the_same_green_as_other_controls(self):
        """준비 30분은 다른 제어와 같은 실제 신호이고, 넘겨받는 순간 보정 고정신호도 같은 녹색이다(42개 날짜·시간대)."""
        counts = load_counts(COUNTS)
        seen = set()
        with tempfile.TemporaryDirectory() as d:
            for s in CFG["situations"]:
                if (s["date"], s["period"]) in seen:
                    continue
                seen.add((s["date"], s["period"]))
                start = ft.warm_start(s["date"], s["period"], counts, Path(d) / "r.rou.xml")
                h = start["handover_s"]
                states, log = pb.fixed_states(s["date"], s["period"], h + CONTROL_S + TAIL_S)
                raw = fs.real_signal_states(s["date"], fs.period_start_s(s["period"]) - ft.WARMUP_S, h)
                self.assertEqual(raw, start["warmup_states"])
                self.assertEqual(states[h], start["handover_state"])
                self.assertEqual(len(states), h + CONTROL_S + TAIL_S)
                self.assertEqual(log["shortened_below_rule"], 0)
        self.assertEqual(len(seen), 42)

    def test_all_red_is_inserted_after_yellow(self):
        """보정은 전환 규칙 전적색을 황색 뒤에 넣는다(전적색 상태 = 황색의 y를 r로 바꾼 것)."""
        self.assertTrue(variant()["all_red"])
        s = CFG["situations"][0]
        states, log = pb.fixed_states(s["date"], s["period"], 3600)
        self.assertGreater(log["with_all_red"], 0)
        self.assertGreater(log["inserted_s"], 0)
        reds = [i for i in range(1, len(states)) if "y" in states[i - 1] and states[i] == fs.all_red_state(states[i - 1])]
        self.assertTrue(reds)


class JobsTest(unittest.TestCase):
    def test_jobs_cover_three_runs_per_situation(self):
        jobs = pb.jobs_for(CFG, Path("/o"), Path("/k"), "fp")
        self.assertEqual(len(jobs), 252)
        self.assertEqual(len({j["target"] for j in jobs}), 252)
        self.assertEqual({(j["world"], j["policy"]) for j in jobs}, set(pb.RUNS))
        for j in jobs[:3]:
            s = next(x for x in CFG["situations"] if ac.situation_id(x) == j["sid"])
            self.assertEqual(j["route_sha256"], s["route_sha256"][j["world"]])
            self.assertEqual(j["model_identity"], f"{j['world']}/{j['policy']}")


class ReportTest(unittest.TestCase):
    def _run(self, folder: Path, sid: str, route: str, delay: float, vehicles: int = 10):
        d = folder / sid
        d.mkdir(parents=True)
        m = {"situation": sid, "route_sha256": route, "actual_total_delay": delay, "vehicles": vehicles,
             "long_waits": 0, "by_movement": {}}
        (d / "metrics.json").write_text(json.dumps(m))
        (d / "cache.json").write_text(json.dumps({"metrics_sha256": w.file_hash(d / "metrics.json")}))

    def test_read_folder_checks_hash_and_route(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d) / "FIXC"
            self._run(folder, "a", "sha_a", 100.0)
            rows = pb.read_folder(folder, ["a"], {"a": "sha_a"}, "t")
            self.assertEqual(rows[0]["actual_total_delay"], 100.0)
            with self.assertRaises(ValueError):
                pb.read_folder(folder, ["a"], {"a": "other"}, "t")
            (folder / "a" / "metrics.json").write_text(json.dumps({"situation": "a"}))
            with self.assertRaises(ValueError):
                pb.read_folder(folder, ["a"], {"a": "sha_a"}, "t")

    def test_policy_summary_averages_models(self):
        mk = lambda delay: {"vehicles": 10, "actual_total_delay": delay, "long_waits": 1,  # noqa: E731
                            "by_movement": {k: {"vehicles": 1, "mean_delay_s": delay / 10} for k in pb.ref.LEFT_KEYS.values()},
                            "missing": 0, "unfinished": 0, "cost_check": {"passed": True}, "mean_delay_s": delay / 10}
        s = pb.policy_summary([[mk(100.0), mk(200.0)], [mk(300.0), mk(400.0)]])
        self.assertAlmostEqual(s["mean_delay_s"], (200.0 + 300.0) / 20)
        self.assertAlmostEqual(s["left_turns_s"]["left_east"], 25.0)
        self.assertEqual(s["long_waits"], 2)
        self.assertEqual(s["quality_errors"], 0)


if __name__ == "__main__":
    unittest.main()
