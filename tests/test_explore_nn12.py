"""[추가 탐색] NN12 검사 — SUMO 실행 없음."""
import gzip
import json
import sys
import tempfile
import unittest
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import explore_nn12 as n12  # noqa: E402
import explore_nn12_package as pkg  # noqa: E402
import explore_platoon1 as pt  # noqa: E402
import explore_pred as ep  # noqa: E402
import explore_weight as w  # noqa: E402
import explore_weight_core as wc  # noqa: E402
import focused_eval  # noqa: E402
import network_eval as ne  # noqa: E402
import network_rl as nr  # noqa: E402

w.configure()
CFG = n12.config()


class ConfigTest(unittest.TestCase):
    def test_fingerprints_match_earlier_runs(self):
        self.assertEqual(ne.evaluation_fingerprint(), CFG["evaluation_fingerprint"])
        self.assertEqual(nr._code_fingerprint(), CFG["network12_fingerprint"])
        self.assertEqual(CFG["evaluation_fingerprint"], "9d3b7504064b")

    def test_windows_and_seeds(self):
        self.assertEqual(len(CFG["windows"]), 46)
        self.assertEqual(CFG["seeds"], [42, 43, 44, 45, 46])
        self.assertEqual({d for d, _ in CFG["windows"]}, {"20220928", "20220813"})

    def test_models_are_the_nn1_start_models(self):
        self.assertTrue(CFG["models"]["N5"]["identities"]["42"].startswith("fd78e01fac2d"))
        self.assertTrue(CFG["models"]["Rn"]["identities"]["42"].startswith("f5aa1f8b84e3"))
        self.assertEqual(CFG["models"]["N5"]["world"], "P")
        self.assertEqual(CFG["models"]["Rn"]["world"], "R")

    def test_correction_is_recorded(self):
        self.assertEqual(CFG["correction"], n12.CORRECTION)
        self.assertEqual(len(CFG["correction"]["earlier_code"]), 64)
        self.assertNotEqual(CFG["correction"]["earlier_code"], n12.code_fingerprint())
        self.assertIn(CFG["correction"]["recheck"]["window"], CFG["windows"])          # 다시 돌려 비교할 판은 끝난 조각이다
        self.assertNotEqual(CFG["correction"]["recheck"]["window"][1], CFG["correction"]["rerun_hour"])

    def test_neighbor_labels_are_one_to_one_and_keep_the_meaning(self):
        labels = CFG["neighbor_labels"]
        self.assertEqual(len(labels), 8)
        self.assertEqual(len(set(labels.values())), 8)
        main, other = pt.release_states()
        inverse = {b: a for a, b in labels.items()}
        straight, left = [5, 6, 7, 8, 9, 10], [16, 17]                 # 12개 지도 204822에서 석천사거리 쪽 직진·좌회전 연결
        self.assertTrue(all(inverse[main][i] == "G" for i in straight))
        self.assertTrue(all(inverse[other][i] == "G" for i in left))

    def test_neighbor_labels_from_the_raw_signals(self):
        self.assertEqual(n12.neighbor_labels(("20220813",)), CFG["neighbor_labels"])


class PlanTest(unittest.TestCase):
    def test_jobs_cover_every_window_and_seed(self):
        with tempfile.TemporaryDirectory() as d:
            for name in n12.POLICIES:
                jobs = n12.plan_jobs(name, CFG, Path(d) / "out", Path(d) / "work", Path(d) / "models")
                self.assertEqual(len(jobs), 230)
                self.assertEqual(len({j["target"] for j in jobs}), 230)
                self.assertEqual(len({j["work"] for j in jobs}), 230)

    def test_models_rotate_evenly(self):
        with tempfile.TemporaryDirectory() as d:
            jobs = n12.plan_jobs("N5", CFG, Path(d), Path(d), Path(d))
        total = Counter(j["model_seed"] for j in jobs)
        self.assertEqual(sorted(total), [42, 43, 44])
        self.assertLessEqual(max(total.values()) - min(total.values()), 1)
        for date, hour in CFG["windows"]:
            here = Counter(j["model_seed"] for j in jobs if (j["date"], j["hour"]) == (date, hour))
            self.assertEqual(sum(here.values()), 5)
            self.assertEqual(len(here), 3)                               # 조각마다 모델 셋을 모두 쓴다

    def test_planners_have_no_model_and_inputs_saved_only_for_one_seed(self):
        with tempfile.TemporaryDirectory() as d:
            planner = n12.plan_jobs("TM4", CFG, Path(d), Path(d))
            net = n12.plan_jobs("Rn", CFG, Path(d), Path(d), Path(d))
        self.assertTrue(all("model_dir" not in j and not j["save_inputs"] for j in planner))
        self.assertEqual(sum(j["save_inputs"] for j in net), 46)
        self.assertTrue(all(j["eval_seed"] == 42 for j in net if j["save_inputs"]))

    def test_finished_refuses_results_of_another_code_or_model(self):
        with tempfile.TemporaryDirectory() as d:
            job = {"target": str(Path(d) / "t"), "model_identity": "m"}
            self.assertFalse(n12.finished(job, "c"))
            w.write(Path(d) / "t" / "metrics.json", {"nn12": {"code": "c", "model_identity": "m"}})
            self.assertTrue(n12.finished(job, "c"))
            with self.assertRaises(ValueError):
                n12.finished(job, "other")
            with self.assertRaises(ValueError):
                n12.finished({**job, "model_identity": "x"}, "c")
            self.assertTrue(n12.finished(job, "other", ("c",)))          # 정정 기록이 그대로 쓰기로 한 이전 코드만 받는다
            with self.assertRaises(ValueError):
                n12.finished(job, "other", ("another",))

    def test_run_policy_resumes_and_records_failures(self):
        calls = Counter()

        def worker(job):
            calls[job["tag"]] += 1
            if job["tag"] == "20220928_01_eval42" and calls[job["tag"]] == 1:
                return {"ok": False, "tag": job["tag"], "seconds": 1.0, "error": "x"}
            w.write(Path(job["target"]) / "metrics.json", {"nn12": {"code": job["code"], "model_identity": job.get("model_identity")}})
            return {"ok": True, "tag": job["tag"], "seconds": 1.0}

        with tempfile.TemporaryDirectory() as d, ThreadPoolExecutor(4) as pool:
            out = Path(d) / "M4"
            first = n12.run_policy("M4", out, Path(d) / "work", 4, CFG, worker=worker, executor=pool)
            self.assertEqual((first["done_now"], len(first["failed"])), (229, 1))
            self.assertTrue((out / "runs" / "20220928_01_eval42.failed.json").exists())
            second = n12.run_policy("M4", out, Path(d) / "work", 4, CFG, worker=worker, executor=pool)
            self.assertEqual((second["done_now"], second["failed"]), (1, []))
            self.assertEqual(sum(calls.values()), 231)

    def test_run_policy_keeps_results_of_the_earlier_code(self):
        earlier = CFG["correction"]["earlier_code"]
        calls = []

        def worker(job):
            calls.append(job["tag"])
            w.write(Path(job["target"]) / "metrics.json", {"nn12": {"code": job["code"], "model_identity": job.get("model_identity")}})
            return {"ok": True, "tag": job["tag"], "seconds": 1.0}

        with tempfile.TemporaryDirectory() as d, ThreadPoolExecutor(4) as pool:
            out = Path(d) / "TM4"
            w.write(out / "experiment.json", {"code": "이전 코드의 기록"})
            for job in n12.plan_jobs("TM4", CFG, out, Path(d) / "work"):
                if job["hour"] != 23:
                    w.write(Path(job["target"]) / "metrics.json", {"nn12": {"code": earlier, "model_identity": None}})
            status = n12.run_policy("TM4", out, Path(d) / "work", 4, CFG, worker=worker, executor=pool)
            self.assertEqual((status["done_now"], status["failed"]), (10, []))
            self.assertTrue(all("_23_" in tag for tag in calls))
            self.assertEqual(w.read(out / "experiment.json"), {"code": "이전 코드의 기록"})
            self.assertEqual(w.read(out / "experiment_fix1.json"), n12.code_record())


class FakeTrafficlight:
    def __init__(self, state):
        self.state = state

    def getRedYellowGreenState(self, tls):
        return self.state


def fake_env(now, state="a"):
    return SimpleNamespace(sim_step=float(now), sumo=SimpleNamespace(trafficlight=FakeTrafficlight(state)))


class SeatTest(unittest.TestCase):
    greens = list(n12.aihub_states()["greens"])

    def test_seat_gives_the_place_back_when_built(self):
        cls = type("S", (n12.Seat,), {"date": "20220813", "record": Counter()})
        with n12.seated(cls):
            self.assertIs(focused_eval.QueueFirst, cls)
            focused_eval.QueueFirst({}, self.greens)
            self.assertIs(focused_eval.QueueFirst, n12.ORIGINAL_SEAT)     # 입력·계획기 클래스가 원래 단순 제어를 상속해야 한다
        self.assertIs(focused_eval.QueueFirst, n12.ORIGINAL_SEAT)

    def test_seated_restores_after_error(self):
        with self.assertRaises(RuntimeError):
            with n12.seated(object):
                raise RuntimeError("x")
        self.assertIs(focused_eval.QueueFirst, n12.ORIGINAL_SEAT)

    def test_wrong_green_order_stops(self):
        with self.assertRaises(ValueError):
            n12.Seat({}, list(reversed(self.greens)))

    def test_timetable_compare_uses_the_read_lag(self):
        record = Counter()
        cls = type("S", (n12.Seat,), {"date": "20220813", "origin_s": 100.0, "labels": {"map": "raw"}, "record": record, "check_timetable": True})
        seat = cls({}, self.greens)
        seat._table = ["other"] * 300
        seat._table[100 + 20 - n12.TABLE_READ_LAG_S] = "raw"
        seat.compare_timetable(fake_env(20, "map"))
        self.assertEqual((record["timetable_checked"], record["timetable_mismatch"]), (1, 0))
        seat.compare_timetable(fake_env(25, "map"))
        self.assertEqual((record["timetable_checked"], record["timetable_mismatch"]), (2, 1))

    def test_timetable_compare_counts_decisions_past_midnight(self):
        record = Counter()
        cls = type("S", (n12.Seat,), {"date": "20220813", "origin_s": 81000.0, "labels": {"map": "raw"}, "record": record, "check_timetable": True})
        seat = cls({}, self.greens)
        self.assertEqual(len(seat._table), 86400)
        seat._table = ["raw"] * 86400
        seat.compare_timetable(fake_env(5400, "map"))                    # 하루의 마지막 칸(86,399)
        self.assertEqual((record["timetable_checked"], record["timetable_mismatch"], record["timetable_past_midnight"]), (1, 0, 0))
        for now in (5401, 5406, 5426):                                   # 자정 뒤에는 읽을 칸이 없다
            seat.compare_timetable(fake_env(now, "map"))
        self.assertEqual((record["timetable_checked"], record["timetable_mismatch"], record["timetable_past_midnight"]), (1, 0, 3))

    def test_net_seat_passes_integers_and_sensed_input(self):
        seen = {}
        policy = SimpleNamespace(world="R", act=lambda env, sensed, allowed: seen.update(sensed=sensed, allowed=allowed) or allowed[0])
        cls = type("N", (n12.NetSeat,), {"date": "20220813", "record": Counter(), "policy": policy})
        seat = cls({}, self.greens)
        seat.ready, seat.feature = True, SimpleNamespace(sensed=lambda env: "x")
        self.assertEqual(seat(fake_env(5), [2.0, 3.0]), 2)
        self.assertEqual(seen, {"sensed": "x", "allowed": [2, 3]})

    def test_plan_seat_counts_forced_and_planned(self):
        record = Counter()
        cls = type("P", (n12.PlanSeat,), {"date": "20220813", "record": record, "kind": "M4"})
        seat = cls({}, self.greens)
        seen = []
        seat.ready, seat.core = True, SimpleNamespace(observe=lambda env: seen.append("observe"), decide=lambda env, allowed: allowed[-1])
        self.assertEqual(seat(fake_env(5), [1]), 1)
        self.assertEqual(seat(fake_env(10), [1, 2]), 2)
        self.assertEqual(seen, ["observe", "observe"])                    # 강제 단계에서도 진입을 기록한다
        self.assertEqual((record["forced"], record["planned"]), (1, 1))

    def test_planner_is_blocked_only_inside(self):
        before = ep.plan, wc.weighted_plan
        with n12.nn.planner_free(None):
            with self.assertRaises(RuntimeError):
                ep.plan(None, None, 0.0, [], [], [])
            with self.assertRaises(RuntimeError):
                wc.weighted_plan(None, None, 0.0, [], [], [], [])
        self.assertEqual((ep.plan, wc.weighted_plan), before)

    def test_seat_class_checks_the_timetable_only_for_platoon_policies(self):
        for name, want in (("N5", True), ("TM4", True), ("Rn", False), ("M4", False)):
            cls = n12.seat_class(n12.PlanSeat, {"name": name, "date": "20220813", "hour": 7}, {}, Counter(), kind=name)
            self.assertEqual(cls.check_timetable, want)
            self.assertEqual(cls.origin_s, 7 * 3600 - nr.WARMUP_S)


def metrics(delay, vehicles=100, network=None, name=None, seed=None, long_delays=0, moves=None):
    m = {"evaluation_fingerprint": n12.EVALUATION_FP, "network12_fingerprint": n12.NETWORK_FP,
         "window_204820": {"mean_delay_s": delay, "vehicles": vehicles, "long_delays": long_delays, "unfinished": 0, "by_movement": moves or {}},
         "collisions_in_204820": 0, "collisions_total": 0, "collisions_204822": 0, "guard": {"changed_by_rules": {"ped": 0, "vehicle": 0, "max_green": 0}}, "action_counts": {"0": 1}}
    if network is not None:
        m.update(network={"mean_delay_s": network, "vehicles": 1000, "long_delays": 2, "not_inserted": 1}, via_204822={"mean_delay_s": network + 5})
    if name is not None:
        m["nn12"] = {"code": "c", "job_seconds": 100.0, "model_seed": seed, "stats": {"planned": 50, "z5_cells": 5, "timetable_mismatch": 0}}
    return m


DELAYS = {"FIXC": 56.0, "B12": 54.0, "P0": 51.0, "P2": 49.0, "M4": 50.0, "TM4": 49.5, "Rn": 50.5, "N5": 50.0}


def fake_root(root: Path, skip=None, bad_vehicles=False):
    joint2, pred, b12, out = root / "joint2", root / "pred", root / "b12", root / "out"
    for date, hour in CFG["windows"]:
        window = f"{date}_{int(hour):02d}"
        for j, seed in enumerate(CFG["seeds"]):
            shift = 0.1 * (int(hour) % 5)
            w.write(joint2 / n12.EVALUATION_FP / "windows" / window / f"fixedc_eval{seed}" / "metrics.json", metrics(DELAYS["FIXC"] + shift, network=90.0))
            w.write(joint2 / n12.EVALUATION_FP / "windows" / window / f"p2_eval{seed}" / "metrics.json", metrics(DELAYS["P2"] + shift, network=89.0))
            w.write(pred / n12.EVALUATION_FP / "windows" / window / f"pred_p0_eval{seed}" / "metrics.json", metrics(DELAYS["P0"] + shift))
            w.write(b12 / n12.EVALUATION_FP / "windows" / window / f"dqn_seed{seed}_{CFG['b12_models'][str(seed)]}_eval{seed}" / "metrics.json", metrics(DELAYS["B12"] + shift))
            for name in n12.POLICIES:
                if name == skip:
                    continue
                m = metrics(DELAYS[name] + shift, network=88.0, name=name, seed=42 + j % 3 if name in n12.MODELS else None)
                rerun = name in CFG["correction"]["rerun_policies"] and int(hour) == CFG["correction"]["rerun_hour"]
                m["nn12"]["code"] = n12.code_fingerprint() if rerun else CFG["correction"]["earlier_code"]
                if bad_vehicles and name == "N5":
                    m["network"]["vehicles"] = 999
                w.write(out / name / "runs" / n12.run_tag(date, hour, seed) / "metrics.json", m)
    return out, joint2, pred, b12


class ReadTest(unittest.TestCase):
    def test_weighted_delay_counts_cars(self):
        rows = [metrics(10.0, 100), metrics(40.0, 300)]
        self.assertAlmostEqual(n12.weighted_delay(rows), 32.5)

    def test_movement_delay(self):
        rows = [metrics(0, moves={"a->b": {"vehicles": 10, "mean_delay_s": 20.0}}), metrics(0, moves={"a->b": {"vehicles": 30, "mean_delay_s": 40.0}}), metrics(0)]
        self.assertAlmostEqual(n12.movement_delay(rows, ("a", "b")), 35.0)
        self.assertIsNone(n12.movement_delay(rows, ("x", "y")))

    def test_per_window_averages_seeds(self):
        runs = {("w1", 42): metrics(10.0), ("w1", 43): metrics(20.0), ("w2", 42): metrics(30.0)}
        self.assertEqual(n12.per_window(runs, n12.segment), {"w1": 15.0, "w2": 30.0})

    def test_paired_sign_and_drop(self):
        a, b = {"w1": 10.0, "w2": 12.0, "w3": 11.0, "x": 0.0}, {"w1": 12.0, "w2": 13.0, "w3": 13.0, "x": 100.0}
        r = n12.paired(a, b, drop="x")
        self.assertEqual(r["n"], 3)
        self.assertAlmostEqual(r["mean"], 5.0 / 3.0)
        self.assertEqual(r["positive"], 3)
        with self.assertRaises(ValueError):
            n12.paired({"w1": 1.0}, {"w2": 1.0})

    def _table(self, **delays):
        base = {"P2": 48.69, "B12": 53.76, "P0": 51.15, "TM4": 50.0, "M4": 50.0, "N5": 50.0, "Rn": 51.0}
        base.update(delays)
        return {k: {"weighted_delay_s": v} for k, v in base.items()}

    def test_reading_branches(self):
        rules = CFG["rules"]
        self.assertEqual(n12.reading(self._table(N5=48.0), rules)["action"], "continue_single")
        self.assertEqual(n12.reading(self._table(N5=50.89 - 1e-9, Rn=52.0), rules)["action"], "continue_single")
        self.assertEqual(n12.reading(self._table(N5=50.90, Rn=52.0), rules)["action"], "labels_on_12map")
        self.assertEqual(n12.reading(self._table(N5=53.76, Rn=53.80), rules)["action"], "stop")
        self.assertEqual(n12.reading(self._table(N5=60.0, Rn=52.0), rules)["best"], "Rn")

    def test_transfer_reading(self):
        t = n12.reading(self._table(N5=52.3, TM4=50.0, Rn=52.2, M4=52.0), CFG["rules"])["transfer"]
        self.assertFalse(t["N5"]["transferred"])
        self.assertTrue(t["Rn"]["transferred"])
        self.assertTrue(t["Rn"]["partner_longer_than_p0"])
        self.assertFalse(t["N5"]["partner_longer_than_p0"])

    def test_report_end_to_end_without_sumo(self):
        with tempfile.TemporaryDirectory() as d:
            out, joint2, pred, b12 = fake_root(Path(d))
            report = n12.collect_report(out, joint2, pred, b12, CFG)
            self.assertEqual(report["status"], "done")
            self.assertAlmostEqual(report["table"]["FIXC"]["reduction_vs_fixc_pct"], 0.0)
            self.assertLess(report["table"]["N5"]["weighted_delay_s"], report["table"]["Rn"]["weighted_delay_s"])
            self.assertEqual(report["reading"]["best"], "N5")
            self.assertEqual(report["reading"]["action"], "continue_single")
            self.assertAlmostEqual(report["pairs"]["N5-FIXC"]["segment"]["mean"], 6.0)
            self.assertAlmostEqual(report["pairs"]["N5-FIXC"]["network"]["mean"], 2.0)
            self.assertNotIn("network", report["pairs"]["N5-B12"])
            self.assertEqual(report["pairs"]["N5-FIXC"]["segment_without_unstable"]["n"], 45)
            self.assertIsNone(report["bands"]["저녁 16~18시"]["B12"]["network_vs_fixc_s"])
            self.assertEqual(report["table"]["N5"]["z5_cells_per_decision"], 0.1)
            self.assertEqual(sum(x["runs"] for x in report["table"]["N5"]["per_model"].values()), 230)
            self.assertEqual(report["code"], {"rerun": n12.code_fingerprint(), "earlier": CFG["correction"]["earlier_code"]})
            text = n12.render(report)
            self.assertIn("후보 N5", text)
            self.assertIn("옮겨짐", text)
            n12.save_report(Path(d) / "judge", report)
            self.assertTrue((Path(d) / "judge" / "summary.md").exists())
            json.loads((Path(d) / "judge" / "judge.json").read_text())

    def test_report_incomplete_lists_missing(self):
        with tempfile.TemporaryDirectory() as d:
            out, joint2, pred, b12 = fake_root(Path(d), skip="M4")
            report = n12.collect_report(out, joint2, pred, b12, CFG)
            self.assertEqual(report, {"experiment": n12.EXPERIMENT, "status": "incomplete", "missing": {"M4": 230}})
            self.assertIn("아직 끝나지 않았습니다", n12.render(report))

    def test_report_refuses_runs_of_the_wrong_code(self):
        for name, tag in (("Rn", "20220928_23_eval42"), ("N5", "20220928_22_eval42"), ("TM4", "20220928_23_eval42")):
            with self.subTest(name=name, tag=tag), tempfile.TemporaryDirectory() as d:
                out, joint2, pred, b12 = fake_root(Path(d))
                path = out / name / "runs" / tag / "metrics.json"
                m = w.read(path)
                m["nn12"]["code"] = "다른 코드"
                w.write(path, m)
                with self.assertRaises(ValueError):
                    n12.collect_report(out, joint2, pred, b12, CFG)

    def test_report_refuses_different_cars(self):
        with tempfile.TemporaryDirectory() as d:
            out, joint2, pred, b12 = fake_root(Path(d), bad_vehicles=True)
            with self.assertRaises(ValueError):
                n12.collect_report(out, joint2, pred, b12, CFG)

    def test_check_rows(self):
        m = {"evaluation_fingerprint": n12.EVALUATION_FP, "network12_fingerprint": n12.NETWORK_FP, "guard": {"changed_by_rules": {"ped": 0}},
             "nn12": {"planner_free": True, "stats": {"planned": 3, "timetable_checked": 4, "timetable_mismatch": 0}}}
        self.assertTrue(all(n12.check_rows("N5", m).values()))
        m["nn12"]["stats"]["timetable_mismatch"] = 1
        self.assertFalse(n12.check_rows("N5", m)["timetable_same"])
        self.assertNotIn("timetable_same", n12.check_rows("Rn", m))
        m["guard"]["changed_by_rules"]["ped"] = 1
        self.assertFalse(n12.check_rows("M4", m)["rules_not_changed"])


class RecheckTest(unittest.TestCase):
    def folder(self, root: Path, delay=50.0, seconds=9.0, rows=b"a,1\n", job_seconds=1.0, inputs=None) -> Path:
        root.mkdir(parents=True)
        w.write(root / "metrics.json", {"seconds": seconds, "window_204820": {"mean_delay_s": delay}, "nn12": {"name": "N5", "code": "c", "job_seconds": job_seconds, "model_seed": 42, "model_identity": "m", "planner_free": True, "stats": {"planned": 3}}})
        with gzip.open(root / "vehicles.csv.gz", "wb") as f:
            f.write(rows)
        if inputs is not None:
            np.savez(root / "inputs.npz", x=np.array(inputs))
        return root

    def test_same_ignores_only_the_time_taken(self):
        with tempfile.TemporaryDirectory() as d:
            old = self.folder(Path(d) / "old", inputs=[1.0, 2.0])
            new = self.folder(Path(d) / "new", seconds=100.0, job_seconds=200.0, inputs=[1.0, 2.0])
            self.assertTrue(all(n12.same_as_stored(new, old).values()))
            self.assertEqual(set(n12.same_as_stored(new, old)), {"metrics", "record", "vehicles", "inputs"})

    def test_any_difference_is_found(self):
        with tempfile.TemporaryDirectory() as d:
            old = self.folder(Path(d) / "old", inputs=[1.0, 2.0])
            self.assertFalse(n12.same_as_stored(self.folder(Path(d) / "a", delay=50.000001, inputs=[1.0, 2.0]), old)["metrics"])
            self.assertFalse(n12.same_as_stored(self.folder(Path(d) / "b", rows=b"a,2\n", inputs=[1.0, 2.0]), old)["vehicles"])
            self.assertFalse(n12.same_as_stored(self.folder(Path(d) / "c", inputs=[1.0, 2.000001]), old)["inputs"])
            self.assertFalse(n12.same_as_stored(self.folder(Path(d) / "e"), old)["inputs"])          # 입력 파일이 없어도 다름

    def test_recheck_only_for_the_policies_that_failed(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("Rn", "M4"):
                with self.assertRaises(ValueError):
                    n12.recheck(name, None, Path(d), Path(d) / "w", CFG)


class PackageTest(unittest.TestCase):
    def test_notebooks_compile_and_ask_for_eight_cpus(self):
        for name in pkg.NOTEBOOKS:
            nb = pkg.notebook(name)
            self.assertEqual(nb["metadata"]["colab"]["machine_shape"], "hm")
            for c in nb["cells"]:
                if c["cell_type"] == "code":
                    compile("".join(c["source"]), name, "exec")

    def test_policy_notebook_order_and_models(self):
        for name, spec in pkg.SPEC.items():
            text = ["".join(c["source"]) for c in pkg.notebook(name)["cells"]]
            self.assertIn(f'POLICY = "{name}"', text[1])
            steps = [next(i for i, t in enumerate(text) if key in t) for key in ("12개 지도 자료(기존", "test_explore_nn12", '"recheck"', "본 실행(남은 판만)")]
            self.assertEqual(steps, sorted(steps))                      # 자료 → 시험 → 점검 → 본 실행
            self.assertIn("FutureTraffic_NN12_fix1.zip", text[1])
            if spec["models"]:
                self.assertIn(spec["models"], text[1])
            else:
                self.assertIn("MODELS = None", text[1])
        self.assertIn("r1", pkg.SPEC["N5"]["models"])
        self.assertEqual(set(pkg.SPEC), set(CFG["correction"]["rerun_policies"]))     # 실패한 정책만 다시 돈다
        self.assertIn("FutureTraffic_NN12_fix1.zip", "".join("".join(c["source"]) for c in pkg.notebook("judge")["cells"]))

    def test_previous_packages_are_protected(self):
        self.assertIn("FutureTraffic_NN1.zip", pkg.PROTECTED_NAMES)
        self.assertIn("FutureTraffic_JOINT2.zip", pkg.PROTECTED_NAMES)
        self.assertIn("FutureTraffic_NN12.zip", pkg.PROTECTED_NAMES)


if __name__ == "__main__":
    unittest.main()
