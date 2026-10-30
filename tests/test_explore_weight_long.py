"""Long-study endpoint, pairing, budget and orchestration: no SUMO execution."""
import copy
from concurrent.futures import Future
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import explore_weight_long as long
import explore_weight as w
from explore_weight_core import LEFTS


def rows(section, gain=0):
    cfg = w.read(long.CONFIG)
    seeds = cfg[section]["eval_seeds"] if section == "confirmation" else cfg[section]["primary_seeds"]
    result = []
    for i, (date, period) in enumerate(long.groups(section)):
        for seed in seeds:
            n, delay = 100+i, 30+i+(seed % 3)-gain
            result.append({"date":date,"period":period,"seed":seed,"vehicles":n,
                           "total_delay":n*delay,"actual_total_delay":n*delay,"mean_delay_s":delay,
                           "long_waits":0,"cost_check":{"passed":True},
                           "route_sha256":f"route-{date}-{period}-{seed}",
                           "cohort_sha256":f"cohort-{date}-{period}-{seed}",
                           "by_movement":{"->".join(m):{"vehicles":10,"mean_delay_s":90-gain} for m in LEFTS.values()}})
    return result


class Decisions(unittest.TestCase):
    def test_development_needs_two_models_and_both_baselines(self):
        refs = {"P0":rows("development"),"M4":rows("development",.02)}
        candidates = {s:rows("development",.1) for s in (42,43,44)}
        self.assertTrue(long.development_reading(candidates,refs)["passed"])
        candidates[43] = rows("development",-.001)
        candidates[44] = rows("development",-.001)
        self.assertFalse(long.development_reading(candidates,refs)["passed"])
        with self.assertRaises(ValueError):long.development_reading({42:candidates[42]},refs)
        with self.assertRaises(ValueError):long.development_reading(candidates,{"P0":refs["P0"]})

    def test_pair_mismatch_and_quality_error_cannot_pass(self):
        candidates = {s:rows("development",1) for s in (42,43,44)}
        refs = {"P0":rows("development"),"M4":rows("development")}
        candidates[42][0]["cohort_sha256"] = "changed"
        with self.assertRaises(ValueError):long.development_reading(candidates,refs)
        candidates[42] = rows("development",1)
        candidates[42][0]["collisions_unexplained"] = 1
        self.assertFalse(long.development_reading(candidates,refs)["passed"])

    def test_final_small_improvement_crossed_ci_is_reproducible(self):
        cfg = w.read(long.CONFIG)
        refs = {"P0":rows("confirmation"),"M4":rows("confirmation",.01)}
        candidates = {s:rows("confirmation",.1+(s-42)*.001) for s in cfg["train_seeds"]}
        first = long.confirmation_reading(candidates,refs,cfg,repeats=200)
        self.assertTrue(first["passed"])
        self.assertEqual(first,long.confirmation_reading(candidates,refs,cfg,repeats=200))
        self.assertGreater(first["comparisons"]["M4"]["ci95_s"][0],0)
        self.assertLess(first["comparisons"]["P0"]["gain_s"],1.3)
        candidates[44] = rows("confirmation",-.001)
        self.assertFalse(long.confirmation_reading(candidates,refs,cfg,repeats=200)["passed"])
        candidates[44].pop()
        with self.assertRaises(ValueError):long.confirmation_reading(candidates,refs,cfg,repeats=20)


class BudgetAndExecution(unittest.TestCase):
    def test_parallel_cost_and_unclean_resume_keep_previous_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = {"compute_hours":96,"prior_compute_s":21060}
            budget = long.Ledger(tmp,record)
            first,second = budget.begin("seed42",100),budget.begin("seed43",150)
            budget.finish(first,60,"complete")
            resumed = long.Ledger(tmp,record)
            self.assertEqual(resumed.used,21060+60+150)
            self.assertFalse(resumed.pending)
            self.assertEqual(resumed.entries[-1]["status"],"interrupted_conservative_cap")
            with self.assertRaises(long.StudyStopped):resumed.begin("too-large",96*3600)

    def test_resource_profile_is_frozen_before_scores(self):
        with self.assertRaises(long.StudyStopped):long.choose_profile(1)
        self.assertEqual(long.choose_profile(2)["primary_episodes"],278)
        self.assertEqual(long.choose_profile(3)["primary_episodes"],417)
        with tempfile.TemporaryDirectory() as tmp:
            source,out = Path(tmp)/"parent",Path(tmp)/"long"
            source.mkdir();w.write(source/"budget.json",{"used_s":18000})
            with patch("explore_weight_long_train.validate_parent",return_value={"fingerprint":"parent","checkpoint_sha256":"model"}),patch.object(w,"configure"),patch.object(long,"available_cpus",return_value=2):
                record,_ = long.register(source,out,"auto",24,96,check_runtime=False)
            with patch("explore_weight_long_train.validate_parent",return_value={"fingerprint":"parent","checkpoint_sha256":"model"}),patch.object(w,"configure"),patch.object(long,"available_cpus",return_value=4):
                resumed,_ = long.register(source,out,"auto",24,96,check_runtime=False)
            self.assertEqual(record,resumed)
            self.assertEqual(record["profile"]["primary_episodes"],278)
            forecast = w.read(out/"forecast.json")
            self.assertEqual(forecast["confirmation_runs_if_gate_passes"],320)
            self.assertLess(forecast["conservative_total_s"],96*3600)

    def test_chunked_scheduler_reaches_all_fixed_endpoints(self):
        calls=[]
        class ImmediatePool:
            def __init__(self,**kwargs):pass
            def submit(self,fn,job):
                calls.append(copy.deepcopy(job))
                future=Future();future.set_result({"completed_episodes":job["episodes"][-1],"seconds":.01,"status":"complete"})
                return future
            def shutdown(self,**kwargs):pass
        cfg = w.read(long.CONFIG)
        record={"config":cfg,"profile":{"milestones":[42,139]},"fingerprint":"study","parent_fingerprint":"parent",
                "compute_hours":96,"prior_compute_s":21060}
        with tempfile.TemporaryDirectory() as tmp,patch.object(long,"ProcessPoolExecutor",ImmediatePool),patch.object(long,"checkpoint_episodes",return_value=None):
            budget = long.Ledger(tmp,record)
            long.train_all(Path(tmp)/"source",Path(tmp),Path(tmp)/"work",record,2,budget,time.time()+10000)
        for seed in cfg["train_seeds"]:
            jobs=[j for j in calls if j["seed"]==seed]
            self.assertEqual(jobs[-1]["episodes"],[139])
            previous=42 if seed==42 else 0
            for j in jobs:
                self.assertLessEqual(j["episodes"][0]-previous,42)
                self.assertEqual(j["preserve_milestone"],j["episodes"][0] in (42,139))
                previous=j["episodes"][0]
        self.assertEqual({j["seed"] for j in calls[:2]},{43,44})

    def test_parent_cache_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp);w.write(source/"experiment.json",{"fingerprint":"parent"})
            date,period=long.groups("development")[0]
            target=source/"runs/development"/f"M0_{date}_{period}_s42";target.mkdir(parents=True)
            w.write(target/"metrics.json",rows("development")[0]);(target/"decisions.jsonl.gz").write_bytes(b"trace")
            w.write(target/"cache.json",{"fingerprint":"parent","metrics_sha256":"wrong","decisions_sha256":w.file_hash(target/"decisions.jsonl.gz")})
            with self.assertRaises(ValueError):long.reference_rows(source,"M0",[42])


if __name__=="__main__":unittest.main()
