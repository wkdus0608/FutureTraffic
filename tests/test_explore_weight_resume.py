"""Versioned resume coordination; fixtures only, no SUMO or real learning."""
from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import explore_weight as w
import explore_weight_long as old
import explore_weight_resume as resume
from explore_weight_core import LEFTS


def fixture(parent):
    source,study,out,work=[Path(parent)/n for n in ("source","study","resume","work")]
    source.mkdir();study.mkdir()
    cfg=w.read(old.CONFIG)
    original={"config":cfg,"profile":old.choose_profile(3),"parent_fingerprint":"parent",
              "compute_hours":96,"code":{}}
    w.write(study/"experiment.json",{"fingerprint":w.digest(original),**original})
    w.write(study/"budget.json",{"used_s":123000,"pending":{}})
    for seed,episode in ((42,333),(43,292),(44,294)):
        d=study/"models"/f"seed_{seed}"
        p=d/"checkpoints"/f"ep_{episode}"
        w.write(p/"manifest.json",{"episodes":episode,"seed":seed})
        w.write(d/"latest.json",{"checkpoint":str(p.relative_to(d)),"manifest_sha256":w.file_hash(p/"manifest.json")})
    return source,study,out,work


def rows(gain=0):
    result=[]
    for date,period in old.groups("development"):
        for seed in (42,43,44,45):
            result.append({"date":date,"period":period,"seed":seed,"vehicles":100,"mean_delay_s":30-gain,
                           "actual_total_delay":100*(30-gain),"total_delay":100*(30-gain),"long_waits":0,
                           "route_sha256":f"r{date}{period}{seed}","cohort_sha256":f"c{date}{period}{seed}",
                           "cost_check":{"passed":True},"by_movement":{"->".join(m):{"vehicles":10,"mean_delay_s":90-gain} for m in LEFTS.values()}})
    return result


class ResumeTests(unittest.TestCase):
    def test_registration_preserves_goal_prior_budget_and_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,study,out,work=fixture(tmp)
            before={str(p.relative_to(study)):p.read_bytes() for p in study.rglob("*") if p.is_file()}
            with patch("explore_weight_long_train.validate_parent",return_value={"fingerprint":"parent"}),patch.object(w,"configure"):
                first=resume.register(source,study,out,check_runtime=False)
                self.assertEqual(first,resume.register(source,study,out,check_runtime=False))
                self.assertEqual(first["profile"]["primary_episodes"],417)
                self.assertEqual(first["prior_compute_s"],123000)
                with self.assertRaises(ValueError):resume.register(source,study,out,compute_hours=48,check_runtime=False)
                w.write(study/"budget.json",{"used_s":123001,"pending":{}})
                with self.assertRaises(ValueError):resume.register(source,study,out,check_runtime=False)
            after={str(p.relative_to(study)):p.read_bytes() for p in study.rglob("*") if p.is_file()}
            self.assertEqual({k:v for k,v in before.items() if k!="budget.json"},{k:v for k,v in after.items() if k!="budget.json"})

    def test_missing_approval_does_not_clone_or_start_learning(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,study,out,work=fixture(tmp)
            record={"fingerprint":"new","compute_hours":96,"prior_compute_s":123000}
            args=["resume","run","--source",str(source),"--study",str(study),"--out",str(out),"--work",str(work)]
            with patch.object(sys,"argv",args),patch.object(sys,"platform","linux"),patch.object(resume,"register",return_value=record),patch.object(resume,"summary"),patch("explore_weight_resume_train.clone_models") as clone,patch.object(old,"train_all") as train,redirect_stdout(io.StringIO()):
                self.assertEqual(resume.main(),78)
                clone.assert_not_called();train.assert_not_called()

    def test_overlap_and_mac_are_rejected_before_simulation(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,study,out,work=fixture(tmp)
            with self.assertRaises(ValueError):resume.safe_paths(source,study,source/"out",work)
            args=["resume","diagnose","--source",str(source),"--study",str(study),"--out",str(out),"--work",str(work)]
            with patch.object(sys,"argv",args),patch.object(sys,"platform","darwin"),patch.object(resume,"register") as register:
                with self.assertRaises(SystemExit):resume.main()
                register.assert_not_called()

    def test_failed_diagnostic_is_charged_and_does_not_approve(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);record={"compute_hours":96,"prior_compute_s":123000}
            ledger=old.Ledger(out,record)
            with patch("explore_weight_collision_probe.run_probe",return_value={"status":"failed","matches":False}):
                with self.assertRaises(RuntimeError):resume.diagnose(out/"source",out/"study",out,out/"work",record,ledger,10**12)
            self.assertEqual(len(ledger.entries),1)
            self.assertFalse(ledger.pending)
            self.assertFalse((out/"approval.json").exists())

    def test_every_baseline_is_re_evaluated_under_new_version(self):
        calls=[]
        def evaluate(out,work,record,workers,ledger,deadline,stage,policies,groups,seeds):
            calls.append((stage,[name for name,_,_ in policies],list(seeds)))
            return {name:[r for r in rows(0 if name in ("P0","M4") else -1) if r["seed"] in seeds] for name,_,_ in policies}
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);cfg=w.read(old.CONFIG)
            record={"config":cfg,"profile":{"milestones":[417],"primary_episodes":417},"parent_fingerprint":"parent"}
            with patch.object(old,"evaluations",side_effect=evaluate),patch.object(old,"reference_rows",side_effect=AssertionError("Old reference reuse forbidden")):
                resume.evaluate(out/"source",out,out/"work",record,2,None,10**12)
            self.assertEqual(calls[0],("development_references",["P0","M4"],[42,43,44,45]))
            self.assertEqual(calls[1][1],["RL42","RL43","RL44"])
            self.assertEqual(w.read(out/"outcome.json")["status"],"primary_development_not_passed")


if __name__=="__main__":unittest.main()
