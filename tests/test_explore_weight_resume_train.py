"""Continuation provenance and strict case classification; no actual SUMO."""
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
import explore_weight_resume_train as resume
import explore_weight_train as learner


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def approval(root):
    root = Path(root)
    report, evidence = root / "probe/report.json", root / "probe/trajectory.json"
    write(report, {"status":"reviewed_fixture","same_collision":True})
    write(evidence, {"synthetic":True})
    value = {"kind":resume.APPROVAL_KIND,"approved":True,"study_fingerprint":"new-study",
             "diagnostic_report_path":str(report),"diagnostic_report_sha256":sha(report),
             "diagnosis_evidence_hashes":{str(evidence):sha(evidence)},
             "case":copy.deepcopy(resume.EXPECTED_CASE)}
    write(root / "out/approval.json",value)
    return value


def episode(events=None, **changes):
    if events is None:events = [copy.deepcopy(resume.EXPECTED_EVENT)]
    metrics = {"date":"20220906","period":"1500","seed":338,"eval_seed":338,
               "route_sha256":resume.EXPECTED_CASE["route_sha256"],"handover":1803.0,
               "collisions_unexplained":len(events),"collisions_at_204820":0,
               "collision_details":{"collisions_total":len(events),"collisions_at_204820":0,
                                    "collisions_known_map_defect":{},"collisions_unexplained":copy.deepcopy(events)},
               "cost_check_passed":True,"cost_check":{"passed":True},"actual_total_delay":12.3,
               "missing":0,"unfinished":0,"mode_action_errors":0,"changed_by_rules":0}
    metrics.update(changes)
    return SimpleNamespace(metrics=metrics,transitions=[{"mode":0,"reward":-.01}],decisions=[],seconds=.1)


def checkpoint(folder, seed, episodes):
    folder.mkdir(parents=True,exist_ok=True)
    for name in ("model.pt","replay.pkl"):
        (folder / name).write_bytes(f"synthetic-{seed}-{episodes}-{name}".encode())
    manifest = {"kind":learner.CHECKPOINT_KIND,"settings":learner.SETTINGS,"fingerprint":"parent",
                "seed":seed,"episodes":episodes,"control_steps":episodes*720,
                "updates":1,"target_syncs":1,
                "files":{name:sha(folder/name) for name in ("model.pt","replay.pkl")}}
    write(folder / "manifest.json",manifest)
    return manifest


def source_study(root):
    source = Path(root) / "source"
    write(source / "experiment.json",{"fingerprint":"old-study","parent_fingerprint":"parent"})
    write(source / "budget.json",{"used_s":123000,"pending":{}})
    for seed,episodes in ((42,333),(43,292),(44,294)):
        model = source / "models" / f"seed_{seed}"
        generation = f"checkpoints/ep_{episodes:06d}_synthetic"
        manifest = checkpoint(model/generation,seed,episodes)
        write(model/"latest.json",{"checkpoint":generation,"manifest_sha256":sha(model/generation/"manifest.json")})
        lineage = {"kind":"WEIGHT1-learning-length-v1","seed":seed,"study_fingerprint":"old-study","parent_fingerprint":"parent"}
        if seed==42:lineage["source"]={"rng_sha256":"original-rng","identity":{"seed":42,"episodes":42},"source_nested":[1,2,3]}
        else:lineage["initialization"]="fresh unchanged WEIGHT1"
        write(model/"long_lineage.json",lineage)
        write(model/"run.json",{"episodes":episodes,"history":[{"synthetic":True}]})
        stages={}
        for stage in (42,139,278):
            checkpoint(model/"stages"/f"episode_{stage:06d}",seed,stage)
            stages[str(stage)]={"episodes":stage,"stage":str(model/"stages"/f"episode_{stage:06d}")}
        write(source/"training_status"/f"seed_{seed}.json",{
            "kind":"WEIGHT1-learning-length-v1","seed":seed,"study_fingerprint":"old-study","parent_fingerprint":"parent",
            "completed_episodes":278,"control_steps":278*720,"status":"failed" if seed==44 else "training",
            "milestones":stages,"completed_milestones":[42,139,278],"error":"old reason"})
    return source


def snapshot(root):
    return {str(p.relative_to(root)):sha(p) for p in root.rglob("*") if p.is_file()}


class ApprovalAndClassification(unittest.TestCase):
    def test_approval_fails_closed_and_binds_report_and_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);value=approval(root)
            self.assertEqual(resume.load_approval(root/"out","new-study"),value)
            with self.assertRaises(resume.ApprovalError):resume.load_approval(root/"missing","new-study")
            with self.assertRaises(resume.ApprovalError):resume.load_approval(root/"out","wrong")
            (root/"probe/report.json").write_text('{}')
            with self.assertRaises(resume.ApprovalError):resume.validate_approval(value)
            value=approval(root);(root/"probe/trajectory.json").write_text('{}')
            with self.assertRaises(resume.ApprovalError):resume.validate_approval(value)
            value=approval(root);value["approved"]=False
            with self.assertRaises(resume.ApprovalError):resume.validate_approval(value)

    def test_exact_event_reclassification_keeps_raw_counts_cost_and_experience(self):
        with tempfile.TemporaryDirectory() as tmp:
            value=approval(Path(tmp));original=episode();before=copy.deepcopy(original.metrics)
            classified=resume.apply_classification(original,value)
            self.assertEqual(original.metrics,before)
            self.assertEqual(classified.metrics["raw_collision_details"],before["collision_details"])
            self.assertEqual(classified.metrics["raw_collisions_unexplained"],1)
            self.assertEqual(classified.metrics["collisions_unexplained"],0)
            self.assertEqual(classified.metrics["collision_details"]["collisions_total"],1)
            self.assertEqual(classified.metrics["actual_total_delay"],original.metrics["actual_total_delay"])
            self.assertIs(classified.transitions,original.transitions)
            learner._check_episode(classified,expected_steps=1)
            with self.assertRaises(ValueError):learner._check_episode(original,expected_steps=1)

    def test_other_route_driver_location_time_and_duplicate_still_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            value=approval(Path(tmp))
            cases=[episode(seed=339,eval_seed=339),episode(date="20220907"),episode(route_sha256="different"),episode(handover=1584.0),episode(handover=1804.0)]
            wrong_event=copy.deepcopy(resume.EXPECTED_EVENT);wrong_event["victim"]="other-truck"
            cases += [episode([wrong_event]),episode([copy.deepcopy(resume.EXPECTED_EVENT),wrong_event]),
                      episode([copy.deepcopy(resume.EXPECTED_EVENT),copy.deepcopy(resume.EXPECTED_EVENT)]),
                      episode(collisions_at_204820=1)]
            for result in cases:
                classified=resume.apply_classification(result,value)
                with self.assertRaises(ValueError):learner._check_episode(classified,expected_steps=1)
            duplicate=resume.apply_classification(episode(),value)
            with self.assertRaises(resume.ApprovalError):resume.apply_classification(duplicate,value)

    def test_zero_collision_other_policies_remain_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            value=approval(Path(tmp));result=episode([],seed=1042,eval_seed=1042,date="20220928")
            classified=resume.apply_classification(result,value)
            self.assertEqual(classified.metrics["collisions_unexplained"],0)
            self.assertFalse(classified.metrics["collision_classification"]["confirmed_events"])
            learner._check_episode(classified,expected_steps=1)


class CloneProvenance(unittest.TestCase):
    def test_model_copy_is_independent_and_keeps_inner_identity_and_rng_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=source_study(root);before=snapshot(source);out=root/"out"
            audit=resume.clone_models(source,out,"new-study","parent")
            self.assertFalse(audit["reused"])
            self.assertEqual(snapshot(source),before)
            self.assertTrue(audit["originalsha"])
            for seed in (42,43,44):
                origin=source/"models"/f"seed_{seed}";target=out/"models"/f"seed_{seed}"
                old=learner.validate_checkpoint(origin,"parent");new=learner.validate_checkpoint(target,"parent")
                self.assertEqual(old["files"],new["files"])
                self.assertNotEqual((Path(old["path"])/"model.pt").stat().st_ino,(Path(new["path"])/"model.pt").stat().st_ino)
                lineage=json.loads((target/"long_lineage.json").read_text())
                original=json.loads((origin/"long_lineage.json").read_text());original["study_fingerprint"]="new-study"
                self.assertEqual(lineage,original)
                status=json.loads((out/"training_status"/f"seed_{seed}.json").read_text())
                self.assertEqual(status["completed_episodes"],new["episodes"])
                self.assertEqual(status["study_fingerprint"],"new-study")
                self.assertEqual((out/"training_status"/f"source_seed_{seed}.json").read_bytes(),(source/"training_status"/f"seed_{seed}.json").read_bytes())
            self.assertTrue(resume.clone_models(source,out,"new-study","parent")["reused"])
            stage=out/"models/seed_43/stages/episode_000139/model.pt";stage.write_bytes(b"changed")
            with self.assertRaises(ValueError):resume.clone_models(source,out,"new-study","parent")

    def test_overlap_and_existing_unowned_directory_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=source_study(root)
            with self.assertRaises(ValueError):resume.clone_models(source,source/"child","new-study","parent")
            out=root/"out";target=out/"models/seed_42";target.mkdir(parents=True);(target/"foreign.txt").write_text('owned elsewhere')
            with self.assertRaises((ValueError,FileNotFoundError)):resume.clone_models(source,out,"new-study","parent")
            self.assertTrue((target/"foreign.txt").is_file())


class WorkerStatus(unittest.TestCase):
    def test_failure_is_raised_even_if_target_reached_and_time_limit_is_returned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);approval(root)
            job={"out":str(root/"out"),"study_fingerprint":"new-study","seed":44,"episodes":[320]}
            original_factory=resume.original_train._episode_runner
            results=[{"status":"failed","completed_episodes":320,"error":"collisions_unexplained=1"},
                     {"status":"complete","completed_episodes":319},
                     {"status":"unexpected","completed_episodes":320}]
            for result in results:
                with patch.object(resume,"_runtime_guard"),patch.object(resume.original_train,"run_training_job",return_value=result):
                    with self.assertRaises(resume.TrainingJobFailed):resume.training_worker(job)
                self.assertIs(resume.original_train._episode_runner,original_factory)
            with patch.object(resume,"_runtime_guard"),patch.object(resume.original_train,"run_training_job",return_value={"status":"time_limit","completed_episodes":300}):
                self.assertEqual(resume.training_worker(job)["status"],"time_limit")

    def test_training_wrapper_classifies_and_always_restores_factory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);approval(root);job={"out":str(root/"out"),"study_fingerprint":"new-study","seed":44,"episodes":[320]}
            def fake_job(_):
                result=resume.original_train._episode_runner()()
                learner._check_episode(result,expected_steps=1)
                return {"status":"complete","completed_episodes":320}
            fake_factory=lambda: lambda: episode()
            with patch.object(resume,"_runtime_guard"),patch.object(resume.original_train,"_episode_runner",fake_factory),patch.object(resume.original_train,"run_training_job",side_effect=fake_job):
                self.assertEqual(resume.training_worker(job)["completed_episodes"],320)
                self.assertIs(resume.original_train._episode_runner,fake_factory)

    def test_eval_saved_original_reference_avoids_parent_patch_recursion(self):
        import explore_weight_episode as source_episode
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);approval(root)
            job={"target":str(root/"out/runs/development/policy_case"),"study_fingerprint":"new-study"}
            calls=[]
            def fake_original(_):
                calls.append(1)
                return source_episode.run_episode()
            fake_runner=lambda:episode()
            with patch.object(resume,"_runtime_guard"),patch.object(resume,"_ORIGINAL_EVAL_WORKER",side_effect=fake_original),patch.object(source_episode,"run_episode",fake_runner),patch.object(resume.original_long,"_eval_worker",resume.eval_worker):
                result=resume.eval_worker(job)
                self.assertEqual(result.metrics["collisions_unexplained"],0)
                self.assertIs(source_episode.run_episode,fake_runner)
            self.assertEqual(calls,[1])


if __name__=="__main__":unittest.main()
