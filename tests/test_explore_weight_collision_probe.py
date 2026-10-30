"""Pure/mock probe checks only: no actual SUMO process or Mac simulation."""
from __future__ import annotations

import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import explore_weight_collision_probe as probe

EVENT = {"time":"1584.000", "type":"junction", "lane":probe.CASE["lane"], "pos":"10.30493",
         "collider":probe.CASE["collider"], "victim":probe.CASE["victim"],
         "colliderType":"truck", "victimType":"truck", "colliderSpeed":"13.76499", "victimSpeed":"0",
         "colliderFront":"0,7", "colliderBack":"0,0", "victimFront":"1,7", "victimBack":"1,0"}
ROUTE = b'<routes><vType id="truck" length="7" width="2"/></routes>'
NET = (f'<net><junction id="SN20220920232934C" intLanes="{probe.CASE["lane"]}" x="0" y="0">'
       '<request index="0" response="0" foes="0"/></junction>'
       f'<edge id=":SN20220920232934C_2"><lane id="{probe.CASE["lane"]}" shape="0,0 0,7"/></edge>'
       '<edge id="from"><lane id="from_1" shape="0,-10 0,0"/></edge>'
       '<edge id="to"><lane id="to_2" shape="0,7 0,20"/></edge>'
       f'<connection from="from" to="to" fromLane="1" toLane="2" via="{probe.CASE["lane"]}"/>'
       '</net>')


def trajectory(path):
    with Path(path).open("w") as stream:
        for t in range(1550, 1620):
            cars = {}
            for vid in (probe.CASE["collider"], probe.CASE["victim"]):
                cars[vid] = {"geometry":{"width":2,"length":7,"route":["from","to"]},
                             "speed":13.7 if vid==probe.CASE["collider"] else 0,
                             "lane":probe.CASE["lane"], "x":0,"y":0,"angle":0}
            stream.write(json.dumps({"time":t,"vehicles":cars})+"\n")


def fake_warmup(work, cap_s):
    work = Path(work)
    (work/"route.xml").write_bytes(ROUTE)
    (work/"net.xml").write_text(NET)
    attributes = " ".join(f'{k}="{v}"' for k,v in EVENT.items())
    (work/"collisions.xml").write_text(f'<collisions><collision {attributes}/></collisions>')
    trajectory(work/"trajectory.jsonl")
    return {"handover_s":1803.0,"route_file":str(work/"route.xml"),"net_file":str(work/"net.xml"),
            "collisions_file":str(work/"collisions.xml"),"trajectory_file":str(work/"trajectory.jsonl"),
            "rl_actions_executed":0}


class PureAnalysis(unittest.TestCase):
    def test_exact_match_checks_numeric_geometry_and_vehicle_ids(self):
        self.assertTrue(probe.event_matches(EVENT,probe.CASE))
        self.assertTrue(probe.event_matches(EVENT,{**EVENT,"time":1584.0}))
        self.assertFalse(probe.event_matches(EVENT,{**EVENT,"time":1584.1}))
        self.assertFalse(probe.event_matches(EVENT,{**EVENT,"victim":"other"}))
        self.assertFalse(probe.event_matches(EVENT,{**EVENT,"victimBack":"2,0"}))
        self.assertFalse(probe.event_matches({**EVENT,"time":"nan"},EVENT))

    def test_rectangles_distinguish_overlap_touch_and_separation(self):
        left=probe.rectangle((0,7),(0,0),2)
        overlap=probe.rectangle((1,7),(1,0),2)
        touch=probe.rectangle((2,7),(2,0),2)
        distant=probe.rectangle((3,7),(3,0),2)
        self.assertTrue(probe.rectangle_overlap(left,overlap)["overlap"])
        self.assertAlmostEqual(probe.rectangle_overlap(left,overlap)["minimum_projection_overlap_m"],1)
        self.assertFalse(probe.rectangle_overlap(left,touch)["overlap"])
        self.assertFalse(probe.rectangle_overlap(left,distant)["overlap"])
        with self.assertRaises(ValueError):probe.rectangle((0,0),(0,0),2)

    def test_network_and_trajectory_report_do_not_change_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"net.xml";path.write_text(NET)
            before=path.read_bytes();report=probe.network_context(path,probe.CASE["lane"])
            self.assertEqual(path.read_bytes(),before)
            self.assertEqual(report["requests"][0]["foes"],"0")
            self.assertEqual(report["connections"][0]["via"],probe.CASE["lane"])
            path=Path(temp)/"trajectory.jsonl";trajectory(path)
            result=probe.analyse_trajectories(path,EVENT)
            self.assertEqual(result["at_collision"]["time"],1584)
            self.assertEqual(result["consecutive_stopped_seconds_before"][probe.CASE["victim"]],30)
            self.assertEqual(result["consecutive_stopped_seconds_before"][probe.CASE["collider"]],0)
            self.assertTrue(result["straight_rectangle_approximation"]["overlap"])

    def test_capture_step_only_calls_original_step_and_getters(self):
        vehicles=MagicMock()
        vehicles.getIDList.return_value=[probe.CASE["collider"],probe.CASE["victim"],"leader"]
        vehicles.getLeader.side_effect=lambda vid,d: ("leader",5) if vid==probe.CASE["victim"] else None
        vehicles.getTypeID.return_value="truck";vehicles.getLength.return_value=7;vehicles.getWidth.return_value=2
        vehicles.getRoute.return_value=("from","to");vehicles.getPosition.return_value=(1,2)
        vehicles.getAngle.return_value=0;vehicles.getSpeed.return_value=0
        vehicles.getLaneID.return_value="lane";vehicles.getLanePosition.return_value=2;vehicles.getRoadID.return_value="from"
        original=MagicMock(return_value="physical-step-result")
        env=SimpleNamespace(_sumo_step=original,sim_step=1584,sumo=SimpleNamespace(vehicle=vehicles))
        stream=io.StringIO();step=probe._capture_step(env,stream)
        self.assertEqual(step(),"physical-step-result")
        original.assert_called_once_with()
        self.assertEqual(vehicles.getLeader.call_count,2)  # leaders do not recursively extend their chain
        self.assertTrue(all(call[0].startswith("get") for call in vehicles.method_calls))
        row=json.loads(stream.getvalue())
        self.assertEqual(set(row["vehicles"]),{probe.CASE["collider"],probe.CASE["victim"],"leader"})

    def test_original_warmup_factory_wrapper_restores_itself_and_closes(self):
        # Fake modules ensure this integration test cannot start SUMO even if
        # the wrapper implementation regresses.
        env=SimpleNamespace(_sumo_step=lambda:None,sim_step=1803,sumo=SimpleNamespace(vehicle=None))
        factory=MagicMock(return_value=env)
        recorder=MagicMock(return_value=None)
        closed=[]
        def fake_episode(date,period,seed,work,deadline):
            from focused_sumo import make_environment
            used=make_environment(net_file=Path(work)/"net.xml",route_file=Path(work)/"route.xml",
                                  observation_class=object,reward_fn=lambda:0,seed=seed)
            used._sumo_step()
            return SimpleNamespace(handover=1803,start={"handover_s":1803},route=Path(work)/"route.xml",
                                   collisions=Path(work)/"collisions.xml",close=lambda:closed.append(True))
        modules={"focused_sumo":SimpleNamespace(make_environment=factory),
                 "explore_weight_episode":SimpleNamespace(_Episode=fake_episode)}
        with tempfile.TemporaryDirectory() as temp,patch.dict(sys.modules,modules),patch.object(probe,"_capture_step",return_value=recorder):
            result=probe._run_warmup(Path(temp),600)
            self.assertIs(modules["focused_sumo"].make_environment,factory)
        self.assertEqual(closed,[True]);recorder.assert_called_once_with()
        self.assertEqual(result["rl_actions_executed"],0)


class ProbeOrchestration(unittest.TestCase):
    def test_mac_guard_prevents_all_simulation(self):
        with patch.object(probe.sys,"platform","darwin"),patch.object(probe,"_run_warmup") as run:
            with self.assertRaises(RuntimeError):probe.run_probe("source","study","output","work")
            run.assert_not_called()

    def test_readonly_inputs_reproduction_evidence_and_cached_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/"source";study=root/"study";output=root/"output";work=root/"scratch"
            original=study/probe.ORIGINAL_ATTEMPT;original.mkdir(parents=True)
            (original/"metrics.json").write_text("original-unmodified-metrics")
            source.mkdir();(source/"keep").write_text("source")
            snapshot={p: p.read_bytes() for folder in (source,study) for p in folder.rglob("*") if p.is_file()}
            metrics={"route_sha256":hashlib.sha256(ROUTE).hexdigest(),"handover":1803.0}
            with patch.object(probe,"_runtime_guard"),patch.object(probe,"_verify_inputs",return_value=({"fingerprint":"study"},metrics,[EVENT])),patch.object(probe,"_run_warmup",side_effect=fake_warmup) as run:
                result=probe.run_probe(source,study,output,work)
                self.assertEqual(result["status"],"reproduced")
                self.assertTrue(result["matches"])
                self.assertTrue(result["collision_before_control"])
                self.assertEqual(result["replay"]["rl_actions_executed"],0)
                for name,digest in result["artifacts"].items():self.assertEqual(probe._sha(output/name),digest)
                reused=probe.run_probe(source,study,output,work)
                self.assertTrue(reused["reused"]);self.assertEqual(run.call_count,1)
                (output/"trajectories.jsonl.gz").write_bytes(b"changed")
                with self.assertRaises(ValueError):probe.run_probe(source,study,output,work)
            self.assertEqual(snapshot,{p:p.read_bytes() for p in snapshot})

    def test_unmatched_collision_and_timeout_do_not_grant_exception(self):
        for mode in ("mismatch","timeout"):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);original=root/"study"/probe.ORIGINAL_ATTEMPT;original.mkdir(parents=True)
                (original/"metrics.json").write_text("source")
                metrics={"route_sha256":hashlib.sha256(ROUTE).hexdigest(),"handover":1803.0}
                def runner(work,cap):
                    if mode=="timeout":raise TimeoutError("test-only")
                    result=fake_warmup(work,cap)
                    path=Path(result["collisions_file"])
                    path.write_text(path.read_text().replace('time="1584.000"','time="1585.000"'))
                    return result
                with patch.object(probe,"_runtime_guard"),patch.object(probe,"_verify_inputs",return_value=({"fingerprint":"study"},metrics,[EVENT])),patch.object(probe,"_run_warmup",side_effect=runner):
                    result=probe.run_probe(root/"source",root/"study",root/"output",root/"work")
                self.assertFalse(result["matches"])
                self.assertEqual(result["status"],"time_limit" if mode=="timeout" else "not_reproduced")
                self.assertTrue((root/"output/report.json").exists())

    def test_frozen_study_and_original_raw_evidence_are_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/"source";study=root/"study";attempt=study/probe.ORIGINAL_ATTEMPT
            attempt.mkdir(parents=True);(root/"code.py").write_text("unchanged")
            frozen={"parent_fingerprint":"parent","code":{"code.py":probe._sha(root/"code.py")}}
            experiment={**frozen,"fingerprint":probe._digest(frozen)}
            probe._write(study/"experiment.json",experiment)
            metrics={"date":"20220906","period":"1500","seed":338,"train_seed":44,
                     "collision_details":{"collisions_unexplained":[EVENT]}}
            probe._write(attempt/"metrics.json",metrics)
            attrs=" ".join(f'{k}="{v}"' for k,v in EVENT.items());raw=f'<collisions><collision {attrs}/></collisions>'.encode()
            with gzip.open(attempt/"collisions.xml.gz","wb") as stream:stream.write(raw)
            evidence={"study_fingerprint":experiment["fingerprint"],
                      "files":{n:probe._sha(attempt/n) for n in ("metrics.json","collisions.xml.gz")},
                      "source_sha256":{"collisions.xml":hashlib.sha256(raw).hexdigest()}}
            probe._write(attempt/"evidence.json",evidence)
            verifier=MagicMock()
            with patch.object(probe,"ROOT",root),patch.dict(sys.modules,{"explore_weight_long_train":SimpleNamespace(verify_parent=verifier)}):
                result=probe._verify_inputs(source,study)
                self.assertEqual(result[2],[EVENT]);verifier.assert_called_once_with(source,"parent")
                (root/"code.py").write_text("changed")
                with self.assertRaises(ValueError):probe._verify_inputs(source,study)


if __name__=="__main__":unittest.main()
