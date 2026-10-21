"""[추가 탐색] 연동 정보 관측(96절)과 COORD 판정 검사 — SUMO 실행 없음(가짜 연결, 지도 파일은 있을 때만 읽음)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import explore_coord12 as ec  # noqa: E402
import focused_rl  # noqa: E402

LAYOUT = {"straight_links": [1, 2], "left_links": [3], "bins": [["a"], ["b"], ["c"], ["d", "e"]],
          "bin_capacity": [10.0, 10.0, 10.0, 4.0]}


class FakeLight:
    def __init__(self, state, phase, spent, phases, program="real"):
        self.state, self.phase, self.spent, self.phases, self.program = state, phase, spent, phases, program

    def getRedYellowGreenState(self, tls):
        return self.state

    def getPhase(self, tls):
        return self.phase

    def getSpentDuration(self, tls):
        return self.spent

    def getProgram(self, tls):
        return self.program

    def getAllProgramLogics(self, tls):
        other = SimpleNamespace(programID="0", phases=[SimpleNamespace(duration=99, state="rrrr")])
        mine = SimpleNamespace(programID=self.program, phases=[SimpleNamespace(duration=d, state=s) for d, s in self.phases])
        return [other, mine]


class FakeEdge:
    def __init__(self, vehicles, halting):
        self.vehicles, self.halting = vehicles, halting

    def getLastStepVehicleNumber(self, e):
        return self.vehicles[e]

    def getLastStepHaltingNumber(self, e):
        return self.halting[e]


class CoordFeatureTest(unittest.TestCase):
    def test_since_change_walks_back_over_same_state(self):
        # 직진 녹색 켜짐: 꺼짐(10) · 켜짐(20) · 켜짐(5) · 꺼짐(3) · 지금 켜짐 2번 현시에서 4초
        on = [False, True, True, False]
        self.assertEqual(ec.since_change([10, 20, 5, 3], on, 2, 4.0), 24.0)
        self.assertEqual(ec.since_change([10, 20, 5, 3], on, 3, 1.0), 1.0)
        self.assertEqual(ec.since_change([10, 20, 5, 3], on, 0, 7.0), 7.0)   # 프로그램 처음이면 멈춤

    def test_program_runs_uses_active_program_and_majority(self):
        light = FakeLight("rGGr", 0, 0.0, [(30, "rGGr"), (4, "ryyr"), (40, "rGrr"), (20, "rrrG")])
        durations, on = ec.program_runs(SimpleNamespace(trafficlight=light), [1, 2])
        self.assertEqual(durations, [30.0, 4.0, 40.0, 20.0])
        self.assertEqual(on, [True, False, True, False])          # 둘 중 하나(절반)가 녹색이면 켜짐

    def test_features(self):
        phases = [(50, "rrrr"), (30, "rGGr"), (6, "rGGG")]
        light = FakeLight("rGGG", 2, 2.0, phases)
        edge = FakeEdge({"a": 5, "b": 0, "c": 30, "d": 3, "e": 4}, {"a": 1, "b": 0, "c": 0, "d": 3, "e": 0})
        sumo = SimpleNamespace(trafficlight=light, edge=edge)
        runs = ec.program_runs(sumo, LAYOUT["straight_links"])
        f = ec.coord_features(sumo, runs, LAYOUT)
        self.assertEqual(len(f), ec.N_EXTRA)
        self.assertEqual(f[0], 1.0)                                # 직진 둘 다 녹색
        self.assertAlmostEqual(f[1], (2 + 30) / ec.SINCE_CAP_S)    # 켜진 뒤 32초
        self.assertEqual(f[2], 1.0)                                # 좌회전 녹색
        self.assertEqual(f[3:], [0.4, 0.0, 1.0, 1.0])              # 달리는 차 ÷ 칸 크기, 최대 1
        light.state, light.phase, light.spent = "rrrr", 0, 400.0
        f = ec.coord_features(sumo, runs, LAYOUT)
        self.assertEqual(f[:3], [0.0, 1.0, 0.0])                   # 180초 넘으면 1

    def test_mask_stays_last(self):
        base = np.array([1, 0, 0, 0, 0.5, 0.2, 0.3, 0, 1, 1, 0], dtype=np.float32)   # 끝 4칸 = 행동 표시
        out = ec.insert_before_mask(base, [0.7] * ec.N_EXTRA, 4)
        self.assertEqual(out.shape, (base.size + ec.N_EXTRA,))
        np.testing.assert_array_equal(out[:4], base[:4])           # HoldAwareDQN이 읽는 현재 현시
        np.testing.assert_array_equal(out[-4:], base[-4:])         # 그대로 실행될 행동 표시
        self.assertEqual(out.dtype, np.float32)

    def test_observation_space_and_install_restores(self):
        original_zone, original_dqn = focused_rl.ZoneObservation, focused_rl.hold_aware_dqn_class
        try:
            ec.install()
            ec.install()                                           # 두 번 불러도 한 겹만
            cls = focused_rl.ZoneObservation
            self.assertTrue(getattr(cls, "coord", False))
            self.assertIs(cls.__mro__[1], original_zone)
            ts = SimpleNamespace(num_green_phases=4, lanes=list(range(20)))
            base_size = original_zone(ts).observation_space().shape[0]
            self.assertEqual(cls(ts).observation_space().shape[0], base_size + ec.N_EXTRA)
            self.assertIsNot(focused_rl.hold_aware_dqn_class, original_dqn)
        finally:
            focused_rl.ZoneObservation, focused_rl.hold_aware_dqn_class = original_zone, original_dqn

    @unittest.skipUnless(sorted((ROOT / "results/network_demand/map").glob("net12_*.net.xml")), "12개 지도 파일 없음")
    def test_layout_on_real_map(self):
        from network_map import network_12
        if not network_12().exists():
            self.skipTest("지금 지문의 12개 지도 파일 없음")
        lay = ec.east_layout(network_12())
        self.assertEqual(lay["chain"][0]["edge"], ec.APPROACH)
        self.assertTrue(400 < lay["length_m"] < 430, lay["length_m"])   # 204822 출구 ~ 정지선 약 415m
        self.assertEqual(len(lay["straight_links"]), 6)
        self.assertEqual(len(lay["left_links"]), 2)
        self.assertEqual([len(b) for b in lay["bins"]], [3, 3, 3, 2])


class CoordDecisionTest(unittest.TestCase):
    """가짜 평가 결과로 COORD 판정이 96절 기준대로 계산되는지 본다."""

    def _model(self, root: Path, name: str, seed: int, explore: dict) -> str:
        d = root / name / f"seed_{seed}"
        d.mkdir(parents=True)
        (d / "training_config.json").write_text(json.dumps({"seed": seed, "timesteps": 100000}))
        (d / "dqn_204820.zip").write_bytes(f"{name}{seed}".encode())
        (d / "explore.json").write_text(json.dumps(explore))
        return str(d)

    def _runs(self, folder: Path, tag, delay, east_evening, long=10):
        for date in ("20220928", "20220813"):
            for hour in (8, 17):
                for s in (42, 43, 44, 45, 46):
                    d = folder / "windows" / f"{date}_{hour:02d}" / f"{tag(s)}_eval{s}"
                    d.mkdir(parents=True)
                    ew = east_evening if hour == 17 else 40.0
                    by = {"477706392_02->477707013_01": {"vehicles": 100, "mean_delay_s": ew},
                          "477706392_02->477706883_02": {"vehicles": 10, "mean_delay_s": 150.0},
                          "477706883_01->477707013_01": {"vehicles": 10, "mean_delay_s": 90.0}}
                    w = {"vehicles": 1000, "mean_delay_s": delay + (s - 44) * 0.1, "long_delays": long, "by_movement": by}
                    (d / "metrics.json").write_text(json.dumps({"window_204820": w}))

    def test_decision(self):
        import explore_decide as ed
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seeds = (42, 43, 44, 45, 46)
            base = [self._model(root, "b12", s, {"double_dqn": True, "n_steps": 5}) for s in seeds]
            cand = [self._model(root, "coord", s, {"observation": "coord", "n_steps": 5, "double_dqn": True}) for s in seeds]
            old = [self._model(root, "dqn75", s, {}) for s in seeds]
            self._runs(root / "base", ed.dqn_tag(base), 54.0, 55.0)
            self._runs(root / "cand", ed.dqn_tag(cand), 51.5, 40.0)
            for p in ("fixed", "fixedc", "greedy"):
                self._runs(root / "ref", lambda s, p=p: p, 55.0, 32.0)
            self._runs(root / "ref", ed.dqn_tag(old), 54.3, 54.5)
            args = SimpleNamespace(base_dir=str(root / "base"), base_models=base, cand_dir=str(root / "cand"), cand_models=cand,
                                   ref_dir=str(root / "ref"), ref_models=old)
            r = ed.decide_coord(args)
            self.assertTrue(r["adopted"], r["checks"])               # 2.5초 감소, 나머지 같음
            self.assertEqual(r["mechanism"]["evening_east_straight_gain_s"], 15.0)
            self.assertIn("작동했다", r["mechanism"]["reading"])
            self.assertEqual(r["evening_east_straight_s"]["fixed"], 32.0)
            # 2.2초 미만이면 채택하지 않는다
            for p in (root / "cand").rglob("metrics.json"):
                m = json.loads(p.read_text())
                m["window_204820"]["mean_delay_s"] += 0.5
                p.write_text(json.dumps(m))
            r = ed.decide_coord(args)
            self.assertFalse(r["adopted"])
            self.assertFalse(r["checks"]["구간 지연 2.2초 이상 감소"])
            # 연동 모델이 아니면 멈춘다
            args.cand_models = base
            with self.assertRaises(SystemExit):
                ed.decide_coord(args)


if __name__ == "__main__":
    unittest.main()
