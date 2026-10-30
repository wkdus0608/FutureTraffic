"""[추가 탐색] 천장 재기(109절) 검사 — SUMO 실행 없음(가짜 차량 파일·가짜 환경·가짜 결과, 지도 파일은 있을 때만 읽음)."""
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import explore_ceiling as ce  # noqa: E402
import explore_pred as ep  # noqa: E402
import test_explore_pred as tep  # noqa: E402
from test_explore_pred import GUARD, _model, _write, fake_env  # noqa: E402

STRAIGHT_EW = ("477706392_02", "477707013_01")      # 길주로 동 → 서 직진
SINGLE_NET = sorted((ROOT / "results/focused_204820/cache").glob("net_dqn_template_*.net.xml"))


def route_file(folder: Path) -> Path:
    path = folder / "routes.rou.xml"
    path.write_text("""<routes>
  <route id="r1" edges="477714896_02 477752894_0101_01 477706392_02 477707013_01"/>
  <route id="r2" edges="477707013_01 477715088_0101_01"/>
  <vehicle id="late" route="r1" depart="120.00"/>
  <vehicle id="early" route="r1" depart="10.00"/>
  <vehicle id="nomove" route="r2" depart="50.00"/>
</routes>""")
    return path


class OracleInputTest(unittest.TestCase):
    def test_movements_carry_sorted_departures(self):
        import focused_eval
        with tempfile.TemporaryDirectory() as d:
            rf = route_file(Path(d))
            m = ce.controller_class("oracle", 120).movements(rf)
            self.assertEqual(dict(m), focused_eval.QueueFirst.movements(rf))       # 이동은 단순 제어와 같음
            self.assertEqual(m.departures, [(10.0, "477714896_02", STRAIGHT_EW), (120.0, "477714896_02", STRAIGHT_EW)])
            self.assertEqual(ce.controller_class("long", 240).movements(rf).departures, [])   # L은 출발 기록을 쓰지 않음

    def test_future_arrivals(self):
        a, b = ("x", "y"), ("x", "z")
        deps = [(1000.0, "eA", a), (1004.0, "eB", b), (1010.0, "eA", a), (1100.0, "eX", a), (1110.0, "eA", a), (1300.0, "eA", a)]
        found, unknown = ce.future_arrivals(deps, [x[0] for x in deps], {"eA": 20.0, "eB": 7.5}, {a: 0, b: 1}, 1000.0, 5.0, 24)
        self.assertEqual(found, [(2, 1), (6, 0)])     # 1004+7.5 → 2단계, 1010+20 → 6단계; 1000(지금)·1110(+20 → 26단계)·1300은 밖
        self.assertEqual(unknown, 1)                  # 출발 도로를 모르는 차는 세기만

    def test_choose_long_horizon(self):
        pick = lambda means: ce.choose_long_horizon(lambda h: means[h])
        self.assertEqual(pick({240: 0.6})[0], 240)
        h, tried = pick({240: 1.4, 180: 0.8})
        self.assertEqual((h, [t["ok"] for t in tried]), (180, [False, True]))
        self.assertEqual(pick({240: None, 180: 1.2})[0], None)                    # 둘 다 느림 → L을 하지 않음

    def test_refuses_other_prediction_code(self):
        original = ep.code_fingerprint
        ep.code_fingerprint = lambda: "000000000000"
        try:
            with self.assertRaises(SystemExit):
                ce.controller_class("long", 240)
        finally:
            ep.code_fingerprint = original


@unittest.skipUnless(SINGLE_NET, "단독 모형 틀 지도 없음")
class ControllerTest(unittest.TestCase):
    def setUp(self):
        helper = tep.ControllerFakeEnvTest()       # 진입로마다 멈춘 차 3대·달리는 차 1대(98절 시험과 같은 차)
        helper.setUp()
        self.helper = helper

    def test_long_is_p0_with_longer_horizon(self):
        vehicles, movement_of = self.helper._vehicles()
        env, greens = fake_env(SINGLE_NET[-1], vehicles)
        c = ce.controller_class("long", 240)(movement_of, greens)
        self.assertIn(c(env, [0, 1, 2, 3]), [0, 1, 2, 3])
        self.assertEqual(c._steps, 48)
        q0, arrivals = c.queues_and_arrivals(env)
        self.assertEqual(sum(map(sum, arrivals)), 0)                               # P0처럼 새로 오는 차 없음
        self.assertEqual(sum(q0), 3 * len(self.helper.movements))
        self.assertEqual((c.stats["variant"], c.stats["horizon_s"], c.stats["pred_code"]), ("long", 240, ce.PRED_CODE))
        self.assertEqual(c.stats["code"], ce.code_fingerprint())

    def test_oracle_adds_exact_future_departures(self):
        vehicles, movement_of = self.helper._vehicles()
        env, greens = fake_env(SINGLE_NET[-1], vehicles)
        served = list(self.helper.movements.values())
        m0 = served[0]
        first = self.helper.ZONE_EDGES[m0[0]][0]
        movement_of = ce.Movements(movement_of)
        movement_of.departures = [(3590.0, first, m0), (3610.0, first, m0), (3615.0, first, m0), (3900.0, first, m0)]
        c = ce.controller_class("oracle", 120)(movement_of, greens)
        c(env, [0, 1, 2, 3])
        _, arrivals = c.queues_and_arrivals(env)
        i, travel = c.index[m0], c._travel_from[first]
        self.assertGreater(travel, 0)
        self.assertAlmostEqual(travel, c.geo["travel"][m0[0]], delta=c.geo["travel"][m0[0]])   # 같은 진입로 길이 수준
        self.assertEqual(sum(map(sum, arrivals)), len(served) + 2)                 # 달리는 차(P1처럼) + 앞으로 2분 안에 출발할 2대
        for depart in (3610.0, 3615.0):
            self.assertGreaterEqual(arrivals[int((depart - 3600.0 + travel) // 5)][i], 1)
        self.assertEqual(c.stats["oracle_unknown_start"], 0)
        self.assertEqual(len(c._entries), 0)                                       # 최근 진입 비율은 쓰지 않음

    def test_oracle_without_departures_stops(self):
        vehicles, movement_of = self.helper._vehicles()
        env, greens = fake_env(SINGLE_NET[-1], vehicles)
        c = ce.controller_class("oracle", 120)(movement_of, greens)
        with self.assertRaises(SystemExit):
            c(env, [0, 1, 2, 3])

    def test_abort_when_too_slow(self):
        vehicles, movement_of = self.helper._vehicles()
        env, greens = fake_env(SINGLE_NET[-1], vehicles)
        c = ce.controller_class("long", 240, abort=True)(movement_of, greens)
        type(c).stats.update(planned=ce.ABORT_AFTER, seconds=ce.ABORT_AFTER * (ce.ABORT_MEAN_S + 0.1))
        with self.assertRaises(ce.TooSlow):
            c(env, [0])
        quiet = ce.controller_class("long", 240)(movement_of, greens)               # 점검이 아니면 멈추지 않음
        type(quiet).stats.update(planned=ce.ABORT_AFTER, seconds=ce.ABORT_AFTER * 10.0)
        self.assertEqual(quiet(env, [0]), 0)


class JobsTest(unittest.TestCase):
    def test_validation_first_and_tags(self):
        split = json.loads((ROOT / "config/focused_split.json").read_text())
        jobs = ce.single_jobs(Path("/drive"), Path("/tmp/ceil"), {"oracle": 120, "long": 240})
        self.assertEqual(len(jobs), 2 * (2 * len(split["validation"]) + len(split["train"])))   # 34개
        first = jobs[:4 * len(split["validation"])]
        self.assertTrue(all(j["date"] in split["validation"] for j, _, _ in first))
        self.assertTrue(all(t.startswith("/drive/") and f"/ceil_{v}_eval42" in t for j, t, v in jobs))
        self.assertEqual(len(ce.single_jobs(Path("/drive"), Path("/tmp/ceil"), {"oracle": 120})), len(jobs) // 2)


# ------------------------------------------------------------------ 판정(가짜 결과)
CEIL_STATS = {"horizon_s": 240, "code": "abc", "planned": 10, "seconds": 3.0, "max_seconds": 0.8, "outside_allowed": 0}


def _single(folder: Path, date: str, tag: str, delay: float, arrivals="random", seed=None, stats_key=None):
    run = folder / date / f"{tag}_eval42{'' if arrivals == 'random' else '_' + arrivals}"
    by = {"477706392_02->477706883_02 (l)": {"vehicles": 50, "mean_delay_s": 140.0},
          "477706883_01->477707013_01 (l)": {"vehicles": 50, "mean_delay_s": 100.0}}
    m = {"date": date, "train_seed": seed, "all": {"vehicles": 1000, "mean_delay_s": delay, "long_waits": 5, "not_arrived": 0,
                                                   "by_movement": by},
         "collisions_unexplained": [], "pedestrian": {"x": {"waits_over_limit": 0}}, "guard": GUARD}
    if stats_key:
        m[stats_key] = CEIL_STATS
    _write(run, m)
    with gzip.open(run / "events.json.gz", "wt") as f:
        json.dump([{"pet": {"clearing_first": {"pet": 2.5}, "entering_first": None}}], f)


def _window(root: Path, w: str, tag: str, seed: int, delay: float, stats_key=None):
    by = {"477706392_02->477706883_02": {"vehicles": 20, "mean_delay_s": 160.0},
          "477706883_01->477707013_01": {"vehicles": 20, "mean_delay_s": 95.0}}
    m = {"window_204820": {"vehicles": 1000, "mean_delay_s": delay + (seed - 44) * 0.1, "long_delays": 10, "by_movement": by},
         "collisions_in_204820": 0, "guard": GUARD}
    if stats_key:
        m[stats_key] = CEIL_STATS
    _write(root / "windows" / w / f"{tag}_eval{seed}", m)


class CeilDecisionTest(unittest.TestCase):
    def _check(self, root: Path, long_h=240) -> str:
        p = root / "check.json"
        p.write_text(json.dumps({"ok": True, "long_horizon_s": long_h}))
        return str(p)

    def test_single(self):
        import explore_decide as ed
        split = json.loads((ROOT / "config/focused_split.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred, v6, h1, ceil = root / "pred", root / "v6", root / "h1", root / "ceil"
            h1m = [_model(root, "h1", s) for s in ed.SEEDS]
            for dates, arrivals_list, training in ((ed.VALIDATION, ("random", "platoon"), False), (split["train"], ("random",), True)):
                for arr in arrivals_list:
                    for d in dates:
                        _single(v6, d, "fixedc", 47.3, arr)
                        _single(v6, d, "greedy", 39.7, arr)
                        _single(pred, d, "pred_p0", 38.0, arr, stats_key="pred")
                        _single(pred, d, "pred_p1", 37.5, arr, stats_key="pred")
                        _single(ceil, d, "ceil_long", 36.0, arr, stats_key="ceil")
                        _single(ceil, d, "ceil_oracle", 37.6, arr, stats_key="ceil")
                        if not training:
                            for s in ed.SEEDS:
                                _single(h1 if arr == "random" else pred, d, ed.dqn_tag(h1m)(s), 39.78, arr, seed=s)
            args = SimpleNamespace(ceil_dir=str(ceil), check=self._check(root), pred_dir=str(pred), v6_dir=str(v6),
                                   h1_dir=str(h1), h1_models=h1m)
            r = ed.decide_ceil_single(args)
        c = r["validation_random"]["comparisons"]
        self.assertEqual(c["1_L_vs_P0"]["word"], "낫다")                          # 2.0초 짧음
        self.assertEqual(c["2_O_vs_P0"]["word"], "차이 없음")                     # 0.4초
        self.assertEqual(c["2_O_vs_P1"]["delay_diff_s"], 0.1)
        self.assertEqual(c["extra_L_vs_dqn_h1"]["word"], "낫다")
        self.assertEqual(r["interpretation"][0], "1. 더 멀리 보면 줄어든다.")
        self.assertEqual(r["gate"], "short")
        self.assertEqual(r["validation_random"]["ceil"]["L"]["horizon_s"], 240)
        self.assertEqual(r["safety_note"], "PET < 0 없음")
        self.assertIn("L", r["training13_random"]["summary"])

    def test_single_without_long(self):
        import explore_decide as ed
        split = json.loads((ROOT / "config/focused_split.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred, v6, h1, ceil = root / "pred", root / "v6", root / "h1", root / "ceil"
            h1m = [_model(root, "h1", s) for s in ed.SEEDS]
            for dates, arrivals_list, training in ((ed.VALIDATION, ("random", "platoon"), False), (split["train"], ("random",), True)):
                for arr in arrivals_list:
                    for d in dates:
                        for t, v, k in (("fixedc", 47.3, None), ("greedy", 39.7, None)):
                            _single(v6, d, t, v, arr, stats_key=k)
                        _single(pred, d, "pred_p0", 38.0, arr, stats_key="pred")
                        _single(pred, d, "pred_p1", 37.5, arr, stats_key="pred")
                        _single(ceil, d, "ceil_oracle", 36.5, arr, stats_key="ceil")      # 1.5초 짧음
                        if not training:
                            for s in ed.SEEDS:
                                _single(h1 if arr == "random" else pred, d, ed.dqn_tag(h1m)(s), 39.78, arr, seed=s)
            args = SimpleNamespace(ceil_dir=str(ceil), check=self._check(root, None), pred_dir=str(pred), v6_dir=str(v6),
                                   h1_dir=str(h1), h1_models=h1m)
            r = ed.decide_ceil_single(args)
        self.assertEqual(r["gate"], "none")
        self.assertTrue(r["interpretation"][0].startswith("1. L을 잴 수 없었다"))
        self.assertTrue(r["interpretation"][-1].startswith("도착을 정확히 알면 줄어든다"))

    def test_network12_and_gate(self):
        import explore_decide as ed
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            b12 = [_model(root, "b12", s) for s in ed.SEEDS]
            for d in ed.VALIDATION:
                for h in (8, 17):
                    w = f"{d}_{h:02d}"
                    for s in ed.SEEDS:
                        for folder, tag, delay, key in (("ref", "fixedc", 56.5, None), ("ref", "greedy", 54.1, None),
                                                        ("pred", "pred_p0", 51.2, "pred"), ("pred", "pred_p2", 48.7, "pred"),
                                                        ("ceil", "ceil_long", 50.0, "ceil")):
                            _window(root / folder, w, tag, s, delay, key)
                        _window(root / "b12", w, ed.dqn_tag(b12)(s), s, 53.8)
            args = SimpleNamespace(ceil_dir=str(root / "ceil"), check=self._check(root), pred_dir=str(root / "pred"),
                                   ref_dir=str(root / "ref"), base_dir=str(root / "b12"), base_models=b12)
            r = ed.decide_ceil_12(args)
            self.assertEqual(r["comparisons"]["1_L_vs_P0"]["word"], "차이 없음")      # 1.2초 < 2.2초
            self.assertEqual(r["gate"], "same")
            self.assertEqual(r["comparisons"]["extra_L_vs_dqn_b12"]["word"], "낫다")
            self.assertEqual(r["safety_note"], "석천사거리 안 충돌 없음")

            def gate(single, network):
                (root / "s.json").write_text(json.dumps({"gate": single, "interpretation": []}))
                (root / "n.json").write_text(json.dumps({"gate": network, "interpretation": []}))
                return ed.decide_ceil(SimpleNamespace(single_decision=str(root / "s.json"),
                                                      network_decision=str(root / "n.json") if network else None))["next"]

            self.assertEqual(gate("short", "same"), "plan_step1")
            self.assertEqual(gate("same", "short"), "plan_step1")
            self.assertEqual(gate("same", "same"), "stop")
            self.assertEqual(gate("same", "long"), "ask_user")
            self.assertEqual(gate("none", None), "ask_user")
        self.assertEqual([ed.ceil_gate(x, 1.3) for x in (-1.3, -1.29, 1.3, None)], ["short", "same", "long", "none"])


if __name__ == "__main__":
    unittest.main()
