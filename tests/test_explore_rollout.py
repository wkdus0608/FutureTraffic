"""[추가 탐색] 0단계 SUMO 탐색(119절) 검사 — SUMO 실행 없음. 정보 범위·표본 미래·상태 고르기·판정 계산을 본다."""
import statistics
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import explore_rollout as er  # noqa: E402

EAST_STRAIGHT = ["477714896_02", "477706392_02", "477707013_01"]
EAST_LEFT = ["477714896_02", "477706392_02", "477706883_02"]


class PlanTest(unittest.TestCase):
    def test_values_match_plan(self):
        self.assertEqual((er.H_MAIN, er.H_SENS, er.TERMINAL_S, er.HISTORY_S), (300, 600, 75.0, 900))
        self.assertEqual(len(er.SELECT_SEEDS), 8)
        self.assertEqual(len(er.CONFIRM_SEEDS), 8)
        self.assertFalse(set(er.SELECT_SEEDS) & set(er.CONFIRM_SEEDS))           # 고르는 용과 확인용은 겹치지 않음
        self.assertEqual(len(er.EPISODES), 12)
        self.assertEqual(er.CLASS_N * len(er.CLASSES), 48)
        self.assertIn("--save-state.rng true", er.STATE_OPTS)
        self.assertIn("--save-state.precision 20", er.STATE_OPTS)
        self.assertEqual(er.P0_CODE, "9440fe1d3869")

    def test_t975(self):
        self.assertEqual(er.t975(7), 2.365)
        self.assertAlmostEqual(er.t975(47), 2.0117, places=2)


class InformationTest(unittest.TestCase):
    def setUp(self):
        self.vehicles = {"old": (100.0, EAST_LEFT, "truck"), "a": (1500.0, EAST_STRAIGHT, "passenger"),
                         "b": (1999.5, EAST_LEFT, "bus"), "now": (2000.0, EAST_STRAIGHT, "passenger"),
                         "later": (2100.0, EAST_LEFT, "motorcycle")}

    def test_history_uses_only_past_900_seconds(self):
        hist = er.history(self.vehicles, 2000.0)
        self.assertEqual(hist, [(tuple(EAST_STRAIGHT), "passenger"), (tuple(EAST_LEFT), "bus")])   # t0와 그 뒤 차는 없음

    def test_sample_future_draws_from_history_only(self):
        hist = er.history(self.vehicles, 2000.0)
        a = er.sample_future(hist, 2000.0, 1001)
        self.assertEqual(a, er.sample_future(hist, 2000.0, 1001))               # 같은 시나리오는 같다
        self.assertNotEqual(a, er.sample_future(hist, 2000.0, 1002))
        self.assertTrue(all(2000.0 < dep <= 2600.0 for _, _, _, dep in a))
        self.assertTrue({(e, t) for _, e, t, _ in a} <= set(hist))               # 미래의 실제 차(motorcycle)는 나오지 않음
        self.assertEqual(len({v for v, _, _, _ in a}), len(a))
        self.assertTrue(all(v.startswith(er.SAMPLE_PREFIX + "1001_") for v, _, _, _ in a))
        n = [len(er.sample_future(hist * 45, 2000.0, s)) for s in range(300)]    # 최근 90대 → 600초에 평균 60대
        self.assertAlmostEqual(statistics.fmean(n), 60, delta=2.5)
        self.assertEqual(er.sample_future([], 2000.0, 1), [])

    def test_future_ids_from_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.xml"
            root = ET.Element("snapshot")
            for vid in ("a", "b", "now", "later"):
                ET.SubElement(root, "vehicle", id=vid)
            ET.SubElement(root, "rngState", default="0")
            ET.ElementTree(root).write(state)
            self.assertEqual(er.future_ids(state, self.vehicles, 2000.0), ["later"])   # 출발 시각이 t0 뒤인 실제 차만

    def test_types_only_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            route = Path(tmp) / "20220811_0730.rou.xml"
            route.write_text('<routes><vType id="passenger"/><vType id="bus"/><route id="r0" edges="a b"/>'
                             '<vehicle id="v" type="bus" route="r0" depart="1"/></routes>')
            out = er.types_only_file(route)
            self.assertEqual(out.name, "20220811_0730.types.rou.xml")
            root = ET.parse(out).getroot()
            self.assertEqual([c.tag for c in root], ["vType", "vType"])

    def test_neutralize_idle_program(self):
        with tempfile.TemporaryDirectory() as tmp:
            xml = Path(tmp) / "s.xml"
            xml.write_text('<snapshot>\n<tlLogic id="204820" programID="0" phase="10" duration="4000" active="0"/>\n'
                           '<tlLogic id="204820" programID="online" phase="3" duration="5" active="1"/>\n'
                           '<tlLogic id="201292" programID="0" phase="7" duration="6" active="1"/>\n</snapshot>')
            self.assertEqual(er.neutralize_idle_program(xml, "204820"), 1)
            text = xml.read_text()
            self.assertIn('programID="0" phase="0" duration="4000" active="0"', text)    # 쓰이지 않는 프로그램만
            self.assertIn('programID="online" phase="3"', text)
            self.assertIn('id="201292" programID="0" phase="7"', text)

    def test_read_routes_and_movement(self):
        with tempfile.TemporaryDirectory() as tmp:
            route = Path(tmp) / "r.rou.xml"
            route.write_text(f'<routes><route id="r0" edges="{" ".join(EAST_LEFT)}"/>'
                             '<vehicle id="v" type="bus" route="r0" depart="12.50"/></routes>')
            vehicles, routes = er.read_routes(route)
            self.assertEqual(vehicles["v"], (12.5, EAST_LEFT, "bus"))
            self.assertEqual(er.movement_of_edges(EAST_LEFT), ("477706392_02", "477706883_02"))
            self.assertIsNone(er.movement_of_edges(["x", "y"]))

    def test_signal_values_keep_only_plain_values(self):
        ts = SimpleNamespace(green_phase=2, is_yellow=False, yellow_time=np.int64(6), last_reward=np.float32(-1.5),
                             next_action_time=100.0, _from_green=None, env=object(), green_phases=[1, 2])
        v = er.signal_values(ts)
        self.assertEqual(set(v), {"green_phase", "is_yellow", "yellow_time", "last_reward", "next_action_time", "_from_green"})


def _point(i, ep, W, le=0, ls=0, les=False, lss=False):
    return {"state": f"s{i}", "episode": ep, "W": W, "left_east": le, "left_south": ls,
            "left_east_served": les, "left_south_served": lss}


class SelectionTest(unittest.TestCase):
    def test_classify_order_and_rules(self):
        pts = [_point(i, f"e{i % 3}", W=i) for i in range(100)]
        pts[50].update(left_east=40)                          # 상위 10% 이상, 녹색 아님 → 좌회전 정체
        pts[51].update(left_east=40, left_east_served=True)   # 지금 녹색이면 아님
        pts[99].update(left_south=40)                         # W 상위여도 좌회전 정체가 먼저
        cut = er.classify(pts)
        self.assertEqual(pts[50]["class"], "left")
        self.assertNotEqual(pts[51]["class"], "left")
        self.assertEqual(pts[99]["class"], "left")
        self.assertEqual(pts[90]["class"], "congested")
        self.assertEqual(pts[10]["class"], "light")
        self.assertIsNone(pts[60]["class"])
        self.assertAlmostEqual(cut["W80"], 79.2)

    def test_left_needs_three_vehicles(self):
        pts = [_point(i, "e", W=i, le=(2 if i < 20 else 0)) for i in range(100)]   # 상위 10%여도 2대면 아님
        er.classify(pts)
        self.assertFalse(any(p["class"] == "left" for p in pts))

    def test_pick_is_fixed_and_spread(self):
        pts = [_point(i, f"e{i % 4}", W=i) for i in range(400)]
        er.classify(pts)
        a = er.pick(pts)
        b = er.pick([dict(p) for p in pts])
        self.assertEqual([p["state"] for p in a], [p["state"] for p in b])
        for cls in er.CLASSES[1:]:
            got = [p for p in a if p["class"] == cls]
            self.assertEqual(len(got), er.CLASS_N)
            self.assertEqual(len({p["episode"] for p in got}), 4)                 # 에피소드를 돌아가며
        few = [_point(i, "e0", W=i) for i in range(10)]
        er.classify(few)
        self.assertLessEqual(len(er.pick(few)), 10)


def _costs(base, gains_select, gains_confirm):
    """후보 0 = P0. 후보 1의 비용 = base − gain(고르는 용·확인용 각각)."""
    out = {0: {}, 1: {}}
    for s, g in zip(er.SELECT_SEEDS, gains_select):
        out[0][s] = {"main": base, "sens": base}
        out[1][s] = {"main": base - g, "sens": base - g}
    for s, g in zip(er.CONFIRM_SEEDS, gains_confirm):
        out[0][s] = {"main": base, "sens": base}
        out[1][s] = {"main": base - g, "sens": base - g}
    return out


class JudgeTest(unittest.TestCase):
    def test_better_candidate_confirmed(self):
        j = er.judge_state(_costs(1000, [50] * 8, [40, 45, 35, 50, 42, 38, 41, 44]), 0, "main")
        self.assertEqual(j["best"], 1)
        self.assertTrue(j["differs"] and j["significantly_better"])
        self.assertAlmostEqual(j["confirm_gain"], 41.88, places=2)
        self.assertEqual(j["select_gain"], 50)

    def test_tie_keeps_p0(self):
        j = er.judge_state(_costs(1000, [0] * 8, [10] * 8), 0, "main")
        self.assertEqual(j["best"], 0)
        self.assertEqual((j["confirm_gain"], j["differs"]), (0.0, False))

    def test_lucky_selection_shows_as_worse(self):
        j = er.judge_state(_costs(1000, [5] * 8, [-20, -25, -18, -30, -22, -19, -21, -24]), 0, "main")
        self.assertTrue(j["worse"] and j["significantly_worse"])                 # 고르는 용에서만 좋아 보인 후보
        self.assertGreater(j["select_gain"], j["confirm_gain"])

    def test_class_reading(self):
        good = [{"confirm_gain": g, "select_gain": g + 5, "differs": True, "significantly_better": True, "worse": False,
                 "significantly_worse": False} for g in (30, 40, 35, 50, 45, 38)]
        cls = {"congested": er.read_class(good), "left": er.read_class([]), "light": er.read_class(good[:2])}
        self.assertGreater(cls["congested"]["ci95"][0], 0)
        self.assertEqual(er.reading(cls), "이득 있음(congested)")
        flat = [{**r, "confirm_gain": g} for r, g in zip(good, (30, -40, 35, -50, 5, -8))]
        self.assertEqual(er.reading({"congested": er.read_class(flat), "left": er.read_class([])}), "이득이 보이지 않음")


class ExactDelayTest(unittest.TestCase):
    def test_finish_counts_time_loss_and_insertion_wait_exactly(self):
        m = er.DelayMeter.__new__(er.DelayMeter)
        t0 = 1000.0
        m.t0, m.at = t0, {}
        m.desired = {"A": 900.0, "B": 950.0, "C": t0 + 5, "D": t0 - 20, "E": t0 + 250, "F": t0 + 1}
        m.tl0 = {"A": 5.0, "B": 3.0}                             # t0에 도로 위였던 차
        m._marked = {300: {"tl": {"A": 12.0, "D": 2.0, "F": 1.5},     # 표시 시점에 도로 위
                           "pending": ["E"], "departed": {"C": t0 + 10, "D": t0 + 30, "F": t0 + 1},
                           "residual": 4, "increment": 90.0}}
        with tempfile.TemporaryDirectory() as tmp:
            trip = Path(tmp) / "trip.xml"
            trip.write_text(f'<tripinfos><tripinfo id="B" arrival="{t0 + 100}" timeLoss="9.0"/>'
                            f'<tripinfo id="C" arrival="{t0 + 200}" timeLoss="4.0"/>'
                            f'<tripinfo id="F" arrival="{t0 + 400}" timeLoss="7.0"/></tripinfos>')   # F는 표시 뒤 도착
            at = m.finish(trip)[300]
        # 시간 손실 (12−5) + (9−3) + 4 + 2 + 1.5, 진입 대기 C 5 + D 30 + F 0 + E (300 − 250)
        self.assertAlmostEqual(at["time_loss"], 7 + 6 + 4 + 2 + 1.5)
        self.assertAlmostEqual(at["depart_wait"], 5 + 30 + 0 + 50)
        self.assertAlmostEqual(at["delay"], 20.5 + 85)
        self.assertEqual((at["residual"], at["terminal"]), (4, 300.0))
        self.assertAlmostEqual(at["cost"], 105.5 + 300)
        self.assertEqual(at["delay_increment"], 90.0)

    def test_vehicle_arriving_at_mark_counted_once(self):
        m = er.DelayMeter.__new__(er.DelayMeter)
        m.t0, m.at, m.desired, m.tl0 = 0.0, {}, {}, {}
        m._marked = {300: {"tl": {"X": 25.0}, "pending": [], "departed": {}, "residual": 0, "increment": 0.0}}
        with tempfile.TemporaryDirectory() as tmp:
            trip = Path(tmp) / "trip.xml"
            trip.write_text('<tripinfos><tripinfo id="X" arrival="300.00" timeLoss="25.0"/>'
                            '<tripinfo id="U" arrival="-1" timeLoss="9.0"/></tripinfos>')   # 끝나지 않은 차(-1)는 도착이 아님
            self.assertEqual(m.finish(trip)[300]["time_loss"], 25.0)

    def test_cost_check_pairs(self):
        runs = {(0, 1001): {300: {"delay": 100.0, "delay_increment": 98.0}, 600: {"delay": 200.0, "delay_increment": 195.0}},
                (2, 1001): {300: {"delay": 90.0, "delay_increment": 89.5}, 600: {"delay": 210.0, "delay_increment": 204.0}}}
        c = er.cost_check(runs, [0, 2])
        self.assertEqual(c["300"]["pairs"], [{"seed": 1001, "exact_diff": 10.0, "increment_diff": 8.5, "error": -1.5}])
        self.assertEqual(c["600"]["exact_minus_increment"], [5.0, 6.0])


def _full_costs(p0_cost_main, alt_main, p0_sens, alt_sens):
    """후보 0 = P0, 후보 1. 확인용·고르는 용 모두 같은 값(지연과 끝 벌점을 반씩)."""
    out = {0: {}, 1: {}}
    for s_ in (*er.SELECT_SEEDS, *er.CONFIRM_SEEDS):
        for c, (m, n) in ((0, (p0_cost_main, p0_sens)), (1, (alt_main, alt_sens))):
            out[c][s_] = {"main": m, "main_delay": m / 2, "main_terminal": m / 2, "sens": n, "sens_delay": n / 2, "sens_terminal": n / 2}
    return out


class SummaryPartsTest(unittest.TestCase):
    def test_parts_and_persistence(self):
        costs = _full_costs(1000, 900, 2000, 2050)      # 300초로는 후보 1이 낫지만 600초로 보면 못함
        j = er.judge_state(costs, 0, "main")
        self.assertEqual(j["best"], 1)
        self.assertEqual((j["confirm_gain"], j["delay_gain"], j["terminal_gain"]), (100.0, 50.0, 50.0))
        keep = er.evaluate_choice(costs, 0, j["best"], "sens")
        self.assertEqual(keep["confirm_gain"], -50.0)                    # 같은 선택이 600초에서는 손해
        self.assertTrue(keep["worse"])

    def test_summary_end_to_end(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            chosen = [{"state": f"2022081{i}_0730_{i}", "class": c, "episode": f"2022081{i}_0730", "W": 10, "allowed": [0, 1],
                       "p0_action": 0} for i, c in enumerate(er.CLASSES)]
            (out / "sample.json").write_text(json.dumps({"chosen": chosen, "episodes": {
                p["episode"]: {"choice_points_per_hour": 90, "vehicles_per_hour": 5000} for p in chosen}}))
            lines = []
            for p in chosen:
                for c in (0, 1):
                    for s_ in (*er.SELECT_SEEDS, *er.CONFIRM_SEEDS):
                        m300 = 1000 - (60 if c == 1 else 0) + s_ % 7
                        m600 = 2000 - (30 if c == 1 else 0) + s_ % 5
                        mk = lambda v: {"cost": v, "delay": v - 300, "terminal": 300, "delay_increment": v - 301}
                        lines.append(json.dumps({"state": p["state"], "candidate": c, "seed": s_, "code": "abc",
                                                 "marks": {"300": mk(m300), "600": mk(m600)}, "injected": 10, "leaks": 0,
                                                 "changed_by_rules": 0, "setup_s": 2.0, "seconds": 5.0}))
            (out / "rollouts.jsonl").write_text("\n".join(lines) + "\n")
            er.summary(SimpleNamespace(out=str(out)))
            r = json.loads((out / "summary.json").read_text())
            self.assertEqual(r["main"]["classes"]["all"]["mean_confirm_gain"], 60.0)
            self.assertEqual(r["main_at_600"]["classes"]["all"]["mean_confirm_gain"], 30.0)
            self.assertEqual(r["main"]["classes"]["all"]["mean_delay_gain"], 60.0)        # 끝 벌점은 같음
            self.assertEqual(set(r["by_date"]), {"20220810", "20220811", "20220812"})
            self.assertNotIn("차당", (out / "summary.md").read_text())
            (out / "rollouts.jsonl").write_text("\n".join(lines[:-1] + [lines[-1].replace('"abc"', '"xyz"')]) + "\n")
            with self.assertRaises(SystemExit):
                er.summary(SimpleNamespace(out=str(out)))                                # 다른 코드 결과가 섞이면 멈춤


if __name__ == "__main__":
    unittest.main()
