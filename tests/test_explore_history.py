"""PI1 수요 이력의 경계·미래 유출·마스크·채택·묶음 제외를 검사한다(SUMO 없음)."""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import explore_history as eh  # noqa: E402


class HistoryTest(unittest.TestCase):
    def test_schema_matches_original_definitions(self):
        spec = eh.schema()
        self.assertEqual((len(spec["movements"]), len(spec["type_ids"]), spec["width"]), (14, 13, 29))
        self.assertEqual(spec["movements"], sorted(spec["movements"]))
        self.assertEqual(spec["type_ids"], sorted(spec["type_ids"]))
        self.assertEqual((eh.BASE_WIDTH, eh.WIDTH, eh.STEPS, eh.SEEDS), (96, 125, 10_000, (0, 1, 2, 3, 4)))

    def test_window_boundaries_unknown_and_first_movement(self):
        spec = eh.schema()
        a, b = spec["movements"][:2]
        vt = spec["type_ids"][0]
        # 100은 창의 왼쪽 끝이므로 포함하고, 현재인 1000은 뺀다. 경로의 첫 알려진 이동을 쓴다.
        cars = {"past": (99.99, a, vt), "left": (100.0, a, vt),
                "inside": (999.99, [*a, *b], vt), "now": (1000.0, b, vt),
                "future": (1000.01, b, vt), "unknown": (500.0, ["outside", "exit"], "new_type")}
        f = eh.history_features(cars, 1000.0)
        self.assertEqual(len(f), 29)
        self.assertAlmostEqual(f[0], math.log1p(2) / math.log1p(1000))
        self.assertEqual(f[1], 0.0)
        self.assertAlmostEqual(f[14], math.log1p(1) / math.log1p(1000))
        self.assertAlmostEqual(f[15], math.log1p(2) / math.log1p(1000))
        self.assertAlmostEqual(f[28], math.log1p(1) / math.log1p(1000))

    def test_log_scale_and_clip(self):
        spec = eh.schema()
        movement, vt = spec["movements"][0], spec["type_ids"][0]
        cars = {f"v{i}": (1.0, movement, vt) for i in range(1001)}
        f = eh.history_features(cars, 10.0)
        self.assertEqual((f[0], f[15]), (1.0, 1.0))
        self.assertTrue(all(0 <= x <= 1 for x in f))

    def test_mask_stays_at_end(self):
        base = list(range(92)) + [1, 0, 1, 0]
        got = eh.observation_with_history(base, [0.25] * 29)
        self.assertEqual(len(got), 125)
        self.assertEqual(got[:92], base[:92])
        self.assertEqual(got[92:121], [0.25] * 29)
        self.assertEqual(got[-4:], [1.0, 0.0, 1.0, 0.0])
        with self.assertRaises(ValueError):
            eh.observation_with_history(base[:-1], [0.0] * 29)
        with self.assertRaises(ValueError):
            eh.observation_with_history(base, [float("nan")] * 29)

    def test_future_replacement_never_returns_real_future(self):
        spec = eh.schema()
        a, b = spec["movements"][:2]
        vt = spec["type_ids"][0]
        real = {"old": (900.0, a, vt), "boundary": (1000.0, a, vt), "real_future": (1010.0, a, vt)}
        sampled = [("sampled", b, vt, 1020.0)]
        got = eh.history_vehicles(real, 1000.0, sampled)
        self.assertEqual(set(got), {"old", "boundary", "sampled"})
        self.assertIn("real_future", real)  # 원본 목록을 고치지 않는다.
        self.assertEqual(got["sampled"], (1020.0, list(b), vt))
        # 진행한 시각의 이력에도 실제 미래는 안 들어가고 표본만 들어간다.
        f = eh.history_features(got, 1030.0)
        self.assertAlmostEqual(f[0], math.log1p(2) / math.log1p(1000))
        self.assertAlmostEqual(f[1], math.log1p(1) / math.log1p(1000))
        self.assertEqual(eh.history_vehicles(real, 1000.0, {"sampled": (1020.0, b, vt)}), got)

    def test_invalid_injected_departures_and_duplicates_fail(self):
        with self.assertRaises(ValueError):
            eh.history_vehicles({}, 100.0, [("x", [], "passenger", 100.0)])
        with self.assertRaises(ValueError):
            eh.history_vehicles({"x": (10.0, [], "passenger")}, 100.0, [("x", [], "passenger", 101.0)])
        with self.assertRaises(ValueError):
            eh.history_vehicles({}, 100.0, [("x", [], "passenger", 101.0), ("x", [], "passenger", 102.0)])


class AdoptionTest(unittest.TestCase):
    def test_all_three_requirements_are_needed(self):
        g = [f"g{i}" for i in range(6)]
        base = dict(zip(g, [-1.0] * 6))
        yes = dict(zip(g, [3.0, 3.0, 3.0, 3.0, -1.0, -1.0]))
        self.assertTrue(eh.adoption(base, yes)["uses_history"])
        self.assertFalse(eh.adoption(base, dict(zip(g, [2, 2, 2, -1, -1, -1])))["uses_history"])  # 세 묶음만
        self.assertFalse(eh.adoption(base, dict(zip(g, [-0.5] * 6)))["uses_history"])  # 총 이득 음수
        self.assertFalse(eh.adoption(yes, yes)["uses_history"])  # 엄격한 개선이 필요
        with self.assertRaises(ValueError):
            eh.adoption({"x": 1.0}, {"x": 2.0})
        with self.assertRaises(ValueError):
            eh.adoption(base, {**yes, "g0": float("nan")})


class CacheTest(unittest.TestCase):
    def test_cache_rejects_new_source_or_configuration(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cache.json"
            ident = {"source": "old", "steps": 10000}
            self.assertIsNone(eh.guard_cache(path, ident))
            eh._write_json(path, {"identity": ident, "answer": 1})
            self.assertEqual(eh.guard_cache(path, ident)["answer"], 1)
            with self.assertRaises(SystemExit):
                eh.guard_cache(path, {**ident, "source": "changed"})
            with self.assertRaises(SystemExit):
                eh.guard_cache(path, {**ident, "steps": 20000})

    def test_preserved_route_comparison_ignores_route_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / "a.xml", Path(temp) / "b.xml"
            a.write_text('<routes><route id="r0" edges="a b"/><vehicle id="v" type="t" route="r0" depart="1.0"/></routes>')
            b.write_text('<routes><route id="renamed" edges="a b"/><vehicle id="v" type="t" route="renamed" depart="1"/></routes>')
            self.assertTrue(eh._same_routes(a, b))
            b.write_text(b.read_text().replace('depart="1"', 'depart="2"'))
            self.assertFalse(eh._same_routes(a, b))

    def test_regeneration_preserves_original_origin_and_tail(self):
        """학습 시각을 원점으로 삼거나 인계 지연만큼 창을 늘리지 않는다."""
        import focused_scenario as scenario
        import focused_train as train
        with tempfile.TemporaryDirectory() as temp:
            route = Path(temp) / "routes" / "day.rou.xml"
            def generated(date, origin, end, target, **kwargs):
                target.write_text("<routes/>")
            with patch.object(train, "warm_start", return_value={"handover_s": 1834.0}) as warm, \
                    patch.object(scenario, "window_route_file", side_effect=generated) as window, \
                    patch("focused_data.load_counts", return_value={"fixture": 1}):
                rec = eh._regenerate("20220812", "1800", route)
            origin = scenario.period_start_s("1800") - train.WARMUP_S
            warm.assert_called_once_with("20220812", "1800", {"fixture": 1}, route)
            window.assert_called_once_with("20220812", origin, origin + train.WARMUP_S + train.CONTROL_S + 900,
                                           route, origin_s=origin, counts={"fixture": 1})
            self.assertEqual((rec["origin_s"], rec["handover"]), (origin, 1834.0))


def _sample(group):
    return {"group": group, "obs96": np.zeros(96, dtype=np.float32), "obs125": np.zeros(125, dtype=np.float32),
            "allowed": [0, 1], "p0": 0, "costs": {0: [100.0] * 8, 1: [90.0] * 8}}


class DiagnosticTest(unittest.TestCase):
    def test_group_exclusion_and_fallback_weight_copy(self):
        """진단은 제외 묶음을 학습에 넣지 않고, 미채택 시 원래 B 파일을 복사한다."""
        samples = [_sample(f"g{i}") for i in range(6)]
        trained = []
        def fake_fit(rows, width):
            trained.append((width, {s["group"] for s in rows}))
            return [width]
        def fake_picks(states, rows, width):
            self.assertNotIn(rows[0]["group"], trained[-1][1])
            return [0] * len(rows)  # 두 관측 모두 이득 0: 미채택.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            work, out, combo = root / "work", root / "out", root / "combo"
            work.mkdir()
            (combo / "students" / "b").mkdir(parents=True)
            for seed in eh.SEEDS:
                (combo / "students" / "b" / f"seed{seed}.pt").write_bytes(f"frozen{seed}".encode())
            (work / "history_manifest.json").write_text(json.dumps({"identity": {"data": {"old": "frozen"}}}))
            with patch.object(eh, "load_samples", return_value=samples), patch.object(eh, "fit", side_effect=fake_fit), patch.object(eh, "picks", side_effect=fake_picks):
                result = eh.diagnostic_and_parent(root / "old", root / "new", combo, out, work)
            self.assertEqual(len(trained), 12)
            self.assertFalse(result["policy"]["uses_history"])
            self.assertEqual(result["policy"]["width"], 96)
            for seed in eh.SEEDS:
                self.assertEqual((out / "parent" / f"seed{seed}.pt").read_bytes(), f"frozen{seed}".encode())
            # 캐시 결과도 고정 부모의 파일 변경을 확인한다.
            (out / "parent" / "seed0.pt").write_bytes(b"changed")
            with patch.object(eh, "load_samples", return_value=samples), self.assertRaises(SystemExit):
                eh.diagnostic_and_parent(root / "old", root / "new", combo, out, work)

    def test_history_adoption_fits_full_parent_only_after_excluded_evaluation(self):
        """125칸 채택 때만 전체 자료로 부모 하나를 만들고, 완료 캐시는 다시 학습하지 않는다."""
        samples = [_sample(f"g{i}") for i in range(6)]
        fitted = []
        def fake_fit(rows, width):
            fitted.append((width, len(rows)))
            return [{"fixture": width}] * 5
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            work, out = root / "work", root / "out"
            work.mkdir()
            (work / "history_manifest.json").write_text(json.dumps({"identity": {"data": {}}}))
            with patch.object(eh, "load_samples", return_value=samples), \
                    patch.object(eh, "fit", side_effect=fake_fit), \
                    patch.object(eh, "picks", side_effect=lambda states, rows, width: [int(width == 125)] * len(rows)):
                result = eh.diagnostic_and_parent(root / "old", root / "new", root / "combo", out, work)
            self.assertEqual(fitted, [(w, 5) for _ in range(6) for w in (96, 125)] + [(125, 6)])
            self.assertTrue(result["policy"]["uses_history"])
            self.assertEqual(result["policy"]["width"], 125)
            with patch.object(eh, "load_samples", return_value=samples), patch.object(eh, "fit") as no_more:
                cached = eh.diagnostic_and_parent(root / "old", root / "new", root / "combo", out, work)
            no_more.assert_not_called()
            self.assertEqual(cached["policy"]["weights"], result["policy"]["weights"])


if __name__ == "__main__":
    unittest.main()
