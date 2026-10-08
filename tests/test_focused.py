"""석천사거리 중심 실험의 기본 정합성 검사."""
import json
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from focused_data import COUNTS, find_network, load_counts  # noqa: E402
from focused_scenario import (FocusedNetwork, aihub_states, replay_phases,  # noqa: E402
                              write_dqn_template_network, write_routes)


@unittest.skipUnless(COUNTS.exists(), "counts_204820.csv가 없으면 건너뜀")
class FocusedScenarioChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fn = FocusedNetwork()
        cls.counts = load_counts(COUNTS)

    def test_vehicles_equal_recorded_counts(self):
        rows = self.counts[("20220810", "1800")]
        with tempfile.TemporaryDirectory() as tmp:
            info = write_routes(self.fn, rows, Path(tmp) / "r.rou.xml", seed=1)
            root = ET.parse(Path(tmp) / "r.rou.xml").getroot()
        self.assertEqual(info["vehicles"], sum(r["count"] for r in rows))
        self.assertEqual(len(root.findall("vehicle")), sum(r["count"] for r in rows))
        types = Counter(v.get("type") for v in root.findall("vehicle"))
        expected = Counter()
        for r in rows:
            expected[r["vehicle_type"]] += r["count"]
        self.assertEqual(types, expected)

    def test_routes_are_connected_and_cross_once(self):
        for (a, b) in self.fn.movements:
            route = self.fn.route_for(a, b)
            for x, y in zip(route, route[1:]):
                self.assertIn(y, self.fn.succ[x])
            crossings = sum(1 for x, y in zip(route, route[1:]) if (x, y) in self.fn.movements)
            self.assertEqual(crossings, 1)
            self.assertNotEqual(route[-1], a)  # 진입 도로에서 끝나는 경로 없음

    def test_entry_branch_split_keeps_records_and_follows_upstream_share(self):
        """길주로 서쪽 접근로(477707013_02) 차량을 두 갈래에 나눠도 이동별 대수는 기록과 같고, 4차로 가지 대수는 상류 기록 비율을 따른다."""
        from focused_scenario import ENTRY_BRANCHES, entry_branch_shares
        rows = self.counts[("20220811", "0830")]
        with tempfile.TemporaryDirectory() as tmp:
            info = write_routes(self.fn, rows, Path(tmp) / "r.rou.xml", seed=1)
            root = ET.parse(Path(tmp) / "r.rou.xml").getroot()
        routes = {r.get("id"): r.get("edges").split() for r in root.findall("route")}
        moves, first = Counter(), Counter()
        for v in root.findall("vehicle"):
            e = routes[v.get("route")]
            mv = next((a, b) for a, b in zip(e, e[1:]) if (a, b) in self.fn.movements)
            moves[mv] += 1
            if mv[0] == "477707013_02":
                first[e[0]] += 1
        expected = Counter()
        for r in rows:
            expected[(r["from_edge"], r["to_edge"])] += r["count"]
        self.assertEqual(moves, expected)
        share = entry_branch_shares(rows)["477707013_02"]
        self.assertAlmostEqual(info["entry_branch_shares"]["477707013_02"], share)
        spec = ENTRY_BRANCHES["477707013_02"]
        volume = sum(first.values())
        self.assertLessEqual(abs(first[spec["main"]] - share * volume), 2)
        self.assertEqual(first[spec["main"]] + first[spec["alternative"]], volume)

    def test_alternative_branch_route_is_connected(self):
        for b in ("477706392_01", "477706883_02", "477706911_02_02"):
            route = self.fn.route_for("477707013_02", b, "alternative")
            self.assertEqual(route[0], "477744390_01")
            for x, y in zip(route, route[1:]):
                self.assertIn(y, self.fn.succ[x])

    def test_template_network_changes_only_target_signal(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = write_dqn_template_network(Path(tmp) / "t.net.xml")
            a = ET.parse(find_network()).getroot()
            b = ET.parse(out).getroot()
        strip = lambda r: [ET.tostring(e) for e in r if not (e.tag == "tlLogic" and e.get("id") == "204820")]
        self.assertEqual(strip(a), strip(b))
        greens = [p.get("state") for p in b.find("tlLogic[@id='204820']").findall("phase") if "y" not in p.get("state")]
        self.assertEqual(len(greens), 4)

    def test_window_routes_keep_each_period_and_mark_its_vehicles(self):
        """여러 15분 시간대를 합친 차량 파일은 시간대별 기록 대수를 그대로 담고, 차량 ID에 시간대를 붙인다."""
        from collections import Counter
        from focused_scenario import period_start_s, window_route_file
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "w.rou.xml"
            period_of = window_route_file("20220811", 8 * 3600, 8 * 3600 + 1800, path, counts=self.counts)
            root = ET.parse(path).getroot()
        self.assertEqual(Counter(period_of.values()),
                         {p: sum(r["count"] for r in self.counts[("20220811", p)]) for p in ("0800", "0815")})
        self.assertTrue(all(v.split("_")[0] == p for v, p in period_of.items()))
        departs = [float(v.get("depart")) for v in root.findall("vehicle")]
        self.assertGreaterEqual(min(departs), 0)
        self.assertLess(max(departs), 1800)
        self.assertEqual(period_start_s("0815"), 8 * 3600 + 900)

    def test_day_route_cache_is_keyed_by_environment_fingerprint(self):
        """조건(코드·설정·자료)이 바뀌면 지문이 바뀌어, 예전 차량 파일을 다시 쓰지 않는다."""
        from focused_data import ENV_FILES, environment_fingerprint, fingerprint
        from focused_scenario import day_route_file
        fp = environment_fingerprint()
        self.assertEqual(fp, fingerprint(ENV_FILES))
        self.assertNotEqual(fp, fingerprint(ENV_FILES[:-1]))
        with tempfile.TemporaryDirectory() as tmp:
            path, period_of = day_route_file("20220811", Path(tmp))
            self.assertIn(fp, path.name)
            self.assertEqual(len(period_of), sum(r["count"] for (d, _), rows in self.counts.items()
                                                 if d == "20220811" for r in rows))

    def test_corrected_replay_uses_same_green_states(self):
        info = aihub_states()
        phases = replay_phases("0800")
        states = {s for _, s in phases if "y" not in s}
        self.assertTrue(states <= set(info["greens"]))
        self.assertGreaterEqual(sum(d for d, _ in phases), 1800)


class CollisionJudgeChecks(unittest.TestCase):
    """충돌은 모두 세고, 원인을 확인한 지도 오류와 모양이 같은 것만 예외로 둔다."""
    # 2022-08-19 점검에서 실제로 기록된 충돌(등록된 예외의 근거)
    SEEN = {"time": "72372.00", "type": "junction", "lane": ":SN20220622173835A_5_6", "pos": "12.00",
            "collider": "2000_b450", "victim": "2000_p449", "colliderType": "bus", "victimType": "passenger",
            "colliderSpeed": "13.89", "victimSpeed": "0.00", "colliderFront": "1957.16,5126.87",
            "colliderBack": "1944.93,5125.72", "victimFront": "1956.63,5123.71", "victimBack": "1951.72,5124.55"}

    def judge(self, events):
        from focused_eval import judge_collisions
        with tempfile.TemporaryDirectory() as tmp:
            root = ET.Element("collisions")
            for e in events:
                ET.SubElement(root, "collision", e)
            ET.ElementTree(root).write(Path(tmp) / "collisions.xml")
            return judge_collisions(Path(tmp) / "collisions.xml")

    def test_only_the_confirmed_split_overlap_is_excepted(self):
        (ax, ay), (bx, by) = (1944.93, 5125.72), (1952.94, 5124.34)     # 5번 갈래 중심선
        n = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
        deep = f"{bx - 4 * (bx - ax) / n:.2f},{by - 4 * (by - ay) / n:.2f}"   # 분기점 끝에서 4m 안쪽
        variants = {
            "seen": {},
            "victim_moving": {"victimSpeed": "1.20"},
            "inside_204820": {"lane": ":204820_12_0"},
            "rear_end_on_lane": {"type": "collision", "lane": "477715088_0101_02_5"},
            "mirror_branch": {"lane": ":SN20220622173835A_5_5"},
            "tail_deep_in_split": {"victimBack": deep},
            "victim_on_other_lane": {"victimBack": "1951.18,5121.40"},
        }
        events = [{**self.SEEN, **change, "collider": name} for name, change in variants.items()]
        result = self.judge(events)
        self.assertEqual(result["collisions_total"], 7)
        self.assertEqual(result["collisions_known_map_defect"], {"west_split_SN20220622173835A": 1})
        self.assertEqual([e["collider"] for e in result["collisions_unexplained"]], list(variants)[1:])
        self.assertEqual(result["collisions_at_204820"], 1)

    def test_exception_still_matches_the_map(self):
        """예외의 근거인 지도 모양(막힌 6번 차로, 5번 차로가 두 갈래로 나뉨, 양보 관계 없음)이 그대로인지."""
        rule = json.loads((ROOT / "config/collision_exceptions.json").read_text())["exceptions"][0]["match"]
        net = ET.parse(find_network()).getroot()
        shapes = {l.get("id"): [tuple(map(float, p.split(","))) for p in l.get("shape").split()] for l in net.iter("lane")}
        self.assertIn(rule["collider_lane"], shapes)
        back = shapes[rule["victim_back_lane"]]
        self.assertEqual([back[0], back[-1]], [tuple(p) for p in rule["victim_back_lane_shape"]])
        junction = next(j for j in net.iter("junction") if j.get("id") == "SN20220622173835A")
        self.assertTrue(all(set(r.get("foes")) == {"0"} for r in junction.iter("request")))
        to_lane6 = [c for c in net.iter("connection") if c.get("from") == "477715088_0102_02" and c.get("toLane") == "6"]
        self.assertEqual([(c.get("fromLane"), c.get("via")) for c in to_lane6], [("5", rule["collider_lane"])])
        lane6 = [c for c in net.iter("connection")
                 if (c.get("from"), c.get("fromLane")) == ("477715088_0102_02", "6") or
                 (c.get("to"), c.get("toLane")) == ("477715088_0102_02", "6")]
        self.assertEqual(lane6, [])

    def test_missing_collision_record_is_not_counted_as_zero(self):
        from focused_eval import judge_collisions
        with self.assertRaises(FileNotFoundError):
            judge_collisions(Path(tempfile.gettempdir()) / "no_such_run" / "collisions.xml")


class EvalBookkeepingChecks(unittest.TestCase):
    """DQN 결과는 모델 파일 지문으로 구분하고, 빠진 평가가 있으면 비교표를 만들지 않는다."""

    @staticmethod
    def model(root: Path, seed: int, content: bytes) -> str:
        root.mkdir(parents=True)
        (root / "training_config.json").write_text(json.dumps({"seed": seed, "timesteps": 1000}))
        (root / "dqn_204820.zip").write_bytes(content)
        return str(root)

    def test_changed_model_gets_its_own_result_folder(self):
        from focused_eval import planned_runs
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self.model(root / "v1/seed_42", 42, b"weights after 100k")
            b = self.model(root / "v2/seed_42", 42, b"weights after 200k")     # 폴더 이름은 같고 모델은 다름
            folder = lambda m: planned_runs(root / "eval", ["20220928"], ["dqn"], [m], 42)[0]["run_dir"]
            self.assertNotEqual(folder(a), folder(b))
            self.assertEqual(folder(a), folder(a))
            Path(a, "dqn_204820.zip").write_bytes(b"trained further")             # 같은 폴더의 모델을 바꾸면 새 결과 폴더
            self.assertNotEqual(folder(a), planned_runs(root / "eval", ["20220928"], ["dqn"], [b], 42)[0]["run_dir"])
            with self.assertRaises(SystemExit):                                   # 한 비교에 같은 seed 모델 둘
                planned_runs(root / "eval", ["20220928"], ["dqn"], [a, b], 42)

    def test_report_refuses_when_a_run_is_missing(self):
        import argparse
        from focused_eval import planned_runs, report
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m = self.model(root / "seed_42", 42, b"weights")
            dates = ["20220928", "20220813"]
            for r in planned_runs(root / "eval/fp", dates, ["fixed", "actuated", "greedy", "dqn"], [m], 42):
                if not (r["policy"] == "dqn" and r["date"] == "20220813"):          # DQN의 하루치가 빠짐
                    Path(r["run_dir"]).mkdir(parents=True)
                    (Path(r["run_dir"]) / "metrics.json").write_text("{}")
            args = argparse.Namespace(dates=dates, models=[m], eval_seed=42, arrivals="random", fingerprint="fp",
                                      out=str(root / "eval"))
            with self.assertRaises(SystemExit) as caught:
                report(args)
            self.assertIn("빠진 평가가 1개", str(caught.exception))


class SimpleControlChecks(unittest.TestCase):
    """단순 제어: 규칙상 켤 수 있는 녹색 가운데 기다리는 차가 가장 많은 것. 같으면 지금 녹색 유지."""

    def test_picks_the_allowed_green_with_most_waiting(self):
        from types import SimpleNamespace
        from focused_eval import QueueFirst
        from focused_rl import MOVEMENT_LINKS
        from focused_scenario import signal_greens
        greens = signal_greens()
        served = [{m for m, links in MOVEMENT_LINKS.items() if any(g[l] == "G" for l in links)} for g in greens]
        only = lambda i: next(m for m in served[i] if all(m not in served[j] for j in range(4) if j != i))
        m1, m2, m3 = only(1), only(2), only(3)          # 녹색 하나에만 속한 이동
        movement_of = {"a": m1, "b": m1, "c": m2, "d": m2, "e": m3, "f": m2}
        pending = ["d"]

        def env(current):
            sumo = SimpleNamespace(
                edge=SimpleNamespace(getLastStepVehicleIDs=lambda e: ["a", "b", "c", "e"] if e == "z" else []),
                vehicle=SimpleNamespace(getSpeed=lambda v: 5.0 if v == "e" else 0.0),   # e는 달리는 중(세지 않음)
                simulation=SimpleNamespace(getPendingVehicles=lambda: pending))
            return SimpleNamespace(sumo=sumo, traffic_signals={"204820": SimpleNamespace(green_phase=current)})

        q = QueueFirst(movement_of, greens)
        q.edges = ["z"]
        # 기다리는 차: 녹색 1 = 2대(a, b), 녹색 2 = 2대(c + 진입 대기 d), 녹색 3 = 0대(e는 달림)
        self.assertEqual(q(env(0), [0, 1, 2, 3]), 1)          # 같으면 번호가 작은 녹색
        self.assertEqual(q(env(2), [0, 1, 2, 3]), 2)          # 같으면 지금 녹색 유지
        self.assertEqual(q(env(0), [0, 3]), 0)                # 켤 수 없는 녹색(1, 2)은 고르지 않는다
        self.assertEqual(q(env(1), [3]), 3)                   # 규칙이 하나만 남기면 그것
        pending.append("f")                                    # 녹색 2에 한 대 더 → 3대
        self.assertEqual(q(env(1), [0, 1, 2, 3]), 2)          # 지금 녹색(1)보다 많으면 바꾼다


class PlatoonVariantChecks(unittest.TestCase):
    """동쪽 무리 도착 민감도: 동쪽 차량의 출발 시각만 이웃 신호 204822의 녹색에 맞춘다."""

    def test_release_greens_are_found_from_data(self):
        config = json.loads((ROOT / "config/east_platoon.json").read_text())
        moves = config["movements"]
        self.assertGreaterEqual(moves["477714888_02"]["correlation"][moves["477714888_02"]["release_state"]], 0.5)   # 직진
        self.assertIsNone(moves["477707011_01"]["release_state"])                                                  # 우회전
        split = json.loads((ROOT / "config/focused_split.json").read_text())
        self.assertFalse(set(config["dates_used"]) & set(split["test"]))      # 시험 날짜는 쓰지 않았다

    def test_only_east_departures_move_into_upstream_greens(self):
        from focused_platoon import EAST_APPROACH, PLATOON_CONFIG, platoon_route_file, upstream_states
        from focused_scenario import UPSTREAM_LIMIT_M, day_route_file
        date = "20220811"
        start = FocusedNetwork()._chain(EAST_APPROACH, True, UPSTREAM_LIMIT_M)[0]
        with tempfile.TemporaryDirectory() as tmp:
            base, _ = day_route_file(date, Path(tmp))
            variant, _ = platoon_route_file(date, Path(tmp))
            def load(f):
                root = ET.parse(f).getroot()
                routes = {r.get("id"): r.get("edges") for r in root.iter("route")}
                return [(v.get("id"), v.get("type"), routes[v.get("route")], float(v.get("depart"))) for v in root.iter("vehicle")]
            a, b = {x[0]: x for x in load(base)}, load(variant)
        self.assertEqual(sorted(a), sorted(x[0] for x in b))
        self.assertEqual([x[3] for x in b], sorted(x[3] for x in b))                      # 출발 순서 정렬
        moved = [x for x in b if x[3] != a[x[0]][3]]
        self.assertTrue(all(x[1:3] == a[x[0]][1:3] for x in b))                            # 차종·경로 그대로
        self.assertTrue(moved and all(x[2].split()[0] == start for x in moved))           # 동쪽 차량만
        through = json.loads(PLATOON_CONFIG.read_text())["movements"]["477714888_02"]["release_state"]
        states = upstream_states(date)
        in_green = sum(states[min(86399, int(x[3]))] == through for x in moved) / len(moved)
        self.assertGreater(in_green, 0.6)                                                   # 대부분 직진 녹색 안에서 출발


if __name__ == "__main__":
    unittest.main()
