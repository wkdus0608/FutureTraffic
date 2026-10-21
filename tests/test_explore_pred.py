"""[추가 탐색] 예측 제어(98절)와 PRED 판정 검사 — SUMO 실행 없음(지도 파일은 있을 때만 읽음)."""
import gzip
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import explore_pred as ep  # noqa: E402
from focused_rl import CROSSWALKS, SignalConstraints  # noqa: E402

PET1 = {(0, 2): 2, (1, 0): 1, (1, 2): 2, (1, 3): 1, (2, 0): 2, (3, 1): 1}
JILJU, SEOKCHEON = tuple(CROSSWALKS)          # 길주로 횡단(녹색 3), 석천로 횡단(녹색 1)


def guard() -> SignalConstraints:
    g = SignalConstraints()
    g.phase_min, g.serving = [15, 33, 15, 46], {JILJU: [3], SEOKCHEON: [1]}
    g.base_yellow, g.all_red = 4, dict(PET1)
    return g


def model(**kw) -> ep.SignalModel:
    served = [[(0, 0.5)], [(1, 1.5), (2, 1.0)], [(3, 0.5)], [(4, 1.0), (5, 1.0)]]
    g = guard()
    return ep.SignalModel(served=served, phase_min=g.phase_min, max_green=120, step=5, guard=g,
                          crosswalks=tuple(CROSSWALKS), **kw)


def state(phase=0, tau=60, now=1000.0, served=(990.0, 990.0), ig=4.0):
    return ep.SignalState(phase, now - tau, ig, served)


class RuleTest(unittest.TestCase):
    def test_hold_and_max_green(self):
        m = model()
        self.assertEqual(m.actions(state(phase=0, tau=10), 1000.0), [0])          # 전환 4초 + 최소 15초 전
        self.assertNotIn(0, m.actions(state(phase=0, tau=125), 1000.0))           # 최대 녹색 120초
        self.assertIn(0, m.actions(state(phase=0, tau=60), 1000.0))

    def test_pedestrian_deadline_matches_guard(self):
        m = model()
        st = state(phase=0, tau=60, served=(1000.0, 1000.0 - 116))                 # 석천로 횡단 120초 한도까지 4초
        self.assertEqual(m.actions(st, 1000.0), [1])
        g, ts = m.guard, SimpleNamespace(env=SimpleNamespace(delta_time=5))
        late = {p: g._outcome(ts, 0, p, 1000.0, {JILJU: 1000.0, SEOKCHEON: 884.0}) for p in range(4)}
        self.assertEqual([p for p in range(4) if late[p] <= 0], [1])

    def test_advance_service_and_crosswalk(self):
        m = model()
        st, eff = m.advance(state(phase=1, tau=60), 1000.0, 1)
        self.assertEqual(eff, 5.0)
        self.assertEqual(st.served[1], 1005.0)                                     # 녹색 1이 석천로 횡단을 연다
        st, eff = m.advance(state(phase=1, tau=60), 1000.0, 3)
        self.assertEqual((st.phase, st.switch_time, st.intergreen, eff), (3, 1000.0, 5.0, 0.0))   # 황색 4 + 전적색 1
        self.assertEqual(st.served, (1005.0, 990.0))                               # 전환 5초가 끝난 1005초에 녹색 3이 길주로 횡단을 엶
        st, _ = m.advance(state(phase=1, tau=60), 1000.0, 2)
        self.assertEqual(st.served, (990.0, 990.0))                                # 1 → 2는 황색 4 + 전적색 2: 1005초에는 아직 전환 중
        st, eff = m.advance(ep.SignalState(3, 996.0, 4.0, (990.0, 990.0)), 1000.0, 3)
        self.assertEqual(eff, 3.0)                                                 # 녹색 1000초 시작 + 출발 손실 2초
        self.assertEqual(st.served[0], 1005.0)


class PlanTest(unittest.TestCase):
    def test_merge_gives_same_best_plan_as_full_search(self):
        rng = random.Random(7)
        m = model()
        for trial in range(25):
            now = 1000.0
            st = state(phase=rng.randrange(4), tau=rng.choice([20, 40, 60, 90]),
                       served=tuple(now - rng.uniform(0, 90) for _ in CROSSWALKS))
            q0 = [rng.uniform(0, 20) for _ in range(6)]
            arrivals = [[rng.uniform(0, 1.5) for _ in range(6)] for _ in range(9)]
            first = m.actions(st, now)
            a1, i1 = ep.plan(m, st, now, q0, arrivals, first, merge=True)
            a2, i2 = ep.plan(m, st, now, q0, arrivals, first, merge=False)
            self.assertAlmostEqual(i1["cost"], i2["cost"], places=6, msg=f"시도 {trial}")
            self.assertEqual(a1, a2, f"시도 {trial}")
            self.assertLessEqual(i1["nodes"], i2["nodes"])

    def test_empty_road_keeps_current(self):
        m = model()
        st = state(phase=2, tau=60)
        a, info = ep.plan(m, st, 1000.0, [0.0] * 6, [[0.0] * 6 for _ in range(24)], m.actions(st, 1000.0))
        self.assertEqual(a, 2)
        self.assertEqual(info["cost"], 0.0)

    def test_switches_to_the_waiting_queue(self):
        m = model()
        st = state(phase=0, tau=60, served=(1000.0, 1000.0))
        q0 = [0, 0, 0, 0, 30, 30]
        a, info = ep.plan(m, st, 1000.0, q0, [[0.0] * 6 for _ in range(24)], m.actions(st, 1000.0))
        self.assertEqual(a, 3)
        self.assertEqual(info["plan"][0], 3)

    def test_first_action_only_from_allowed(self):
        m = model()
        st = state(phase=0, tau=60, served=(1000.0, 1000.0))
        a, _ = ep.plan(m, st, 1000.0, [0, 0, 0, 0, 30, 30], [[0.0] * 6 for _ in range(24)], [0, 1])
        self.assertIn(a, [0, 1])

    def test_insert_dominance_and_tie(self):
        nodes = []
        ep._insert(nodes, 10.0, (1.0, 2.0), (False, 0, (0,)))
        ep._insert(nodes, 11.0, (1.0, 3.0), (False, 0, (0,)))             # 밀림
        self.assertEqual(len(nodes), 1)
        ep._insert(nodes, 9.0, (2.0, 1.0), (True, 1, (1,)))               # 서로 못 밀어냄
        self.assertEqual(len(nodes), 2)
        ep._insert(nodes, 8.0, (0.5, 0.5), (True, 2, (2,)))               # 둘 다 밀어냄
        self.assertEqual(nodes, [(8.0, (0.5, 0.5), (True, 2, (2,)))])
        ep._insert(nodes, 8.0, (0.5, 0.5), (False, 0, (0,)))              # 모두 같으면 동점 규칙(지금 녹색 유지)
        self.assertEqual(nodes, [(8.0, (0.5, 0.5), (False, 0, (0,)))])


class InformationTest(unittest.TestCase):
    def test_neighbor_history_and_prediction(self):
        durations = [30, 5, 60, 40, 10, 50]
        on = [True, False, False, True, True, False]
        cur, elapsed, med = ep.neighbor_history(durations, on, 5, 20.0)
        self.assertEqual((cur, elapsed), (False, 20.0))
        self.assertEqual(med, {True: 40.0, False: 65.0})                          # 켜짐 50·30, 꺼짐 65
        self.assertFalse(ep.predicted_on(cur, elapsed, med, 0.0))
        self.assertFalse(ep.predicted_on(cur, elapsed, med, 44.0))
        self.assertTrue(ep.predicted_on(cur, elapsed, med, 46.0))
        self.assertFalse(ep.predicted_on(cur, elapsed, med, 87.0))
        _, _, short = ep.neighbor_history(durations, on, 5, 20.0, window=60.0)
        self.assertEqual(short, {True: 50.0, False: ep.DEFAULT_RUN_S[False]})
        self.assertTrue(ep.predicted_on(True, 100.0, {True: 40.0, False: 60.0}, 0.0))   # t = 0은 늘 지금 상태

    def test_entry_rates(self):
        entries = [(990.0, "a", 0, None), (900.0, "a", 0, None), (600.0, "a", 0, None), (995.0, "b", 1, None)]
        r = ep.entry_rates(entries, 1000.0, first_time=970.0)
        self.assertEqual(r, {("a", 0): 2 / 60, ("b", 1): 1 / 60})                  # 기록 30초뿐 → 60초로 나눔
        r = ep.entry_rates(entries, 1000.0, first_time=0.0)
        self.assertEqual(r[("a", 0)], 2 / 300)

    def test_controller_class(self):
        import focused_eval
        from focused_scenario import aihub_states
        original = focused_eval.QueueFirst
        try:
            cls = ep.install("p1", 120)
            self.assertIs(focused_eval.QueueFirst, cls)
            c = cls({}, aihub_states()["greens"])
            self.assertEqual((c.stats["variant"], c.stats["horizon_s"], c.stats["planned"]), ("p1", 120, 0))
            with self.assertRaises(SystemExit):
                ep.controller_class("p9", 120)
        finally:
            focused_eval.QueueFirst = original

    @unittest.skipUnless(sorted((ROOT / "results/focused_204820/cache").glob("net_dqn_template_*.net.xml")), "단독 모형 틀 지도 없음")
    def test_zone_geometry_on_real_map(self):
        import sumolib
        from focused_rl import ZONE_EDGES
        net = sumolib.net.readNet(str(sorted((ROOT / "results/focused_204820/cache").glob("net_dqn_template_*.net.xml"))[-1]))
        g = ep.zone_geometry(net, ZONE_EDGES)
        self.assertEqual(set(g["zone_of"]), {e for edges in ZONE_EDGES.values() for e in edges})
        for approach, edges in ZONE_EDGES.items():
            self.assertEqual(g["offset"][approach], 0.0)
            far = max(g["offset"][e] + g["length"][e] for e in edges)
            self.assertGreater(far, 100.0)
            self.assertGreater(g["travel"][approach], far / 30.0)                 # 시속 108km보다 빠를 수는 없음


# ------------------------------------------------------------------ 가짜 SUMO 연결로 제어 전체(지도 파일은 진짜)
class FakeSumo:
    def __init__(self, links, vehicles, neighbor=None):
        self.vehicles = vehicles                      # 이름 → (도로, 위치 m, 속도 m/s)
        self.pending = []
        self.links, self.neighbor = links, neighbor or {}
        self.edge = SimpleNamespace(getLastStepVehicleIDs=lambda e: [v for v, (x, _, _) in self.vehicles.items() if x == e])
        self.vehicle = SimpleNamespace(getSpeed=lambda v: self.vehicles[v][2], getLanePosition=lambda v: self.vehicles[v][1])
        self.simulation = SimpleNamespace(getPendingVehicles=lambda: tuple(self.pending))
        n = self.neighbor
        self.trafficlight = SimpleNamespace(
            getControlledLinks=lambda tls: self.links,
            getRedYellowGreenState=lambda tls: n["state"], getPhase=lambda tls: n["phase"],
            getSpentDuration=lambda tls: n["spent"], getProgram=lambda tls: "real",
            getAllProgramLogics=lambda tls: [SimpleNamespace(programID="real",
                                                             phases=[SimpleNamespace(duration=d, state=s) for d, s in n["phases"]])])


def fake_env(net_file: Path, vehicles: dict, neighbor=None):
    import sumolib
    from focused_scenario import aihub_states
    net = sumolib.net.readNet(str(net_file))
    conns = sorted(net.getTLS("204820").getConnections(), key=lambda c: c[2])
    links = [[] for _ in range(conns[-1][2] + 1)]
    for inl, outl, i in conns:
        links[i].append((inl.getID(), outl.getID(), ""))
    g = guard()
    g.last_served = {JILJU: 3540.0, SEOKCHEON: 3540.0}
    ts = SimpleNamespace(green_phase=0, time_since_last_phase_change=60, yellow_time=4, max_green=120, is_yellow=False)
    env = SimpleNamespace(sumo=FakeSumo(links, vehicles, neighbor), _net=str(net_file), delta_time=5, sim_step=3600.0,
                          traffic_signals={"204820": ts}, signal_constraints=g)
    return env, aihub_states()["greens"]


class ControllerFakeEnvTest(unittest.TestCase):
    def setUp(self):
        from focused_rl import MOVEMENT_LINKS, ZONE_EDGES
        from focused_scenario import aihub_states
        greens = aihub_states()["greens"]
        self.ZONE_EDGES = ZONE_EDGES
        self.movements = {}
        for m, links in sorted(MOVEMENT_LINKS.items()):  # 진입로마다 보호 녹색이 있는 이동 하나씩
            if any(g[l] == "G" for g in greens for l in links):
                self.movements.setdefault(m[0], m)
        self.assertEqual(set(self.movements), set(ZONE_EDGES))

    def _vehicles(self):
        """진입로마다 멈춘 차 3대(정지선 쪽 도로)와 달리는 차 1대(구간 가장 위 도로)."""
        out, movement_of = {}, {}
        for approach, edges in self.ZONE_EDGES.items():
            m = self.movements[approach]
            for k in range(3):
                out[f"{approach}_stop{k}"] = (approach, 5.0, 0.0)
                movement_of[f"{approach}_stop{k}"] = m
            out[f"{approach}_run"] = (edges[0], 1.0, 12.0)
            movement_of[f"{approach}_run"] = m
        return out, movement_of

    @unittest.skipUnless(sorted((ROOT / "results/focused_204820/cache").glob("net_dqn_template_*.net.xml")), "단독 모형 틀 지도 없음")
    def test_p0_p1_on_single_map(self):
        import focused_eval
        net = sorted((ROOT / "results/focused_204820/cache").glob("net_dqn_template_*.net.xml"))[-1]
        original = focused_eval.QueueFirst
        try:
            for variant in ("p0", "p1"):
                vehicles, movement_of = self._vehicles()
                env, greens = fake_env(net, vehicles)
                c = ep.controller_class(variant, 120)(movement_of, greens)
                a = c(env, [0, 1, 2, 3])
                self.assertIn(a, [0, 1, 2, 3])
                q0, arrivals = c.queues_and_arrivals(env)
                served = list(self.movements.values())
                self.assertTrue(all(m in c.index for m in served))
                self.assertEqual(sum(q0), 3 * len(served))                     # 멈춘 차만 줄(단순 제어와 같음)
                moving = sum(map(sum, arrivals))
                self.assertEqual(moving, 0 if variant == "p0" else len(served))  # P1: 달리는 차가 정지선 도착으로
                for k in range(1, 7):                                          # 30초 동안 차가 새로 들어옴
                    env.sim_step += 5
                    env.sumo.vehicles[f"new{k}"] = (self.ZONE_EDGES[served[0][0]][0], 1.0, 12.0)
                    movement_of[f"new{k}"] = served[0]
                    c(env, [0])
                if variant == "p1":
                    self.assertEqual(len(c._entries), 6)
                    _, arrivals = c.queues_and_arrivals(env)
                    self.assertGreater(sum(arrivals[-1]), 0)                   # 구간 밖에서 올 차(최근 진입 비율)
                self.assertEqual(c.stats["planned"], 1)
                self.assertEqual(c.stats["decisions"], 7)
        finally:
            focused_eval.QueueFirst = original

    @unittest.skipUnless(sorted((ROOT / "results/network_demand/map").glob("net12_*_dqn_template.net.xml"))
                         or sorted((ROOT / "results/network_demand/rl").glob("net12_*_dqn_template.net.xml")), "12개 지도 틀 없음")
    def test_p2_on_network12_map(self):
        import explore_coord12 as ec
        import network_rl as nr
        net = nr.template_network_12()
        lay = ec.east_layout()
        on = "".join("G" if i in lay["straight_links"] else "r" for i in range(26))
        off = "r" * 26
        # 지금 현시 2(직진 녹색)에서 10초. 지난 켜짐 50초·꺼짐 90초 → 앞으로 40초 더 켜짐, 그 뒤 90초 꺼짐으로 본다
        neighbor = {"state": on, "phase": 2, "spent": 10.0, "phases": [(50, on), (90, off), (45, on), (95, off)]}
        vehicles, movement_of = self._vehicles()
        env, greens = fake_env(net, vehicles, neighbor)
        c = ep.controller_class("p2", 120)(movement_of, greens)
        self.assertIn(c(env, [0, 1, 2, 3]), [0, 1, 2, 3])
        east_m = self.movements[ec.APPROACH]
        for k in range(1, 13):                                                 # 동쪽으로 1분 동안 차가 들어옴(직진 녹색일 때)
            env.sim_step += 5
            env.sumo.vehicles[f"e{k}"] = (self.ZONE_EDGES[ec.APPROACH][0], 1.0, 12.0)
            movement_of[f"e{k}"] = east_m
            c(env, [0])
        self.assertEqual(len(c._neighbor_log), 12)                             # 첫 판단은 세지 않음(그때 들어온 차도 세지 않음)
        self.assertTrue(all(o for _, o in c._neighbor_log))
        _, arrivals = c.queues_and_arrivals(env)
        i = c.index[east_m]
        travel = c.geo["travel"][ec.APPROACH]
        on_steps = [k for k in range(len(arrivals)) if 0 <= k * 5 - travel <= 40]
        off_steps = [k for k in range(len(arrivals)) if 40 < k * 5 - travel <= 130]
        self.assertTrue(on_steps and off_steps)
        rate = 12 / ep.MIN_RATE_WINDOW_S                                       # 켜짐 60초 동안 12대(기록이 짧아 60초로 나눔)
        for k in on_steps:
            self.assertAlmostEqual(arrivals[k][i], rate * 5)                   # 켜짐일 때 비율로 구간 밖에서 옴
        for k in off_steps:
            self.assertEqual(arrivals[k][i], 0.0)                              # 꺼짐일 때 들어온 차는 없었음
        self.assertEqual(c.stats["planned"], 1)


# ------------------------------------------------------------------ 판정(가짜 결과)
def _write(path: Path, data: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "metrics.json").write_text(json.dumps(data))


def _model(root: Path, name: str, seed: int) -> str:
    d = root / name / f"seed_{seed}"
    d.mkdir(parents=True)
    (d / "training_config.json").write_text(json.dumps({"seed": seed, "timesteps": 100000}))
    (d / "dqn_204820.zip").write_bytes(f"{name}{seed}".encode())
    return str(d)


PRED_STATS = {"horizon_s": 120, "code": "abc", "planned": 10, "seconds": 0.5, "max_seconds": 0.1, "outside_allowed": 0}
GUARD = {"changed_by_rules": {"ped": 0, "vehicle": 0, "max_green": 0}}


class PredDecisionTest(unittest.TestCase):
    def _single(self, folder: Path, date: str, tag: str, delay: float, arrivals="random", seed=None, pred=False, long=5):
        run = folder / date / f"{tag}_eval42{'' if arrivals == 'random' else '_' + arrivals}"
        by = {"477706392_02->477706883_02 (l)": {"vehicles": 50, "mean_delay_s": 140.0},     # 단독 모형 이름(focused_v6_report)
              "477706883_01->477707013_01 (l)": {"vehicles": 50, "mean_delay_s": 100.0}}
        m = {"date": date, "train_seed": seed, "all": {"vehicles": 1000, "mean_delay_s": delay, "long_waits": long,
                                                       "not_arrived": 0, "by_movement": by},
             "collisions_unexplained": [], "pedestrian": {"x": {"waits_over_limit": 0}}, "guard": GUARD}
        if pred:
            m["pred"] = PRED_STATS
        _write(run, m)
        with gzip.open(run / "events.json.gz", "wt") as f:
            json.dump([{"pet": {"clearing_first": {"pet": 2.5}, "entering_first": None}}], f)

    def test_single(self):
        import explore_decide as ed
        split = json.loads((ROOT / "config/focused_split.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred, v6, h1 = root / "pred", root / "v6", root / "h1"
            v6m = [_model(root, "v6", s) for s in ed.SEEDS]
            h1m = [_model(root, "h1", s) for s in ed.SEEDS]
            for dates, arrivals_list, training in ((ed.VALIDATION, ("random", "platoon"), False), (split["train"], ("random",), True)):
                for arr in arrivals_list:
                    for d in dates:
                        self._single(pred if training else v6, d, "fixed", 46.0, arr)
                        self._single(v6, d, "fixedc", 47.3, arr)
                        self._single(v6, d, "greedy", 39.7, arr)
                        self._single(pred, d, "pred_p0", 38.0, arr, pred=True)
                        self._single(pred, d, "pred_p1", 37.5, arr, pred=True)
                        if not training:
                            for s, m in zip(ed.SEEDS, v6m):
                                self._single(v6, d, ed.dqn_tag(v6m)(s), 42.25, arr, seed=s)
                            for s, m in zip(ed.SEEDS, h1m):
                                self._single(h1 if arr == "random" else pred, d, ed.dqn_tag(h1m)(s), 39.78, arr, seed=s)
            args = SimpleNamespace(pred_dir=str(pred), v6_dir=str(v6), v6_models=v6m, h1_dir=str(h1), h1_models=h1m)
            r = ed.decide_pred_single(args)
            c = r["validation_random"]["comparisons"]
            self.assertEqual(c["P0_vs_greedy"]["word"], "낫다")                    # 1.7초 짧음, 나머지 같음
            self.assertEqual(c["P1_vs_P0"]["word"], "차이 없음")                   # 0.5초
            self.assertEqual(c["4_P1_vs_dqn_v6"]["word"], "낫다")
            self.assertEqual(c["4_P1_vs_dqn_h1"]["word"], "낫다")                  # 2.28초
            self.assertEqual(c["5_vs_fixedc"]["P1"]["delay_diff_s"], -9.8)
            self.assertTrue(r["interpretation"][0].startswith("1. 같은 정보에서 내다보는 계산이 단순 규칙보다 낫다"))
            self.assertTrue(r["interpretation"][-1].startswith("6. 날짜를 넓혀도 같다"))
            self.assertEqual(r["safety_note"], "PET < 0 없음")
            self.assertEqual(r["validation_random"]["pred"]["P1"]["horizon_s"], 120)

    def _window(self, root: Path, w: str, tag: str, seed: int, delay: float, east: float, pred=False):
        by = {"477706392_02->477706883_02": {"vehicles": 20, "mean_delay_s": 160.0},
              "477706883_01->477707013_01": {"vehicles": 20, "mean_delay_s": 95.0},
              "477706392_02->477707013_01": {"vehicles": 200, "mean_delay_s": east}}
        m = {"window_204820": {"vehicles": 1000, "mean_delay_s": delay + (seed - 44) * 0.1, "long_delays": 10, "by_movement": by},
             "collisions_in_204820": 0, "guard": GUARD}
        if pred:
            m["pred"] = PRED_STATS
        _write(root / "windows" / w / f"{tag}_eval{seed}", m)

    def test_network12(self):
        import explore_decide as ed
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = {k: [_model(root, k, s) for s in ed.SEEDS] for k in ("m75", "b12", "coord")}
            values = {"fixed": (54.7, 32.0), "fixedc": (56.5, 34.0), "greedy": (54.1, 56.0), "pred_p0": (53.8, 55.0),
                      "pred_p1": (53.5, 54.0), "pred_p2": (51.0, 41.0)}
            for d in ed.VALIDATION:
                for h in (8, 17):
                    w = f"{d}_{h:02d}"
                    for s in ed.SEEDS:
                        for tag, (delay, east) in values.items():
                            folder = root / ("pred" if tag.startswith("pred") else "ref")
                            self._window(folder, w, tag, s, delay, east, pred=tag.startswith("pred"))
                        for k, folder, delay in (("m75", "ref", 54.3), ("b12", "b12", 53.8), ("coord", "coord", 52.9)):
                            self._window(root / folder, w, ed.dqn_tag(models[k])(s), s, delay, 50.0)
            args = SimpleNamespace(pred_dir=str(root / "pred"), ref_dir=str(root / "ref"), ref_models=models["m75"],
                                   base_dir=str(root / "b12"), base_models=models["b12"],
                                   cand_dir=str(root / "coord"), cand_models=models["coord"])
            r = ed.decide_pred_12(args)
            c = r["comparisons"]
            self.assertEqual(c["P0_vs_greedy"]["word"], "차이 없음")
            self.assertEqual(c["P2_vs_P1"]["word"], "낫다")                        # 2.5초
            self.assertEqual(c["mechanism"]["evening_east_straight_gain_s"], 13.0)
            self.assertEqual(c["4_P2_vs_dqn_coord"]["word"], "차이 없음")          # 1.9초 < 2.2초
            self.assertTrue(r["interpretation"][0].startswith("1. 120초를 내다봐도 단순 규칙과 같다"))
            self.assertEqual(r["interpretation"][2], "3. 이웃 신호 정보가 계산형 제어에서는 작동한다.")
            self.assertEqual(r["safety_note"], "석천사거리 안 충돌 없음")


if __name__ == "__main__":
    unittest.main()
