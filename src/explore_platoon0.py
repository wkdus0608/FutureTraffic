"""[추가 탐색] 0단계 PLATOON0 — 동쪽 무리 도착 경기장의 여지 재기(학습 없음, 지문 밖, 진행 기록 238·239절).

사전 등록: results/explore/platoon0_20261031/사전등록.md(코드 전 18:51 고정). 잠긴 파일(arena3_common·explore_weight_*·
explore_pred·explore_ceiling·focused_*)은 고치지 않는다. 실행할 때만 focused_scenario.window_route_file을 감싸 차량 파일을 고친다.

- 세계 R = ARENA3 그대로(무작위 도착). 세계 P = 같은 차에서 동쪽 접근로 경로 시작 도로의 차만 출발 시각을 204822 실제 녹색에
  맞춘다(focused_platoon.platoon_route_file과 같은 규칙, 난수만 차마다 seed32("PLATOON0", 배치, 날짜, 차량 번호)).
- 제어: M4(ARENA3 그대로) / P1M4 / TM4 / OM4 = 같은 계획기·같은 M4 가중(동·남 좌회전 2배)에 정보만 다름.
  P1M4 = P1 정보, TM4 = P1 + 동쪽 구간 밖 도착을 204822 시간표로 예측, OM4 = 차량 파일의 정확한 도착(111절 O).
  P0M4는 M4와 같은 정보로 이 감싸개가 M4를 그대로 재현하는지 보는 작은 실행 점검용이다.

사용(FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240):
  python src/explore_platoon0.py make-config                 # 맥: P 세계 차량 파일 SHA를 설정에 고정(SUMO 없음)
  python src/explore_platoon0.py check-world --workers 8     # 84상황 R·P 차량 파일 점검(SUMO 없음)
  python src/explore_platoon0.py smoke --out <dir> --work /content/p0s --workers 8 [--reference <Drive>/arena3/referee/baseline]
  python src/explore_platoon0.py run --out <Drive>/explore_platoon0 --work /content/p0 --workers 8
  python src/explore_platoon0_report.py --out <Drive>/explore_platoon0 --reference <Drive>/arena3/referee/baseline
"""
from __future__ import annotations

import argparse
import bisect
import contextlib
import functools
import json
import math
import multiprocessing
import random
import shutil
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import arena3_common as ac  # noqa: E402  (w.configure()로 pet1·북쪽 240 m를 먼저 정한다)
import arena3_referee as ref  # noqa: E402
import explore_ceiling as ec  # noqa: E402
import explore_pred as ep  # noqa: E402
import explore_weight as w  # noqa: E402
import explore_weight_core as wc  # noqa: E402
import focused_platoon as fp  # noqa: E402
import focused_scenario as fs  # noqa: E402
import focused_train as ft  # noqa: E402
from focused_data import TARGET_TLS  # noqa: E402

EXPERIMENT = "PLATOON0_V1"
CONFIG = ROOT / "config/explore_platoon0_v1.json"
PREREG = ROOT / "results/explore/platoon0_20261031/사전등록.md"
HORIZON_S = 120                       # P1·O와 같은 계획 길이(111절)
TIMETABLE_WINDOW_S = 900.0            # TM4: 동쪽 진입을 204822 상태별로 세는 길이(112절 P3·P2와 같음)
MAX_EPISODE_S = 3600.0
RUNS = {"P": ("M4", "P1M4", "TM4", "OM4"), "R": ("M4", "OM4")}
INFO = {"P0M4": "p0", "P1M4": "p1", "TM4": "timetable", "OM4": "oracle"}
SMOKE_RUNS = (("P", "M4"), ("P", "P0M4"), ("P", "P1M4"), ("P", "TM4"), ("P", "OM4"), ("R", "M4"))
EXTRA_FILES = ("src/explore_platoon0.py", "src/explore_ceiling.py", "src/focused_platoon.py", "config/east_platoon.json",
               "data/derived/east_upstream_204822.csv", "config/explore_platoon0_v1.json")


# ----------------------------------------------------------------------------- 세계 P(무리 도착)
@functools.lru_cache(maxsize=None)
def _upstream_counts() -> dict:
    return fp.load_upstream_counts()


@functools.lru_cache(maxsize=None)
def timetable(date: str) -> tuple[str, ...]:
    """204822의 1초마다 실제 신호 상태(하루 86,400초). 무리를 만드는 기록과 TM4가 보는 시간표가 같다."""
    states = tuple(fp.upstream_states(str(date)))
    if len(states) != 86400:
        raise ValueError(f"204822 시간표가 하루 길이가 아닙니다: {date} {len(states)}초")
    return states


@functools.lru_cache(maxsize=None)
def _green_windows(date: str) -> dict[str, list[int]]:
    release, _ = _release()
    states = timetable(date)
    return {s: [t for t, x in enumerate(states) if x == s] for s in set(release.values()) if s}


@functools.lru_cache(maxsize=1)
def east_geometry():
    """(지도, 동쪽 경로 시작 도로, 204822 → 그 도로 빈 도로 시간). focused_platoon.platoon_route_file과 같은 계산."""
    fn = fs.FocusedNetwork()
    start = fn._chain(fp.EAST_APPROACH, True, fs.UPSTREAM_LIMIT_M)[0]
    if fp.LINK_EDGE not in fn.pred[start]:
        raise ValueError(f"{fp.LINK_EDGE}가 동쪽 경로의 시작 도로 {start} 앞에 있지 않습니다.")
    return fn, start, fn.length[fp.LINK_EDGE] / fn.speed[fp.LINK_EDGE]


@functools.lru_cache(maxsize=1)
def _release() -> tuple[dict[str, str | None], str]:
    movements = json.loads(fp.PLATOON_CONFIG.read_text())["movements"]
    release = {o: m["release_state"] for o, m in movements.items()}
    return release, max(movements, key=lambda o: movements[o]["vehicles"])


def retime_east(path: Path, date: str, origin_s: int, period_of: dict[str, str], tag: str) -> Counter:
    """창 차량 파일에서 동쪽 시작 도로 차의 출발 시각만 204822 녹색에 맞춘다(파일 시각 0 = 하루 origin_s초).

    정지선 도착은 원래 15분 시간대 [p0, p0 + 900) 안에 든다: 출발 = (그 창 안의 녹색 초 + 0~1초) + 204822→시작 도로 시간.
    """
    fn, start, link_tt = east_geometry()
    release, main_origin = _release()
    origins = list(release)
    counts, windows = _upstream_counts(), _green_windows(date)
    root = ET.parse(path).getroot()
    routes = {r.get("id"): r.get("edges").split() for r in root.iter("route")}
    lead_of, share_of, changed = {}, {}, Counter()
    for v in root.findall("vehicle"):
        rid = v.get("route")
        edges = routes[rid]
        if edges[0] != start:
            continue
        vid = v.get("id")
        if rid not in lead_of:
            lead_of[rid] = fn.travel_time_to_stopline(edges, fp.EAST_APPROACH)
        p0 = fs.period_start_s(period_of[vid])
        if p0 not in share_of:
            share_of[p0] = [sum(counts.get((date, m, o), 0) for m in range(p0 // 60, p0 // 60 + 15)) for o in origins]
        weights = share_of[p0]
        rng = random.Random(ac.seed32("PLATOON0", tag, date, vid))
        origin = rng.choices(origins, weights=weights)[0] if sum(weights) else main_origin
        lo, hi = p0 - lead_of[rid] - link_tt, p0 + 900 - lead_of[rid] - link_tt
        state = release.get(origin)
        seconds = windows.get(state, []) if state else []
        a, b = bisect.bisect_left(seconds, lo), bisect.bisect_left(seconds, hi)
        if b > a:
            t, kind = seconds[rng.randrange(a, b)] + rng.random(), "platoon"
        else:
            t, kind = rng.uniform(lo, hi), ("uniform" if state is None else "no_green_window")
        depart = t + link_tt - origin_s
        if depart < 0:
            depart, kind = 0.0, "clamped"
        v.set("depart", f"{depart:.2f}")
        changed[kind] += 1
    vehicles = sorted(root.findall("vehicle"), key=lambda v: (float(v.get("depart")), v.get("id")))
    for v in vehicles:
        root.remove(v)
    root.extend(vehicles)
    tmp = path.with_name(path.name + ".platoon.tmp")
    ET.ElementTree(root).write(tmp, encoding="utf-8", xml_declaration=True)
    tmp.replace(path)
    return changed


@contextlib.contextmanager
def platoon_world(tag: str):
    """with 안에서 만드는 창 차량 파일(에피소드·SHA 계산)을 세계 P로 바꾼다. 바뀐 차 수를 센 Counter를 돌려준다."""
    original = fs.window_route_file
    if getattr(original, "platoon0", False):
        raise RuntimeError("무리 도착 세계가 이미 켜져 있습니다.")
    stats = Counter()

    def window_route_file(date, start_s, end_s, output, origin_s=None, counts=None):
        period_of = original(date, start_s, end_s, output, origin_s=origin_s, counts=counts)
        origin = start_s // 900 * 900 if origin_s is None else origin_s
        stats.update(retime_east(Path(output), str(date), int(origin), period_of, str(tag)))
        return period_of

    window_route_file.platoon0 = True
    fs.window_route_file = window_route_file
    try:
        yield stats
    finally:
        fs.window_route_file = original


def world_context(world: str, tag: str):
    if world == "P":
        return platoon_world(tag)
    if world == "R":
        return contextlib.nullcontext(Counter())
    raise ValueError(f"세계는 R 또는 P: {world}")


def route_sha(situation: dict, world: str, work: Path) -> str:
    with world_context(world, situation["tag"]):
        return ac.route_file_sha(situation, work)


# ----------------------------------------------------------------------------- TM4의 시간표 예측(순수 함수, 시험 대상)
def state_rates(log, lo: float, now: float, state_at, floor_s: float) -> dict[str, dict[int, float]]:
    """[lo, now]에 동쪽 구간에 들어온 차를 '204822를 떠났을 때 상태'별·이동별로 세어, 그 상태였던 초로 나눈 비율(대/초)."""
    exposure = Counter(state_at(u + 0.5) for u in range(math.ceil(lo), math.floor(now)))
    counts: dict[str, Counter] = defaultdict(Counter)
    for t, i, s in log:
        if lo <= t <= now:
            counts[s][i] += 1
    return {s: {i: n / max(floor_s, exposure[s]) for i, n in c.items()} for s, c in counts.items()}


def add_timetable_arrivals(arrivals: list[list[float]], rates: dict, now: float, step: float, travel: float,
                           state_at) -> None:
    """앞으로 k단계 정지선 도착 = Σ(그 단계의 초 τ) 비율[204822 상태(τ − 구간 통과 시간 − 204822→구간 시간)]. 구간에 이미 든 차는 따로 센다."""
    for k in range(math.ceil(travel / step), len(arrivals)):
        for d in range(int(round(step))):
            for i, r in rates.get(state_at(now + k * step + d + 0.5 - travel), {}).items():
                arrivals[k][i] += r


def timetable_class(horizon_s: int = HORIZON_S):
    base = ep.controller_class("p1", horizon_s)

    class TimetableControl(base):
        """P1 정보 + 동쪽 구간 밖 도착을 204822 시간표로 예측(사전 등록 TM4). 다른 접근로는 P1과 같다."""

        def configure(self, date: str, origin_s: float):
            self._tt = timetable(str(date))
            self._origin = float(origin_s)
            self._link_tt = east_geometry()[2]
            self._east_log: deque = deque()          # (진입을 본 시각, 이동 번호, 그때 204822 상태)
            self.tt_stats = Counter()
            return self

        def _state_at(self, sim_t: float) -> str:
            """시각 sim_t에 동쪽 구간에 들어온 차가 204822를 떠났을 때의 상태."""
            u = math.floor(self._origin + sim_t - self._link_tt)
            if 0 <= u < len(self._tt):
                return self._tt[u]
            self.tt_stats["outside_day"] += 1
            return ""

        def _observe(self, env) -> None:
            super()._observe(env)
            now, step = float(env.sim_step), float(env.delta_time)
            for t, approach, i, _ in reversed(self._entries):      # 이번 판단에 새로 적힌 진입(오른쪽 끝)
                if t != now:
                    break
                if approach == fp.EAST_APPROACH:
                    self._east_log.append((now, i, self._state_at(now - step / 2)))
                    self.tt_stats["east_entries"] += 1
            while self._east_log and self._east_log[0][0] < now - TIMETABLE_WINDOW_S:
                self._east_log.popleft()

        def _outside(self, arrivals, sumo, now: float, step: float) -> None:
            east = fp.EAST_APPROACH
            for (approach, i), r in ep.entry_rates(self._entries, now, self._first_time).items():
                if approach == east:
                    continue
                for k in range(math.ceil(self.geo["travel"][approach] / step), len(arrivals)):
                    arrivals[k][i] += r * step
            first = self._first_time if self._first_time is not None else now
            rates = state_rates(self._east_log, max(now - TIMETABLE_WINDOW_S, first), now, self._state_at,
                                ep.MIN_RATE_WINDOW_S)
            add_timetable_arrivals(arrivals, rates, now, step, self.geo["travel"][east], self._state_at)
            self.tt_stats["planned"] += 1

    TimetableControl.__name__ = "TimetableControl"
    return TimetableControl


# ----------------------------------------------------------------------------- 정책(ARENA3 실행기에 끼움)
class InfoM4(ac.P0Policy):
    """M4 가중 계획을 정보만 바꾼 계획기로 다시 푼다. P0Policy 종류라 실행기가 모든 제어 단계에서 부른다(관측 기록용)."""

    def __init__(self, name: str):
        if name not in INFO:
            raise ValueError(f"정보 제어 이름이 아닙니다: {name}")
        self.name, self.kind = name, INFO[name]
        self.ctl = None
        self.weights = None
        self.stats = Counter()

    def _build(self, ctx) -> None:
        e = ctx.ep
        greens = [p.state for p in e.ts.green_phases]
        if self.kind == "oracle":
            cls = ec.controller_class("oracle", HORIZON_S)
        elif self.kind == "timetable":
            cls = timetable_class(HORIZON_S)
        else:
            cls = ep.controller_class(self.kind, HORIZON_S)
        ctl = cls(cls.movements(e.route), greens)
        if self.kind == "timetable":
            ctl.configure(e.date, fs.period_start_s(e.period) - ft.WARMUP_S)
        ctl._setup(e.env)
        if list(ctl.movements) != list(ctx.planner.p0.movements):
            raise RuntimeError("정보 제어의 이동 순서가 M4 계획기와 다릅니다.")
        self.ctl, self.weights = ctl, ac.lefts_weights(ac.MODES6[ac.M4], ctl.movements)

    def decide(self, snap, ctx) -> int:
        if self.ctl is None:
            self._build(ctx)
        env = ctx.ep.env
        self.ctl._observe(env)                      # 강제 단계에서도 진입 기록(P1·TM4)
        allowed = [int(a) for a in ctx.allowed]
        if len(allowed) == 1:
            return allowed[0]
        ts = env.traffic_signals[TARGET_TLS]
        now = float(env.sim_step)
        served = env.signal_constraints._served_now(ts, now)
        state = ep.SignalState(ts.green_phase, now - ts.time_since_last_phase_change, float(ts.yellow_time),
                               tuple(served[c] for c in self.ctl._model.crosswalks))
        q0, arrivals = self.ctl.queues_and_arrivals(env)
        action, _ = wc.weighted_plan(self.ctl._model, state, now, q0, arrivals, allowed, self.weights)
        if action not in allowed:
            raise RuntimeError("정보 제어가 허용 밖 녹색을 냈습니다.")
        self.stats["planned"] += 1
        self.stats["differs_from_m4"] += int(action != snap.greens[ac.M4])
        return int(action)

    def info(self) -> dict:
        out = {"kind": self.kind, **self.stats}
        if self.ctl is not None:
            out["controller"] = {k: v for k, v in type(self.ctl).stats.items() if isinstance(v, (int, float, str))}
            out.update({f"timetable_{k}": v for k, v in getattr(self.ctl, "tt_stats", {}).items()})
        return out


def build_policy(name: str):
    return ac.fixed_policy("M4") if name == "M4" else InfoM4(name)


# ----------------------------------------------------------------------------- 설정·작업
def config() -> dict:
    cfg = json.loads(CONFIG.read_text())
    if cfg.get("experiment") != EXPERIMENT:
        raise ValueError("PLATOON0 설정이 아닙니다.")
    if cfg.get("prereg_sha256") != w.file_hash(PREREG):
        raise ValueError("사전 등록 문서가 설정을 만든 때와 다릅니다.")
    if cfg.get("lock_sha256") != w.file_hash(ac.CONFIG):
        raise ValueError("ARENA3 잠금 설정이 이 설정을 만든 때와 다릅니다.")
    return cfg


def _situation(s: dict) -> dict:
    return {k: s[k] for k in ("date", "period", "tag", "sumo_seed")}


def _sha_job(args) -> tuple[str, str, str]:
    situation, world, work = args
    w.configure()
    return ac.situation_id(situation), world, route_sha(situation, world, Path(work))


def _pool(workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                               initializer=ref._worker_init)


def make_config(workers: int) -> dict:
    """맥에서 한 번: 84상황의 P 세계 차량 파일 SHA를 고정한다(SUMO 없음). R은 ARENA3 잠금 값을 그대로 옮긴다."""
    lock = ac.config()
    sits = [_situation(s) for s in lock["situations_eval"]]
    with tempfile.TemporaryDirectory() as d, _pool(workers) as pool:
        p_sha = {sid: sha for sid, _, sha in pool.map(_sha_job, [(s, "P", d) for s in sits])}
    situations = [{**_situation(s), "route_sha256": {"R": s["route_sha256"], "P": p_sha[ac.situation_id(s)]}}
                  for s in lock["situations_eval"]]
    cfg = {"experiment": EXPERIMENT, "design": "PROJECT_PROGRESS.md 238·239절, results/explore/platoon0_20261031/사전등록.md",
           "prereg": PREREG.relative_to(ROOT).as_posix(), "prereg_sha256": w.file_hash(PREREG),
           "lock_sha256": w.file_hash(ac.CONFIG), "runs": {k: list(v) for k, v in RUNS.items()},
           "information": {"M4": "ARENA3 기준선(멈춘 차 + 진입 대기)", "P1M4": "P1 정보", "TM4": "P1 + 204822 시간표(동쪽)",
                           "OM4": "정확한 도착(차량 파일)"},
           "horizon_s": HORIZON_S, "timetable_window_s": TIMETABLE_WINDOW_S,
           "world_P": "동쪽 시작 도로 차 출발 = 204822 녹색 초(그 15분 창) + 204822→시작 도로 시간, 난수 seed32(PLATOON0, 배치, 날짜, 차 번호)",
           "verdict": {"room_min_gain_s": 2.0, "no_room_below_s": 1.0, "timetable_share": 0.5,
                       "main": "P 세계 OM4 − M4 날짜 평균 ≥ 2.0초이고 t95 하한 > 0 → 여지 있음; < 1.0초 → 없음; 그 밖 불분명",
                       "cause": "날짜별 G_O(P) − G_O(R)의 t95 하한 > 0 → 무리 도착 때문",
                       "share": "S = G_T ÷ G_O ≥ 0.5 → 시간표로 절반 넘게 잡힘"},
           "situations": situations}
    CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=1) + "\n")
    return {"config": str(CONFIG), "sha256": w.file_hash(CONFIG), "situations": len(situations)}


def _world_check_job(args) -> dict:
    """한 상황의 R·P 차량 파일을 만들어 비교한다: 차 번호·차종·경로·시간대는 같고 동쪽 시작 도로 차의 출발만 다르다."""
    s, work = args
    w.configure()
    sid = ac.situation_id(s)
    _, start, _ = east_geometry()
    out = {"situation": sid}
    files = {}
    for world in ("R", "P"):
        path = Path(work) / f"{sid}_{world}.rou.xml"
        origin = fs.period_start_s(s["period"]) - ft.WARMUP_S
        from explore_weight_episode import CONTROL_S, TAIL_S
        from focused_data import COUNTS, load_counts
        with world_context(world, s["tag"]) as stats, ac.arrival_placement(s["tag"]):
            fs.window_route_file(s["date"], origin, origin + ft.WARMUP_S + CONTROL_S + TAIL_S, path, origin_s=origin,
                                 counts=load_counts(COUNTS))
        out[f"sha_{world}"] = w.file_hash(path)
        out[f"changed_{world}"] = dict(stats)
        root = ET.parse(path).getroot()
        edges = {r.get("id"): r.get("edges") for r in root.iter("route")}
        files[world] = {v.get("id"): (v.get("type"), edges[v.get("route")], float(v.get("depart")))
                        for v in root.iter("vehicle")}
        departs = [float(v.get("depart")) for v in root.iter("vehicle")]
        out[f"sorted_{world}"] = departs == sorted(departs)
        path.unlink()
    r, p = files["R"], files["P"]
    out["same_vehicles"] = set(r) == set(p) and all(r[v][:2] == p[v][:2] for v in r)
    moved = [v for v in r if r[v][2] != p[v][2]]
    out["moved"] = len(moved)
    out["east_start"] = sum(1 for v in r if r[v][1].split()[0] == start)
    out["moved_only_east"] = all(r[v][1].split()[0] == start for v in moved)
    out["max_shift_s"] = max((abs(r[v][2] - p[v][2]) for v in moved), default=0.0)
    return out


def check_world(workers: int, cfg: dict | None = None) -> dict:
    cfg = cfg or config()
    with tempfile.TemporaryDirectory() as d, _pool(workers) as pool:
        rows = list(pool.map(_world_check_job, [(_situation(s), d) for s in cfg["situations"]]))
    pins = {ac.situation_id(s): s["route_sha256"] for s in cfg["situations"]}
    bad = [r["situation"] for r in rows
           if not (r["sha_R"] == pins[r["situation"]]["R"] and r["sha_P"] == pins[r["situation"]]["P"]
                   and r["same_vehicles"] and r["moved_only_east"] and r["sorted_R"] and r["sorted_P"])]
    changed = Counter()
    for r in rows:
        changed.update(r["changed_P"])
    return {"ok": not bad, "bad": bad, "situations": len(rows), "east_start_vehicles": sum(r["east_start"] for r in rows),
            "moved": sum(r["moved"] for r in rows), "changed_P": dict(changed),
            "max_shift_s": max(r["max_shift_s"] for r in rows)}


def jobs_for(cfg: dict, out: Path, work: Path, fingerprint: str, pairs=None, situations=None) -> list[dict]:
    pairs = pairs or [(world, name) for world, names in RUNS.items() for name in names]
    jobs = []
    for s in (situations or cfg["situations"]):
        sid = ac.situation_id(s)
        for world, name in pairs:
            jobs.append({"world": world, "policy": name, "sid": sid, "situation": _situation(s),
                         "route_sha256": s["route_sha256"][world], "fingerprint": fingerprint,
                         "model_identity": f"{world}/{name}", "target": str(Path(out) / "runs" / world / name / sid),
                         "work": str(Path(work) / world / name / sid)})
    return jobs


def platoon_worker(job: dict) -> dict:
    started = time.monotonic()
    work = Path(job["work"])
    name = f"{job['world']}/{job['policy']}/{job['sid']}"
    try:
        w.configure()
        shutil.rmtree(work, ignore_errors=True)
        policy = build_policy(job["policy"])
        with world_context(job["world"], job["situation"]["tag"]) as stats:
            result = ac.run_arena_episode(job["situation"], work, policy, max_seconds=MAX_EPISODE_S)
        if result.metrics["route_sha256"] != job["route_sha256"]:
            raise ValueError(f"차량 파일 SHA가 고정값과 다릅니다: {result.metrics['route_sha256']} ≠ {job['route_sha256']}")
        extra = {"policy": job["policy"], "world": job["world"], "fingerprint": job["fingerprint"],
                 "model_identity": job["model_identity"], "job_seconds": time.monotonic() - started,
                 "world_changes": dict(stats), "information": policy.info() if hasattr(policy, "info") else {"kind": "M4"}}
        ac.write_run(Path(job["target"]), result, extra)
        return {"ok": True, "seconds": time.monotonic() - started, "name": name}
    except BaseException as error:
        return {"ok": False, "seconds": time.monotonic() - started, "name": name, "error": f"{type(error).__name__}: {error}"}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def prepare_caches() -> None:
    """작업자가 동시에 지도 캐시를 처음 만들다 이름이 부딪히지 않게(218절) 한 프로세스에서 먼저 만든다."""
    fs.dqn_template_network(ft.CACHE)


def run_all(out: Path, work: Path, workers: int, pairs=None, situations=None, role: str = "run") -> dict:
    cfg = config()
    fingerprint = ref.register(out, EXTRA_FILES)
    prepare_caches()
    jobs = jobs_for(cfg, out, work, fingerprint, pairs, situations)
    ref.stamp(f"PLATOON0 {role}: 작업 {len(jobs)}개, 작업자 {workers}개, 지문 {fingerprint[:12]}")
    result = ref.run_jobs(jobs, workers, ref.Ledger(out, role), worker=platoon_worker)
    done = sum(1 for j in jobs if ac.read_run(Path(j["target"]), fingerprint, j["model_identity"]) is not None)
    ref.stamp(f"PLATOON0 {role}: 끝난 판 {done}/{len(jobs)}, 실패 {len(result['failed'])}")
    return {"jobs": len(jobs), "done": done, "failed": result["failed"], "fingerprint": fingerprint}


def read_runs(out: Path, world: str, name: str, cfg: dict, fingerprint: str) -> dict[str, dict]:
    rows = {}
    for s in cfg["situations"]:
        sid = ac.situation_id(s)
        m = ac.read_run(Path(out) / "runs" / world / name / sid, fingerprint, f"{world}/{name}")
        if m is not None:
            rows[sid] = m
    return rows


def smoke(out: Path, work: Path, workers: int, reference: Path | None) -> dict:
    """작은 실행: 84상황 차량 파일 점검(SUMO 없음) + 첫 상황 6판(P: M4·P0M4·P1M4·TM4·OM4, R: M4)."""
    cfg = config()
    world = check_world(workers, cfg)
    first = cfg["situations"][0]
    sid = ac.situation_id(first)
    runs = run_all(out, work, workers, pairs=list(SMOKE_RUNS), situations=[first], role="smoke")
    fingerprint = runs["fingerprint"]
    m = {f"{wd}/{n}": ac.read_run(Path(out) / "runs" / wd / n / sid, fingerprint, f"{wd}/{n}") for wd, n in SMOKE_RUNS}
    checks = {"world": world["ok"], "all_runs": all(v is not None for v in m.values())}
    if checks["all_runs"]:
        checks["quality_clean"] = all(not ref.quality_errors(v) for v in m.values())
        checks["p0m4_reproduces_m4"] = (m["P/P0M4"]["executed_sha16"] == m["P/M4"]["executed_sha16"]
                                        and m["P/P0M4"]["mean_delay_s"] == m["P/M4"]["mean_delay_s"])
        checks["oracle_knows_start"] = m["P/OM4"]["information"]["controller"].get("oracle_unknown_start") == 0
        tm = m["P/TM4"]["information"]
        checks["timetable_inside_day"] = tm.get("timetable_outside_day", 0) == 0 and tm.get("timetable_east_entries", 0) > 0
        checks["platoon_moved"] = m["P/M4"]["world_changes"].get("platoon", 0) > 0
        if reference is not None:
            base = Path(reference) / "runs" / "M4" / sid / "metrics.json"
            if base.exists():
                r = w.read(base)
                checks["r_m4_equals_arena3"] = (r["mean_delay_s"] == m["R/M4"]["mean_delay_s"]
                                                and r["executed_sha16"] == m["R/M4"]["executed_sha16"]
                                                and r["route_sha256"] == m["R/M4"]["route_sha256"])
    summary = {"ok": all(checks.values()), "checks": checks, "world": world, "situation": sid,
               "delay_s": {k: (v or {}).get("mean_delay_s") for k, v in m.items()},
               "information": {k: (v or {}).get("information") for k, v in m.items()},
               "seconds": {k: (v or {}).get("seconds") for k, v in m.items()}}
    w.write(Path(out) / "smoke.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("make-config")
    c.add_argument("--workers", type=int, default=8)
    c = sub.add_parser("check-world")
    c.add_argument("--workers", type=int, default=8)
    for name in ("smoke", "run"):
        c = sub.add_parser(name)
        c.add_argument("--out", type=Path, required=True)
        c.add_argument("--work", type=Path, required=True)
        c.add_argument("--workers", type=int, default=8)
        if name == "smoke":
            c.add_argument("--reference", type=Path)
    args = parser.parse_args()
    w.configure()
    if args.cmd == "make-config":
        result = make_config(args.workers)
    elif args.cmd == "check-world":
        result = check_world(args.workers)
    elif args.cmd == "smoke":
        result = smoke(args.out, args.work, args.workers, args.reference)
    else:
        result = run_all(args.out, args.work, args.workers)
    print(json.dumps(w.plain(result), ensure_ascii=False, indent=1))
    ok = result.get("ok", True) and not result.get("failed")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
