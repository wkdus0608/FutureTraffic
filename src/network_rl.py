"""12개 지도에서 DQN(석천사거리만 제어) 학습 환경과 학습 시간 측정 — 지문 밖.

단독 모형의 학습 환경(focused_train.WarmStartTrainingEnv)과 같은 틀이다. 에피소드마다 학습 날짜의 한 시각을 골라
준비 30분은 실제 신호로 대기열을 만들고, 그다음 60분을 DQN에게 넘긴다. 다른 것은 세 가지뿐이다.
- 지도: 12개 지도(network_map.network_12)에 석천사거리 프로그램만 '녹색 4개 틀'로 바꾼 사본
  (focused_scenario.write_dqn_template_network와 같은 방식)
- 차량: 12개 지도 하루 수요(network_demand)에서 그 90분에 출발하는 차만 잘라 시각을 당긴 파일
- 이웃 11곳 신호: 그 90분의 실제 신호(network_signals)를 추가 파일로. 석천사거리는 SUMO-RL이 제어한다.
관측·보상·운영 규칙(최소 녹색·보행 등)은 단독 모형 그대로다(석천사거리의 차로 번호가 12개 지도에서도 같음).
다만 '지도에 못 들어온 차'는 석천사거리 구간에 들어가려고 기다리는 차(구간에서 출발하려다 못 들어온 차 + 구간 바로 위
도로에 멈춰 선 석천사거리행 차)로 바꾸고, 출발 대기·누적 대기도 그에 맞춘다(LocalView). 단독 모형은 모든 차가 그 구간 입구에서
출발하므로(넘친 줄은 못 들어온 차가 됨) 같은 뜻이다.

최종 환경 설정(배경 교통 비율·201293 유턴 판·출발 시각 보정)은 config/network12_environment.json에서 읽는다.
학습 지문(network12_fingerprint)에 이 파일과 12개 지도 도구 코드가 들어간다.

사용:
  .venv/bin/python src/network_rl.py prepare                  # 학습·검증 날짜 수요(출발 시각 보정 포함)를 미리 만든다
  .venv/bin/python src/network_rl.py pack                     # 수요·지도를 Colab용 묶음으로(results/colab/FutureTraffic_network12.zip)
  .venv/bin/python src/network_rl.py train --seed 42          # 12개 지도 DQN 학습(10만 단계)
  .venv/bin/python src/network_rl.py time --steps 1440 --scenarios 20220811:0800,20220811:1000   # 학습 시간 측정
"""
from __future__ import annotations

import argparse
import json
import random
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import gymnasium as gym

import focused_data as fd
import network_demand as nd
import network_signals as ns
from focused_scenario import AIHUB_YELLOW_S, aihub_states, period_start_s, real_signal_states
from network_map import network_12

TARGET_TLS = "204820"
WARMUP_S, CONTROL_S = 1800, 3600          # focused_train과 같음
CACHE = fd.PROJECT / "results/network_demand/rl"
ENV12 = fd.PROJECT / "config/network12_environment.json"
PACK = fd.PROJECT / "results/colab/FutureTraffic_network12.zip"
CODE_FILES = ["src/network_rl.py", "src/network_demand.py", "src/network_signals.py", "src/network_map.py",
              "src/network_day.py", "src/network_windows.py", "config/network12_capacity.json",
              "config/network12_environment.json", "config/network12_windows.json"]


def final_environment() -> dict:
    """12개 지도 최종 환경 설정(본격 학습·평가가 쓰는 값)."""
    if not ENV12.exists():
        raise SystemExit(f"{ENV12}이 없습니다. 최종 시험 뒤에 정합니다.")
    cfg = json.loads(ENV12.read_text())
    return {"background": float(cfg["background"]), "uturn": cfg["uturn_201293"], "depart_fix": bool(cfg["depart_fix"])}


def template_network_12() -> Path:
    """12개 지도에서 석천사거리 프로그램만 AI Hub 녹색 4개(+황색) 틀로 바꾼 사본(SUMO-RL이 시작할 때 바꿔 끼움)."""
    src = network_12()
    path = CACHE / f"{src.stem}_dqn_template.net.xml"
    if path.exists():
        return path
    info = aihub_states()
    text = src.read_text(encoding="utf-8")
    logic = ET.fromstring(text).find(f"tlLogic[@id='{TARGET_TLS}']")
    start = text.index(f'<tlLogic id="{TARGET_TLS}"')
    end = text.index("</tlLogic>", start) + len("</tlLogic>")
    greens, phases = info["greens"], []
    for i, g in enumerate(greens):
        nxt = greens[(i + 1) % len(greens)]
        yellow = "".join("y" if a in "Gg" and b in "rs" else a for a, b in zip(g, nxt))
        phases += [f'<phase duration="30" state="{g}"/>', f'<phase duration="3" state="{yellow}"/>']
    attrs = " ".join(f'{k}="{v}"' for k, v in logic.attrib.items())
    block = f"<tlLogic {attrs}>" + "".join("\n        " + p for p in phases) + "\n    </tlLogic>"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.xml")
    tmp.write_text(text[:start] + block + text[end:], encoding="utf-8")
    tmp.replace(path)
    return path


def window_routes(day_routes: Path, origin: int, length: int, out: Path,
                  near: set[str] | None = None) -> tuple[int, set[str]]:
    """하루 수요에서 [origin, origin + length)에 출발하는 차만, 출발 시각을 origin만큼 당겨 쓴다.
    돌려줌: (차 수, 석천사거리 구간에서 출발하는 차 이름). 후자는 near(단독 모형 도로)가 있을 때만 채운다:
    경로가 석천사거리 이동을 지나고 첫 도로가 단독 모형 도로인 차(단독 모형과 같은 출발 도로)."""
    from focused_rl import MOVEMENT_LINKS
    root = ET.parse(day_routes).getroot()
    new = ET.Element("routes")
    new.extend(root.findall("vType"))
    edges_of = {r.get("id"): r.get("edges").split() for r in root.findall("route")}
    local_route = {rid: bool(near) and e[0] in near and any(m in MOVEMENT_LINKS for m in zip(e, e[1:]))
                   for rid, e in edges_of.items()}
    used, vehicles, local = set(), [], set()
    for v in root.findall("vehicle"):
        d = float(v.get("depart"))
        if origin <= d < origin + length:
            v = ET.fromstring(ET.tostring(v))
            v.set("depart", f"{d - origin:.2f}")
            vehicles.append(v)
            used.add(v.get("route"))
            if local_route[v.get("route")]:
                local.add(v.get("id"))
    new.extend(r for r in root.findall("route") if r.get("id") in used)
    new.extend(vehicles)
    out.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(new).write(out, encoding="utf-8", xml_declaration=True)
    return len(vehicles), local


class _LocalSimulation:
    """SUMO simulation 도메인: 도로에 못 들어온 차(getPendingVehicles)를 '석천사거리 구간에 들어가려고 기다리는 차'로 바꾼다
    = 석천사거리 구간에서 출발하려다 못 들어온 차 + 구간 바로 위 도로(feeder)에 멈춰 선, 석천사거리로 가는 차."""

    def __init__(self, inner, view: "LocalView"):
        self._inner, self._view = inner, view

    def getPendingVehicles(self):
        v = self._view
        return tuple(x for x in self._inner.getPendingVehicles() if x in v._keep) + tuple(v._queued)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _LocalVehicle:
    """SUMO vehicle 도메인.
    - 출발 대기(getDepartDelay): 석천사거리 구간에서 출발한 차는 SUMO 값 그대로. 구간 위 도로에서 기다린 차는 그 기다린 초
      (단독 모형에서 그 차는 구간 입구에서 못 들어온 차라 출발 대기로 셈). 나머지는 0.
    - 누적 대기(getAccumulatedWaitingTime): 석천사거리 진입 구간에 처음 들어올 때까지 쌓인 대기(이웃 신호 등)를 뺀다.
      SUMO-RL은 최근 1,000초의 대기를 세므로(--waiting-time-memory 1000) 빼지 않으면 이웃 신호에서 기다린 시간이 들어간다."""

    def __init__(self, inner, view: "LocalView"):
        self._inner, self._view = inner, view

    def getDepartDelay(self, vehicle):
        v = self._view
        if vehicle in v._keep:
            return self._inner.getDepartDelay(vehicle)
        if vehicle in v._queued:
            return v._now - v._queued[vehicle]
        return v._feeder_wait.get(vehicle, 0.0)

    def getAccumulatedWaitingTime(self, vehicle):
        return max(0.0, self._inner.getAccumulatedWaitingTime(vehicle) - self._view._baseline.get(vehicle, 0.0))

    def __getattr__(self, name):
        return getattr(self._inner, name)


class LocalView:
    """SUMO 연결을 감싸 '석천사거리 앞에서 기다리는 차'의 뜻을 단독 모형과 같게 한다.

    단독 모형은 모든 차가 석천사거리 구간 입구에서 출발한다. 그래서 지도에 못 들어온 차·출발 대기는 곧 석천사거리 앞 줄이고,
    줄이 구간(진입 도로에서 거슬러 300m)을 넘으면 넘친 차가 '못 들어온 차'로 보상·관측에 잡힌다. 12개 지도에서는
    (1) 멀리 떨어진 다른 출발 도로(예: cluster 서쪽 지도 끝)의 못 들어온 차가 섞이고, (2) 구간을 넘친 줄은 구간 위 도로에
    서 있어 보이지 않는다. 그대로 두면 보상(감지 구간 정지 차 + 못 들어온 차)·관측(진입로별 못 들어온 차)·운영 규칙(차량
    누적 대기 = 출발 대기 + 최근 대기)이 석천사거리와 관계없는 대기를 세거나, 줄을 위로 넘겨도 벌을 받지 않는다.
    그래서 세 값을 바꾼다(나머지 SUMO 호출은 그대로 넘김).
    - 못 들어온 차 = 석천사거리 구간에서 출발하려다 못 들어온 차(keep) + 구간 바로 위 도로(feeder: 구간 입구에서 거슬러
      다른 신호 교차로 앞까지, 1km 이내)에 멈춰 선 적이 있고 아직 구간에 들어오지 못한, 석천사거리로 가는 차
    - 출발 대기·누적 대기: _LocalVehicle
    """

    def __init__(self, conn, keep: set[str], entry_edges: list[str], feeder_edges: list[str] | None = None):
        self._conn, self._keep, self._entry_edges = conn, keep, entry_edges
        self._feeder_edges = feeder_edges or []
        self._entry_set = set(entry_edges)
        self._baseline: dict[str, float] = {}         # 진입 구간에 처음 보인 때의 누적 대기(다른 곳 출발 차만)
        self._seen: set[str] = set()
        self._queued: dict[str, float] = {}           # 구간 위 도로에서 멈춰 선 시각(구간에 들어오면 뺌)
        self._feeder_wait: dict[str, float] = {}      # 구간에 들어올 때까지 구간 위 도로에서 기다린 초
        self._bound: dict[str, bool] = {}             # 석천사거리로 가는 차인가(경로로 판단, 한 번만 물음)
        self._now = 0.0
        self.simulation = _LocalSimulation(conn.simulation, self)
        self.vehicle = _LocalVehicle(conn.vehicle, self)

    def _to_204820(self, v: str) -> bool:
        b = self._bound.get(v)
        if b is None:
            from focused_rl import MOVEMENT_LINKS
            route = self._conn.vehicle.getRoute(v)
            b = self._bound[v] = any(m in MOVEMENT_LINKS for m in zip(route, route[1:]))
        return b

    def track(self) -> None:
        """1초마다: 구간 위 도로에서 멈춘 석천사거리행 차를 '기다리는 차'로 적고, 진입 구간에 처음 보인 차는 기다림을 끝내고
        그때까지의 누적 대기를 적어 둔다."""
        conn = self._conn
        self._now = conn.simulation.getTime()
        for edge in self._feeder_edges:
            if conn.edge.getLastStepHaltingNumber(edge) == 0:
                continue
            for v in conn.edge.getLastStepVehicleIDs(edge):
                if v not in self._queued and v not in self._seen and self._to_204820(v) and conn.vehicle.getSpeed(v) < 0.1:
                    self._queued[v] = self._now
        for edge in self._entry_edges:
            for v in conn.edge.getLastStepVehicleIDs(edge):
                if v not in self._seen:
                    self._seen.add(v)
                    if v in self._queued:
                        self._feeder_wait[v] = self._now - self._queued.pop(v)
                    if v not in self._keep:
                        self._baseline[v] = conn.vehicle.getAccumulatedWaitingTime(v)

    def __getattr__(self, name):
        return getattr(self._conn, name)


def approach_chain_edges() -> list[str]:
    """석천사거리 진입 구간 도로: 진입 도로마다 거슬러 300m(단독 모형과 같은 규칙) + 서쪽 진입로의 다른 갈래."""
    from focused_rl import MOVEMENT_LINKS
    graph = nd.Graph(network_12())
    approaches = {a for a, _ in MOVEMENT_LINKS}
    edges = set(nd.EXTRA_ENTRIES) | approaches
    for approach in approaches:
        edges.update(graph.chain(approach, True, nd.CHAIN_UP_M))
    return sorted(edges)


def approach_feeder_edges(limit_m: float = 1000.0) -> list[str]:
    """석천사거리 진입 구간 바로 위 큰길: 구간 입구에서 직진 연결로만 거슬러 올라가며 다른 신호 교차로 앞(그 교차로의 진입
    도로)에서 멈추고, 1km까지. 넘친 줄은 이 길에 선다. 이 길에 멈춰 선 석천사거리행 차는 단독 모형이라면 구간 입구에서
    못 들어온 차다. 옆길·회전 연결·석천사거리 출구 도로는 넣지 않는다."""
    import sumolib
    from network_map import single_model_edges
    chain, near = set(approach_chain_edges()), single_model_edges()
    net = sumolib.net.readNet(str(network_12()))
    out, frontier = set(), [(e, 0.0) for e in chain]
    while frontier:
        eid, dist = frontier.pop()
        edge = net.getEdge(eid)
        for pred in edge.getIncoming():
            pid = pred.getID()
            if pid in near or pid in out or pred.getToNode().getType() == "traffic_light":
                continue
            if not any(c.getDirection() == "s" for c in pred.getConnections(edge)):
                continue
            d = dist + pred.getLength()
            if d > limit_m:
                continue
            out.add(pid)
            frontier.append((pid, d))
    return sorted(out)


def use_local_view(env, keep: set[str], entry_edges: list[str], feeder_edges: list[str] | None = None) -> None:
    """SUMO-RL 환경이 SUMO를 새로 시작할 때마다(reset) 연결을 LocalView로 감싸고, 1초마다 진입 구간과 그 위 도로를 본다.
    신호 객체·관측·보상·운영 규칙이 모두 감싼 연결을 쓴다."""
    start, step = env._start_simulation, env._sumo_step

    def start_local():
        start()
        env.sumo = LocalView(env.sumo, keep, entry_edges, feeder_edges)

    def step_local():
        step()
        env.sumo.track()

    env._start_simulation, env._sumo_step = start_local, step_local


def neighbor_window(programs: dict[str, list[tuple[int, str]]], origin: int, length: int, out: Path) -> Path:
    """이웃 11곳의 하루 신호에서 [origin, origin + length) 구간(시각 0부터)만. 석천사거리는 뺀다."""
    window = {}
    for tls, phases in programs.items():
        if tls == TARGET_TLS:
            continue
        per_second = [s for d, s in phases for _ in range(d)][origin:origin + length]
        cut = ns.compress(per_second)
        if "y" not in cut[-1][1]:                  # SUMO 되돌이 검사용 마감 황색
            last, first = cut[-1][1], cut[0][1]
            cut.append((4, "".join("y" if a in "Gg" and b == "r" else a for a, b in zip(last, first))))
        window[tls] = cut
    return ns.write_additional(window, out)


class WarmStart12Env(gym.Env):
    """focused_train.WarmStartTrainingEnv와 같은 에피소드 구조의 12개 지도 환경."""

    def __init__(self, scenarios: list[tuple[str, str]], seed: int, workdir: Path, constraints=None,
                 uturn: str = "drop", background: float = 1.0, depart_fix: bool = False, min_green: int = 15,
                 max_green: int = 120, yellow: int = AIHUB_YELLOW_S, extra_sumo_cmd: str = "",
                 record_per_second_queue: bool = False, extra_queue_edges: list[str] | None = None):
        """extra_sumo_cmd·record_per_second_queue·extra_queue_edges는 조각 평가(network_eval --windows)용이다(학습은 안 씀)."""
        super().__init__()
        from focused_rl import ZoneObservation, zone_queue_reward
        from focused_sumo import make_environment
        from network_map import single_model_edges
        self.scenarios, self.uturn, self.background, self.depart_fix = scenarios, uturn, background, depart_fix
        self.near = single_model_edges()
        self.local: set[str] = set()                   # 석천사거리 구간에서 출발하는 차(에피소드마다 바뀜, LocalView가 봄)
        self.rng = random.Random(seed)
        self.seed_base, self.episode = seed, 0
        self.route, self.neighbors = workdir / "train_routes.rou.xml", workdir / "neighbors.add.xml"
        self.constraints = constraints
        self.history: list[tuple[str, str]] = []
        self._programs: dict[str, dict] = {}
        first = self._prepare()
        self.current = make_environment(seed=seed, net_file=template_network_12(), route_file=self.route,
                                        simulation_duration_sec=first["handover_s"] + CONTROL_S,
                                        observation_class=ZoneObservation, reward_fn=zone_queue_reward,
                                        additional_sumo_cmd=f"-a {self.neighbors} {extra_sumo_cmd}".strip(),
                                        record_per_second_queue=record_per_second_queue,
                                        extra_queue_edges=extra_queue_edges,
                                        min_green=min_green, max_green=max_green, yellow_time=yellow)
        use_local_view(self.current, self.local, approach_chain_edges(), approach_feeder_edges())
        if self.constraints is not None:
            self.constraints.bind(self.current)
        self._apply(first)
        self.observation_space = self.current.observation_space
        self.action_space = self.current.action_space
        self.started = False

    def _prepare(self) -> dict:
        from focused_rl import CROSSWALKS
        date, period = self.rng.choice(self.scenarios)
        self.history.append((date, period))
        origin = period_start_s(period) - WARMUP_S
        demand = nd.demand_dir(date, self.uturn, background=self.background, depart_fix=self.depart_fix)
        if not (demand / "routes.rou.xml").exists():
            raise FileNotFoundError(f"{demand}에 수요가 없습니다. 먼저 `network_rl.py prepare`로 만들거나 묶음을 푸세요.")
        _, local = window_routes(demand / "routes.rou.xml", origin, WARMUP_S + CONTROL_S, self.route, self.near)
        self.local.clear()
        self.local.update(local)
        if date not in self._programs:
            self._programs[date] = ns.day_programs(date)[0]
        neighbor_window(self._programs[date], origin, WARMUP_S + CONTROL_S + 900, self.neighbors)
        states = real_signal_states(date, origin, WARMUP_S + 600)
        handover = next(i for i in range(WARMUP_S, len(states)) if "y" not in states[i] and states[i] != states[i - 1])
        served = {name: max((t + 1 for t in range(handover) if all(states[t][l] == "G" for l in c["links"])), default=0.0)
                  for name, c in CROSSWALKS.items()}
        return {"warmup_states": states[:handover], "handover_state": states[handover], "handover_s": handover,
                "last_served": served}

    def _apply(self, start: dict) -> None:
        env = self.current
        env.warmup_states, env.warmup_handover = start["warmup_states"], start["handover_state"]
        env.sim_max_time = start["handover_s"] + CONTROL_S
        self.last_served = start["last_served"]
        self.handover_s = start["handover_s"]

    def reset(self, *, seed=None, options=None):
        if self.started:
            self.episode += 1
            self._apply(self._prepare())
        self.started = True
        if self.constraints is not None:
            self.constraints.reset(self.last_served)
        return self.current.reset(seed=self.seed_base + self.episode, options=options)

    def step(self, action):
        if self.constraints is not None:
            action = self.constraints(self.current, action)
        obs, reward, terminated, truncated, info = self.current.step(action)
        info = dict(info or {})
        info["executed_action"] = int(self.current.traffic_signals[TARGET_TLS].green_phase)
        return obs, reward, terminated, truncated, info

    def close(self):
        self.current.close()


def time_training(kind: str, scenarios: list[tuple[str, str]], steps: int, seed: int = 42, background: float = 1.0) -> dict:
    """kind: 'single'(단독 모형) 또는 'network12'. 같은 설정의 HoldAwareDQN을 steps만큼 학습시키며 걸린 시간을 잰다."""
    import torch
    torch.set_num_threads(1)
    from focused_rl import SignalConstraints, hold_aware_dqn_class
    from focused_train import DQN_SETTINGS, WarmStartTrainingEnv
    from focused_data import COUNTS, load_counts
    DQN = hold_aware_dqn_class()
    constraints = SignalConstraints()
    workdir = Path(tempfile.mkdtemp(prefix=f"time_{kind}_"))
    t0 = time.time()
    if kind == "single":
        env = WarmStartTrainingEnv(scenarios, load_counts(COUNTS), seed, workdir, constraints=constraints)
    else:
        env = WarmStart12Env(scenarios, seed, workdir, constraints=constraints, background=background)
    setup = time.time() - t0
    model = DQN(policy="MlpPolicy", env=env, seed=seed, verbose=0, **DQN_SETTINGS)
    t1 = time.time()
    model.learn(total_timesteps=steps, progress_bar=False)
    learn = time.time() - t1
    env.close()
    return {"kind": kind, "steps": steps, "episodes": len(env.history), "setup_s": round(setup, 1),
            "learn_s": round(learn, 1), "ms_per_step": round(1000 * learn / steps, 1),
            "estimate_100k_min": round(learn / steps * 100_000 / 60, 1)}


def all_training_scenarios(split_file: Path = nd.SPLIT) -> list[tuple[str, str]]:
    """(날짜, 시각) 후보: 학습 날짜에서 준비 30분과 제어 60분이 하루 안에 들어가는 15분 시각(focused_train과 같은 규칙)."""
    train = json.loads(Path(split_file).read_text())["train"]
    periods = [f"{h:02d}{m:02d}" for h in range(24) for m in (0, 15, 30, 45)]
    return sorted((d, p) for d in train for p in periods if WARMUP_S <= period_start_s(p) <= 86400 - CONTROL_S)


def training_scenarios(split_file: Path = nd.SPLIT) -> list[tuple[str, str]]:
    """학습 시나리오 = 모든 후보 − 막힌 조각과 제어 시간이 겹치는 것(config/network12_windows.json, 10-16 사용자 결정)."""
    import network_windows as nw
    cfg = nw.load_config()
    if cfg is None:
        raise SystemExit(f"{nw.CONFIG}가 없습니다. 먼저 network_windows.py screen·config로 막히는 조각을 찾으세요.")
    final = final_environment()
    if (cfg["background"], cfg["uturn_201293"]) != (final["background"], final["uturn"]):
        raise SystemExit(f"조각 점검({cfg['background']}, {cfg['uturn_201293']})이 최종 환경({final['background']}, "
                         f"{final['uturn']})과 다릅니다. 같은 환경으로 다시 점검하세요.")
    train = json.loads(Path(split_file).read_text())["train"]
    missing = sorted(set(train) - set(cfg["screened_dates"]))
    if missing:
        raise SystemExit(f"점검하지 않은 학습 날짜 {missing}가 있습니다.")
    return [(d, p) for d, p in all_training_scenarios(split_file) if not nw.excluded(d, period_start_s(p), cfg)]


def prepare(dates: list[str], uturn: str, background: float, depart_fix: bool) -> None:
    """학습 전에 날짜마다 12개 지도 수요를 만든다(여러 학습이 같은 날짜를 동시에 만들지 않게).
    출발 시각 보정이면 보정 없는 하루 점검을 먼저 돌려야 해서 날짜마다 약 8분(맥) 걸린다."""
    import network_day
    template_network_12()
    for date in dates:
        demand = nd.demand_dir(date, uturn, background=background, depart_fix=depart_fix)
        if (demand / "routes.rou.xml").exists():
            continue
        t0 = time.time()
        demand = network_day.demand_for(date, uturn, -1, background, depart_fix)
        meta = json.loads((demand / "ledger.json").read_text())["meta"]
        print(f"수요 {date} 만듦 ({meta['vehicles']:,}대, {time.time() - t0:.0f}초)", flush=True)


def _sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pack(dates: list[str], uturn: str, background: float, depart_fix: bool) -> Path:
    """Colab 학습용 묶음: 12개 지도·DQN 틀 지도와 날짜별 수요(경로·장부·보정표). 맥에서 최종 시험한 바로 그 파일을
    쓰도록 sha256 목록을 함께 넣는다(Colab에서 풀 때 검사)."""
    import zipfile
    files = [network_12(), network_12().with_suffix(".check.json"), template_network_12()]
    for date in dates:
        demand = nd.demand_dir(date, uturn, background=background, depart_fix=depart_fix)
        names = ["routes.rou.xml", "ledger.json", "audit.md"] + (["depart_fix.json"] if depart_fix else [])
        missing = [n for n in names if not (demand / n).exists()]
        if missing:
            raise SystemExit(f"{demand}에 {missing}이 없습니다. 먼저 prepare(또는 network_day check)를 돌리세요.")
        files += [demand / n for n in names]
    import network_day
    import network_windows as nw
    screened = sorted((nw.ROOT / nw.tag(uturn, background)).glob("2*_[0-9][0-9].json"))
    if not screened:
        raise SystemExit("조각 점검 결과가 없습니다. 먼저 network_windows.py screen을 돌리세요.")
    files += screened                                   # 조각 점검 결과(작음): Colab에서 몇 개를 다시 돌려 맞대 본다
    reference = network_day.run_dir("20220811", uturn, -1, background, depart_fix) / "result.json"
    if not reference.exists():
        raise SystemExit(f"{reference}가 없습니다. 재현성 비교용 맥 결과(08-11 하루 점검)를 먼저 만드세요.")
    manifest = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "dates": dates, "uturn": uturn, "background": background,
                "depart_fix": depart_fix, "network12_fingerprint": _code_fingerprint(),
                "files": {str(f.relative_to(fd.PROJECT)): _sha(f) for f in files}}
    manifest["files"]["reference/network12_day_20220811_mac.json"] = _sha(reference)
    PACK.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(PACK, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, str(f.relative_to(fd.PROJECT)))
        z.write(reference, "reference/network12_day_20220811_mac.json")     # 맥 결과(Colab 재현성 비교용)
        z.writestr("network12_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=1))
    return PACK


def _code_fingerprint() -> str:
    """12개 지도 코드·설정 파일의 지문(12자리). focused_data.fingerprint와 달리 전환 규칙(FOCUSED_RULESET)을 섞지 않는다.
    규칙은 환경 지문(environment_fingerprint)과 학습 기록의 rule_set에 따로 남는다."""
    import hashlib
    h = hashlib.sha256()
    for rel in CODE_FILES:
        h.update(rel.encode())
        h.update(hashlib.sha256((fd.PROJECT / rel).read_bytes()).digest())
    return h.hexdigest()[:12]


def train(args) -> None:
    """focused_train.train과 같은 설정·체크포인트·이어 학습. 환경만 12개 지도(WarmStart12Env)."""
    import torch
    torch.set_num_threads(1)
    from stable_baselines3.common.monitor import Monitor
    from focused_data import environment_fingerprint
    from focused_rl import SignalConstraints, hold_aware_dqn_class
    from focused_train import DQN_SETTINGS, _checkpoint_callback
    DQN = hold_aware_dqn_class()
    scenarios = training_scenarios()
    out = Path(args.out) / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "checkpoint"
    code_fp = _code_fingerprint()
    resume = args.resume and (checkpoint / "checkpoint.json").exists()
    env_fp = environment_fingerprint()
    if resume:
        saved = json.loads((out / "run.json").read_text())
        if (saved["environment_fingerprint"], saved["network12_fingerprint"]) != (env_fp, code_fp):
            raise SystemExit("체크포인트의 환경 지문이 지금과 다릅니다. 처음부터 다시 학습하세요.")
    constraints = SignalConstraints()
    workdir = Path(tempfile.mkdtemp(prefix=f"network12_dqn_{args.seed}_"))
    env = WarmStart12Env(scenarios, args.seed + (1000 if resume else 0), workdir, constraints=constraints,
                         uturn=args.uturn, background=args.background, depart_fix=args.depart_fix)
    monitored = Monitor(env, filename=str(out / "monitor.csv"), override_existing=not resume)
    if resume:
        model = DQN.load(checkpoint / "model", env=monitored)
        model.load_replay_buffer(checkpoint / "replay_buffer")
        print(f"이어서 학습: {model.num_timesteps}단계부터", flush=True)
    else:
        model = DQN(policy="MlpPolicy", env=monitored, seed=args.seed, verbose=0, **DQN_SETTINGS)
        (out / "run.json").write_text(json.dumps({"environment_fingerprint": env_fp, "network12_fingerprint": code_fp,
                                                  "rule_set": fd.variant()["rule_set"], "seed": args.seed,
                                                  "background": args.background, "uturn": args.uturn,
                                                  "depart_fix": args.depart_fix, "started": time.strftime("%Y-%m-%d %H:%M:%S")}))
    remaining = max(0, args.timesteps - model.num_timesteps)
    started = time.time()
    model.learn(total_timesteps=remaining, reset_num_timesteps=not resume, progress_bar=False,
                callback=_checkpoint_callback(checkpoint, args.checkpoint_freq))
    model.save(out / f"dqn_{TARGET_TLS}")
    monitored.close()
    from focused_rl import CROSSWALK_MIN_GREEN, CROSSWALKS
    from focused_scenario import driver_vtype_attributes
    config = {"environment_fingerprint": env_fp, "network12_fingerprint": code_fp, "variant": fd.variant_signature(),
              "rule_set": fd.variant()["rule_set"], "map": "network12", "network": network_12().name,
              "seed": args.seed, "timesteps": args.timesteps, "resumed": resume, "background": args.background,
              "uturn": args.uturn, "depart_fix": args.depart_fix, "local_view": True,
              "train_dates": sorted({d for d, _ in scenarios}),
              "episode": {"warmup_s": WARMUP_S, "control_s": CONTROL_S,
                          "start": "실제 운영 신호로 30분 돌린 뒤, 시각 P 이후 첫 녹색 시작에 넘겨받음(단독 모형과 같음)"},
              "scenarios_available": len(scenarios), "scenarios_excluded": len(all_training_scenarios()) - len(scenarios),
              "windows_config": json.loads((fd.PROJECT / "config/network12_windows.json").read_text())["failed"],
              "episodes_this_run": len(env.history),
              "scenarios_used_this_run": env.history, "driver_model_vtype_attributes": driver_vtype_attributes(),
              "seconds_this_run": round(time.time() - started, 1),
              "dqn": {**DQN_SETTINGS, "algorithm": "HoldAwareDQN(단독 모형과 같음)"},
              "signal_constraints": {
                  "vehicle_max_cumulative_wait_s": constraints.max_wait, "phase_min_green_s": constraints.phase_min,
                  "crosswalks": {n: {"min_green_s": CROSSWALK_MIN_GREEN[n], "max_wait_s": c["max_wait"],
                                     "phases": constraints.serving.get(n, [])} for n, c in CROSSWALKS.items()},
                  "decisions": constraints.decisions, "rule_decided": dict(constraints.rule_decided),
                  "ped_interventions": constraints.ped_interventions,
                  "vehicle_interventions": constraints.vehicle_interventions,
                  "max_green_switches": constraints.max_green_switches},
              "environment": {"delta_time": 5, "yellow_time": AIHUB_YELLOW_S, "min_green": 15, "max_green": 120,
                              "time_to_teleport": -1}}
    (out / "training_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in config.items() if k != "scenarios_used_this_run"}, ensure_ascii=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def env_options(p):                       # 기본값은 최종 환경 설정(config/network12_environment.json)
        p.add_argument("--uturn", choices=("drop", "straight"), default=None, help="기본: 최종 환경 설정")
        p.add_argument("--background", type=float, default=None, help="기본: 최종 환경 설정")
        p.add_argument("--no-depart-fix", action="store_true", help="출발 시각 보정 없는 수요(시험용)")

    pr = sub.add_parser("prepare", help="날짜별 수요를 미리 만든다(출발 시각 보정이면 하루 점검 1번 포함)")
    env_options(pr)
    pr.add_argument("--dates", default="train+validation", help="train, validation, train+validation 또는 쉼표로 나눈 날짜")
    pk = sub.add_parser("pack", help="Colab용 묶음(지도·수요·sha256 목록)")
    env_options(pk)
    pk.add_argument("--dates", default="train+validation")
    tr = sub.add_parser("train", help="12개 지도에서 DQN 학습(석천사거리만 제어)")
    env_options(tr)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--timesteps", type=int, default=100_000)
    tr.add_argument("--out", default=str(fd.PROJECT / "results/network_demand/rl/models"))
    tr.add_argument("--checkpoint-freq", type=int, default=5_000)
    tr.add_argument("--resume", action="store_true")
    t = sub.add_parser("time")
    t.add_argument("--steps", type=int, default=1440)
    t.add_argument("--scenarios", default="20220811:0800,20220811:1730")
    t.add_argument("--kinds", default="single,network12")
    args = ap.parse_args()
    split = json.loads(nd.SPLIT.read_text())
    if args.cmd in ("prepare", "pack", "train"):
        final = final_environment()
        args.uturn = args.uturn or final["uturn"]
        args.background = final["background"] if args.background is None else args.background
        args.depart_fix = final["depart_fix"] and not args.no_depart_fix
    if args.cmd in ("prepare", "pack"):
        names = {"train": split["train"], "validation": split["validation"],
                 "train+validation": sorted(split["train"] + split["validation"])}
        dates = names.get(args.dates) or args.dates.split(",")
        if set(dates) & set(split["test"]):
            raise SystemExit("시험 날짜는 쓰지 않습니다.")
        if args.cmd == "prepare":
            prepare(sorted(dates), args.uturn, args.background, args.depart_fix)
        else:
            print("저장:", pack(sorted(dates), args.uturn, args.background, args.depart_fix))
        return
    if args.cmd == "train":
        train(args)
        return
    scenarios = [tuple(x.split(":")) for x in args.scenarios.split(",")]
    test = set(json.loads(nd.SPLIT.read_text())["test"])
    if any(d in test for d, _ in scenarios):
        raise SystemExit("시험 날짜는 쓰지 않습니다.")
    out = []
    background = final_environment()["background"]
    for kind in args.kinds.split(","):
        r = time_training(kind, scenarios, args.steps, background=background)
        out.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
    path = CACHE / f"timing_{time.strftime('%Y%m%d_%H%M')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"scenarios": scenarios, "results": out}, ensure_ascii=False, indent=1))
    print("저장:", path)


if __name__ == "__main__":
    main()
