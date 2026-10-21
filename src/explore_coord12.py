"""[추가 탐색] 연동 정보 관측(COORD) — 12개 지도, 지문 밖, 잠근 파일·network_rl.py·network_eval.py를 고치지 않는다(진행 기록 96절).

95절 진단: 저녁 첨두에 길주로 동 → 서 직진에서 DQN이 연동된 고정신호에 진다(32.3 대 54.5초). 그 차의 96%는 상류 204822를
지나오는데, DQN은 204822 신호도, 오고 있는 무리의 위치도 보지 못한다. 그래서 기존 관측(64칸) 끝의 '그대로 실행될 행동 표시'
앞에 7칸을 더한다(71칸, 표시는 계속 맨 끝).
  1. 204822에서 석천사거리 쪽 길주로로 곧장 가는 직진 연결 중 지금 녹색인 비율
  2. 그 직진 녹색이 켜지거나 꺼진 뒤 지난 시간 ÷ 180초(최대 1)
  3. 같은 신호에서 그 길로 들어오는 좌회전 연결 중 지금 녹색인 비율
  4~7. 204822 출구 ~ 석천사거리 정지선(약 415m)을 정지선에서 0~100·100~200·200~300·300m 넘음으로 나눠(도로 단위,
       도로 가운데 위치로 칸을 정함), 칸마다 달리는 차(멈추지 않은 차) ÷ 그 칸에 설 수 있는 차 수(최대 1)
지금 상태와 지난 시간뿐이다(앞으로의 시간표는 넣지 않는다). 학습 방법은 B12(88절: n = 5 + Double)와 같고 관측만 바꾼다.

실행할 때만 focused_rl.ZoneObservation·hold_aware_dqn_class를 바꿔 끼운다(network_rl.WarmStart12Env·train,
network_eval.run_window가 함수 안에서 이 이름을 가져온다). 12개 지도 지문·평가 지문은 그대로다. 모델 폴더의 explore.json에
이 파일의 지문을 남기고, 평가는 같은 지문의 모델만 받는다.

사용(Colab 28번 칸):
  FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240 FOCUSED_DQN_NSTEPS=5 python src/explore_coord12.py check --out <Drive>/explore_12map_coord/check.json
  FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240 FOCUSED_DQN_NSTEPS=5 python src/explore_coord12.py train --seed 42 --timesteps 100000 --out <모델 폴더> --resume
  FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240 python src/explore_coord12.py eval --windows --dates 20220928 20220813 --policies dqn \
      --models <모델 폴더>/seed_42 --eval-seed 42 --workers 8 --out <Drive>/explore_12map_coord
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

NEIGHBOR = "204822"                   # 석천사거리 동쪽 상류 신호(연동, 29절)
APPROACH = "477706392_02"             # 석천사거리 동쪽 진입 도로(정지선 쪽 끝)
BIN_LIMITS_M = (100.0, 200.0, 300.0)  # 정지선에서의 거리 칸 경계
SINCE_CAP_S = 180.0
N_EXTRA = 3 + len(BIN_LIMITS_M) + 1
CHAIN_LIMIT_M = 2000.0
GREEN = "Gg"


def code_fingerprint() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


# ------------------------------------------------------------------ 지도(한 번만)
_LAYOUT: dict | None = None


def east_layout(net_file: Path | None = None) -> dict:
    """동쪽 접근 도로 사슬(정지선에서 직진 연결로 거슬러 204822 출구까지), 거리 칸, 204822 연결 번호.
    net_file을 주지 않으면 SUMO가 읽는 12개 지도(DQN 틀)로 한 번만 계산해 둔다."""
    global _LAYOUT
    if net_file is None:
        if _LAYOUT is None:
            import network_rl as nr
            _LAYOUT = east_layout(nr.template_network_12())
        return _LAYOUT
    from focused_sumo import configure_sumo
    configure_sumo()
    import sumolib
    from focused_rl import VEHICLE_SPACING_M
    net = sumolib.net.readNet(str(net_file))
    tls = net.getTLS(NEIGHBOR)
    leaving = {out.getEdge().getID() for _, out, _ in tls.getConnections()}
    chain, edge, dist = [], net.getEdge(APPROACH), 0.0
    while True:
        chain.append({"edge": edge.getID(), "offset_m": dist, "length_m": edge.getLength(),
                      "lane_m": sum(l.getLength() for l in edge.getLanes())})
        dist += edge.getLength()
        if edge.getID() in leaving:
            break
        preds = [p for p in edge.getIncoming() if any(c.getDirection() == "s" for c in p.getConnections(edge))]
        if len(preds) != 1 or dist > CHAIN_LIMIT_M:
            raise SystemExit(f"{edge.getID()}: 직진 앞 도로가 {len(preds)}개이거나 {NEIGHBOR}까지 {CHAIN_LIMIT_M:.0f}m를 넘습니다.")
        edge = preds[0]
    first = chain[-1]["edge"]                          # 204822에서 나온 첫 도로
    links = {"s": set(), "l": set()}
    for inl, out, idx in tls.getConnections():
        if out.getEdge().getID() != first:
            continue
        direction = next(c.getDirection() for c in inl.getOutgoing() if c.getToLane() == out)
        if direction in links:
            links[direction].add(idx)
    bins = [[] for _ in range(len(BIN_LIMITS_M) + 1)]
    for c in chain:
        middle = c["offset_m"] + c["length_m"] / 2
        bins[sum(middle >= b for b in BIN_LIMITS_M)].append(c)
    layout = {"chain": chain, "length_m": dist, "straight_links": sorted(links["s"]), "left_links": sorted(links["l"]),
              "bins": [[c["edge"] for c in b] for b in bins],
              "bin_capacity": [max(1.0, sum(c["lane_m"] for c in b) / VEHICLE_SPACING_M) for b in bins]}
    if not layout["straight_links"] or not all(layout["bins"]):
        raise SystemExit(f"{NEIGHBOR} 직진 연결 또는 거리 칸이 비었습니다: {layout}")
    return layout


# ------------------------------------------------------------------ 관측 7칸
def green_share(state: str, links: list[int]) -> float:
    return sum(state[i] in GREEN for i in links) / len(links) if links else 0.0


def since_change(durations: list[float], on: list[bool], phase: int, spent: float) -> float:
    """지금 현시 안에서 지난 시간 + 같은 켜짐/꺼짐이 이어진 앞 현시들의 길이(프로그램 처음에서 멈춤)."""
    t, k = spent, phase - 1
    while k >= 0 and on[k] == on[phase]:
        t += durations[k]
        k -= 1
    return t


def program_runs(sumo, straight: list[int]) -> tuple[list[float], list[bool]]:
    """204822의 지금 프로그램(에피소드마다 이웃 신호 추가 파일)에서 현시 길이와 '직진 녹색이 켜졌는가'."""
    pid = sumo.trafficlight.getProgram(NEIGHBOR)
    logic = next(lg for lg in sumo.trafficlight.getAllProgramLogics(NEIGHBOR) if lg.programID == pid)
    return [float(p.duration) for p in logic.phases], [green_share(p.state, straight) >= 0.5 for p in logic.phases]


def coord_features(sumo, runs: tuple[list[float], list[bool]], layout: dict) -> list[float]:
    state = sumo.trafficlight.getRedYellowGreenState(NEIGHBOR)
    durations, on = runs
    since = since_change(durations, on, sumo.trafficlight.getPhase(NEIGHBOR), sumo.trafficlight.getSpentDuration(NEIGHBOR))
    moving = [sum(sumo.edge.getLastStepVehicleNumber(e) - sumo.edge.getLastStepHaltingNumber(e) for e in edges)
              for edges in layout["bins"]]
    return ([green_share(state, layout["straight_links"]), min(1.0, since / SINCE_CAP_S),
             green_share(state, layout["left_links"])]
            + [min(1.0, m / cap) for m, cap in zip(moving, layout["bin_capacity"])])


def insert_before_mask(base, extra, n_actions: int):
    """기존 관측 끝 n칸(그대로 실행될 행동 표시)은 맨 끝에 둔다. HoldAwareDQN이 끝 n칸을 표시로, 앞 n칸을 현시로 읽는다."""
    import numpy as np
    return np.concatenate([base[:-n_actions], np.asarray(extra, dtype=np.float32), base[-n_actions:]]).astype(np.float32)


def coord_observation_class(base):
    from gymnasium import spaces
    import numpy as np

    class CoordObservation(base):
        coord = True

        def __init__(self, ts):
            super().__init__(ts)
            self._runs = None                    # 신호 객체가 에피소드마다 새로 만들어지므로 에피소드마다 한 번 읽는다

        def __call__(self):
            layout = east_layout()
            if self._runs is None:
                self._runs = program_runs(self.ts.sumo, layout["straight_links"])
            return insert_before_mask(super().__call__(), coord_features(self.ts.sumo, self._runs, layout),
                                      self.ts.num_green_phases)

        def observation_space(self):
            size = super().observation_space().shape[0] + N_EXTRA
            return spaces.Box(low=np.zeros(size, dtype=np.float32), high=np.ones(size, dtype=np.float32))

    return CoordObservation


def install() -> None:
    """focused_rl의 관측 클래스와 학습기 이름을 바꿔 끼운다(여러 번 불러도 한 번만)."""
    import focused_rl
    from explore_dqn import double_hold_aware_dqn_class
    if not getattr(focused_rl.ZoneObservation, "coord", False):
        focused_rl.ZoneObservation = coord_observation_class(focused_rl.ZoneObservation)
    focused_rl.hold_aware_dqn_class = double_hold_aware_dqn_class


def explore_record() -> dict:
    return {"observation": "coord", "observation_code": code_fingerprint(), "extra_features": N_EXTRA,
            "double_dqn": True, "n_steps": int(os.environ.get("FOCUSED_DQN_NSTEPS", "0")), "reward": "queue",
            "baseline": "B12(88절, n = 5 + Double)", "note": "진행 기록 96절, src/explore_coord12.py"}


def _same_training_as_b12() -> None:
    if os.environ.get("FOCUSED_DQN_NSTEPS") != "5":
        raise SystemExit("B12와 같은 학습 방법이어야 한다: FOCUSED_DQN_NSTEPS=5로 돌린다.")
    for name in ("FOCUSED_DQN_GAMMA", "FOCUSED_DQN_TARGET_UPDATE", "FOCUSED_DQN_DOUBLE"):
        if os.environ.get(name):
            raise SystemExit(f"{name}은 비워 둔다(B12와 같게, Double은 이 파일이 끼운다).")
    if os.environ.get("FOCUSED_REWARD", "queue") != "queue":
        raise SystemExit("보상은 기본(queue)이어야 한다.")


def _arg(args: list[str], name: str, default: str | None = None) -> str | None:
    return args[args.index(name) + 1] if name in args else default


# ------------------------------------------------------------------ 명령
def train(args: list[str]) -> None:
    import network_rl
    _same_training_as_b12()
    out, seed = _arg(args, "--out"), _arg(args, "--seed", "42")
    if out is None:
        raise SystemExit("--out(모델 폴더)이 필요하다.")
    record_file = Path(out) / f"seed_{seed}" / "explore.json"
    record = explore_record()
    if record_file.exists():
        saved = json.loads(record_file.read_text())
        if (saved.get("observation"), saved.get("observation_code")) != (record["observation"], record["observation_code"]):
            raise SystemExit(f"{record_file}: 이어 학습할 모델의 관측 코드({saved.get('observation_code')})가 지금"
                             f"({record['observation_code']})과 다르다. 처음부터 다시 학습한다.")
    record_file.parent.mkdir(parents=True, exist_ok=True)
    record_file.write_text(json.dumps(record, ensure_ascii=False))
    install()
    sys.argv = [sys.argv[0], "train", *args]
    network_rl.main()


def evaluate(args: list[str]) -> None:
    import multiprocessing
    import network_eval
    if not sys.platform.startswith("linux"):
        raise SystemExit("바꿔 끼운 관측을 하위 실행에 넘기려면 fork가 필요하다(Colab·리눅스에서 실행).")
    multiprocessing.set_start_method("fork", force=True)
    policies = args[args.index("--policies") + 1:] if "--policies" in args else []
    policies = policies[:next((i for i, a in enumerate(policies) if a.startswith("--")), len(policies))]
    if policies != ["dqn"]:
        raise SystemExit("이 평가는 --policies dqn만 받는다(다른 정책은 network_eval.py로, 같은 결과를 다시 쓴다).")
    models = args[args.index("--models") + 1:] if "--models" in args else []
    models = models[:next((i for i, a in enumerate(models) if a.startswith("--")), len(models))]
    if not models:
        raise SystemExit("--models가 필요하다.")
    for m in models:
        rec = Path(m) / "explore.json"
        saved = json.loads(rec.read_text()) if rec.exists() else {}
        if (saved.get("observation"), saved.get("observation_code")) != ("coord", code_fingerprint()):
            raise SystemExit(f"{m}: 이 관측 코드({code_fingerprint()})로 학습한 모델이 아니다({saved.get('observation_code')}).")
        if not (Path(m) / "training_config.json").exists():
            raise SystemExit(f"{m}: 학습이 끝나지 않았다.")
    install()
    sys.argv = [sys.argv[0], "run", *args]
    network_eval.main()


def check(args: list[str]) -> None:
    """학습 전 점검(SUMO, 약 2~3분): 학습 날짜 조각 하나에서 관측 71칸의 모양·범위·행동 표시 위치와, 이웃 신호 값이
    SUMO 신호 상태와 맞는지 본다. 하나라도 어긋나면 멈춘다(학습하지 않는다)."""
    import random
    import tempfile
    import network_rl as nr
    import focused_rl
    _same_training_as_b12()
    install()
    layout = east_layout()
    date, period = _arg(args, "--date", "20220811"), _arg(args, "--period", "1800")
    decisions = int(_arg(args, "--decisions", "180"))
    final = nr.final_environment()
    env = nr.WarmStart12Env([(date, period)], 42, Path(tempfile.mkdtemp(prefix="coord_check_")),
                            constraints=focused_rl.SignalConstraints(), uturn=final["uturn"],
                            background=final["background"], depart_fix=final["depart_fix"])
    n = env.action_space.n
    rng = random.Random(0)
    problems, rows = [], []
    try:
        obs, _ = env.reset()
        size = env.observation_space.shape[0]
        prev = None
        for k in range(decisions):
            sumo = env.current.sumo
            if obs.shape != (size,) or obs.min() < 0 or obs.max() > 1:
                problems.append(f"{k}: 관측 모양 {obs.shape} 또는 범위 {obs.min():.2f}~{obs.max():.2f}")
            extra = obs[-n - N_EXTRA:-n]
            state = sumo.trafficlight.getRedYellowGreenState(NEIGHBOR)
            on_now = green_share(state, layout["straight_links"]) >= 0.5
            if abs(extra[0] - green_share(state, layout["straight_links"])) > 1e-6 or abs(extra[2] - green_share(state, layout["left_links"])) > 1e-6:
                problems.append(f"{k}: 녹색 비율이 SUMO 상태와 다름")
            if program_runs(sumo, layout["straight_links"])[1][sumo.trafficlight.getPhase(NEIGHBOR)] != on_now:
                problems.append(f"{k}: 읽은 프로그램의 지금 현시가 SUMO 신호 상태와 다름")
            since = extra[1] * SINCE_CAP_S
            if prev is not None and extra[1] < 1.0:      # 판단 간격은 5초다. 그 사이에 바뀌었으면 지난 시간은 5초 이하
                same = prev[0] == on_now
                if same and prev[1] < 1.0 and abs(since - prev[1] * SINCE_CAP_S - 5.0) > 1.01 and since > 5.0 + 1e-6:
                    problems.append(f"{k}: 같은 상태인데 지난 시간이 {prev[1] * SINCE_CAP_S:.0f} → {since:.0f}초")
                if not same and since > 5.0 + 1e-6:
                    problems.append(f"{k}: 상태가 바뀌었는데 지난 시간 {since:.0f}초")
            prev = (on_now, float(extra[1]))
            rows.append([float(x) for x in extra])
            allowed = [i for i in range(n) if obs[-n + i] > 0.5] or list(range(n))
            obs, _, terminated, truncated, _ = env.step(rng.choice(allowed))
            if terminated or truncated:
                break
    finally:
        env.close()
    flips = sum(1 for a, b in zip(rows, rows[1:]) if (a[0] >= 0.5) != (b[0] >= 0.5))
    report = {"date": date, "period": period, "decisions": len(rows), "observation_size": size, "extra": N_EXTRA,
              "observation_code": code_fingerprint(), "layout": {k: v for k, v in layout.items() if k != "chain"},
              "straight_green_share": round(sum(r[0] for r in rows) / max(1, len(rows)), 3), "straight_flips": flips,
              "mean_moving_by_bin": [round(sum(r[3 + i] for r in rows) / max(1, len(rows)), 3) for i in range(N_EXTRA - 3)],
              "problems": problems[:20], "problem_count": len(problems), "ok": not problems and size == 64 + N_EXTRA and flips > 0}
    out = _arg(args, "--out")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if not report["ok"]:
        raise SystemExit("점검 실패: 학습하지 않는다.")
    print("점검 통과", flush=True)


def main() -> None:
    commands = {"train": train, "eval": evaluate, "check": check}
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        raise SystemExit(f"사용: explore_coord12.py {{{'|'.join(commands)}}} …")
    commands[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
