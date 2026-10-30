"""[추가 탐색] 관측 정보 × 학습 방법 4조합(COMBO, 진행 기록 148절). 지문 밖.

질문: 학생이 탐색의 판단을 새 상황에 옮기지 못하는 것이 관측 정보 때문인가, 학습 방법 때문인가, 둘이 함께 맞아야 하는가?
- 관측: O0 = 72칸. O1 = 72칸 + 이동 8개 × 도착 시간 3칸 [0, 10), [10, 20), [20초 이상) = 96칸. 센 차는 감지 구간 도로 위에서 달리는
  차(0.1m/s 이상), 도착 시간 = 정지선까지 거리 ÷ 지금 도로의 제한속도(예측 제어 P1과 같은 셈), 값 = 차 수 ÷ 20(최대 1; 149절: 감지 구간
  크기로 나누면 값이 0.1 아래로 몰려 실행 전에 바꿈). 끝 4칸(그대로 실행될 행동 표시) 앞에 놓는다.
- 학습: L0 = 후보 상대 비용을 맞히고 가장 작은 것(134절 고정 설정, 문턱 0). L1 = 같은 MLP 출력을 허용 녹색 안 소프트맥스 확률 π로
  보고 Σ π × 상대 비용을 바로 줄인다(실행은 다섯 모델 π 평균이 가장 큰 녹색). 둘 다 10,000번, Adam 1e-3, 감쇠 1e-3, 시작 seed 0~4.
- 칸: A = O0·L0(146절 S6 가중치 그대로), B = O1·L0, C = O0·L1, D = O1·L1. 자료는 6묶음 탐색 시점 314번(기존 229 + 신규 85).
- 새 24칸은 탐색 실행 9개를 기록된 실행 행동으로 다시 돌려 읽는다(마지막 탐색 시점까지 모든 선택 시점의 시각·허용·P0·관측 64·72칸이
  기록과 같아야 함, 139·146절과 같은 범위). 맥과 Colab은 소수점이 조금 달라(10-06) 맥에서는 어긋날 수 있으므로 Colab에서 만든다.
- 평가: 141절과 같은 16개 에피소드에서 P0·A·B·C·D 연속 제어(80번). 학생은 제어 60분 동안 좌회전 정체 시점에서만 고르고 나머지와
  꼬리는 P0(P0+신경망 혼합 제어).

명령(Colab 41번 칸)
  features --old <explore_online/runs> --new <explore_collect/runs> --work --out --workers
  train    --old --new --s6 <explore_collect/students/s6> --out
  evaluate --work --out --workers
  offline  --old --new --out                                   (참고만: 6묶음 하나씩 빼기 저장 장면 평가)
  summary  --out --previous <explore_collect>                  (148절 판정, 146절 P0·S6와 재현 확인)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import time
from pathlib import Path

import numpy as np

import explore_collect as ecl
import explore_distill as ed
import explore_online as eo
import explore_remeasure as rm
import explore_rollout as er
import explore_seedsplit as ess
import explore_student as es

PROJECT = ed.PROJECT
ARRIVAL_EDGES_S = (10.0, 20.0)                 # 도착 시간 칸 [0, 10), [10, 20), [20초 이상)
MOVING_MIN = 0.1                               # 이보다 느리면 멈춘 차(72칸의 기다리는 차, 단순 제어와 같은 문턱)
ARRIVAL_SCALE = 20.0                           # 칸 하나의 차 수 ÷ 20(최대 1). 맥 장치 확인 284번에서 칸 하나 최대 22대, 99%가 13대 이하(149절)
N_MOVEMENTS = 8
N_ARRIVALS = N_MOVEMENTS * (len(ARRIVAL_EDGES_S) + 1)
WIDTH = {"o0": 72, "o1": 72 + N_ARRIVALS}
CELLS = {"a": ("o0", "l0"), "b": ("o1", "l0"), "c": ("o0", "l1"), "d": ("o1", "l1")}
MODES = ("p0", *CELLS)
NEW_CELLS = ("b", "c", "d")
STEPS = ess.STEPS
SEEDS = es.SEEDS
EVAL_GROUPS, EVAL_SEEDS = ecl.EVAL_GROUPS, ecl.EVAL_SEEDS
OLD_RUNS = [(d, p, s) for d, p in eo.EPISODES for s in eo.BASE_SEEDS]
NEW_RUNS = [(d, p, ecl.NEW_SEED) for d, p in ecl.NEW_GROUPS]
SAMPLES = 314                                  # 146절 S6와 같은 자료
SAME_DELAY = 1e-3                              # 재현 확인: 146절과 차량당 지연이 이만큼 안이면 같음


def code_fingerprint() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


# ------------------------------------------------------------------ 새 24칸(순수 계산)
def arrival_bin(eta: float) -> int:
    return sum(eta >= edge for edge in ARRIVAL_EDGES_S)


def arrival_features(vehicles, movements: list) -> list[float]:
    """vehicles = [(이동, 정지선까지 거리 m, 도로 제한속도 m/s, 지금 속도 m/s)]. 이동마다 [0, 10), [10, 20), [20초 이상)에 오는 달리는 차
    ÷ ARRIVAL_SCALE(최대 1). 순서는 이동 먼저(이동 0의 세 칸, 이동 1의 세 칸, …)."""
    n_bins = len(ARRIVAL_EDGES_S) + 1
    index = {m: i for i, m in enumerate(movements)}
    counts = [[0] * n_bins for _ in movements]
    for m, dist, limit, speed in vehicles:
        if speed < MOVING_MIN or m not in index:
            continue
        counts[index[m]][arrival_bin(max(0.0, dist) / limit)] += 1
    return [min(1.0, c / ARRIVAL_SCALE) for row in counts for c in row]


def observation96(obs72, arrivals) -> list[float]:
    """72칸의 끝 4칸(그대로 실행될 행동 표시) 앞에 24칸을 넣는다(72칸이 이동별 8칸을 넣은 자리와 같은 방식)."""
    n = ed.N_GREENS
    return [float(x) for x in (*obs72[:-n], *arrivals, *obs72[-n:])]


# ------------------------------------------------------------------ 새 24칸(SUMO)
_GEO: dict = {}


def _geometry(net_file: str) -> dict:
    if net_file not in _GEO:
        import sumolib
        import explore_pred
        from focused_rl import ZONE_EDGES
        _GEO[net_file] = explore_pred.zone_geometry(sumolib.net.readNet(net_file), ZONE_EDGES)
    return _GEO[net_file]


def approach_vehicles(sumo, geo: dict, movement_of: dict) -> list[tuple]:
    """감지 구간 도로 위 차마다 (이동, 정지선까지 거리, 도로 제한속도, 지금 속도). 거리 셈은 예측 제어 P1과 같다."""
    out = []
    for e in geo["zone_of"]:
        for v in sumo.edge.getLastStepVehicleIDs(e):
            d = geo["offset"][e] + geo["length"][e] - sumo.vehicle.getLanePosition(v)
            out.append((movement_of.get(v), d, geo["speed"][e], sumo.vehicle.getSpeed(v)))
    return out


def arrivals24(ep) -> list[float]:
    """에피소드의 지금 순간 24칸(eo.observation72(ep) 뒤에 부른다: 다시 돌리기·평가 모두 같은 순서)."""
    import explore_choice as ech
    movements = ech.protected_movements([p.state for p in ep.ts.green_phases])
    if len(movements) != N_MOVEMENTS:
        raise SystemExit(f"보호 녹색 이동이 {len(movements)}개다({N_MOVEMENTS}개여야 함)")
    vehicles = approach_vehicles(ep.env.sumo, _geometry(ep.env._net), ep.movement_of)
    return [round(x, 9) for x in arrival_features(vehicles, movements)]


def replay_features(runs: Path, date: str, period: str, seed: int, work: Path) -> dict:
    """탐색 실행 하나를 기록된 '실행한 행동'으로 다시 돌려, 마지막 탐색 시점까지의 모든 선택 시점에서 24칸을 읽는다(139·146절의
    상태 다시 만들기와 같은 범위; 학습에 쓰는 것은 탐색 시점뿐). 탐색 시점에서는 원래처럼 상태를 저장했다가 지운다
    (explore_remeasure.replay와 같은 순서). 선택 시점마다 시각·허용·P0·관측 64·72칸이 기록과 다르면 멈춘다."""
    tag = f"{date}_{period}_s{seed}"
    rows = [json.loads(line) for line in (Path(runs) / f"{tag}_search" / "decisions.jsonl").read_text().splitlines()]
    last = max(i for i, r in enumerate(rows) if r["searched"])
    started = time.time()
    shutil.rmtree(work, ignore_errors=True)
    ep = eo.OnlineEpisode(date, period, work / "routes", seed)
    er.types_only_file(ep.route)
    out, k = [], 0
    try:
        while not ep.done and k <= last:
            allowed, a_p0 = ep.decide()
            action = a_p0
            if len(allowed) > 1:
                t0 = float(ep.env.sim_step)
                if k >= len(rows):
                    raise SystemExit(f"{tag}: 기록보다 선택 시점이 많다")
                row = rows[k]
                k += 1
                er._features(ep, allowed, a_p0)
                obs64, obs72 = eo.observation72(ep)
                bad = rm.mismatches(row, t0, allowed, a_p0, obs64, obs72)
                if bad:
                    raise SystemExit(f"{tag} 시각 {t0}: 기록과 다름 {bad}")
                out.append({"t0": t0, "searched": row["searched"], "arrivals": arrivals24(ep)})
                if row["searched"]:
                    stem = work / "scratch" / "snap"
                    ep.snapshot(stem, allowed, a_p0)
                    for suffix in (".xml", ".pkl"):
                        stem.with_suffix(suffix).unlink(missing_ok=True)
                action = row["executed"]
            ep.apply(action)
    finally:
        ep.close()
    if k != last + 1:
        raise SystemExit(f"{tag}: 마지막 탐색 시점까지 선택 시점 {k}번을 만났다(기록 {last + 1}번)")
    return {"rows": out, "decisions_checked": k, "seconds": round(time.time() - started, 1)}


def _features_job(job: tuple) -> str:
    runs, date, period, seed, work, out = job
    tag = f"{date}_{period}_s{seed}"
    d = Path(out) / "features"
    if (d / f"{tag}.done.json").exists():
        return f"이미 있음 → 건너뜀: {tag}"
    r = replay_features(Path(runs), date, period, seed, Path(work) / "features" / tag)
    d.mkdir(parents=True, exist_ok=True)
    with (d / f"{tag}.jsonl").open("w") as f:
        for x in r["rows"]:
            f.write(json.dumps(x) + "\n")
    done = {"tag": tag, "choice_points": len(r["rows"]), "searched": sum(x["searched"] for x in r["rows"]),
            "decisions_checked": r["decisions_checked"], "seconds": r["seconds"], "features": rm.file_hash(d / f"{tag}.jsonl"),
            "decisions": rm.file_hash(Path(runs) / f"{tag}_search" / "decisions.jsonl"), "code": code_fingerprint(),
            "online_code": eo.code_fingerprint(), "rollout_code": er.code_fingerprint(), "replay_code": rm.code_fingerprint()}
    (d / f"{tag}.done.json").write_text(json.dumps(done, ensure_ascii=False, indent=1))
    shutil.rmtree(Path(work) / "features" / tag, ignore_errors=True)
    return (f"24칸 {tag}: 선택 시점 {r['decisions_checked']}번 기록과 같음(탐색 시점 {done['searched']}번), {r['seconds']}초")


def features(args) -> None:
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed
    rm._guard()
    jobs = [(args.old, d, p, s, args.work, args.out) for d, p, s in OLD_RUNS] + [(args.new, d, p, s, args.work, args.out) for d, p, s in NEW_RUNS]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs)), mp_context=multiprocessing.get_context("spawn")) as pool:
        for f in as_completed([pool.submit(_features_job, j) for j in jobs]):
            print(f.result(), flush=True)


# ------------------------------------------------------------------ 자료
def load_samples(old: Path, new: Path, out: Path) -> list[dict]:
    """6묶음 탐색 시점 314번. obs72 = 기록된 72칸, obs96 = 72칸 + 다시 돌려 읽은 24칸(날짜·seed·시각으로 맞춤, 없으면 멈춤)."""
    samples = ed.load(Path(old)) + ed.load(Path(new))
    feats = {}
    for d, p, s in OLD_RUNS + NEW_RUNS:
        f = Path(out) / "features" / f"{d}_{p}_s{s}.jsonl"
        if not (Path(out) / "features" / f"{d}_{p}_s{s}.done.json").exists():
            raise SystemExit(f"24칸이 없다: {f}")
        for line in f.read_text().splitlines():
            x = json.loads(line)
            feats[(f"{d}_{p}", s, x["t0"])] = x["arrivals"]
    return attach(samples, feats)


def attach(samples: list[dict], feats: dict) -> list[dict]:
    out = []
    for s in samples:
        key = (s["group"], s["seed"], s["t0"])
        if key not in feats:
            raise SystemExit(f"24칸이 붙지 않은 탐색 시점: {key}")
        obs72 = [float(x) for x in s["obs"]]
        out.append({**s, "obs72": np.asarray(obs72, dtype=np.float32),
                    "obs96": np.asarray(observation96(obs72, feats[key]), dtype=np.float32)})
    return out


def obs_key(cell: str) -> str:
    return "obs72" if CELLS[cell][0] == "o0" else "obs96"


def allowed_mask(samples: list[dict]) -> np.ndarray:
    m = np.zeros((len(samples), ed.N_GREENS), dtype=bool)
    for i, s in enumerate(samples):
        m[i, s["allowed"]] = True
    return m


# ------------------------------------------------------------------ 학습
def make_net(width: int):
    """explore_student.make_net과 같은 모양(입력 폭만 바꿀 수 있음)."""
    import torch as th
    layers, w = [], width
    for h in ed.MLP_HIDDEN:
        layers += [th.nn.Linear(w, h), th.nn.ReLU()]
        w = h
    return th.nn.Sequential(*layers, th.nn.Linear(w, ed.N_GREENS))


def train_value(X: np.ndarray, Y: np.ndarray, seed: int, steps: int = STEPS) -> dict:
    """L0: explore_student.train과 같은 학습(허용된 녹색만 제곱 오차). 72칸이면 같은 가중치가 나온다(시험)."""
    import torch as th
    th.manual_seed(seed)
    net = make_net(X.shape[1])
    opt = th.optim.Adam(net.parameters(), lr=ed.MLP_LR, weight_decay=ed.MLP_DECAY)
    x, mask, y = th.as_tensor(X), th.as_tensor(~np.isnan(Y)), th.as_tensor(np.nan_to_num(Y))
    for _ in range(steps):
        opt.zero_grad()
        loss = (((net(x) - y) ** 2) * mask).sum() / mask.sum()
        loss.backward()
        opt.step()
    return {k: v.detach().clone() for k, v in net.state_dict().items()}


def train_policy(X: np.ndarray, Y: np.ndarray, seed: int, steps: int = STEPS) -> dict:
    """L1: 출력을 허용 녹색 안 소프트맥스 확률 π로 보고 시점마다 Σ π × 상대 비용(L0와 같은 목표)의 평균을 줄인다.
    신경망 모양·시작값·최적화·길이는 L0와 같다."""
    import torch as th
    th.manual_seed(seed)
    net = make_net(X.shape[1])
    opt = th.optim.Adam(net.parameters(), lr=ed.MLP_LR, weight_decay=ed.MLP_DECAY)
    x, mask, y = th.as_tensor(X), th.as_tensor(~np.isnan(Y)), th.as_tensor(np.nan_to_num(Y))
    for _ in range(steps):
        opt.zero_grad()
        pi = th.softmax(net(x).masked_fill(~mask, float("-inf")), dim=1)
        loss = (pi * y).sum(dim=1).mean()
        loss.backward()
        opt.step()
    return {k: v.detach().clone() for k, v in net.state_dict().items()}


def fit(samples: list[dict], cell: str, seeds=SEEDS, steps: int = STEPS) -> list[dict]:
    X = np.stack([s[obs_key(cell)] for s in samples])
    Y = np.stack([ed.targets(s) for s in samples])
    trainer = train_value if CELLS[cell][1] == "l0" else train_policy
    return [trainer(X, Y, seed, steps) for seed in seeds]


def _outputs(states: list[dict], X: np.ndarray) -> list[np.ndarray]:
    import torch as th
    outs = []
    for state in states:
        net = make_net(X.shape[1])
        net.load_state_dict(state)
        net.eval()
        with th.no_grad():
            outs.append(net(th.as_tensor(X)).numpy())
    return outs


def policy_probs(states: list[dict], X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """모델마다 허용 녹색 안 소프트맥스 π, 다섯 모델 평균. 허용 안 된 녹색은 0."""
    probs = []
    for z in _outputs(states, X):
        z = np.where(mask, z.astype(np.float64), -np.inf)
        z = np.exp(z - z.max(axis=1, keepdims=True))
        probs.append(z / z.sum(axis=1, keepdims=True))
    return np.mean(probs, axis=0)


def pick_policy(prob: np.ndarray, sample: dict) -> int:
    """π 평균이 가장 큰 허용 녹색(같으면 P0, 그다음 작은 번호)."""
    high = max(prob[c] for c in sample["allowed"])
    if prob[sample["p0"]] >= high - 1e-9:
        return sample["p0"]
    return min(c for c in sample["allowed"] if prob[c] >= high - 1e-9)


def picks(states: list[dict], cell: str, samples: list[dict]) -> tuple[list[int], np.ndarray]:
    """칸의 실행 규칙으로 고른 녹색과 예측(L0: 다섯 모델 평균 상대 비용, L1: π 평균)."""
    X = np.stack([s[obs_key(cell)] for s in samples])
    if CELLS[cell][1] == "l0":
        pred = np.mean(_outputs(states, X), axis=0)
        return [es.hybrid(p, s, ess.TAU) for p, s in zip(pred, samples)], pred
    prob = policy_probs(states, X, allowed_mask(samples))
    return [pick_policy(p, s) for p, s in zip(prob, samples)], prob


def _data_hashes(old: Path, new: Path, out: Path) -> dict:
    return {"old": ecl.data_hashes(Path(old)), "new": ecl.data_hashes(Path(new)),
            "features": {f.name: rm.file_hash(f) for f in sorted((Path(out) / "features").glob("*.jsonl"))}}


def train(args) -> None:
    import torch as th
    out = Path(args.out)
    samples = load_samples(Path(args.old), Path(args.new), out)
    if len(samples) != SAMPLES or len({s["group"] for s in samples}) != 6:
        raise SystemExit(f"자료가 148절과 다르다: {len(samples)}번, 묶음 {len({s['group'] for s in samples})}개")
    a = out / "students" / "a"
    if not (a / "policy.json").exists():                              # A = 146절 S6 가중치 그대로
        s6 = json.loads((Path(args.s6) / "policy.json").read_text())
        if s6["samples"] != SAMPLES or s6["steps"] != STEPS or s6["tau"] != ess.TAU or s6["model_seeds"] != list(SEEDS):
            raise SystemExit("S6가 146절 설정과 다르다")
        a.mkdir(parents=True, exist_ok=True)
        for s in SEEDS:
            if rm.file_hash(Path(args.s6) / f"seed{s}.pt") != s6["weights"][f"seed{s}"]:
                raise SystemExit("S6 가중치가 저장 때와 다르다")
            shutil.copy2(Path(args.s6) / f"seed{s}.pt", a / f"seed{s}.pt")
        policy = {"cell": "a", "obs": "o0", "method": "l0", "source": "146절 S6", "samples": s6["samples"], "steps": STEPS,
                  "model_seeds": list(SEEDS), "weights": s6["weights"], "s6_policy": s6, "code": code_fingerprint()}
        (a / "policy.json").write_text(json.dumps(policy, ensure_ascii=False, indent=1))
        print(f"학생 a: 146절 S6 가중치 복사(해시 같음)", flush=True)
    for cell in NEW_CELLS:
        d = out / "students" / cell
        if (d / "policy.json").exists():
            print(f"이미 있음 → 건너뜀: {d}")
            continue
        started = time.time()
        states = fit(samples, cell)
        d.mkdir(parents=True, exist_ok=True)
        for seed, state in zip(SEEDS, states):
            th.save(state, d / f"seed{seed}.pt")
        chosen, _ = picks(states, cell, samples)
        policy = {"cell": cell, "obs": CELLS[cell][0], "method": CELLS[cell][1], "width": WIDTH[CELLS[cell][0]], "samples": len(samples),
                  "groups": sorted({s["group"] for s in samples}), "steps": STEPS, "model_seeds": list(SEEDS),
                  "weights": {f"seed{s}": rm.file_hash(d / f"seed{s}.pt") for s in SEEDS}, "data": _data_hashes(args.old, args.new, out),
                  "train_changes_from_p0": sum(p != s["p0"] for p, s in zip(chosen, samples)),
                  "seconds": round(time.time() - started, 1), "code": code_fingerprint(),
                  "student_code": es.code_fingerprint(), "distill_code": ed.code_fingerprint()}
        (d / "policy.json").write_text(json.dumps(policy, ensure_ascii=False, indent=1))
        print(f"학생 {cell}({CELLS[cell][0]}·{CELLS[cell][1]}): 탐색 시점 {len(samples)}번, 학습 자료 안에서 P0와 다르게 고름 "
              f"{policy['train_changes_from_p0']}번, {policy['seconds']}초", flush=True)


def load_cell(out: Path, cell: str) -> tuple[list, dict]:
    import torch as th
    d = Path(out) / "students" / cell
    policy = json.loads((d / "policy.json").read_text())
    for s in policy["model_seeds"]:
        if rm.file_hash(d / f"seed{s}.pt") != policy["weights"][f"seed{s}"]:
            raise SystemExit(f"학생 {cell}의 가중치가 저장 때와 다르다")
    return [th.load(d / f"seed{s}.pt") for s in policy["model_seeds"]], policy


# ------------------------------------------------------------------ 평가(Colab)
def eval_jobs(work: str, out: str) -> list[tuple]:
    return [(d, p, s, m, work, out) for d, p in EVAL_GROUPS for s in EVAL_SEEDS for m in MODES]


def run_episode(job: tuple) -> str:
    """평가 에피소드 하나: P0만, 또는 제어 60분 동안 좌회전 정체 시점에서만 학생 칸(나머지·꼬리는 P0). 146절 평가와 같은 셈이고,
    모든 방식이 모든 선택 시점에서 24칸을 읽어 기록한다(쓰는 것은 B·D뿐)."""
    import focused_train as ft
    from focused_eval import judge_collisions
    date, period, seed, mode, work, out = job
    tag = f"{date}_{period}_s{seed}_{mode}"
    d = Path(out) / "eval" / tag
    if (d / "metrics.json").exists():
        return f"{tag}: 이미 있음"
    states, policy = load_cell(Path(out), mode) if mode != "p0" else (None, None)
    w = Path(work) / "eval" / tag
    shutil.rmtree(w, ignore_errors=True)
    started = time.time()
    ep = eo.OnlineEpisode(date, period, w / "routes", seed)
    end_control = ep.handover + ft.CONTROL_S
    decisions = []
    try:
        while not ep.done:
            allowed, a_p0 = ep.decide()
            action = a_p0
            if len(allowed) > 1:
                t0 = float(ep.env.sim_step)
                feat = er._features(ep, allowed, a_p0)
                obs64, obs72 = eo.observation72(ep)
                arrivals = arrivals24(ep)
                row = {"t0": t0, "in_control": t0 < end_control, "allowed": allowed, "p0_action": a_p0, "W": feat["W"],
                       **{k: feat[k] for k in (*er.LEFTS, *[f"{n}_served" for n in er.LEFTS])}, "left_point": eo.is_left_point(feat),
                       "obs64": obs64, "obs72": obs72, "arrivals": arrivals, "student": False}
                if states is not None and row["in_control"] and row["left_point"]:
                    s = {"allowed": allowed, "p0": a_p0, "obs72": np.asarray(obs72, dtype=np.float32),
                         "obs96": np.asarray(observation96(obs72, arrivals), dtype=np.float32)}
                    chosen, pred = picks(states, mode, [s])
                    action = chosen[0]
                    row.update(student=True, pred=[round(float(x), 6) for x in pred[0]])
                row["executed"] = action
                decisions.append(row)
            ep.apply(action)
    finally:
        ep.close()
    rows = eo.read_tripinfo(ep.tripinfo)
    metrics = eo.episode_metrics(rows, ep.vehicles, ep.handover, ft.CONTROL_S)
    coll = judge_collisions(ep.collisions)
    st = [x for x in decisions if x["student"]]
    metrics.update({"date": date, "period": period, "seed": seed, "mode": mode, "handover": ep.handover, "code": code_fingerprint(),
                    "online_code": eo.code_fingerprint(), "p0_code": er.P0_CODE, "student_weights": policy["weights"] if policy else None,
                    "seconds": round(time.time() - started, 1), "choice_points": len(decisions),
                    "left_points_in_control": sum(x["in_control"] and x["left_point"] for x in decisions),
                    "student_decisions": len(st), "changes": sum(x["executed"] != x["p0_action"] for x in st),
                    "student_greens": {str(g): sum(x["executed"] == g for x in st) for g in range(ed.N_GREENS)},
                    "changed_by_rules": ep.cons.interventions, "collisions_at_204820": coll["collisions_at_204820"],
                    "collisions_unexplained": len(coll["collisions_unexplained"])})
    d.mkdir(parents=True, exist_ok=True)
    with (d / "decisions.jsonl").open("w") as f:
        for x in decisions:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    (d / "vehicle_delays.json").write_text(json.dumps({v: float(r["timeLoss"]) + float(r["departDelay"]) for v, r in rows.items()}))
    tmp = d / "metrics.json.tmp"
    tmp.write_text(json.dumps(metrics, ensure_ascii=False, indent=1))
    tmp.replace(d / "metrics.json")
    shutil.rmtree(w, ignore_errors=True)
    return (f"{tag}: 차량당 지연 {metrics['mean_delay_s']}초, 좌회전 동/남 {metrics['left_east']}/{metrics['left_south']}초, "
            f"300초 초과 {metrics['long_waits']}, 학생 판단 {len(st)}번(변경 {metrics['changes']}), {metrics['seconds']}초")


def evaluate(args) -> None:
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed
    rm._guard()
    for cell in CELLS:
        load_cell(Path(args.out), cell)                                # 가중치가 다 있고 저장 때와 같은지 먼저
    jobs = [j for j in eval_jobs(args.work, args.out) if not (Path(args.out) / "eval" / f"{j[0]}_{j[1]}_s{j[2]}_{j[3]}" / "metrics.json").exists()]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for f in as_completed([pool.submit(run_episode, j) for j in jobs]):
            print(f.result(), flush=True)


# ------------------------------------------------------------------ 참고: 저장 장면(판정에 안 씀)
def offline(args) -> None:
    """6묶음 하나씩 빼기: 네 칸 모두 같은 방법으로 남은 5묶음에 다시 학습해 뺀 묶음의 8개 평균 비용으로 잰다(A도 다시 학습)."""
    out = Path(args.out)
    if (out / "offline.json").exists():
        print(f"이미 있음 → 건너뜀: {out / 'offline.json'}")
        return
    samples = load_samples(Path(args.old), Path(args.new), out)
    groups = sorted({s["group"] for s in samples})
    result = {"code": code_fingerprint(), "samples": len(samples), "groups": groups, "cells": {}}
    for cell in CELLS:
        tests, chosen, per_group = [], [], {}
        for g in groups:
            fold = [s for s in samples if s["group"] != g]
            test = [s for s in samples if s["group"] == g]
            p, _ = picks(fit(fold, cell), cell, test)
            per_group[g] = ess.measure(test, p)
            tests += test
            chosen += p
        result["cells"][cell] = {"total": ess.measure(tests, chosen), "per_group": per_group}
        t = result["cells"][cell]["total"]
        print(f"참고 저장 장면 {cell}: 이득 {t['gain']}대·초(몫 {t['share']}), 변경 {t['changes']}번(이득 {t['good_changes']}·손해 {t['bad_changes']})",
              flush=True)
    (out / "offline.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))


# ------------------------------------------------------------------ 판정
def episode_table(out: Path) -> tuple[list[dict], list[str]]:
    rows, missing = [], []
    for d, p in EVAL_GROUPS:
        for s in EVAL_SEEDS:
            got = {}
            for m in MODES:
                f = Path(out) / "eval" / f"{d}_{p}_s{s}_{m}" / "metrics.json"
                if f.exists():
                    got[m] = json.loads(f.read_text())
                else:
                    missing.append(f.parent.name)
            if len(got) == len(MODES):
                rows.append({"group": f"{d}_{p}", "seed": s, **got})
    return rows, missing


def improve_vs_a(rows: list[dict], mode: str) -> dict:
    """148절 판정 1(141절 주 판정과 같은 조건): 칸이 기준 A보다 나은가(A 지연 − 칸 지연)."""
    c = ecl.compare(rows, "a", mode)
    enough_changes = c["changes"][mode] >= ecl.CHANGE_MIN * c["changes"]["a"]
    if (c["mean"] > 0 and c["interval"][0] > 0 and c["positive_groups"] >= ecl.GROUPS_MIN and not c["lefts_worse"] and not c["long_worse"]
            and enough_changes):
        word = "기준보다 나아졌다"
    elif c["mean"] <= 0:
        word = "변화 없음 또는 나빠짐"
    else:
        word = "불분명"
    return {"word": word, "enough_changes": enough_changes, **c}


def effect_values(rows: list[dict]) -> dict:
    """에피소드마다 나눈 효과(줄인 지연, 양수가 좋음). 관측 = ½[(A−B)+(C−D)], 학습 = ½[(A−C)+(B−D)], 상호작용 = (C−D)−(A−B)."""
    y = lambda r, m: r[m]["mean_delay_s"]
    return {"observation": [0.5 * ((y(r, "a") - y(r, "b")) + (y(r, "c") - y(r, "d"))) for r in rows],
            "learning": [0.5 * ((y(r, "a") - y(r, "c")) + (y(r, "b") - y(r, "d"))) for r in rows],
            "interaction": [(y(r, "c") - y(r, "d")) - (y(r, "a") - y(r, "b")) for r in rows]}


def effect_reading(rows: list[dict], values: list[float]) -> dict:
    mean, lo, hi = er.t_interval(values)
    groups = {}
    for r, v in zip(rows, values):
        groups.setdefault(r["group"], []).append(v)
    g = {k: statistics.fmean(v) for k, v in sorted(groups.items())}
    pos, neg = sum(v > 0 for v in g.values()), sum(v < 0 for v in g.values())
    if mean > 0 and lo > 0 and pos >= ecl.GROUPS_MIN:
        word = "좋아지는 쪽 단서"
    elif mean < 0 and hi < 0 and neg >= ecl.GROUPS_MIN:
        word = "나빠지는 쪽 단서"
    else:
        word = "가를 수 없음"
    return {"word": word, "mean": round(mean, 3), "interval": [round(lo, 3), round(hi, 3)], "groups": {k: round(v, 3) for k, v in g.items()},
            "positive_groups": pos, "negative_groups": neg}


def readings(rows: list[dict]) -> dict:
    vs_a = {m: improve_vs_a(rows, m) for m in NEW_CELLS}
    vs_p0 = {m: ecl.beats_p0(rows, m) for m in CELLS}
    eff = {k: effect_reading(rows, v) for k, v in effect_values(rows).items()}
    together = (vs_a["d"]["word"] == "기준보다 나아졌다" and all(vs_a[m]["word"] != "기준보다 나아졌다" for m in ("b", "c"))
                and eff["interaction"]["word"] == "좋아지는 쪽 단서")
    improved = [m for m in NEW_CELLS if vs_a[m]["word"] == "기준보다 나아졌다"]
    if any(vs_p0[m]["word"] == "P0를 넘었다" for m in improved):
        overall = "학습한 혼합 제어가 P0를 넘은 첫 단서(개발 평가)"
    elif improved:
        overall = "기준보다 나아졌지만 아직 P0를 넘지 못함"
    else:
        overall = "이 4조합으로는 개선을 확인하지 못했다"
    return {"overall": overall, "improved": improved, "together": together, "vs_a": vs_a, "vs_p0": vs_p0, "effects": eff}


def reproduction(rows: list[dict], previous: Path) -> dict:
    """새 실행의 P0·A가 146절의 P0·S6와 에피소드마다 같은가(차량당 지연 SAME_DELAY 안)."""
    out = {"p0": [], "a": []}
    for r in rows:
        d, p = r["group"].split("_")
        for mode, old in (("p0", "p0"), ("a", "s6")):
            f = Path(previous) / "eval" / f"{d}_{p}_s{r['seed']}_{old}" / "metrics.json"
            if not f.exists():
                out[mode].append({"group": r["group"], "seed": r["seed"], "previous": None})
                continue
            m = json.loads(f.read_text())
            out[mode].append({"group": r["group"], "seed": r["seed"], "previous": m["mean_delay_s"], "now": r[mode]["mean_delay_s"],
                              "changes_previous": m.get("changes"), "changes_now": r[mode]["changes"]})
    same = lambda xs: all(x["previous"] is not None and abs(x["previous"] - x["now"]) <= SAME_DELAY for x in xs)
    return {"p0_same": same(out["p0"]), "a_same": same(out["a"]), **out}


def summary(args) -> None:
    out = Path(args.out)
    rows, missing = episode_table(out)
    codes = {r[m]["code"] for r in rows for m in MODES}
    if len(codes) > 1:
        raise SystemExit(f"서로 다른 코드의 평가가 섞였다: {sorted(codes)}")
    complete = not missing and len(rows) == len(EVAL_GROUPS) * len(EVAL_SEEDS)
    off = json.loads((out / "offline.json").read_text()) if (out / "offline.json").exists() else None
    mean_of = lambda m, k: round(statistics.fmean(r[m][k] for r in rows), 3) if rows else None
    result = {"code": code_fingerprint(), "eval_code": codes.pop() if codes else None, "missing": missing, "episodes": len(rows),
              "reading": readings(rows) if complete else None,
              "reproduction": reproduction(rows, Path(args.previous)) if complete and args.previous else None,
              "means": {m: {k: mean_of(m, k) for k in ("mean_delay_s", "left_east", "left_south")} for m in MODES},
              "totals": {m: {k: sum(r[m][k] for r in rows) for k in ("long_waits", "unfinished", "changes", "student_decisions",
                                                                    "changed_by_rules", "collisions_unexplained", "missing")} for m in MODES},
              "student_greens": {m: {g: sum(r[m]["student_greens"][g] for r in rows) for g in map(str, range(ed.N_GREENS))} for m in CELLS},
              "offline_reference": {c: off["cells"][c]["total"] for c in CELLS} if off else None,
              "per_episode": [{"group": r["group"], "seed": r["seed"],
                               **{m: {k: r[m][k] for k in ("mean_delay_s", "left_east", "left_south", "long_waits", "unfinished", "changes",
                                                          "student_decisions", "changed_by_rules", "collisions_unexplained")}
                                  for m in MODES}} for r in rows]}
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    names = {"a": "A(72칸·비용 예측, 기준)", "b": "B(96칸·비용 예측)", "c": "C(72칸·확률 직접)", "d": "D(96칸·확률 직접)"}
    lines = ["# 관측 정보 × 학습 방법 4조합(진행 기록 148절)", ""]
    if not complete:
        lines.append(f"- 평가가 다 모이지 않음: {', '.join(missing)}")
    else:
        r = result["reading"]
        lines += [f"- 전체: **{r['overall']}**" + (" — 정보와 학습 방법이 함께 맞아야 나아진다(단서)" if r["together"] else ""), ""]
        for m in NEW_CELLS:
            v = r["vs_a"][m]
            lines.append(f"- {names[m]} 대 A: **{v['word']}** — 평균 {v['mean']}초 [{v['interval'][0]}, {v['interval'][1]}], 묶음 {v['positive_groups']}/4, "
                         f"좌회전 나빠짐 {v['lefts_worse']}, 300초 초과 나빠짐 {v['long_worse']}, 변경 A {v['changes']['a']}번 → {v['changes'][m]}번")
        for m in CELLS:
            b = r["vs_p0"][m]
            lines.append(f"- {names[m]} 대 P0: **{b['word']}** — Δ 평균 {b['mean']}초 [{b['interval'][0]}, {b['interval'][1]}], "
                         f"에피소드 {b['positive_episodes']}/16, 묶음 {b['positive_groups']}/4")
        label = {"observation": "관측 효과", "learning": "학습 효과", "interaction": "상호작용"}
        for k, e in r["effects"].items():
            lines.append(f"- {label[k]}: **{e['word']}** — 평균 {e['mean']}초 [{e['interval'][0]}, {e['interval'][1]}], 좋아진 묶음 {e['positive_groups']}/4")
        rep = result["reproduction"]
        if rep:
            lines.append(f"- 재현 확인(146절과 같은가): P0 {'같음' if rep['p0_same'] else '다름'}, A(= S6) {'같음' if rep['a_same'] else '다름'}")
        lines += ["", "| 방식 | 차량당 지연 | 동쪽 좌회전 | 남쪽 좌회전 | 300초 초과 | 학생 판단(변경) | 고른 녹색 0/1/2/3 |", "|---|---:|---:|---:|---:|---|---|"]
        for m in MODES:
            mm, t = result["means"][m], result["totals"][m]
            greens = "/".join(str(result["student_greens"][m][g]) for g in map(str, range(ed.N_GREENS))) if m in CELLS else "—"
            lines.append(f"| {names.get(m, 'P0')} | {mm['mean_delay_s']} | {mm['left_east']} | {mm['left_south']} | {t['long_waits']} | "
                         f"{t['student_decisions']}({t['changes']}) | {greens} |")
        lines += ["", "| 묶음 | seed | P0 | A | B | C | D | 변경 A/B/C/D |", "|---|---:|---:|---:|---:|---:|---:|---|"]
        for e in result["per_episode"]:
            lines.append(f"| {e['group']} | {e['seed']} | " + " | ".join(str(e[m]["mean_delay_s"]) for m in MODES) + " | "
                         + "/".join(str(e[m]["changes"]) for m in CELLS) + " |")
    if off:
        lines.append("")
        for c in CELLS:
            t = off["cells"][c]["total"]
            lines.append(f"- 참고(판정 안 씀) 저장 장면 6묶음 하나씩 빼기 {names[c]}: 이득 {t['gain']}대·초(몫 {t['share']}), 변경 {t['changes']}번"
                         f"(이득 {t['good_changes']}·손해 {t['bad_changes']})")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features")
    t = sub.add_parser("train")
    o = sub.add_parser("offline")
    for p in (f, t, o):
        p.add_argument("--old", required=True)
        p.add_argument("--new", required=True)
        p.add_argument("--out", required=True)
    f.add_argument("--work", required=True)
    f.add_argument("--workers", type=int, default=4)
    t.add_argument("--s6", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--work", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--workers", type=int, default=4)
    m = sub.add_parser("summary")
    m.add_argument("--out", required=True)
    m.add_argument("--previous", default=None, help="146절 결과 폴더(explore_collect): P0·S6 재현 확인")
    args = ap.parse_args()
    {"features": features, "train": train, "evaluate": evaluate, "offline": offline, "summary": summary}[args.cmd](args)


if __name__ == "__main__":
    main()
