"""[추가 탐색] NN3 — 신경망 강화학습 라운드 1(계획서 3A단계, 사전 등록 results/explore/nn3_20261103/사전등록.md).

준비(prep): 학습하지 않은 새벽·심야 기록을 더한 흉내 모델 N6를 N5와 견주어 출발 모델을 고른다.
라운드(round): 출발 모델이 직접 운전하는 판의 25% 지점에서 허용 녹색을 모두 같은 표본 미래로 굴린다(뒤도 그 신경망이 잇는다)
  → 계획기 없는 학생을 라벨 10개로 학습 → 한 시간 SUMO로 라벨·문턱을 거르고 확인한다. 예측 이득이 문턱 이상일 때만 바꾼다.
잠긴 파일·이전 실험 코드와 결과는 고치지 않고 가져다 쓴다. 실행 때 계획기를 부르면 멈춘다.

사용(FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240):
  python src/explore_nn3.py make-config --workers 8                    (맥, SUMO 없음: 새 상황의 차량 파일 SHA 고정)
  python src/explore_nn3.py check --models <N5 폴더> --out <dir> --work /content/n3c --workers 8
  python src/explore_nn3.py prep --nn1 <Drive>/explore_nn1/P/aug --teacher <Drive>/explore_platoon1/students/seed_42 --out <Drive>/explore_nn3/prep --work /content/n3p --workers 8
  python src/explore_nn3.py round --prep <Drive>/explore_nn3/prep --out <Drive>/explore_nn3/r1 --work /content/n3r --shard 0 --shards 3 --workers 8
  python src/explore_nn3.py judge --root <Drive>/explore_nn3 --out <Drive>/explore_nn3/judge
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import multiprocessing
import os
import pickle
import shutil
import statistics
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import arena3_bprime as bp  # noqa: E402
import arena3_common as ac  # noqa: E402
import arena3_referee as ref  # noqa: E402
import arena3_stage2_eval as s2e  # noqa: E402
import explore_nn1 as nn  # noqa: E402
import explore_platoon0 as p0  # noqa: E402
import explore_platoon1 as pt  # noqa: E402
import explore_weight as w  # noqa: E402

EXPERIMENT = "NN3_V1"
ROUND = 1
CONFIG = ROOT / "config/explore_nn3_v1.json"
PREREG = ROOT / "results/explore/nn3_20261103/사전등록.md"
REFERENCE = ROOT / "results/explore/nn3_20261103/smoke_reference/check.json"
NN1_CONFIG, NN12_CONFIG = ROOT / "config/explore_nn1_v1.json", ROOT / "config/explore_nn12_v1.json"
SOURCES = ("src/explore_nn3.py", "config/explore_nn3_v1.json", "results/explore/nn3_20261103/사전등록.md", *nn.SOURCES)
WORLD = "P"
GREENS = nn.GREENS
SCALE = nn.SCALE
INPUT_SIZE = len(nn.COLUMNS[WORLD])
TRAIN_DATES, VALID_DATES = nn.TRAIN_DATES, nn.VALID_DATES
LOW_SLOTS = {"dawn": tuple(f"{h:02d}{m:02d}" for h in range(1, 5) for m in (0, 15, 30, 45)), "late": tuple(f"22{m:02d}" for m in (0, 15, 30, 45))}
LOW_LOG_COUNTS = {"dawn": 3, "late": 1}
LOW_ROUND_COUNTS = {"dawn": 1, "late": 1}
LOW_VALID_PERIODS = ("0200", "0400", "2215")
FINAL_PERIODS = tuple(f"{(360 + 45 * j) // 60:02d}{(360 + 45 * j) % 60:02d}" for j in range(21))
SLOT_SALT = "NN3_SLOTS"
ROLLOUT = {"futures": 8, "horizon_s": 600, "long_horizon_s": 900, "long_fraction": 0.10, "long_salt": "NN3_LONG_POINTS", "marks": [30, 60, 150, 300, 600], "long_marks": [30, 60, 150, 300, 600, 900],
           "move_marks_from": 150, "future_salt": "NN3_FUTURE_SEED", "min_valid_futures": 4, "max_failed_rollout_fraction": 0.02, "max_backlog_jobs": 1500, "main_episode_max_seconds": 10800.0,
           "restore_check": {"tolerance": 1e-6, "max_mismatch_rate": 0.08, "min_points": 200}}
POINTS = {"fraction": 0.25, "salt": "NN3_R1_POINTS"}
LEARN = {"label_marks": [30, 60, 150, 300, 600], "label_weights": [25.0, 75.0], "shares": [0.05, 0.10, 0.20], "screen_share": 0.10, "top_labels": 3, "finalists": 2}
RULES = {"n6_low_gain_s": 0.5, "n6_day_tolerance_s": 0.2, "adopt_gain_s": 0.2}
T975 = {83: 1.98896, 45: 2.01410}
BASE = "BASE"
EAST_EXIT = "477706392_01"            # 석천사거리에서 동쪽(204822 방향)으로 나가는 도로
STAGES = ("screen", "thresholds", "final")


# ----------------------------------------------------------------------------- 설정
def config() -> dict:
    cfg = w.read(CONFIG)
    if cfg.get("experiment") != EXPERIMENT:
        raise ValueError("NN3 설정이 아닙니다.")
    for key, path in (("prereg_sha256", PREREG), ("lock_sha256", ac.CONFIG), ("nn1_config_sha256", NN1_CONFIG), ("nn12_config_sha256", NN12_CONFIG)):
        if cfg.get(key) != w.file_hash(path):
            raise ValueError(f"NN3 설정을 만든 때와 파일이 다릅니다: {Path(path).relative_to(ROOT)}")
    return cfg


def _picked(slots, salt: str, count: int) -> list[str]:
    return sorted(sorted(slots, key=lambda p: hashlib.sha256(f"{SLOT_SALT}|{salt}|{p}".encode()).hexdigest())[:count])


def low_log_situations() -> list[dict]:
    """N6에 더할 기록: 학습 6일 × 새벽 3칸 + 심야 1칸 × 라운드 2개(0 = 선생님 운전, 1 = N5 운전)."""
    out, index = [], 0
    for r in (0, 1):
        for date in TRAIN_DATES:
            for band, count in LOW_LOG_COUNTS.items():
                for period in _picked(LOW_SLOTS[band], f"low|{r}|{date}|{band}", count):
                    index += 1
                    out.append({**ac.make_situation(date, period, f"N3L{r}x{index}"), "round": r, "band": band})
    return out


def low_validation_situations() -> list[dict]:
    return [ac.make_situation(d, p, f"VL{j}") for d in VALID_DATES for p in LOW_VALID_PERIODS for j in (1, 2)]


def round_situations(round_: int = ROUND) -> list[dict]:
    """학습 6일 × 날짜당 10칸(NN1 자료 강화와 같은 묶음 8칸 + 새벽 1칸 + 심야 1칸), 새 도착 배치. 칸은 결과와 무관한 해시 규칙."""
    bands = {**{k: (n, slots) for k, (n, slots) in nn.AUG_BANDS.items()}, **{k: (LOW_ROUND_COUNTS[k], LOW_SLOTS[k]) for k in LOW_SLOTS}}
    if sum(n for n, _ in bands.values()) != 10:
        raise ValueError("날짜당 칸이 10개가 아닙니다.")
    out, index = [], 0
    for date in TRAIN_DATES:
        for band, (count, slots) in bands.items():
            for period in _picked(slots, f"r{round_}|{date}|{band}", count):
                index += 1
                out.append({**ac.make_situation(date, period, f"N3R{round_}x{index}"), "round": round_, "band": band})
    return out


def final_validation_situations() -> list[dict]:
    return [ac.make_situation(d, p, f"VF{j}") for d in VALID_DATES for p in FINAL_PERIODS for j in (1, 2)]


def make_config(workers: int) -> dict:
    """맥, SUMO 없음. 새 상황의 무리 세계 차량 파일 SHA를 고정한다."""
    lock, nn1, nn12 = ac.config(), nn.config(), w.read(NN12_CONFIG)
    logs, low, data, final = low_log_situations(), low_validation_situations(), round_situations(), final_validation_situations()
    rows = [*logs, *low, *data, *final]
    if len({ac.situation_id(s) for s in rows}) != len(rows):
        raise ValueError("새 상황 이름이 겹칩니다.")
    previous = w.read(CONFIG) if CONFIG.exists() else {}
    keep = {ac.situation_id(s): s["route_sha256"] for part in ("situations_low_logs", "situations_low_validation", "situations_round", "situations_final") for s in previous.get(part, [])}
    need = [s for s in rows if ac.situation_id(s) not in keep]
    if need:
        with tempfile.TemporaryDirectory() as d, p0._pool(workers) as pool:
            for sid, _, sha in pool.map(p0._sha_job, [(s, WORLD, d) for s in need]):
                keep[sid] = sha
    for s in rows:
        s["route_sha256"] = keep[ac.situation_id(s)]
    used = {x["route_sha256"] for part in ("eval", "confirmation", "bprime_data") for x in lock[f"situations_{part}"]}
    pcfg = pt.config()
    used |= {s["route_sha256"] for part in ("situations_data", "situations_eval", "situations_confirmation") for s in pcfg[part]}
    used |= {s["route_sha256"]["P"] for part in ("situations_aug", "situations_validation") for s in nn1[part]}
    new = [s["route_sha256"] for s in rows]
    if len(set(new)) != len(new) or set(new) & used:
        raise ValueError("새 차량 파일이 서로 같거나 기존 차량 파일과 같습니다.")
    cfg = {"experiment": EXPERIMENT, "round": ROUND, "world": WORLD, "design": "docs/신경망_강화학습_계획서_20261102.md 3A단계, PROJECT_PROGRESS.md 271절, results/explore/nn3_20261103/사전등록.md",
           "prereg": PREREG.relative_to(ROOT).as_posix(), "prereg_sha256": w.file_hash(PREREG), "lock_sha256": w.file_hash(ac.CONFIG), "nn1_config_sha256": w.file_hash(NN1_CONFIG), "nn12_config_sha256": w.file_hash(NN12_CONFIG),
           "n5": nn12["models"]["N5"], "teachers": nn1["teachers"], "training": nn1["training"], "points": POINTS, "rollout": ROLLOUT, "learn": LEARN, "rules": RULES,
           "input": nn1["input"]["P"], "reporting": "검증 날짜의 개발 비교다. 출발 모델은 흉내 값이고 강화학습의 몫은 라운드 정책 − 출발 모델로만 말한다. 무리 도착은 모형이다.",
           "situations_low_logs": logs, "situations_low_validation": low, "situations_round": data, "situations_final": final}
    CONFIG.write_text(json.dumps(w.plain(cfg), ensure_ascii=False, indent=1) + "\n")
    return {"config": str(CONFIG), "sha256": w.file_hash(CONFIG), "situations": len(rows), "new_sha": len(need)}


def day_validation(cfg: dict) -> list[dict]:
    """NN1 검증 24상황(0928·0814 × 6칸 × 배치 2개). N5의 저장 결과와 같은 상황이다."""
    return nn.named_situations(nn.config(), "situations_validation", WORLD)


# ----------------------------------------------------------------------------- 고르기(순수 계산)
def choose(net, stack, x: np.ndarray, allowed: list[int]) -> tuple[int, list[tuple[int, int]]]:
    """신경망의 녹색에서 시작해 쌓은 순서대로 학생을 적용한다. (녹색, 실제로 바꾼 쌍)."""
    action = nn.green_choice(net, x, allowed)
    changes = []
    for student, tau in stack:
        new, _ = nn.student_choice(student, x, action, allowed, float(tau))
        if new != action:
            changes.append((action, new))
            action = new
    return action, changes


def candidate_name(mark: int, weight: float, share: float) -> str:
    return f"h{int(mark)}w{int(weight)}s{int(round(share * 100)):02d}"


def parse_candidate(name: str) -> tuple[int, float, float]:
    h, rest = name[1:].split("w")
    weight, share = rest.split("s")
    return int(h), float(weight), int(share) / 100.0


def screen_candidates(cfg: dict) -> list[str]:
    learn = cfg["learn"]
    return [candidate_name(h, a, learn["screen_share"]) for h in learn["label_marks"] for a in learn["label_weights"]]


def rank_key(name: str, delay: float) -> tuple:
    """짧은 지연 → 작은 비율 → 긴 h."""
    h, _, share = parse_candidate(name)
    return (delay, share, -h)


def threshold_candidates(screen_means: dict[str, float], cfg: dict) -> list[str]:
    """거르기 상위 라벨 × 나머지 비율."""
    learn = cfg["learn"]
    top = sorted(screen_means, key=lambda n: rank_key(n, screen_means[n]))[:int(learn["top_labels"])]
    out = []
    for name in top:
        h, weight, _ = parse_candidate(name)
        out += [candidate_name(h, weight, s) for s in learn["shares"] if s != learn["screen_share"]]
    return out


def finalists(means: dict[str, float], cfg: dict) -> list[str]:
    return sorted(means, key=lambda n: rank_key(n, means[n]))[:int(cfg["learn"]["finalists"])]


def n6_decision(low_gain: float, day_gain: float, rules: dict) -> dict:
    """gain = N5 − N6(양수면 N6가 짧다)."""
    take = low_gain >= float(rules["n6_low_gain_s"]) and day_gain >= -float(rules["n6_day_tolerance_s"])
    return {"start": "N6" if take else "N5", "low_gain_s": low_gain, "day_gain_s": day_gain}


def weighted_mean(rows: list[dict]) -> float:
    return math.fsum(float(r["total"]) for r in rows) / sum(int(r["vehicles"]) for r in rows)


def paired_gain(base: dict[str, list[dict]], cand: dict[str, list[dict]], t975: float | None = None) -> dict:
    """상황 → [모델별 행]. 차 가중 평균 차이(양수면 cand가 짧다)와 상황을 하나씩 뺀 jackknife t95."""
    sids = sorted(base)
    if sorted(cand) != sids:
        raise ValueError("두 정책의 상황이 다릅니다.")

    def gain(keys):
        b = [r for k in keys for r in base[k]]
        c = [r for k in keys for r in cand[k]]
        return weighted_mean(b) - weighted_mean(c)

    whole = gain(sids)
    n = len(sids)
    if n < 2:
        return {"gain_s": whole, "t95": [None, None], "n": n, "positive": None}
    leave = [gain([x for x in sids if x != s]) for s in sids]
    center = statistics.fmean(leave)
    se = math.sqrt((n - 1) / n * sum((x - center) ** 2 for x in leave))
    half = (T975.get(n - 1, 2.0) if t975 is None else t975) * se
    positive = sum(weighted_mean(base[s]) > weighted_mean(cand[s]) for s in sids)
    return {"gain_s": whole, "t95": [whole - half, whole + half], "n": n, "positive": positive}


def adoption(gain: dict, rules: dict) -> bool:
    return gain["t95"][0] is not None and gain["gain_s"] >= float(rules["adopt_gain_s"]) and gain["t95"][0] > 0.0


def shard_jobs(jobs: list[dict], shard: int, shards: int) -> list[dict]:
    return [j for i, j in enumerate(jobs) if i % shards == shard]


# ----------------------------------------------------------------------------- 라벨(순수 계산)
def long_point(cfg: dict, state_id: str) -> bool:
    r = cfg["rollout"]
    value = int(hashlib.sha256(f"{r['long_salt']}|{state_id}".encode()).hexdigest()[:8], 16) % 10000
    return value < round(float(r["long_fraction"]) * 10000)


def point_row(point: dict, cfg: dict) -> tuple[dict | None, str | None]:
    """지점 하나 → 표. 복원 불일치·유효 미래 부족은 빼고, 형식이 틀리면 멈춘다. 값은 (표시 지점, 녹색, 미래)."""
    if not point.get("restore_check", {}).get("ok"):
        return None, "restore_mismatch"
    x = np.asarray(point["x"], np.float32)
    if x.shape != (INPUT_SIZE,) or not np.isfinite(x).all():
        raise ValueError(f"입력이 올바르지 않습니다: {point.get('state')}")
    allowed = sorted(int(a) for a in point["allowed"])
    if len(allowed) < 2 or int(point["executed"]) not in allowed or sorted(int(k) for k in point["rollouts"]) != allowed:
        raise ValueError(f"허용 녹색·실행 녹색·굴림 후보가 맞지 않습니다: {point.get('state')}")
    futures = int(cfg["rollout"]["futures"])
    if {len(v) for v in point["rollouts"].values()} != {futures}:
        raise ValueError("후보마다 미래 수가 다르거나 설정과 다릅니다.")
    valid = pt.valid_futures(point)
    if len(valid) < int(cfg["rollout"]["min_valid_futures"]):
        return None, "insufficient_valid_futures"
    marks = list(cfg["learn"]["label_marks"])
    delay, residual = np.full((len(marks), GREENS, futures), np.nan), np.full((len(marks), GREENS, futures), np.nan)
    for green, runs in point["rollouts"].items():
        if len({r["future_seed"] for r in (runs[j] for j in valid)}) != len(valid):
            raise ValueError("미래 난수가 겹칩니다.")
        for j in valid:
            for m, mark in enumerate(marks):
                cell = runs[j]["marks"][str(mark)]
                delay[m, int(green), j], residual[m, int(green), j] = float(cell["delay"]), float(cell["residual"])
    for j in valid:
        if len({point["rollouts"][str(a)][j]["future_seed"] for a in allowed}) != 1:
            raise ValueError("후보끼리 같은 미래를 쓰지 않았습니다.")
    mask = nn.mask_of(allowed)
    return {"x": x, "allowed": mask, "base": int(point["executed"]), "state": str(point["state"]), "delay": delay, "residual": residual, "valid_futures": len(valid)}, None


def label_costs(delay: np.ndarray, residual: np.ndarray, mark_index: int, weight: float) -> np.ndarray:
    """(지점, 녹색, 미래) 천 차량·초. 허용 밖·뺀 미래는 nan."""
    return (delay[:, mark_index] + float(weight) * residual[:, mark_index]) / SCALE


def back_half_score(best: np.ndarray, predicted: np.ndarray, base: np.ndarray, back_costs: np.ndarray, tau: float) -> dict:
    """날짜 보류 예측이 문턱 이상으로 고른 바꿈의 실제 이득을 뒤 절반 미래로 잰다(천 차량·초, 결정당)."""
    rows = np.arange(len(base))
    diff = back_costs[rows, base] - back_costs[rows, best]
    realized = np.nanmean(np.where(np.isfinite(diff), diff, np.nan), axis=1)
    chosen = predicted >= tau
    return {"changes": int(chosen.sum()), "share": float(chosen.mean()), "gain_per_change": float(realized[chosen].mean()) if chosen.any() else 0.0,
            "gain_per_decision": float(realized[chosen].sum() / len(base)), "win_rate": float((realized[chosen] > 0).mean()) if chosen.any() else 0.0}


# ----------------------------------------------------------------------------- 실행 정책
class StackPolicy(nn.NNPolicy):
    """운전 신경망 + 쌓은 학생들. 실행 때 계획기가 없다. 꼬리도 같은 act로 잇는다(FeaturePlanner)."""

    def __init__(self, net, name: str, stack=()):
        super().__init__(WORLD, net, name)
        self.stack = list(stack)

    def pick(self, x: np.ndarray, allowed: list[int]) -> int:
        action, changes = choose(self.net, self.stack, x, allowed)
        self.stats["planned"] += 1
        self.stats["z5_cells"] += int((np.abs((x - self._mean) / self._std) > 5.0).sum())
        if changes:
            self.stats["student_changed"] += 1
            for a, b in changes:
                self.pairs[f"{a}->{b}"] += 1
        return action

    def act(self, env, sensed: np.ndarray, allowed: list[int]) -> int:
        core = pt.episode_core(self, self.ctx)
        core.observe(env)
        if len(allowed) == 1:
            self.stats["forced"] += 1
            return allowed[0]
        return self.pick(self.features(env, sensed, core), allowed)

    def info(self) -> dict:
        return {"kind": "stack", "students": len(self.stack), **self.stats, "pairs": dict(self.pairs)}


@functools.lru_cache(maxsize=4)
def _cached_policy(spec_text: str):
    spec = json.loads(spec_text)
    net = nn.load_model(Path(spec["net"]))[0]
    return net, [(nn.load_model(Path(d))[0], float(t)) for d, t in spec["stack"]]


def policy_spec(model_dir: Path, stack=()) -> str:
    return json.dumps({"net": str(model_dir), "stack": [[str(d), float(t)] for d, t in stack]}, sort_keys=True)


def meter_class():
    base = bp._meter_class()

    class AllMovesMeter(base):
        """이동별 지연도 남긴다(moves_from초 이상의 표시 지점만). 셈은 전체 지연과 같다."""
        moves_from = int(ROLLOUT["move_marks_from"])

        def finish(self, tripinfo: Path) -> dict:
            import xml.etree.ElementTree as ET
            arrived = {}
            if Path(tripinfo).exists():
                arrived = {t.get("id"): (float(t.get("arrival")), float(t.get("timeLoss"))) for t in ET.parse(tripinfo).getroot().iter("tripinfo")}
            at = super().finish(tripinfo)
            self.east_exits = sorted(round(arr - self.t0, 1) for v, (arr, _) in arrived.items() if arr >= 0 and (self.move.get(v) or ("", ""))[1] == EAST_EXIT)
            for h, m in sorted(self._marked.items()):
                if h < self.moves_from:
                    continue
                M = self.t0 + h
                done = {v for v, (arr, _) in arrived.items() if 0 <= arr <= M}
                total = Counter()
                for v, tl in m["tl"].items():
                    if v not in done:
                        total[self.move.get(v)] += tl - self.tl0.get(v, 0.0)
                for v, (arr, tl) in arrived.items():
                    if 0 <= arr <= M:
                        total[self.move.get(v)] += tl - self.tl0.get(v, 0.0)
                for v, dep in m["departed"].items():
                    total[self.move.get(v)] += dep - max(self.desired.get(v, self.t0), self.t0)
                for v in m["pending"]:
                    total[self.move.get(v)] += M - max(self.desired.get(v, self.t0), self.t0)
                at[h]["moves"] = {f"{k[0]}->{k[1]}": round(v, 3) for k, v in sorted(total.items(), key=str) if k}
            return at

    return AllMovesMeter


MARK_KEEP = ("delay", "depart_wait", "residual", "left_east", "left_south", "moves")


def rollout_worker(job: dict) -> dict:
    """저장 상태에서 표본 미래 하나: 첫 5초 후보 녹색, 이후 그 판을 운전한 정책(계획기 없음). 꼬리 구간도 같다."""
    import explore_rollout as er
    from focused_eval import judge_collisions
    started = time.monotonic()
    env = None
    fd, trip_name = tempfile.mkstemp(prefix="n3_trip_", suffix=".xml")
    os.close(fd)
    fd, coll_name = tempfile.mkstemp(prefix="n3_coll_", suffix=".xml")
    os.close(fd)
    trip, coll = Path(trip_name), Path(coll_name)
    try:
        w.configure()
        torch.set_num_threads(1)
        horizon = int(job["horizon_s"])
        kinds = Counter()
        net, stack = _cached_policy(job["policy_spec"])
        with nn.planner_free(None):
            with pt.platoon_futures(job["date"], float(job["origin_s"]), kinds):
                env, cons, ts, ctl, snap, injected, vehicles = bp._restore_env(job, "sampled", int(job["future_seed"]), horizon, trip, coll)
            greens = [p.state for p in ts.green_phases]
            feature = nn.FeaturePlanner(SimpleNamespace(env=env, movement_of=ctl.movement_of, ts=ts), float(snap["handover"]) + pt.CONTROL_S)
            core = pt.TM4Core().build(ctl.movement_of, greens, job["date"], float(job["origin_s"]), env, movements=feature.p0.movements)
            core.import_state(snap["tm4_state"])
            allowed = [int(a) for a in cons.allowed(env)]
            if allowed != job["allowed"]:
                raise ValueError("표본 미래를 넣은 뒤 허용 녹색이 저장 순간과 다릅니다.")
            x0 = np.asarray([*feature.sensed(env), *nn.east_cells(core, env)], np.float32)
            if np.max(np.abs(x0 - np.asarray(job["x"], np.float32))) > 1e-6:
                raise ValueError("표본 미래를 넣은 뒤 입력 120칸이 저장 순간과 다릅니다(미래가 입력에 샜거나 복원 오차).")
            if int(job["candidate"]) not in allowed:
                raise ValueError("후보가 허용 녹색이 아닙니다.")
            meter = meter_class()(env, snap["t0"], vehicles, injected, marks=tuple(job["marks"]))

            def follow(e, permitted):
                core.observe(e)
                permitted = [int(a) for a in permitted]
                if len(permitted) == 1:
                    return permitted[0]
                return choose(net, stack, np.asarray([*feature.sensed(e), *nn.east_cells(core, e)], np.float32), permitted)[0]

            _, changed = er.drive(env, cons, follow, int(job["candidate"]), snap["t0"], horizon, meter.step)
        env.close()
        env = None
        meter.finish(trip)
        missing = [h for h in job["marks"] if h not in meter.at]
        if missing:
            raise ValueError(f"표시 지점 값이 없습니다: {missing}")
        collision = judge_collisions(coll)
        return {"ok": True, "candidate": int(job["candidate"]), "future": int(job["index"]), "future_seed": int(job["future_seed"]), "leaks": len(meter.leaks), "changed_by_rules": int(changed),
                "injected": len(injected), "east_platoon": int(kinds["east_platoon"]), "east_kept": int(kinds["east_kept"]), "collisions_unexplained": len(collision["collisions_unexplained"]),
                "collisions_at_204820": int(collision["collisions_at_204820"]), "seconds": time.monotonic() - started, "east_exits": meter.east_exits,
                "marks": {str(h): {k: meter.at[h][k] for k in MARK_KEEP if k in meter.at[h]} for h in job["marks"]}}
    except BaseException as error:
        return {"ok": False, "candidate": job.get("candidate"), "future": job.get("index"), "error": f"{type(error).__name__}: {error}", "seconds": time.monotonic() - started}
    finally:
        if env is not None:
            env.close()
        trip.unlink(missing_ok=True)
        coll.unlink(missing_ok=True)


class Collector(StackPolicy):
    """본 판: 행동은 언제나 지금 정책. 표본 지점에서는 상태를 저장하고 굴림을 맡긴 뒤 기다리지 않는다. limit(점검용) = 굴릴 최대 지점 수."""

    def __init__(self, pool, cfg: dict, situation: dict, state_dir: Path, net, stack, spec: str, *, limit: int | None = None):
        super().__init__(net, f"R{ROUND}_COLLECT", stack)
        self.pool, self.cfg, self.situation, self.state_dir, self.spec, self.limit = pool, cfg, situation, Path(state_dir), spec, limit
        self.sid = ac.situation_id(situation)
        self.points: list[dict] = []
        self.rollouts: list[tuple] = []
        self.verifies: list[tuple] = []
        self.counts = Counter()

    def decide(self, snap, ctx) -> int:
        self.ctx = ctx
        env = ctx.ep.env
        allowed = [int(a) for a in ctx.allowed]
        core = pt.episode_core(self, ctx)
        core.observe(env)
        if len(allowed) == 1:
            self.stats["forced"] += 1
            return allowed[0]
        x = self.features(env, snap.observation, core)
        action = self.pick(x, allowed)
        if pt.point_sampled(self.cfg, self.sid, ctx.step):
            self.counts["hash_sampled"] += 1
            if self.limit is None or len(self.points) < self.limit:
                self._submit(snap, ctx, x, action)
        return action

    def _submit(self, snap, ctx, x: np.ndarray, action: int) -> None:
        import explore_m4probe_pending as pend
        import explore_rollout as er
        e, r = ctx.ep, self.cfg["rollout"]
        k = len(self.points)
        state_id = f"{self.sid}_p{k:03d}"
        stem = self.state_dir / state_id
        saved = er.Episode.snapshot(e, stem, snap.allowed, snap.p0_action)
        saved["pending_snapshot"] = pend.capture_pending(e.env.sumo)
        saved["tm4_state"] = self.core.export_state()
        stem.with_suffix(".pkl").write_bytes(pickle.dumps(saved))
        er.types_only_file(e.route)
        far = long_point(self.cfg, state_id)
        horizon, marks = (r["long_horizon_s"], r["long_marks"]) if far else (r["horizon_s"], r["marks"])
        base = {"stem": str(stem), "route_dir": str(e.route.parent), "sumo_seed": int(self.situation["sumo_seed"]), "allowed": [int(a) for a in snap.allowed], "horizon_s": int(horizon), "marks": list(marks),
                "x": x.tolist(), "date": str(self.situation["date"]), "origin_s": pt.origin_of(self.situation), "policy_spec": self.spec}
        self.points.append({"state": state_id, "step": int(ctx.step), "now": float(snap.now), "allowed": base["allowed"], "executed": int(action), "x": x.tolist(), "horizon_s": int(horizon)})
        for a in snap.allowed:
            for j in range(int(r["futures"])):
                job = {**base, "candidate": int(a), "index": j, "future_seed": pt.future_seed(self.cfg, state_id, j)}
                self.rollouts.append((k, int(a), j, self.pool.submit(rollout_worker, job)))
        self.verifies.append((k, self.pool.submit(bp.verify_worker, {**base, "green": int(action)})))

    def after_step(self, ctx, snap, action, executed) -> None:
        if self.points and self.points[-1]["step"] == ctx.step and "main_after" not in self.points[-1]:
            self.points[-1]["main_after"] = bp.trace_table(ctx.ep.env.sumo)

    def outstanding(self) -> int:
        return sum(not f.done() for *_, f in self.rollouts) + sum(not f.done() for _, f in self.verifies)

    def futures_list(self) -> list:
        return [f for *_, f in self.rollouts] + [f for _, f in self.verifies]

    def spent(self) -> float:
        done = [f for f in self.futures_list() if f.done() and not f.cancelled() and f.exception() is None]
        return sum(float(f.result().get("seconds") or 0.0) for f in done)

    def finish(self) -> dict:
        r = self.cfg["rollout"]
        tol = float(r["restore_check"]["tolerance"])
        cpu, errors, failures = 0.0, [], []
        east = Counter()
        for p in self.points:
            p["rollouts"] = {str(a): [None] * int(r["futures"]) for a in p["allowed"]}
        for k, a, j, future in self.rollouts:
            got = future.result()
            cpu += float(got.get("seconds") or 0.0)
            if not got["ok"]:
                failures.append(f"{self.points[k]['state']} 녹색 {a} 미래 {j}: {got['error']}")
            elif got["leaks"] or got["changed_by_rules"]:
                errors.append(f"{self.points[k]['state']} 녹색 {a} 미래 {j}: 미래 엿봄 또는 규칙 변경")
            else:
                self.points[k]["rollouts"][str(a)][j] = {key: got[key] for key in ("future_seed", "collisions_unexplained", "collisions_at_204820", "seconds", "east_exits", "marks")}
                east.update({"east_platoon": got["east_platoon"], "east_kept": got["east_kept"]})
        allowed_failures = max(3, int(float(r["max_failed_rollout_fraction"]) * len(self.rollouts)))
        if len(failures) > allowed_failures:
            errors.insert(0, f"굴림 실패 {len(failures)}/{len(self.rollouts)}개(허용 {allowed_failures}), 첫 번째: {failures[0]}")
        checked = mismatched = 0
        for k, future in self.verifies:
            got = future.result()
            cpu += float(got.get("seconds") or 0.0)
            point = self.points[k]
            main = point.pop("main_after", None)
            ok = bool(got["ok"]) and main is not None and bp.same_table(got["table"], main, tol)
            point["restore_check"] = {"ok": ok, "error": got.get("error")}
            checked += 1
            mismatched += not ok
        for p in self.points:
            p.pop("main_after", None)
            p["valid_futures"] = len(pt.valid_futures(p))
        self.cleanup()
        if errors:
            raise RuntimeError(f"수집 판 실패({len(errors)}건): {errors[0]}")
        return {"checked": checked, "mismatched": mismatched, "rollout_cpu_seconds": cpu, "rollout_failures": len(failures), "rollouts": len(self.rollouts), "east_future": dict(east)}

    def cleanup(self) -> None:
        for p in self.points:
            stem = self.state_dir / p["state"]
            stem.with_suffix(".xml").unlink(missing_ok=True)
            stem.with_suffix(".pkl").unlink(missing_ok=True)


# ----------------------------------------------------------------------------- 한 판 실행(작업자)
def free_job(job: dict) -> dict:
    """평가 한 판. 신경망 + 학생은 계획기 없는 경로, TM4는 참고용 기존 경로."""
    started = time.monotonic()
    work = Path(job["work"])
    name = f"{job['policy']}/{job['sid']}"
    try:
        w.configure()
        torch.set_num_threads(1)
        shutil.rmtree(work, ignore_errors=True)
        if job["mode"] == "tm4":
            policy = pt.TM4Policy()
            result = pt.run_episode(job["situation"], work, policy, max_seconds=nn.MAX_EPISODE_S)
            extra = {"planner_free": False}
        else:
            net = nn.load_model(Path(job["model_dir"]))[0]
            policy = StackPolicy(net, job["policy"], [(nn.load_model(Path(d))[0], float(t)) for d, t in job["stack"]])
            with nn.planner_free(policy, "NN") as guard:
                result = pt.run_episode(job["situation"], work, policy, max_seconds=nn.MAX_EPISODE_S)
            extra = {"planner_free": True, "tail": "NN", "tail_planner_calls": guard["tail_calls"]}
        if result.metrics["route_sha256"] != job["situation"]["route_sha256"]:
            raise ValueError(f"차량 파일의 SHA가 설정과 다릅니다: {result.metrics['route_sha256']}")
        ac.write_run(Path(job["target"]), result, {"policy": job["policy"], "fingerprint": job["fingerprint"], "model_identity": job.get("model_identity"),
                                                   "job_seconds": time.monotonic() - started, "information": policy.info(), **extra})
        return {"ok": True, "seconds": time.monotonic() - started, "name": name}
    except BaseException as error:
        return {"ok": False, "seconds": time.monotonic() - started, "name": name, "error": f"{type(error).__name__}: {error}"}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run_row(m: dict) -> dict:
    info = m.get("information") or {}
    return {"vehicles": int(m["vehicles"]), "total": float(m["actual_total_delay"]), "planned": int(info.get("planned", 0)), "changed": int(info.get("student_changed", 0)), "long": int(m.get("long_waits", 0)),
            "left_east": m.get("left_east"), "left_south": m.get("left_south"), "collisions": int(m.get("collisions_unexplained", 0)), "at_204820": int(m.get("collisions_at_204820", 0)), "rules": int(m.get("changed_by_rules", 0)),
            "dirty": ref.quality_errors(m), "seconds": float(m.get("job_seconds", 0.0)), "pairs": info.get("pairs", {}), "z5": int(info.get("z5_cells", 0))}


def job_rows(jobs: list[dict]) -> dict:
    """정책 → 상황 → 행. 끝난 판의 결과를 읽어 작은 요약으로 만든다(다른 런타임은 이 요약만 읽는다)."""
    with ThreadPoolExecutor(16) as pool:
        got = list(pool.map(ref.cached, jobs))
    rows: dict[str, dict] = {}
    for job, m in zip(jobs, got):
        if m is None:
            raise ValueError(f"끝나지 않은 판이 있습니다: {job['policy']}/{job['sid']}")
        rows.setdefault(job["policy"], {})[job["sid"]] = run_row(m)
    return rows


def wait_for(paths: list[Path], label: str, every_s: float = 60.0, sleep=time.sleep, stamp=ref.stamp) -> None:
    """다른 런타임이 끝낼 때까지 기다린다(Drive가 늦게 보여도 다음 확인에서 잡힌다)."""
    while True:
        missing = [p for p in paths if not Path(p).exists()]
        if not missing:
            return
        stamp(f"{label}: 다른 런타임을 기다립니다(남은 파일 {len(missing)}개).")
        sleep(every_s)


# ----------------------------------------------------------------------------- 준비: 출발 모델
def n5_models(nn1: Path, cfg: dict) -> dict[int, Path]:
    out = {}
    for seed in cfg["n5"]["seeds"]:
        folder = Path(nn1) / "models" / "r1" / f"seed_{seed}"
        if nn.model_identity(folder) != cfg["n5"]["identities"][str(seed)]:
            raise ValueError(f"N5 모델이 설정에 고정한 것과 다릅니다: {folder}")
        out[int(seed)] = folder
    return out


def low_jobs(round_: int, out: Path, work: Path, fp_: str, teacher: Path, models: dict[int, Path], cfg: dict) -> list[dict]:
    sits = [s for s in cfg["situations_low_logs"] if s["round"] == round_]
    seeds = sorted(models)
    jobs = []
    for i, s in enumerate(sits):
        driver = None if round_ == 0 else models[seeds[i % len(seeds)]]
        identity = "teacher" if driver is None else nn.model_identity(driver)
        job = ref.jobs_for(Path(out), Path(work), fp_, f"r{round_}", [s], model_identity=identity, kind="log")[0]
        job.update(world=WORLD, mode="log", teacher_dir=str(teacher), driver_dir=str(driver) if driver else None, cfg={"teachers": cfg["teachers"]})
        jobs.append(job)
    return jobs


def low_situations_of(cfg: dict, round_: int) -> list[dict]:
    return [s for s in cfg["situations_low_logs"] if s["round"] == round_]


def train_n6(nn1: Path, out: Path, record: dict, cfg: dict, models: dict[int, Path]) -> dict:
    """N5의 기록(라운드 0·1) + 새벽·심야 기록. 표준화·학습 횟수는 N5와 같고, 비용 차의 중앙값만 낮·새벽을 따로 잰다."""
    torch.set_num_threads(1)
    st = cfg["training"]
    nn1cfg = nn.config()
    day_record = s2e.folder_record(nn1)
    day, low, skipped = [], [], {}
    for k in (0, 1):
        got, skip = nn.log_rows(nn1, k, nn.named_situations(nn1cfg, "situations_aug", WORLD, k), day_record)
        day += got
        skipped[f"day_r{k}"] = skip
        got, skip = nn.log_rows(out, k, low_situations_of(cfg, k), record)
        low += got
        skipped[f"low_r{k}"] = skip
    n5_fingerprint = w.digest({"round": 1, "logs": [[r["sid"], r["log_sha256"]] for r in day]})
    for seed, folder in models.items():
        if w.read(folder / "manifest.json").get("dataset_fingerprint") != n5_fingerprint:
            raise ValueError("읽은 낮 기록이 N5를 학습한 자료와 다릅니다.")
    rows = [*day, *low]
    x = np.concatenate([r["x"] for r in rows]).astype(np.float32)
    y = np.concatenate([r["teacher"] for r in rows])
    allowed = np.concatenate([r["allowed"] for r in rows])
    if not np.isfinite(x).all() or not allowed[np.arange(len(y)), y].all():
        raise ValueError("기록의 입력이 유한하지 않거나 선생님 녹색이 허용 밖입니다.")
    weights_day, median_day = nn.margin_weights(np.concatenate([r["costs"] for r in day]), st["margin_clip"])
    weights_low, median_low = nn.margin_weights(np.concatenate([r["costs"] for r in low]), st["margin_clip"])
    weights = np.concatenate([weights_day, weights_low])
    fingerprint = w.digest({"n5": n5_fingerprint, "low": [[r["sid"], r["log_sha256"]] for r in low]})
    summary = {"decisions_day": int(sum(len(r["x"]) for r in day)), "decisions_low": int(sum(len(r["x"]) for r in low)), "skipped": skipped, "margin_median": {"day": median_day, "low": median_low}, "dataset_fingerprint": fingerprint, "models": {}}
    for seed in st["seeds"]:
        folder = Path(out) / "models" / "N6" / f"seed_{seed}"
        if (folder / "manifest.json").exists():
            if w.read(folder / "manifest.json").get("dataset_fingerprint") != fingerprint:
                raise ValueError(f"이미 있는 N6가 다른 기록으로 학습됐습니다: {folder}")
            summary["models"][str(seed)] = {"model_sha256": nn.model_identity(folder), "reused": True}
            continue
        started = time.monotonic()
        net = nn.fit_green(x, y, allowed, seed, int(st["epochs_imitation"]), st, weights=weights)
        info = {"agreement_in_sample": float((nn.greens_of(net, x, allowed) == y).mean()), "decisions": int(len(x)), "seconds": time.monotonic() - started}
        sha = nn.save_model(folder, net, _meta(nn.KIND_GREEN, "N6", seed, fingerprint, st, training=info))
        summary["models"][str(seed)] = {"model_sha256": sha, "reused": False, **info}
    s2e._write(Path(out) / "models" / "N6" / "training.json", summary)
    ref.stamp(f"N6 학습: 낮 결정 {summary['decisions_day']}, 새벽·심야 결정 {summary['decisions_low']}, 제외 {skipped}")
    return summary


def _meta(kind: str, candidate: str, seed: int, dataset: str, st: dict, **more) -> dict:
    return {"kind": kind, "world": WORLD, "candidate": candidate, "seed": int(seed), "input_size": INPUT_SIZE, "network": list(st["network"]), "dataset_fingerprint": dataset,
            "config_sha256": w.file_hash(CONFIG), "source_sha256": w.file_hash(ROOT / "src/explore_nn3.py"), **more}


def nn_jobs(folder: Path, work: Path, fp_: str, name: str, models: dict[int, Path], sits: list[dict], *, stacks: dict | None = None) -> list[dict]:
    """모델마다 '이름_m번호' 정책. stacks = 난수 → [(학생 폴더, 문턱)]."""
    jobs = []
    for i, (seed, model) in enumerate(sorted(models.items())):
        stack = (stacks or {}).get(seed, [])
        identity = "|".join([nn.model_identity(model), *(f"{nn.model_identity(Path(d))}|tau={float(t):.6f}" for d, t in stack)])
        for job in ref.jobs_for(Path(folder), Path(work), fp_, f"{name}_m{i}", sits, model_identity=identity, kind="nn"):
            job.update(mode="free", model_dir=str(model), stack=[[str(d), float(t)] for d, t in stack])
            jobs.append(job)
    return jobs


def _by_situation(rows: dict, name: str, sids: list[str]) -> dict[str, list[dict]]:
    """'이름_m번호' 정책들의 행을 상황별로 묶는다."""
    keys = sorted(k for k in rows if k.rsplit("_m", 1)[0] == name)
    if not keys:
        raise ValueError(f"결과가 없습니다: {name}")
    return {s: [rows[k][s] for k in keys] for s in sids}


def _stored_n5_day(nn1: Path, cfg: dict, models: dict[int, Path]) -> dict:
    """NN1 검증의 N5(라운드 1) 저장 결과."""
    folder = Path(nn1) / "valid"
    record = s2e.folder_record(folder)
    sits = day_validation(cfg)
    rows = {}
    for i, (seed, model) in enumerate(sorted(models.items())):
        rows[f"N5_m{i}"] = {ac.situation_id(s): run_row(_required(ac.read_run(folder / "runs" / f"N5_r1_m{i}" / ac.situation_id(s), record["fingerprint"], nn.model_identity(model)), s)) for s in sits}
    return rows


def _required(m: dict | None, s: dict) -> dict:
    if m is None:
        raise ValueError(f"NN1 검증 결과가 없습니다: {ac.situation_id(s)}")
    return m


def prep(nn1: Path, teacher: Path, out: Path, work: Path, workers: int, cfg: dict) -> int:
    """새벽·심야 기록 → N6 학습 → 검증 → 출발 모델 고르기. 끊겨도 다시 실행하면 이어서 돈다."""
    nn1, teacher, out, work = Path(nn1), Path(teacher), Path(out), Path(work)
    fp_ = ref.register(out, extra_files=SOURCES)
    record = s2e.folder_record(out)
    nn.load_teacher(WORLD, teacher, {"teachers": cfg["teachers"]})
    n5 = n5_models(nn1, cfg)
    nn.prepare_caches(WORLD)
    ledger = ref.Ledger(out, "prep")
    for k in (0, 1):
        status = ref.run_jobs(low_jobs(k, out, work / f"r{k}", fp_, teacher, n5, cfg), max(1, workers), ledger, worker=nn.run_job)
        if status["failed"]:
            ref.stamp(f"새벽·심야 기록 라운드 {k} 실패 {len(status['failed'])}판. 다시 실행하면 끝난 판은 건너뜁니다.")
            return 1
    training = train_n6(nn1, out, record, cfg, n5)
    n6 = {int(s): out / "models" / "N6" / f"seed_{s}" for s in cfg["training"]["seeds"]}
    valid = out / "valid"
    fp_valid = ref.register(valid, extra_files=SOURCES)
    day, low = day_validation(cfg), cfg["situations_low_validation"]
    jobs = [*nn_jobs(valid, work / "valid", fp_valid, "N6", n6, day), *nn_jobs(valid, work / "valid", fp_valid, "N6", n6, low), *nn_jobs(valid, work / "valid", fp_valid, "N5", n5, low)]
    for job in ref.jobs_for(valid, work / "valid", fp_valid, "TM4", low, model_identity="TM4", kind="tm4"):
        job.update(mode="tm4")
        jobs.append(job)
    status = ref.run_jobs(jobs, max(1, workers), ledger, worker=free_job)
    if status["failed"]:
        ref.stamp(f"검증 실패 {len(status['failed'])}판. 다시 실행하면 끝난 판은 건너뜁니다.")
        return 1
    rows = job_rows(jobs)
    day_ids, low_ids = [ac.situation_id(s) for s in day], [ac.situation_id(s) for s in low]
    n5_rows = {**{k: dict(v) for k, v in rows.items() if k.startswith("N5_")}}
    for k, v in _stored_n5_day(nn1, cfg, n5).items():
        n5_rows.setdefault(k, {}).update(v)
    n6_rows = {k: v for k, v in rows.items() if k.startswith("N6_")}
    day_gain = paired_gain(_by_situation(n5_rows, "N5", day_ids), _by_situation(n6_rows, "N6", day_ids))
    low_gain = paired_gain(_by_situation(n5_rows, "N5", low_ids), _by_situation(n6_rows, "N6", low_ids))
    decision = n6_decision(low_gain["gain_s"], day_gain["gain_s"], cfg["rules"])
    chosen = n6 if decision["start"] == "N6" else n5
    start_dir = out / "start"
    identities = {}
    for seed, folder in chosen.items():
        target = start_dir / f"seed_{seed}"
        if not (target / "manifest.json").exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
            shutil.rmtree(tmp, ignore_errors=True)
            shutil.copytree(folder, tmp)
            os.replace(tmp, target)
        identities[str(seed)] = nn.model_identity(target)
        if identities[str(seed)] != nn.model_identity(folder):
            raise ValueError(f"출발 모델 사본의 해시가 다릅니다: {target}")
    means = {"N5_day": weighted_mean([r for v in _by_situation(n5_rows, "N5", day_ids).values() for r in v]), "N6_day": weighted_mean([r for v in _by_situation(n6_rows, "N6", day_ids).values() for r in v]),
             "N5_low": weighted_mean([r for v in _by_situation(n5_rows, "N5", low_ids).values() for r in v]), "N6_low": weighted_mean([r for v in _by_situation(n6_rows, "N6", low_ids).values() for r in v]),
             "TM4_low": weighted_mean(list(rows["TM4"].values()))}
    s2e._write(valid / "rows.json", {"N5": n5_rows, "N6": n6_rows, "TM4": rows["TM4"]})
    s2e._write(out / "start.json", {"start": decision["start"], "identities": identities, "means": means, "day_gain": day_gain, "low_gain": low_gain, "training": {k: training[k] for k in ("decisions_day", "decisions_low", "margin_median", "dataset_fingerprint")},
                                    "note": "gain = N5 − N6(양수면 N6가 짧음). 흉내 값이다."})
    ref.stamp(f"출발 모델: {decision['start']} | 낮 {means['N5_day']:.2f} → {means['N6_day']:.2f}, 새벽 {means['N5_low']:.2f} → {means['N6_low']:.2f}, TM4 새벽 {means['TM4_low']:.2f}")
    return 0


def start_models(prep_dir: Path, cfg: dict) -> dict[int, Path]:
    saved = w.read(Path(prep_dir) / "start.json")
    out = {}
    for seed in cfg["training"]["seeds"]:
        folder = Path(prep_dir) / "start" / f"seed_{seed}"
        if nn.model_identity(folder) != saved["identities"][str(seed)]:
            raise ValueError(f"출발 모델 사본이 기록과 다릅니다: {folder}")
        out[int(seed)] = folder
    return out


# ----------------------------------------------------------------------------- 라운드: 자료
def write_status(path: Path, **values) -> None:
    w.write(path, {**values, "updated_unix": time.time()})


def finalize(out: Path, fp_: str, identity: str, item: dict) -> dict:
    policy, result = item["policy"], item["result"]
    rc = policy.finish()
    target = Path(out) / "runs" / "DATA" / item["sid"]
    sha = pt.write_points(target, policy.points)
    result.metrics.update({"nn3_round": ROUND, "nn3_driver_seed": item["seed"], "nn3_points": len(policy.points), "nn3_counts": dict(policy.counts), "nn3_restore_check": {k: rc[k] for k in ("checked", "mismatched")},
                           "nn3_rollouts": rc["rollouts"], "nn3_rollout_failures": rc["rollout_failures"], "nn3_rollout_cpu_seconds": rc["rollout_cpu_seconds"], "nn3_main_seconds": item["main_seconds"], "nn3_east_future": rc["east_future"], "information": policy.info()})
    ac.write_run(target, result, {"policy": "DATA", "fingerprint": fp_, "model_identity": identity, "points_sha256": sha, "role": "data"})
    return {**rc, "cpu_seconds": rc["rollout_cpu_seconds"] + item["main_seconds"], "points": len(policy.points)}


def collect(models: dict[int, Path], cfg: dict, out: Path, work: Path, shard: int, shards: int, workers: int, *, executor=None) -> int:
    out, work = Path(out), Path(work)
    fp_ = ref.register(out, extra_files=SOURCES)
    r = cfg["rollout"]
    seeds = sorted(models)
    ledger = ref.Ledger(out, f"collect_shard{shard}")
    status_path = out / "status" / f"collect_shard{shard}.json"
    rc_cfg = r["restore_check"]
    mine = [(i, s) for i, s in enumerate(cfg["situations_round"]) if i % shards == shard]
    checked = mismatched = 0
    failed, inflight = [], []
    stopped = False

    def status():
        done = sum((out / "runs" / "DATA" / ac.situation_id(s) / "cache.json").exists() for _, s in mine)
        write_status(status_path, done=done, total=len(mine), failed=failed, restore_checked=checked, restore_mismatched=mismatched, inflight=[i["sid"] for i in inflight], stopped_by_restore_rule=stopped)

    def harvest():
        nonlocal checked, mismatched
        for item in [i for i in inflight if i["policy"].outstanding() == 0]:
            inflight.remove(item)
            try:
                got = finalize(out, fp_, item["identity"], item)
                checked += got["checked"]
                mismatched += got["mismatched"]
                ledger.add(got["cpu_seconds"])
                ref.stamp(f"자료 {item['sid']}: 지연 {item['result'].metrics['mean_delay_s']:.3f}초, 지점 {got['points']}개, 복원 불일치 {got['mismatched']}/{got['checked']}, 굴림 실패 {got['rollout_failures']}, {got['cpu_seconds'] / 3600:.2f} CPU-시간")
            except BaseException as error:
                failed.append({"situation": item["sid"], "error": f"{type(error).__name__}: {error}"})
                ref.stamp(f"자료 실패 {item['sid']}: {failed[-1]['error']}")
                ledger.add(item["main_seconds"] + item["policy"].spent())
                if isinstance(error, KeyboardInterrupt):
                    raise
            finally:
                shutil.rmtree(work / item["sid"], ignore_errors=True)
            status()

    def backlog() -> int:
        return sum(i["policy"].outstanding() for i in inflight)

    def wait_some():
        pending = [f for i in inflight for f in i["policy"].futures_list() if not f.done()]
        if pending:
            wait(pending, timeout=30, return_when=FIRST_COMPLETED)

    pool = executor or pt.rollout_pool(workers)
    with pool:
        for i, s in mine:
            sid = ac.situation_id(s)
            seed = seeds[i % len(seeds)]
            model = models[seed]
            identity = nn.model_identity(model)
            target = out / "runs" / "DATA" / sid
            if ac.read_run(target, fp_, identity) is not None:
                m = w.read(target / "metrics.json")
                pt.read_points(target)
                checked += m["nn3_restore_check"]["checked"]
                mismatched += m["nn3_restore_check"]["mismatched"]
                continue
            while backlog() > int(r["max_backlog_jobs"]):
                wait_some()
                harvest()
            began = time.monotonic()
            shutil.rmtree(work / sid, ignore_errors=True)
            net = nn.load_model(model)[0]
            policy = Collector(pool, cfg, s, work / sid / "states", net, [], policy_spec(model))
            try:
                with nn.planner_free(policy, "NN"):
                    result = pt.run_episode(s, work / sid / "episode", policy, max_seconds=float(r["main_episode_max_seconds"]), on_step=policy.after_step)
                if result.metrics["route_sha256"] != s["route_sha256"]:
                    raise ValueError("차량 파일 SHA가 설정과 다릅니다.")
                dirty = ref.quality_errors(result.metrics)
                if dirty:
                    raise RuntimeError(f"수집 본 판 품질 오류: {dirty}")
                inflight.append({"situation": s, "sid": sid, "seed": seed, "identity": identity, "policy": policy, "result": result, "main_seconds": time.monotonic() - began})
            except BaseException as error:
                failed.append({"situation": sid, "error": f"{type(error).__name__}: {error}"})
                ref.stamp(f"본 판 실패 {sid}: {failed[-1]['error']}")
                for future in policy.futures_list():
                    future.cancel()
                wait([f for f in policy.futures_list() if not f.cancelled()])
                policy.cleanup()
                shutil.rmtree(work / sid, ignore_errors=True)
                ledger.add(time.monotonic() - began + policy.spent())
                if isinstance(error, KeyboardInterrupt):
                    raise
            harvest()
            status()
            total_checked, total_mismatched = pt.restore_totals(out)
            rate = total_mismatched / total_checked if total_checked else 0.0
            if total_checked >= int(rc_cfg["min_points"]) and rate > float(rc_cfg["max_mismatch_rate"]):
                stopped = True
                ref.stamp(f"모든 조각 합 복원 불일치 {rate:.1%}({total_mismatched}/{total_checked})가 기준을 넘어 새 판을 시작하지 않습니다.")
                break
        while inflight:
            wait_some()
            harvest()
    status()
    if stopped:
        return 3
    return 1 if failed else 0


# ----------------------------------------------------------------------------- 라운드: 학생
def load_points(data: Path, cfg: dict) -> dict:
    """끝난 자료 전체를 표로 읽는다. 지점이 왜 빠졌는지 센다."""
    data = Path(data)
    record = s2e.folder_record(data)
    rows, skipped, runs, moves = [], Counter(), [], Counter()
    far = {"delay": {}, "residual": {}}
    for s in cfg["situations_round"]:
        sid = ac.situation_id(s)
        target = data / "runs" / "DATA" / sid
        m = w.read(target / "metrics.json")
        if m.get("fingerprint") != record["fingerprint"] or m.get("route_sha256") != s["route_sha256"] or m.get("nn3_round") != ROUND:
            raise ValueError(f"자료의 지문·차량 파일이 설정과 다릅니다: {sid}")
        dirty = ref.quality_errors(m)
        if dirty:
            raise ValueError(f"자료 본 판 품질 오류 {sid}: {dirty}")
        runs.append({"sid": sid, "band": s["band"], "driver": m["nn3_driver_seed"], "points": m["nn3_points"], "cpu_seconds": m["nn3_rollout_cpu_seconds"], "main_seconds": m["nn3_main_seconds"], "points_sha256": m["points_sha256"]})
        if m.get("collisions_unexplained", 0) or m.get("collisions_at_204820", 0):
            skipped["collision_runs"] += 1
            continue
        for point in pt.read_points(target):
            row, reason = point_row(point, cfg)
            if row is None:
                skipped[reason] += 1
                continue
            rows.append({**row, "date": s["date"], "sid": sid, "band": s["band"], "driver": m["nn3_driver_seed"]})
            if int(point.get("horizon_s", 0)) == int(cfg["rollout"]["long_horizon_s"]):
                far["delay"][row["state"]] = {g: [None if r is None else float(r["marks"]["900"]["delay"]) for r in v] for g, v in point["rollouts"].items()}
                far["residual"][row["state"]] = {g: [None if r is None else float(r["marks"]["900"]["residual"]) for r in v] for g, v in point["rollouts"].items()}
    if not rows:
        raise ValueError("쓸 지점이 없습니다.")
    return {"x": np.stack([r["x"] for r in rows]).astype(np.float32), "allowed": np.stack([r["allowed"] for r in rows]), "base": np.asarray([r["base"] for r in rows], np.int64), "dates": np.asarray([r["date"] for r in rows]),
            "delay": np.stack([r["delay"] for r in rows]), "residual": np.stack([r["residual"] for r in rows]), "states": [r["state"] for r in rows], "bands": np.asarray([r["band"] for r in rows]), "drivers": np.asarray([r["driver"] for r in rows]),
            "far": far, "skipped": dict(skipped), "runs": runs, "fingerprint": w.digest({"runs": [[r["sid"], r["points_sha256"]] for r in runs]})}


def _student_task(args: tuple) -> dict:
    """라벨 하나 × 난수 하나: 날짜 보류 예측(앞 절반 미래) → 문턱·뒤 절반 채점 → 모든 날짜·미래로 배포 학생."""
    d, mark_index, mark, weight, seed, folder, st, shares = args
    torch.set_num_threads(1)
    started = time.monotonic()
    x, base, allowed, dates = d["x"], d["base"], d["allowed"], d["dates"]
    costs = label_costs(d["delay"], d["residual"], mark_index, weight)
    front, back = nn.half_costs(costs)
    gains_f, se_f = nn.gains_against(front, base, allowed)
    predicted, best = np.zeros(len(x)), np.zeros(len(x), np.int64)
    for date in sorted(set(dates.tolist())):
        held = dates == date
        fold = nn.fit_gain(x[~held], base[~held], gains_f[~held], se_f[~held], allowed[~held], seed, int(st["epochs_student"]), st)
        best[held], predicted[held] = nn.predicted_best(fold, x[held], base[held], allowed[held])
    taus = {f"{s:.2f}": float(np.quantile(predicted, 1.0 - float(s))) for s in shares}
    scores = {k: back_half_score(best, predicted, base, back, t) for k, t in taus.items()}
    gains, se = nn.gains_against(costs, base, allowed)
    net = nn.fit_gain(x, base, gains, se, allowed, seed, int(st["epochs_student"]), st)
    full = nn.predicted_best(net, x, base, allowed)[1]
    info = {"mark": int(mark), "weight": float(weight), "taus": taus, "scores": scores, "share_in_sample": {k: float((full >= t).mean()) for k, t in taus.items()}, "points": int(len(x)), "seconds": time.monotonic() - started}
    meta = _meta(nn.KIND_GAIN, f"h{int(mark)}w{int(weight)}", seed, d["fingerprint"], st, tau=taus[f"{0.10:.2f}"], label_mark=int(mark), label_end_weight=float(weight), training=info)
    sha = nn.save_model(Path(folder), net, meta)
    return {"seed": int(seed), "label": f"h{int(mark)}w{int(weight)}", "model_sha256": sha, **info}


def learn(data: Path, out: Path, cfg: dict, workers: int) -> dict:
    """모든 라벨·난수의 학생을 만든다. 이미 있으면 같은 자료인지 보고 건너뛴다."""
    d = load_points(data, cfg)
    st, le = cfg["training"], cfg["learn"]
    out = Path(out)
    tasks, done = [], []
    payload = {k: d[k] for k in ("x", "base", "allowed", "dates", "delay", "residual", "fingerprint")}
    for m, mark in enumerate(le["label_marks"]):
        for weight in le["label_weights"]:
            for seed in st["seeds"]:
                folder = out / f"h{int(mark)}w{int(weight)}" / f"seed_{seed}"
                if (folder / "manifest.json").exists():
                    manifest = w.read(folder / "manifest.json")
                    if manifest.get("dataset_fingerprint") != d["fingerprint"]:
                        raise ValueError(f"이미 있는 학생이 다른 자료로 학습됐습니다: {folder}")
                    done.append({"seed": int(seed), "label": f"h{int(mark)}w{int(weight)}", "model_sha256": nn.model_identity(folder), **manifest["training"]})
                else:
                    tasks.append((payload, m, mark, weight, seed, str(folder), st, le["shares"]))
    if tasks:
        with ProcessPoolExecutor(max_workers=max(1, workers), mp_context=multiprocessing.get_context("spawn"), initializer=ref._worker_init) as pool:
            done += list(pool.map(_student_task, tasks))
    summary = {"round": ROUND, "points": int(len(d["x"])), "skipped": d["skipped"], "runs": d["runs"], "dataset_fingerprint": d["fingerprint"], "students": sorted(done, key=lambda r: (r["label"], r["seed"])), "agreement": label_agreement(d, cfg)}
    s2e._write(out / "done.json", summary)
    ref.stamp(f"학생 학습 끝: 지점 {summary['points']}, 제외 {d['skipped']}, 학생 {len(done)}개")
    return summary


def label_agreement(d: dict, cfg: dict) -> dict:
    """굴림 길이별 이득의 일치(결과와 무관한 자료 점검): 지점·녹색별 평균 이득을 긴 길이(600초·900초)와 견준다."""
    le = cfg["learn"]
    rows = np.arange(len(d["base"]))

    def gains_at(m: int, weight: float) -> np.ndarray:
        costs = label_costs(d["delay"], d["residual"], m, weight)
        return nn.gains_against(costs, d["base"], d["allowed"])[0]

    reference = gains_at(len(le["label_marks"]) - 1, 25.0)
    mask = d["allowed"].copy()
    mask[rows, d["base"]] = False
    out = {}
    for m, mark in enumerate(le["label_marks"]):
        for weight in le["label_weights"]:
            g = gains_at(m, weight)
            out[f"h{int(mark)}w{int(weight)}"] = {"corr_with_h600w25": float(np.corrcoef(g[mask], reference[mask])[0, 1]), "sign_agreement": float(((g[mask] > 0) == (reference[mask] > 0)).mean()), "mean_abs_gain": float(np.abs(g[mask]).mean())}
    states = {s: i for i, s in enumerate(d["states"])}
    far_rows = []
    for state, by_green in d["far"]["delay"].items():
        i = states[state]
        base = str(int(d["base"][i]))
        for green, values in by_green.items():
            if green == base:
                continue
            pairs = [(b, v, d["far"]["residual"][state][base][j], d["far"]["residual"][state][green][j]) for j, (b, v) in enumerate(zip(by_green[base], values)) if b is not None and v is not None]
            if pairs:
                far_rows.append((reference[i, int(green)], statistics.fmean((a - b + 25.0 * (c - e)) / SCALE for a, b, c, e in pairs)))
    if len(far_rows) > 2:
        a, b = np.asarray(far_rows).T
        out["h600w25_vs_h900w25"] = {"pairs": len(far_rows), "corr": float(np.corrcoef(a, b)[0, 1]), "sign_agreement": float(((a > 0) == (b > 0)).mean())}
    return out


def student_stacks(students: Path, name: str, seeds: list[int]) -> dict[int, list]:
    """후보 이름 → 난수별 [(학생 폴더, 문턱)]."""
    h, weight, share = parse_candidate(name)
    out = {}
    for seed in seeds:
        folder = Path(students) / f"h{h}w{int(weight)}" / f"seed_{seed}"
        tau = w.read(folder / "manifest.json")["training"]["taus"][f"{share:.2f}"]
        out[int(seed)] = [[str(folder), float(tau)]]
    return out


# ----------------------------------------------------------------------------- 라운드: 고르기
def _stage_status(out: Path, stage: str, shard: int) -> Path:
    return Path(out) / "status" / f"{stage}_shard{shard}.json"


def run_stage(stage: str, jobs: list[dict], out: Path, shard: int, shards: int, workers: int, ledger) -> bool:
    mine = shard_jobs(jobs, shard, shards)
    path = _stage_status(out, stage, shard)
    if path.exists() and w.read(path).get("jobs") == len(mine):
        return True
    status = ref.run_jobs(mine, max(1, workers), ledger, worker=free_job)
    if status["failed"]:
        ref.stamp(f"{stage}: 실패 {len(status['failed'])}판. 다시 실행하면 끝난 판은 건너뜁니다.")
        return False
    write_status(path, jobs=len(mine), rows=job_rows(mine))
    return True


def stage_rows(out: Path, stage: str, shards: int) -> dict:
    """모든 런타임의 요약을 합친다."""
    rows: dict[str, dict] = {}
    for shard in range(shards):
        for policy, row in w.read(_stage_status(out, stage, shard))["rows"].items():
            rows.setdefault(policy, {}).update(row)
    return rows


def candidate_means(rows: dict, names: list[str], sids: list[str]) -> dict[str, float]:
    return {n: weighted_mean([r for v in _by_situation(rows, n, sids).values() for r in v]) for n in names}


def selection(out: Path, shards: int, cfg: dict) -> dict:
    """거르기·문턱 결과에서 문턱 후보와 최종 후보를 정한다. 런타임마다 같은 답이 나온다."""
    sids = [ac.situation_id(s) for s in day_validation(cfg)]
    screen = stage_rows(out, "screen", shards)
    names = screen_candidates(cfg)
    means = candidate_means(screen, names, sids)
    answer = {"screen_means": means, "base_day": candidate_means(screen, [BASE], sids)[BASE], "thresholds": threshold_candidates(means, cfg)}
    if all(_stage_status(out, "thresholds", s).exists() for s in range(shards)):
        more = stage_rows(out, "thresholds", shards)
        extra = candidate_means(more, answer["thresholds"], sids)
        answer["threshold_means"] = extra
        answer["finalists"] = finalists({**means, **extra}, cfg)
    return answer


def run_round(prep_dir: Path, out: Path, work: Path, shard: int, shards: int, workers: int, cfg: dict, *, every_s: float = 60.0) -> int:
    """자료 → 학생 → 거르기 → 문턱 → 확인. 런타임 사이는 Drive의 상태 파일로 맞춘다. 다시 실행하면 끝난 부분은 건너뛴다."""
    prep_dir, out, work = Path(prep_dir), Path(out), Path(work)
    if not (prep_dir / "start.json").exists():
        ref.stamp("출발 모델이 아직 없습니다. NN3_P_prep을 먼저 끝내세요.")
        return 2
    models = start_models(prep_dir, cfg)
    seeds = sorted(models)
    data = out / "data"
    nn.prepare_caches(WORLD)
    code = collect(models, cfg, data, work / "data", shard, shards, workers)
    if code:
        return code
    write_status(_stage_status(out, "data", shard), done=True)
    wait_for([_stage_status(out, "data", s) for s in range(shards)], "자료", every_s)
    students = out / "students"
    if shard == 0:
        learn(data, students, cfg, workers)
    wait_for([students / "done.json"], "학생", every_s)
    evaluation = out / "eval"
    fp_ = ref.register(evaluation, extra_files=SOURCES)
    ledger = ref.Ledger(evaluation, f"shard{shard}")
    day, low, final = day_validation(cfg), cfg["situations_low_validation"], cfg["situations_final"]

    def jobs_of(names: list[str], sits: list[dict]) -> list[dict]:
        rows = []
        for name in names:
            stacks = None if name == BASE else student_stacks(students, name, seeds)
            rows += nn_jobs(evaluation, work / "eval", fp_, name, models, sits, stacks=stacks)
        return rows

    if not run_stage("screen", jobs_of([BASE, *screen_candidates(cfg)], day), out, shard, shards, workers, ledger):
        return 1
    wait_for([_stage_status(out, "screen", s) for s in range(shards)], "거르기", every_s)
    chosen = selection(out, shards, cfg)
    ref.stamp(f"거르기 끝. 문턱 후보: {chosen['thresholds']}")
    if not run_stage("thresholds", jobs_of(chosen["thresholds"], day), out, shard, shards, workers, ledger):
        return 1
    wait_for([_stage_status(out, "thresholds", s) for s in range(shards)], "문턱", every_s)
    chosen = selection(out, shards, cfg)
    ref.stamp(f"문턱 끝. 확인 후보: {chosen['finalists']}")
    if not run_stage("final", jobs_of([BASE, *chosen["finalists"]], [*final, *low]), out, shard, shards, workers, ledger):
        return 1
    ref.stamp("이 런타임의 확인 끝. 모든 런타임이 끝나면 NN3_P_R1_judge를 실행하세요.")
    return 0


# ----------------------------------------------------------------------------- 판정
def read_stage_runs(evaluation: Path, names: list[str], sits: list[dict], models: int = 3) -> dict:
    """지표 파일에서 직접 읽는다(상태 요약과 따로 계산해 맞는지 본다)."""
    evaluation = Path(evaluation)
    fingerprint = w.read(evaluation / "experiment.json")["fingerprint"]
    keys = [(f"{n}_m{i}", ac.situation_id(s)) for n in names for i in range(models) for s in sits]

    def one(key):
        path = evaluation / "runs" / key[0] / key[1] / "metrics.json"
        if not path.exists():
            return key, None
        m = w.read(path)
        if m.get("fingerprint") != fingerprint:
            raise ValueError(f"결과의 지문이 폴더와 다릅니다: {path}")
        return key, run_row(m)

    rows: dict[str, dict] = {}
    with ThreadPoolExecutor(16) as pool:
        for (policy, sid), row in pool.map(one, keys):
            if row is None:
                raise FileNotFoundError(f"{policy}/{sid}")
            rows.setdefault(policy, {})[sid] = row
    return rows


def policy_summary(by_sit: dict[str, list[dict]]) -> dict:
    rows = [r for v in by_sit.values() for r in v]
    planned = sum(r["planned"] for r in rows)
    pairs = Counter()
    for r in rows:
        pairs.update(r["pairs"])
    models = len(next(iter(by_sit.values())))
    east = [r["left_east"] for r in rows if r["left_east"] is not None]
    south = [r["left_south"] for r in rows if r["left_south"] is not None]
    return {"mean_delay_s": weighted_mean(rows), "per_model_s": [weighted_mean([v[i] for v in by_sit.values()]) for i in range(models)], "runs": len(rows), "change_share": sum(r["changed"] for r in rows) / max(planned, 1), "decisions_per_run": planned / max(len(rows), 1),
            "long_per_run": sum(r["long"] for r in rows) / len(rows), "left_east_s": statistics.fmean(east) if east else None, "left_south_s": statistics.fmean(south) if south else None, "collisions_unexplained": sum(r["collisions"] for r in rows), "collisions_at_204820": sum(r["at_204820"] for r in rows),
            "rules_changed": sum(r["rules"] for r in rows), "quality_errors": sum(len(r["dirty"]) for r in rows), "pairs": dict(pairs.most_common(6)), "seconds_per_run": statistics.fmean(r["seconds"] for r in rows)}


def collect_report(root: Path, cfg: dict) -> dict:
    root = Path(root)
    start = w.read(root / "prep" / "start.json")
    done = root / "r1" / "students" / "done.json"
    report = {"experiment": EXPERIMENT, "start": start}
    if not done.exists():
        return {**report, "status": "prep_only"}
    summary = w.read(done)
    report["students"] = {"points": summary["points"], "skipped": summary["skipped"], "agreement": summary["agreement"], "cpu_hours_rollouts": sum(r["cpu_seconds"] for r in summary["runs"]) / 3600, "runs": len(summary["runs"]),
                          "scores": {s["label"]: {} for s in summary["students"]}}
    for s in summary["students"]:
        report["students"]["scores"][s["label"]][str(s["seed"])] = s["scores"]
    evaluation = root / "r1" / "eval"
    day, low, final = day_validation(cfg), cfg["situations_low_validation"], cfg["situations_final"]
    day_ids, low_ids, final_ids = [ac.situation_id(s) for s in day], [ac.situation_id(s) for s in low], [ac.situation_id(s) for s in final]
    try:
        screen = read_stage_runs(evaluation, [BASE, *screen_candidates(cfg)], day)
        means = candidate_means(screen, screen_candidates(cfg), day_ids)
        names = threshold_candidates(means, cfg)
        more = read_stage_runs(evaluation, names, day)
        extra = candidate_means(more, names, day_ids)
        best = finalists({**means, **extra}, cfg)
        last = read_stage_runs(evaluation, [BASE, *best], [*final, *low])
    except FileNotFoundError as error:
        return {**report, "status": "incomplete", "missing": str(error)}
    base_day = _by_situation(screen, BASE, day_ids)
    combined = {**screen, **more}
    report["screen"] = {n: {"mean_delay_s": v, "gain_vs_base": paired_gain(base_day, _by_situation(combined, n, day_ids)), "summary": policy_summary(_by_situation(combined, n, day_ids))} for n, v in sorted({**means, **extra}.items(), key=lambda kv: kv[1])}
    report["base_day_s"] = candidate_means(screen, [BASE], day_ids)[BASE]
    report["finalists"] = best
    base_final = _by_situation(last, BASE, final_ids)
    base_low = _by_situation(last, BASE, low_ids)
    report["final"] = {"BASE": policy_summary(base_final), "low_BASE": policy_summary(base_low)}
    gains = {}
    for name in best:
        by_final, by_low = _by_situation(last, name, final_ids), _by_situation(last, name, low_ids)
        gains[name] = {"gain": paired_gain(base_final, by_final), "low_gain": paired_gain(base_low, by_low), "summary": policy_summary(by_final),
                       "per_date": part_gains(base_final, by_final, [(d, lambda k, x=d: k.startswith(x)) for d in VALID_DATES]),
                       "per_model": [weighted_mean([v[i] for v in base_final.values()]) - weighted_mean([v[i] for v in by_final.values()]) for i in range(3)],
                       "per_period": part_gains(base_final, by_final, [(p, lambda k, x=p: f"_{x}_" in k) for p in FINAL_PERIODS])}
    report["final"]["candidates"] = gains
    winner = max(best, key=lambda n: gains[n]["gain"]["gain_s"])
    report["reading"] = {"winner": winner, "gain": gains[winner]["gain"], "adopted": adoption(gains[winner]["gain"], cfg["rules"])}
    report["agreement_one_hour"] = surrogate_agreement(report["screen"], report["students"]["scores"])
    return {**report, "status": "done"}


def part_gains(base: dict, cand: dict, parts: list[tuple]) -> dict:
    """부분(날짜·시간대)별 차 가중 차이. 상황이 없는 부분은 뺀다."""
    out = {}
    for name, keep in parts:
        keys = [k for k in base if keep(k)]
        if keys:
            out[name] = paired_gain({k: base[k] for k in keys}, {k: cand[k] for k in keys})["gain_s"]
    return out


def surrogate_agreement(screen: dict, scores: dict) -> dict:
    """라벨별 뒤 절반 이득(결정당)과 한 시간 이득의 순위 일치(거르기 10개, 비율 10%)."""
    a, b = [], []
    for name, row in screen.items():
        h, weight, share = parse_candidate(name)
        if abs(share - 0.10) > 1e-9:
            continue
        label = f"h{h}w{int(weight)}"
        a.append(statistics.fmean(s[f"{share:.2f}"]["gain_per_decision"] for s in scores[label].values()))
        b.append(row["gain_vs_base"]["gain_s"])
    ranks = lambda v: np.argsort(np.argsort(v)).astype(float)  # noqa: E731
    return {"labels": len(a), "spearman": float(np.corrcoef(ranks(a), ranks(b))[0, 1]) if len(a) > 2 else None}


def gain_text(g: dict) -> str:
    interval = "구간 없음" if g["t95"][0] is None else f"{g['t95'][0]:+.2f}, {g['t95'][1]:+.2f}"
    return f"{g['gain_s']:+.2f} [{interval}]"


def render(report: dict) -> str:
    s = report["start"]
    lines = ["# NN3 라운드 1 결과(무리 세계, 검증 날짜 개발 비교)", "", f"**출발 모델: {s['start']}** (흉내 값). 낮 24상황 N5 {s['means']['N5_day']:.2f} → N6 {s['means']['N6_day']:.2f}초, 새벽·심야 12상황 N5 {s['means']['N5_low']:.2f} → N6 {s['means']['N6_low']:.2f}초, TM4 {s['means']['TM4_low']:.2f}초.",
             f"N5 − N6(+면 N6가 짧음): 낮 {gain_text(s['day_gain'])}초, 새벽 {gain_text(s['low_gain'])}초.", ""]
    if report["status"] != "done":
        lines.append(f"라운드 1은 아직 끝나지 않았습니다: {report.get('missing', '자료·학생 전')}")
        return "\n".join(lines) + "\n"
    st = report["students"]
    lines += [f"자료: {st['runs']}판, 지점 {st['points']}개, 제외 {st['skipped']}, 굴림 {st['cpu_hours_rollouts']:.0f} CPU-시간.", "", "## 거르기·문턱(24상황 × 모델 3개)", "", f"출발 모델 {report['base_day_s']:.2f}초", "",
              "| 후보 | 지연 | 출발 모델 대비 [t95] | 바꾼 비율 | 동 좌회전 | 남 좌회전 | 미확인 충돌 |", "|---|---:|---|---:|---:|---:|---:|"]
    for name, row in report["screen"].items():
        g, m = row["gain_vs_base"], row["summary"]
        lines.append(f"| {name} | {row['mean_delay_s']:.2f} | {gain_text(g)} | {m['change_share']:.1%} | {m['left_east_s']:.1f} | {m['left_south_s']:.1f} | {m['collisions_unexplained']} |")
    base = report["final"]["BASE"]
    lines += ["", f"라벨 뒤 절반 이득과 한 시간 이득의 순위 일치(10개): {report['agreement_one_hour']['spearman']:+.2f}", "", "## 확인(84상황 × 모델 3개)", "", f"출발 모델 {base['mean_delay_s']:.2f}초(모델별 {', '.join(f'{x:.2f}' for x in base['per_model_s'])}), 동 좌회전 {base['left_east_s']:.1f}, 남 좌회전 {base['left_south_s']:.1f}, 300초 초과 판당 {base['long_per_run']:.1f}", "",
              "| 후보 | 지연 | 출발 모델 대비 [t95] | 앞선 상황 | 날짜별 | 모델별 | 새벽·심야 12상황 | 바꾼 비율 | 동 좌회전 | 남 좌회전 | 300초 초과 | 미확인 충돌 |", "|---|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in report["final"]["candidates"].items():
        g, m = row["gain"], row["summary"]
        lines.append(f"| {name} | {m['mean_delay_s']:.2f} | {gain_text(g)} | {g['positive']}/{g['n']} | {', '.join(f'{v:+.2f}' for v in row['per_date'].values())} | {', '.join(f'{v:+.2f}' for v in row['per_model'])} | {row['low_gain']['gain_s']:+.2f} | {m['change_share']:.1%} | {m['left_east_s']:.1f} | {m['left_south_s']:.1f} | {m['long_per_run']:.1f} | {m['collisions_unexplained']} |")
    r = report["reading"]
    lines += ["", f"**읽기:** {r['winner']} − 출발 모델 {gain_text(r['gain'])}초 → **{'채택' if r['adopted'] else '미채택'}** (기준: 평균 +0.2초 이상, t95 하한 > 0).",
              "", "- 검증 날짜의 개발 비교다. 출발 모델은 흉내 값이고, 강화학습의 몫은 위 차이만이다. 무리 도착은 모형이다."]
    return "\n".join(lines) + "\n"


def save_report(out: Path, report: dict) -> None:
    s2e._write(Path(out) / "judge.json", w.plain(report))
    (Path(out) / "summary.md").write_text(render(report), encoding="utf-8")


# ----------------------------------------------------------------------------- 점검(맥·Colab에서 같은 코드)
def check(models: Path, out: Path, work: Path, workers: int, cfg: dict, points: int = 3) -> dict:
    """(1) 상태 저장·굴림을 해도 본 판 신호가 같다 (2) 학습 전 학생을 쌓아도 신호가 같다 (3) 복원·입력 대조·엿봄 없음 (4) 새벽·심야에서 끝까지 돈다 (5) 시간 어림 (6) 맥 기록과 같은 신호."""
    out, work = Path(out), Path(work)
    shutil.rmtree(work, ignore_errors=True)
    nn.prepare_caches(WORLD)
    model = Path(models) / "seed_42"
    if nn.model_identity(model) != cfg["n5"]["identities"]["42"]:
        raise ValueError("점검 모델이 설정의 N5 난수 42가 아닙니다.")
    net = nn.load_model(model)[0]
    s = cfg["situations_round"][0]

    def plain_run(policy, tag, situation=s):
        with nn.planner_free(policy, "NN") as guard:
            result = pt.run_episode(situation, work / tag, policy, max_seconds=nn.MAX_EPISODE_S)
        if result.metrics["route_sha256"] != situation["route_sha256"]:
            raise ValueError("차량 파일 SHA가 설정과 다릅니다.")
        return result, guard

    plain, guard = plain_run(StackPolicy(net, "N5"), "plain")
    untrained = nn.GainNet(INPUT_SIZE)
    stacked, _ = plain_run(StackPolicy(net, "stacked", [(untrained.eval(), 0.5)]), "stacked")
    with pt.rollout_pool(workers) as pool:
        policy = Collector(pool, cfg, s, work / "same" / "states", net, [], policy_spec(model), limit=points)
        with nn.planner_free(policy, "NN"):
            same = pt.run_episode(s, work / "same" / "episode", policy, max_seconds=float(cfg["rollout"]["main_episode_max_seconds"]), on_step=policy.after_step)
        wait(policy.futures_list())
        rc = policy.finish()
    rows = [point_row(p, cfg) for p in policy.points]
    low = {}
    for name, situation in (("dawn", next(x for x in cfg["situations_low_logs"] if x["band"] == "dawn")), ("late", next(x for x in cfg["situations_low_logs"] if x["band"] == "late" and x["period"] == max(z["period"] for z in cfg["situations_low_logs"] if z["band"] == "late")))):
        r, g = plain_run(StackPolicy(net, f"N5_{name}"), f"low_{name}", situation)
        tm4 = pt.run_episode(situation, work / f"tm4_{name}", pt.TM4Policy(), max_seconds=nn.MAX_EPISODE_S)
        low[name] = {"period": situation["period"], "vehicles": int(r.metrics["vehicles"]), "n5_delay_s": r.metrics["mean_delay_s"], "tm4_delay_s": tm4.metrics["mean_delay_s"], "quality_errors": ref.quality_errors(r.metrics) + ref.quality_errors(tm4.metrics), "tail_calls": g["tail_calls"], "seconds": r.seconds}
    seconds = [float(x.get("seconds") or 0.0) for *_, f in policy.rollouts for x in [f.result()]]
    exits = [len(x.get("east_exits") or []) for *_, f in policy.rollouts for x in [f.result()]]
    keys = ("executed_sha16", "route_sha256", "actual_total_delay")
    report = {"same_as_plain": all(same.metrics[k] == plain.metrics[k] for k in keys), "stacked_same": all(stacked.metrics[k] == plain.metrics[k] for k in keys), "planner_calls": guard["tail_calls"], "points": len(policy.points), "restore": {k: rc[k] for k in ("checked", "mismatched")}, "rollouts": rc["rollouts"],
              "rollout_failures": rc["rollout_failures"], "usable_points": sum(r[0] is not None for r in rows), "skip_reasons": [r[1] for r in rows if r[0] is None], "east_future": rc["east_future"], "rollout_seconds": {"mean": statistics.fmean(seconds), "max": max(seconds)}, "east_exits_per_rollout": statistics.fmean(exits), "main_seconds": same.seconds, "plain_seconds": plain.seconds,
              "low": low, "plain": {k: plain.metrics[k] for k in keys}, "decisions": StackPolicy.info(policy), "quality_errors": ref.quality_errors(plain.metrics)}
    if REFERENCE.exists():
        saved = w.read(REFERENCE)
        report["same_as_mac"] = all(saved["plain"][k] == report["plain"][k] for k in keys) and all(saved["low"][n]["n5_delay_s"] == low[n]["n5_delay_s"] for n in low)
    report["ok"] = bool(report["same_as_plain"] and report["stacked_same"] and report["usable_points"] > 0 and report["east_exits_per_rollout"] > 0 and not report["quality_errors"] and rc["mismatched"] == 0 and not rc["rollout_failures"] and report.get("same_as_mac", True)
                        and all(not v["quality_errors"] for v in low.values()))
    s2e._write(out / "check.json", w.plain(report))
    shutil.rmtree(work, ignore_errors=True)
    return report


# ----------------------------------------------------------------------------- 명령
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("make-config")
    p.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("check")
    p.add_argument("--models", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--points", type=int, default=3)
    p = sub.add_parser("prep")
    p.add_argument("--nn1", required=True)
    p.add_argument("--teacher", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    p = sub.add_parser("round")
    p.add_argument("--prep", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--shards", type=int, default=4)
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    p = sub.add_parser("judge")
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    w.configure()
    if args.cmd == "make-config":
        print(json.dumps(make_config(args.workers), ensure_ascii=False, indent=1))
        return 0
    cfg = config()
    if args.cmd == "check":
        report = check(Path(args.models), Path(args.out), Path(args.work), args.workers, cfg, args.points)
        print(json.dumps(w.plain(report), ensure_ascii=False, indent=1))
        return 0 if report["ok"] else 1
    if args.cmd == "prep":
        return prep(Path(args.nn1), Path(args.teacher), Path(args.out), Path(args.work), args.workers, cfg)
    if args.cmd == "round":
        return run_round(Path(args.prep), Path(args.out), Path(args.work), args.shard, args.shards, args.workers, cfg)
    report = collect_report(Path(args.root), cfg)
    save_report(Path(args.out), report)
    print(render(report))
    return 0 if report["status"] == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
