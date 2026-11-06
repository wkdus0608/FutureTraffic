"""[추가 탐색] TUNE1 — 무리 세계에서 처음부터 DQN·PPO와 은닉층 크기(사전 등록 results/explore/tune1_20261105/사전등록.md).

신경망이 허용 녹색 안에서 녹색을 직접 고른다. 입력은 N5와 같은 120칸이고, 학습에도 실행에도 계획기·선생님이 없다(부르면 멈춘다).
보상은 심판과 같은 정확한 지연이다. 평가는 N5와 같은 신경망만 경로로 하므로, 시연 조건을 그대로 잰다.
잠긴 파일·이전 실험 코드와 결과는 고치지 않고 가져다 쓴다.

사용(FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240):
  python src/explore_tune.py make-config --prep <Drive>/explore_nn3/prep --workers 8                  (맥, SUMO 없음)
  python src/explore_tune.py check --cell DQN_64 --prep <…/prep> --out <dir> --work /content/tc
  python src/explore_tune.py prepare --cell DQN_64 --out <Drive>/explore_tune/tune1
  python src/explore_tune.py train --cell DQN_64 --seed 42 --out <Drive>/explore_tune/tune1 --work /content/tt
  python src/explore_tune.py evaluate --cell DQN_64 --out <Drive>/explore_tune/tune1 --work /content/te --workers 8
  python src/explore_tune.py baseline --prep <…/prep> --out <Drive>/explore_tune/tune1 --work /content/tb --workers 8
  python src/explore_tune.py judge --root <Drive>/explore_tune/tune1 --out <Drive>/explore_tune/tune1/judge
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import math
import multiprocessing
import os
import random
import shutil
import statistics
import sys
import tempfile
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn as tnn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import arena3_common as ac  # noqa: E402
import arena3_g1 as g1  # noqa: E402
import arena3_ppo as pp  # noqa: E402
import arena3_referee as ref  # noqa: E402
import arena3_stage2_eval as s2e  # noqa: E402
import explore_nn1 as nn  # noqa: E402
import explore_nn3 as n3  # noqa: E402
import explore_nn3r2 as r2  # noqa: E402
import explore_nn3r3 as r3  # noqa: E402
import explore_platoon0 as p0  # noqa: E402
import explore_platoon1 as pt  # noqa: E402
import explore_weight as w  # noqa: E402

EXPERIMENT = "TUNE1_V1"
CONFIG = ROOT / "config/explore_tune1_v1.json"
PREREG = ROOT / "results/explore/tune1_20261105/사전등록.md"
REFERENCE = ROOT / "results/explore/tune1_20261105/smoke_reference/check.json"
KIND = "tune1-net-v1"
SOURCES = ("src/explore_tune.py", "config/explore_tune1_v1.json", "results/explore/tune1_20261105/사전등록.md", "src/arena3_g1.py", "src/arena3_ppo.py", "src/explore_nn3r3.py", *r3.SOURCES)
WORLD = "P"
INPUT_SIZE = n3.INPUT_SIZE
GREENS = nn.GREENS
SEEDS = (42, 43, 44)
N5, FIXC = "N5", "FIXC"
BATCHES = ("VU1", "VU2")
CURVE_PERIODS = ("0730", "1030", "1330", "1630", "1800", "2015")
NETWORKS = {"64": [64, 64], "256": [256, 256], "64x3": [64, 64, 64]}
DQN = {"gamma_per_5s": 0.99, "n_choice_steps": 3, "learning_rate": 0.001, "batch_size": 64, "replay_size": 100000, "learning_starts": 500, "updates_per_decision": 8, "target_every_updates": 1250,
       "epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_decay_decisions": 30000, "grad_clip": 10.0, "checkpoint_every_episodes": 24}
PPO = {"episodes_per_update": 8, "critic_warmup_updates": 2, "learning_rate": 0.0003, "adam_eps": 1e-05, "n_epochs": 10, "batch_size": 64, "clip_range": 0.2, "gamma_per_5s": 0.99, "gae_lambda_per_5s": 0.95, "normalize_advantage": True,
       "ent_coef": 0.0, "vf_coef": 0.5, "max_grad_norm": 0.5, "target_kl": None}
TRAINING = {"seeds": list(SEEDS), "episodes": 600, "parallel_episodes": 2, "marks": [150, 300, 450], "max_discarded": 30, "episode_max_seconds": 3600.0}
RULES = {"size_gain_s": 0.3, "size_models_positive": 2, "beats_n5_gain_s": 0.2, "curve_rising_s": 0.5, "target_reduction_pct": 35.0}


def cells() -> dict[str, dict]:
    return {f"{algo}_{tag}": {"algo": algo, "network": net, "settings": settings} for algo, settings in (("DQN", DQN), ("PPO", PPO)) for tag, net in NETWORKS.items()}


# ----------------------------------------------------------------------------- 설정
def config() -> dict:
    cfg = w.read(CONFIG)
    if cfg.get("experiment") != EXPERIMENT:
        raise ValueError("TUNE1 설정이 아닙니다.")
    for key, path in (("prereg_sha256", PREREG), ("lock_sha256", ac.CONFIG), ("nn3r3_config_sha256", r3.CONFIG)):
        if cfg.get(key) != w.file_hash(path):
            raise ValueError(f"TUNE1 설정을 만든 때와 파일이 다릅니다: {Path(path).relative_to(ROOT)}")
    return cfg


def screen_situations() -> list[dict]:
    return [ac.make_situation(d, p, f"VU{j}") for d in n3.VALID_DATES for p in n3.FINAL_PERIODS for j in (1, 2)]


def curve_ids() -> list[str]:
    return [ac.situation_id(ac.make_situation(d, p, "VU1")) for d in n3.VALID_DATES for p in CURVE_PERIODS]


def scenario_for(seed: int, episode: int) -> dict:
    """난수·판 번호로 정해지는 학습 상황. 난수가 같으면 모든 칸이 같은 순서의 같은 상황을 쓴다."""
    rng = random.Random(ac.seed32("TUNE1_SCENARIO", seed, episode))
    return ac.make_situation(rng.choice(n3.TRAIN_DATES), rng.choice(ac.TRAIN_SLOTS), f"TU{seed}e{episode}")


def make_config(prep: Path, workers: int) -> dict:
    """맥, SUMO 없음. N5 해시·입력 눈금과 VU 상황의 무리 세계 차량 파일 SHA를 고정한다."""
    r3cfg = r3.config()
    n5 = {key: value["n5_sha256"] for key, value in r3cfg["base"]["frozen"].items()}
    for seed in SEEDS:
        if nn.model_identity(Path(prep) / "start" / f"seed_{seed}") != n5[str(seed)]:
            raise ValueError(f"N5 모델이 라운드 3 설정과 다릅니다: 난수 {seed}")
    net = nn.load_model(Path(prep) / "start" / f"seed_{SEEDS[0]}")[0]
    rows = screen_situations()
    previous = w.read(CONFIG) if CONFIG.exists() else {}
    keep = {ac.situation_id(s): s["route_sha256"] for s in previous.get("situations_screen", [])}
    need = [s for s in rows if ac.situation_id(s) not in keep]
    if need:
        with tempfile.TemporaryDirectory() as d, p0._pool(workers) as pool:
            for sid, _, sha in pool.map(p0._sha_job, [(s, WORLD, d) for s in need]):
                keep[sid] = sha
    for s in rows:
        s["route_sha256"] = keep[ac.situation_id(s)]
    new = [s["route_sha256"] for s in rows]
    used = r3.previous_routes(n3.config(), r2.config()) | {s["route_sha256"] for part in ("situations_round", "situations_screen", "situations_final", "situations_low") for s in r3cfg[part]}
    if len(set(new)) != len(new) or set(new) & used:
        raise ValueError("새 차량 파일이 서로 같거나 기존 차량 파일과 같습니다.")
    cfg = {"experiment": EXPERIMENT, "world": WORLD, "design": "PROJECT_PROGRESS.md 281·282절, results/explore/tune1_20261105/사전등록.md", "prereg": PREREG.relative_to(ROOT).as_posix(), "prereg_sha256": w.file_hash(PREREG), "lock_sha256": w.file_hash(ac.CONFIG), "nn3r3_config_sha256": w.file_hash(r3.CONFIG),
           "dates": {"train": list(n3.TRAIN_DATES), "validation": list(n3.VALID_DATES)}, "cells": cells(), "training": TRAINING, "rules": RULES, "n5": n5, "input": {**r3cfg["input"], "scale_from": "N5 난수 42(입력 눈금만, 가중치 안 씀)", "mean": net.mean.tolist(), "std": net.std.tolist()},
           "curve_situations": curve_ids(), "reporting": "검증 날짜의 개발 비교다. 처음부터 강화학습한 값이다. 무리 도착은 모형이다.", "situations_screen": rows}
    CONFIG.write_text(json.dumps(w.plain(cfg), ensure_ascii=False, indent=1) + "\n")
    return {"config": str(CONFIG), "sha256": w.file_hash(CONFIG), "situations": len(rows), "new_sha": len(need), "cells": sorted(cfg["cells"])}


# ----------------------------------------------------------------------------- 신경망
class TuneNet(tnn.Module):
    """입력(고정 눈금으로 맞춤) → 은닉층 → 녹색 4개의 점수. DQN에서는 Q, PPO에서는 행위자 점수. nn.NNPolicy가 그대로 쓴다."""

    def __init__(self, hidden, mean, std, size: int = INPUT_SIZE):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        layers, last = [], size
        for width in hidden:
            layers += [tnn.Linear(last, int(width)), tnn.ReLU()]
            last = int(width)
        layers.append(tnn.Linear(last, GREENS))
        self.net = tnn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / self.std)

    def small_head(self, gain: float = 0.01) -> "TuneNet":
        """PPO 행위자: 처음 정책이 거의 고르게 뽑도록 마지막 층을 작게 시작한다(SB3 기본)."""
        head = self.net[-1]
        tnn.init.orthogonal_(head.weight, gain)
        tnn.init.zeros_(head.bias)
        return self


class TuneCritic(tnn.Module):
    """SB3 PPO 기본 평가자: tanh, 직교 초기화(은닉 √2, 출력 1). 은닉층은 칸을 따른다."""

    def __init__(self, hidden, mean, std, size: int = INPUT_SIZE):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        layers, last = [], size
        for width in hidden:
            layers += [tnn.Linear(last, int(width)), tnn.Tanh()]
            last = int(width)
        self.body = tnn.Sequential(*layers)
        self.out = tnn.Linear(last, 1)
        for layer in self.body:
            if isinstance(layer, tnn.Linear):
                tnn.init.orthogonal_(layer.weight, math.sqrt(2.0))
                tnn.init.zeros_(layer.bias)
        tnn.init.orthogonal_(self.out.weight, 1.0)
        tnn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.body((x - self.mean) / self.std)).squeeze(1)


def net_bytes(net: TuneNet) -> bytes:
    buf = io.BytesIO()
    torch.save(net.state_dict(), buf)
    return buf.getvalue()


def net_from_bytes(raw: bytes, hidden, scale: dict) -> TuneNet:
    net = TuneNet(hidden, scale["mean"], scale["std"], size=len(scale["mean"]))
    net.load_state_dict(torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True))
    net.eval()
    return net


def mask_array(allowed: list[int]) -> np.ndarray:
    return np.isin(np.arange(GREENS), np.asarray(allowed, dtype=np.int64))


def epsilon_at(decisions: int, settings: dict) -> float:
    frac = min(1.0, max(0.0, decisions / float(settings["epsilon_decay_decisions"])))
    return float(settings["epsilon_start"]) + frac * (float(settings["epsilon_end"]) - float(settings["epsilon_start"]))


def masked_log_probs(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)


def pick(net: TuneNet, x: np.ndarray, allowed: list[int], algo: str, settings: dict, decisions: int, rng: random.Random) -> dict:
    """학습 때의 선택(순수 계산). DQN은 ε-탐욕, PPO는 확률로 뽑는다. 항상 허용 녹색 안에서 고른다."""
    mask = mask_array(allowed)
    with torch.no_grad():
        scores = net(torch.from_numpy(np.asarray(x, np.float32)).view(1, -1))[0]
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("녹색 점수가 유한하지 않습니다.")
    if algo == "DQN":
        eps = epsilon_at(decisions, settings)
        greedy = int(torch.where(torch.from_numpy(mask), scores.double(), torch.tensor(float("-inf"), dtype=torch.float64)).argmax())
        return {"action": int(rng.choice(allowed)) if rng.random() < eps else greedy, "epsilon": eps}
    logp = masked_log_probs(scores.double(), torch.from_numpy(mask)).numpy()
    valid = [int(a) for a in allowed]
    action = rng.choices(valid, weights=[math.exp(logp[a]) for a in valid])[0]
    return {"action": int(action), "log_prob": float(logp[action])}


class TrainPolicy(nn.NNPolicy):
    """학습용 운전. 선택 지점(허용 녹색 2개 이상)마다 입력·마스크·행동을 남긴다. 꼬리는 NNPolicy.act의 탐욕 선택이 잇는다."""

    def __init__(self, net: TuneNet, algo: str, settings: dict, base_decisions: int, rng_seed: int, name: str = "TUNE"):
        super().__init__(WORLD, net, name)
        self.algo, self.settings, self.base = algo, settings, int(base_decisions)
        self.rng = random.Random(int(rng_seed))
        self.taken: list[dict] = []

    def decide(self, snap, ctx) -> int:
        self.ctx = ctx
        env = ctx.ep.env
        core = pt.episode_core(self, ctx)
        core.observe(env)
        allowed = [int(a) for a in ctx.allowed]
        if len(allowed) == 1:
            self.stats["forced"] += 1
            return allowed[0]
        x = self.features(env, snap.observation, core)
        got = pick(self.net, x, allowed, self.algo, self.settings, self.base + len(self.taken), self.rng)
        self.stats["planned"] += 1
        self.taken.append({"step": int(ctx.step), "observation": x, "mode": got["action"], "mask": mask_array(allowed), **{k: v for k, v in got.items() if k != "action"}})
        return got["action"]


# ----------------------------------------------------------------------------- 에피소드 작업자
KEEP = ("mean_delay_s", "vehicles", "actual_total_delay", "long_waits", "left_east", "left_south", "collisions_unexplained", "route_sha256", "seconds", "executed_sha16")


def episode_worker(job: dict) -> dict:
    started = time.monotonic()
    work = Path(job["work"])
    try:
        w.configure()
        torch.set_num_threads(1)
        shutil.rmtree(work, ignore_errors=True)
        net = net_from_bytes(job["net"], job["network"], job["scale"])
        policy = TrainPolicy(net, job["algo"], job["settings"], job["base_decisions"], job["rng_seed"])
        try:
            with nn.planner_free(policy, "NN") as guard:
                result = pt.run_episode(job["situation"], work, policy, max_seconds=float(job["max_seconds"]), trace=False)
        except TimeoutError as error:
            return {"ok": True, "status": "discard", "errors": [], "metrics": {}, "choices": [], "rewards": [], "seconds": time.monotonic() - started, "reason": f"시간 초과: {error}"}
        m = result.metrics
        if guard["tail_calls"]:
            raise RuntimeError("학습 중 계획기를 불렀습니다.")
        if len(result.rewards) != 720:
            raise RuntimeError(f"보상 단계 수가 720이 아닙니다: {len(result.rewards)}")
        errors = ref.quality_errors(m)
        if errors:
            return {"ok": False, "status": "error", "errors": errors, "seconds": time.monotonic() - started}
        status = "discard" if m.get("collisions_unexplained") else "ok"
        return {"ok": True, "status": status, "errors": [], "metrics": {k: m.get(k) for k in KEEP}, "choices": policy.taken, "rewards": list(result.rewards), "seconds": time.monotonic() - started,
                "reason": "확인 안 된 충돌" if status == "discard" else None}
    except BaseException as error:
        return {"ok": False, "status": "error", "errors": [f"{type(error).__name__}: {error}"], "seconds": time.monotonic() - started}
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ----------------------------------------------------------------------------- DQN
class Replay:
    def __init__(self, capacity: int, size: int = INPUT_SIZE):
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, size), np.float32)
        self.next_obs = np.zeros((capacity, size), np.float32)
        self.mode = np.zeros(capacity, np.int64)
        self.reward = np.zeros(capacity, np.float32)
        self.discount = np.zeros(capacity, np.float32)
        self.done = np.zeros(capacity, np.bool_)
        self.next_mask = np.zeros((capacity, GREENS), np.bool_)
        self.position = self.size = 0

    FIELDS = ("obs", "next_obs", "mode", "reward", "discount", "done", "next_mask")

    def add(self, rows: list[dict]) -> None:
        for t in rows:
            p = self.position
            self.obs[p], self.next_obs[p], self.mode[p] = t["obs"], t["next_obs"], t["mode"]
            self.reward[p], self.discount[p], self.done[p], self.next_mask[p] = t["reward"], t["discount"], t["done"], t["next_mask"]
            self.position = (p + 1) % self.capacity
            self.size = min(self.capacity, self.size + 1)

    def sample(self, batch: int, rng: np.random.Generator) -> dict:
        ids = rng.integers(0, self.size, size=batch)
        return {k: torch.from_numpy(getattr(self, k)[ids]) for k in self.FIELDS}

    def state(self) -> dict:
        return {"capacity": self.capacity, "position": self.position, "size": self.size, **{k: getattr(self, k)[:self.size].copy() for k in self.FIELDS}}

    def load(self, s: dict) -> None:
        if int(s["capacity"]) != self.capacity:
            raise ValueError("재생 크기가 다릅니다.")
        for k in self.FIELDS:
            getattr(self, k)[:int(s["size"])] = s[k]
        self.position, self.size = int(s["position"]), int(s["size"])


def dqn_transitions(choices: list[dict], rewards: list[float], settings: dict) -> list[dict]:
    """SMDP n-step 전이(arena3_g1과 같은 계산) + 다음 지점의 허용 녹색."""
    n = int(settings["n_choice_steps"])
    rows = g1.smdp_transitions(choices, rewards, float(settings["gamma_per_5s"]), n)
    for k, t in enumerate(rows):
        t["next_mask"] = np.ones(GREENS, np.bool_) if t["done"] else np.asarray(choices[k + n]["mask"], np.bool_)
        if not np.asarray(choices[k]["mask"], np.bool_)[t["mode"]]:
            raise ValueError("허용되지 않은 녹색이 학습 자료에 있습니다.")
    return rows


class DQNLearner:
    algo = "DQN"

    def __init__(self, seed: int, hidden, scale: dict, settings: dict):
        torch.manual_seed(int(seed))
        self.seed, self.settings = int(seed), settings
        self.online = TuneNet(hidden, scale["mean"], scale["std"], size=len(scale["mean"]))
        self.target = copy.deepcopy(self.online)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=float(settings["learning_rate"]))
        self.replay = Replay(int(settings["replay_size"]), size=len(scale["mean"]))
        self.sample_rng = np.random.default_rng(self.seed + 100_000)
        self.updates = self.syncs = 0

    @property
    def net(self) -> TuneNet:
        return self.online

    def round_size(self, parallel: int) -> int:
        return int(parallel)

    def gradient_step(self) -> float:
        s = self.settings
        d = self.replay.sample(int(s["batch_size"]), self.sample_rng)
        with torch.no_grad():
            chosen = self.online(d["next_obs"]).masked_fill(~d["next_mask"], -torch.inf).argmax(dim=1, keepdim=True)
            value = self.target(d["next_obs"]).gather(1, chosen).squeeze(1)
            value = torch.where(d["done"], torch.zeros_like(value), value)
            goal = d["reward"] + d["discount"] * value
        pred = self.online(d["obs"]).gather(1, d["mode"].view(-1, 1)).squeeze(1)
        loss = F.smooth_l1_loss(pred, goal)
        if not bool(torch.isfinite(loss)):
            raise ValueError("DQN 손실이 유한하지 않습니다.")
        self.optimizer.zero_grad()
        loss.backward()
        tnn.utils.clip_grad_norm_(self.online.parameters(), float(s["grad_clip"]))
        self.optimizer.step()
        self.updates += 1
        if self.updates % int(s["target_every_updates"]) == 0:
            self.target.load_state_dict(self.online.state_dict())
            self.syncs += 1
        return float(loss.detach())

    def absorb(self, results: list[dict]) -> dict:
        losses, decisions = [], 0
        for r in results:
            if not r["choices"]:
                continue
            rows = dqn_transitions(r["choices"], r["rewards"], self.settings)
            decisions += len(rows)
            self.replay.add(rows)
            if self.replay.size >= int(self.settings["learning_starts"]):
                for _ in range(int(self.settings["updates_per_decision"]) * len(rows)):
                    losses.append(self.gradient_step())
        return {"decisions": decisions, "updates": len(losses), "mean_loss": float(np.mean(losses)) if losses else None, "replay": self.replay.size, "updates_total": self.updates, "syncs": self.syncs}

    def checkpoint_due(self, episodes: int) -> bool:
        return episodes % int(self.settings["checkpoint_every_episodes"]) < 2

    def payload(self) -> dict:
        return {"online": self.online.state_dict(), "target": self.target.state_dict(), "optimizer": self.optimizer.state_dict(), "replay": self.replay.state(), "sample_rng": self.sample_rng.bit_generator.state, "updates": self.updates, "syncs": self.syncs}

    def restore(self, p: dict) -> None:
        self.online.load_state_dict(p["online"])
        self.target.load_state_dict(p["target"])
        self.optimizer.load_state_dict(p["optimizer"])
        self.replay.load(p["replay"])
        self.sample_rng.bit_generator.state = p["sample_rng"]
        self.updates, self.syncs = int(p["updates"]), int(p["syncs"])


# ----------------------------------------------------------------------------- PPO
class PPOLearner:
    algo = "PPO"

    def __init__(self, seed: int, hidden, scale: dict, settings: dict):
        torch.manual_seed(int(seed))
        self.seed, self.settings = int(seed), settings
        self.actor = TuneNet(hidden, scale["mean"], scale["std"], size=len(scale["mean"])).small_head()
        self.critic = TuneCritic(hidden, scale["mean"], scale["std"], size=len(scale["mean"]))
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(settings["learning_rate"]), eps=float(settings["adam_eps"]))
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=float(settings["learning_rate"]), eps=float(settings["adam_eps"]))
        self.rng = np.random.default_rng(self.seed + 300_000)
        self.updates = 0

    @property
    def net(self) -> TuneNet:
        return self.actor

    def round_size(self, parallel: int) -> int:
        return int(self.settings["episodes_per_update"])

    def absorb(self, results: list[dict]) -> dict:
        warm = self.updates < int(self.settings["critic_warmup_updates"])
        lam = 1.0 if warm else float(self.settings["gae_lambda_per_5s"])
        parts = []
        for r in results:
            if not r["choices"]:
                continue
            part = pp.episode_batch(r, self.critic, self.settings, lam)      # ARENA3 PPO와 같은 SMDP GAE
            parts.append({**part, "mask": np.stack([np.asarray(c["mask"], np.bool_) for c in r["choices"]])})
        self.updates += 1
        if not parts:
            return {"decisions": 0, "warmup": warm, "actor_updated": False}
        return {"warmup": warm, **self.update(pp.concat(parts), update_actor=not warm)}

    def update(self, batch: dict, update_actor: bool) -> dict:
        """SB3 PPO.train()과 같은 순서: 에포크마다 섞고, 미니배치마다 이득 정규화 → 클립 목적 → 평가자 MSE. 행위자·평가자는 옵티마이저를 나눈다."""
        s = self.settings
        n = len(batch["obs"])
        obs, mode, mask = torch.from_numpy(batch["obs"]), torch.from_numpy(batch["mode"]), torch.from_numpy(batch["mask"])
        old, adv_all, ret = torch.from_numpy(batch["log_prob"]).float(), torch.from_numpy(batch["advantages"]).float(), torch.from_numpy(batch["returns"]).float()
        clip, size = float(s["clip_range"]), int(s["batch_size"])
        log = {"policy_loss": [], "value_loss": [], "approx_kl": [], "clip_fraction": [], "entropy": []}
        stopped = False
        for _ in range(int(s["n_epochs"])):
            order = self.rng.permutation(n)
            for start in range(0, n, size):
                ix = torch.from_numpy(order[start:start + size])
                if update_actor:
                    logits = self.actor(obs[ix]).masked_fill(~mask[ix], float("-inf"))
                    logp = pp.log_probs(logits).gather(1, mode[ix].view(-1, 1)).squeeze(1)
                    log_ratio = logp - old[ix]
                    ratio = torch.exp(log_ratio)
                    adv = adv_all[ix]
                    if s["normalize_advantage"] and len(ix) > 1:
                        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                    pg = -torch.min(ratio * adv, torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv).mean()
                    ent = pp.entropy(logits).mean()
                    loss = pg - float(s["ent_coef"]) * ent
                    if not bool(torch.isfinite(loss)):
                        raise ValueError("PPO 행위자 손실이 유한하지 않습니다.")
                    with torch.no_grad():
                        kl = float(((ratio - 1.0) - log_ratio).mean())
                    if s["target_kl"] is not None and kl > 1.5 * float(s["target_kl"]):
                        stopped = True
                        break
                    self.actor_opt.zero_grad()
                    loss.backward()
                    tnn.utils.clip_grad_norm_(self.actor.parameters(), float(s["max_grad_norm"]))
                    self.actor_opt.step()
                    log["policy_loss"].append(float(pg.detach()))
                    log["approx_kl"].append(kl)
                    log["clip_fraction"].append(float(((ratio.detach() - 1.0).abs() > clip).float().mean()))
                    log["entropy"].append(float(ent.detach()))
                vloss = float(s["vf_coef"]) * F.mse_loss(self.critic(obs[ix]), ret[ix])
                if not bool(torch.isfinite(vloss)):
                    raise ValueError("PPO 평가자 손실이 유한하지 않습니다.")
                self.critic_opt.zero_grad()
                vloss.backward()
                tnn.utils.clip_grad_norm_(self.critic.parameters(), float(s["max_grad_norm"]))
                self.critic_opt.step()
                log["value_loss"].append(float(vloss.detach()))
            if stopped:
                break
        var = float(np.var(batch["returns"]))
        explained = 1.0 - float(np.var(batch["returns"] - batch["values"])) / var if var > 0 else None
        return {"decisions": int(n), "actor_updated": bool(update_actor), "stopped_by_kl": stopped, "explained_variance": explained, "advantage_std": float(np.std(batch["advantages"])), **{k: (float(np.mean(v)) if v else None) for k, v in log.items()}}

    def checkpoint_due(self, episodes: int) -> bool:
        return True

    def payload(self) -> dict:
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(), "actor_opt": self.actor_opt.state_dict(), "critic_opt": self.critic_opt.state_dict(), "rng": self.rng.bit_generator.state, "updates": self.updates}

    def restore(self, p: dict) -> None:
        self.actor.load_state_dict(p["actor"])
        self.critic.load_state_dict(p["critic"])
        self.actor_opt.load_state_dict(p["actor_opt"])
        self.critic_opt.load_state_dict(p["critic_opt"])
        self.rng.bit_generator.state = p["rng"]
        self.updates = int(p["updates"])


LEARNERS = {"DQN": DQNLearner, "PPO": PPOLearner}


# ----------------------------------------------------------------------------- 저장
def save_net(folder: Path, net: TuneNet, meta: dict) -> str:
    folder = Path(folder)
    folder.parent.mkdir(parents=True, exist_ok=True)
    tmp = folder.with_name(f".{folder.name}.{uuid.uuid4().hex}.tmp")
    tmp.mkdir()
    torch.save({**meta, "kind": KIND, "state": net.state_dict()}, tmp / "model.pt")
    sha = w.file_hash(tmp / "model.pt")
    s2e._write(tmp / "manifest.json", {**meta, "kind": KIND, "model_sha256": sha})
    if folder.exists():
        shutil.rmtree(folder)
    os.replace(tmp, folder)
    return sha


def net_identity(folder: Path) -> str:
    manifest = w.read(Path(folder) / "manifest.json")
    sha = w.file_hash(Path(folder) / "model.pt")
    if manifest.get("kind") != KIND or sha != manifest["model_sha256"]:
        raise ValueError(f"TUNE1 모델이 아니거나 파일 해시가 다릅니다: {folder}")
    return sha


def load_net(folder: Path) -> TuneNet:
    net_identity(folder)
    payload = torch.load(Path(folder) / "model.pt", map_location="cpu", weights_only=False)
    state = payload["state"]
    net = TuneNet(payload["network"], state["mean"], state["std"], size=len(state["mean"]))
    net.load_state_dict(state)
    net.eval()
    return net


def save_checkpoint(seed_dir: Path, learner, state: dict, identity: dict) -> None:
    tmp = seed_dir / "checkpoint.pt.tmp"
    torch.save({"kind": KIND, "identity": identity, "seed": learner.seed, "state": state, "learner": learner.payload(), "torch_rng": torch.get_rng_state()}, tmp)
    sha = w.file_hash(tmp)
    os.replace(tmp, seed_dir / "checkpoint.pt")
    s2e._write(seed_dir / "checkpoint.json", {"sha256": sha, "episodes": state["episodes"], "updated_unix": time.time()})


def load_checkpoint(seed_dir: Path, seed: int, identity: dict, make) -> tuple | None:
    path = seed_dir / "checkpoint.pt"
    if not path.exists():
        return None
    meta = w.read(seed_dir / "checkpoint.json")
    if w.file_hash(path) != meta["sha256"]:
        raise ValueError("체크포인트 해시가 다릅니다.")
    p = torch.load(path, map_location="cpu", weights_only=False)
    if p.get("kind") != KIND or p.get("identity") != identity or p.get("seed") != seed or p["state"]["episodes"] != meta["episodes"]:
        raise ValueError("체크포인트의 칸·설정·난수가 다릅니다. 기존 결과를 보존하고 새 폴더를 쓰세요.")
    learner = make()
    learner.restore(p["learner"])
    torch.set_rng_state(p["torch_rng"])
    return learner, p["state"]


# ----------------------------------------------------------------------------- 학습
def train(cell: str, seed: int, out: Path, work: Path, cfg: dict, *, executor=None, worker=episode_worker, overrides: dict | None = None, training: dict | None = None, situation_of=scenario_for) -> dict:
    """칸 하나, 난수 하나. 끊겼다가 다시 부르면 체크포인트에서 이어 간다."""
    spec = cfg["cells"][cell]
    run = {**cfg["training"], **(training or {})}
    settings = {**spec["settings"], **(overrides or {})}
    algo, hidden = spec["algo"], spec["network"]
    scale = {k: cfg["input"][k] for k in ("mean", "std")}
    if seed not in cfg["training"]["seeds"]:
        raise ValueError(f"난수는 {cfg['training']['seeds']} 중 하나여야 합니다.")
    seed_dir = Path(out) / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    identity = {"cell": cell, "algo": algo, "network": hidden, "settings": settings, "training": run}
    make = lambda: LEARNERS[algo](seed, hidden, scale, settings)  # noqa: E731
    spend = g1.SpendLog(seed_dir / "spend.jsonl")
    loaded = load_checkpoint(seed_dir, seed, identity, make)
    if loaded is None:
        learner = make()
        state = {"episodes": 0, "decisions": 0, "discarded": [], "history": [], "log": [], "saved_marks": []}
        save_checkpoint(seed_dir, learner, state, identity)
    else:
        learner, state = loaded
    budget = int(run["episodes"])
    meta = lambda: {"cell": cell, "algo": algo, "seed": seed, "network": hidden, "settings": settings, "episodes": state["episodes"], "decisions": state["decisions"], "world": WORLD}  # noqa: E731

    def status(finished: bool = False) -> dict:
        recent = [h["mean_delay_s"] for h in state["history"][-20:] if h.get("mean_delay_s") is not None]
        result = {"cell": cell, "seed": seed, "finished": finished, "episodes": state["episodes"], "budget": budget, "decisions": state["decisions"], "discarded": len(state["discarded"]), "recent_delay_s": statistics.fmean(recent) if recent else None, "cpu_hours": spend.total() / 3600, "last": state["log"][-1] if state["log"] else {}, "updated_unix": time.time()}
        s2e._write(seed_dir / "status.json", result)
        return result

    status()
    if state["episodes"] < budget:
        pool = executor or ProcessPoolExecutor(max_workers=int(run["parallel_episodes"]), mp_context=multiprocessing.get_context("spawn"), initializer=g1._worker_init)
        with pool:
            while state["episodes"] < budget:
                raw = net_bytes(learner.net)
                jobs = []
                for b in range(min(learner.round_size(int(run["parallel_episodes"])), budget - state["episodes"])):
                    idx = state["episodes"] + b
                    s = situation_of(seed, idx)
                    if s["date"] not in cfg["dates"]["train"]:
                        raise ValueError("학습 날짜가 아닌 상황이 뽑혔습니다.")
                    jobs.append({"situation": s, "work": str(Path(work) / cell / f"seed_{seed}" / f"e{idx}"), "net": raw, "network": hidden, "scale": scale, "algo": algo, "settings": settings, "base_decisions": state["decisions"], "rng_seed": ac.seed32("TUNE1_SAMPLE", cell, seed, idx), "max_seconds": float(run["episode_max_seconds"])})
                results = list(pool.map(worker, jobs))
                bad = [(job, r) for job, r in zip(jobs, results) if not r["ok"]]
                if bad:
                    spend.add(sum(float(r.get("seconds") or 0.0) for r in results), f"failed e{state['episodes']}")
                    job, r = bad[0]
                    raise RuntimeError(f"{cell} 난수 {seed} 에피소드 {ac.situation_id(job['situation'])} 실패: {r['errors']}")
                learn_started = time.monotonic()
                keep = []
                for job, r in zip(jobs, results):
                    idx = state["episodes"]
                    state["episodes"] += 1
                    sid = ac.situation_id(job["situation"])
                    if r["status"] == "discard":
                        state["discarded"].append({"episode": idx, "situation": sid, "reason": r.get("reason")})
                        if len(state["discarded"]) > int(run["max_discarded"]):
                            raise RuntimeError(f"버린 판이 {run['max_discarded']}개를 넘었습니다.")
                        continue
                    keep.append(r)
                    state["decisions"] += len(r["choices"])
                    state["history"].append({"episode": idx, "situation": sid, **r["metrics"], "decisions": len(r["choices"]), "epsilon": r["choices"][0].get("epsilon") if r["choices"] else None})
                info = learner.absorb(keep)
                state["log"].append({"episodes": state["episodes"], "learn_seconds": time.monotonic() - learn_started, **info})
                spend.add(sum(float(r["seconds"]) for r in results) + time.monotonic() - learn_started, f"e{state['episodes']}")
                for mark in run["marks"]:
                    if state["episodes"] >= int(mark) and int(mark) not in state["saved_marks"]:
                        save_net(seed_dir / "marks" / f"e{mark}", learner.net, meta())
                        state["saved_marks"].append(int(mark))
                if learner.checkpoint_due(state["episodes"]) or state["episodes"] >= budget:
                    save_checkpoint(seed_dir, learner, state, identity)
                    now = status()
                    print(f"{cell} 난수 {seed}: {state['episodes']}/{budget}판, 결정 {state['decisions']}, 최근 지연 {now['recent_delay_s']}, 버림 {len(state['discarded'])}, 작업시간 {now['cpu_hours']:.2f}h", flush=True)
    finished = state["episodes"] >= budget
    if finished and not (seed_dir / "final" / "manifest.json").exists():
        save_net(seed_dir / "final", learner.net, meta())
    if finished:
        s2e._write(seed_dir / "history.json", w.plain({"history": state["history"], "log": state["log"], "discarded": state["discarded"]}))
    return status(finished)


# ----------------------------------------------------------------------------- 평가
def eval_worker(job: dict) -> dict:
    """평가 한 판: N5와 같은 신경망만 경로(탐욕 선택, 꼬리까지 신경망)."""
    started = time.monotonic()
    work = Path(job["work"])
    name = f"{job['policy']}/{job['sid']}"
    try:
        w.configure()
        torch.set_num_threads(1)
        shutil.rmtree(work, ignore_errors=True)
        policy = nn.NNPolicy(WORLD, load_net(Path(job["model_dir"])), job["policy"])
        with nn.planner_free(policy, "NN") as guard:
            result = pt.run_episode(job["situation"], work, policy, max_seconds=float(job.get("max_seconds", nn.MAX_EPISODE_S)))
        if result.metrics["route_sha256"] != job["situation"]["route_sha256"]:
            raise ValueError(f"차량 파일의 SHA가 설정과 다릅니다: {result.metrics['route_sha256']}")
        ac.write_run(Path(job["target"]), result, {"policy": job["policy"], "fingerprint": job["fingerprint"], "model_identity": job.get("model_identity"), "job_seconds": time.monotonic() - started, "information": policy.info(), "planner_free": True, "tail": "NN", "tail_planner_calls": guard["tail_calls"]})
        return {"ok": True, "seconds": time.monotonic() - started, "name": name}
    except BaseException as error:
        return {"ok": False, "seconds": time.monotonic() - started, "name": name, "error": f"{type(error).__name__}: {error}"}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def cell_jobs(evaluation: Path, work: Path, fp_: str, name: str, folders: list[Path], sits: list[dict]) -> list[dict]:
    jobs = []
    for i, folder in enumerate(folders):
        for job in ref.jobs_for(Path(evaluation), Path(work), fp_, f"{name}_m{i}", sits, model_identity=net_identity(folder), kind="tune"):
            job.update(model_dir=str(folder))
            jobs.append(job)
    return jobs


def evaluate(cell: str, root: Path, work: Path, workers: int, cfg: dict, *, worker=eval_worker, executor=None) -> dict:
    root = Path(root)
    train_dir, evaluation = root / cell / "train", root / cell / "eval"
    sits = cfg["situations_screen"]
    curve = [s for s in sits if ac.situation_id(s) in set(cfg["curve_situations"])]
    if len(curve) != len(cfg["curve_situations"]):
        raise ValueError("곡선 상황을 찾지 못했습니다.")
    seeds = cfg["training"]["seeds"]
    for seed in seeds:
        if not (train_dir / f"seed_{seed}" / "final" / "manifest.json").exists():
            raise FileNotFoundError(f"학습이 끝나지 않았습니다: {cell} 난수 {seed}")
    fp_ = ref.register(evaluation, extra_files=SOURCES)
    if worker is eval_worker and executor is None:
        nn.prepare_caches(WORLD)
    jobs = cell_jobs(evaluation, Path(work) / "final", fp_, cell, [train_dir / f"seed_{s}" / "final" for s in seeds], sits)
    for mark in cfg["training"]["marks"]:
        jobs += cell_jobs(evaluation, Path(work) / f"e{mark}", fp_, f"{cell}_e{mark}", [train_dir / f"seed_{s}" / "marks" / f"e{mark}" for s in seeds], curve)
    status = ref.run_jobs(jobs, max(1, workers), ref.Ledger(evaluation, "eval"), worker=worker, executor=executor)
    s2e._write(evaluation / "status.json", {**status, "total": len(jobs), "updated_unix": time.time()})
    return status


def baseline(prep: Path, root: Path, work: Path, workers: int, cfg: dict, *, executor=None) -> dict:
    """같은 84상황의 고정신호와 N5 모델 3개."""
    folder = Path(root) / "baseline"
    fp_ = ref.register(folder, extra_files=SOURCES)
    if executor is None:
        nn.prepare_caches(WORLD)
    sits = cfg["situations_screen"]
    models = {}
    for seed in cfg["training"]["seeds"]:
        model = Path(prep) / "start" / f"seed_{seed}"
        if nn.model_identity(model) != cfg["n5"][str(seed)]:
            raise ValueError(f"N5 모델이 설정에 고정한 것과 다릅니다: 난수 {seed}")
        models[seed] = model
    jobs = [*r2.fixed_jobs(folder, Path(work), fp_, sits), *n3.nn_jobs(folder, Path(work), fp_, N5, models, sits)]
    status = ref.run_jobs(jobs, max(1, workers), ref.Ledger(folder, "baseline"), worker=r2.job_worker, executor=executor)
    s2e._write(folder / "status.json", {**status, "total": len(jobs), "updated_unix": time.time()})
    return status


# ----------------------------------------------------------------------------- 판정
def model_gains(base: dict, cand: dict) -> list[float]:
    models = len(next(iter(base.values())))
    return [n3.weighted_mean([v[i] for v in base.values()]) - n3.weighted_mean([v[i] for v in cand.values()]) for i in range(models)]


def size_reading(gains: dict[str, dict], delays: dict[str, float], rules: dict) -> str:
    """64-64 대비 다른 크기가 기준을 넘으면 지연이 짧은 쪽, 아니면 64-64. gains = 크기 → {'gain', 'models'}."""
    good = [k for k, g in gains.items() if g["gain"]["t95"][0] is not None and g["gain"]["gain_s"] >= float(rules["size_gain_s"]) and g["gain"]["t95"][0] > 0.0 and sum(x > 0 for x in g["models"]) >= int(rules["size_models_positive"])]
    return min(good, key=lambda k: delays[k]) if good else "64"


def beats(gain: dict, threshold: float) -> bool:
    return gain["t95"][0] is not None and gain["gain_s"] >= float(threshold) and gain["t95"][0] > 0.0


def by_band_gain(base: dict, cand: dict) -> dict:
    return n3.part_gains(base, cand, [(n, lambda k, a=lo, z=hi: a <= int(k.split("_")[1]) < z) for n, lo, hi in (("아침 06~08", 600, 900), ("낮 09~15", 900, 1600), ("저녁 16~18", 1600, 1900), ("밤 19~21", 1900, 2400))])


def collect_report(root: Path, cfg: dict) -> dict:
    root = Path(root)
    sits = cfg["situations_screen"]
    ids = [ac.situation_id(s) for s in sits]
    report = {"experiment": EXPERIMENT}
    try:
        fixed = r2.read_fixed(root / "baseline", sits)[FIXC]
        n5_rows = n3.read_stage_runs(root / "baseline", [N5], sits)
        by_cell = {}
        for cell in cfg["cells"]:
            by_cell[cell] = n3._by_situation(n3.read_stage_runs(root / cell / "eval", [cell], sits), cell, ids)
    except FileNotFoundError as error:
        return {**report, "status": "incomplete", "missing": str(error)}
    n5 = n3._by_situation(n5_rows, N5, ids)
    fixed_mean = n3.weighted_mean([fixed[s] for s in ids])
    fixed_rows = [fixed[s] for s in ids]
    report["fixed"] = {"mean_delay_s": fixed_mean, "left_east_s": statistics.fmean(r["left_east"] for r in fixed_rows if r["left_east"] is not None), "left_south_s": statistics.fmean(r["left_south"] for r in fixed_rows if r["left_south"] is not None), "long_per_run": sum(r["long"] for r in fixed_rows) / len(fixed_rows)}
    n5_summary = n3.policy_summary(n5)
    report["n5"] = {"summary": n5_summary, "percent_vs_fixed": r2.percent(n5_summary["mean_delay_s"], fixed_mean)}
    curve_rows = {}
    report["cells"] = {}
    curve_sits = [s for s in sits if ac.situation_id(s) in set(cfg["curve_situations"])]
    curve_list = [ac.situation_id(s) for s in curve_sits]
    for cell, rows in by_cell.items():
        for s in ids:
            if {r["vehicles"] for r in rows[s]} != {fixed[s]["vehicles"]}:
                raise ValueError(f"고정신호와 차 수가 다릅니다: {cell}/{s}")
        summary = n3.policy_summary(rows)
        row = {"summary": summary, "percent_vs_fixed": r2.percent(summary["mean_delay_s"], fixed_mean), "gain_vs_n5": n3.paired_gain(n5, rows), "models_vs_n5": model_gains(n5, rows),
               "per_date": n3.part_gains(n5, rows, [(d, lambda k, x=d: k.startswith(x)) for d in n3.VALID_DATES]), "per_batch": n3.part_gains(n5, rows, [(b, lambda k, x=b: k.endswith(x)) for b in BATCHES]), "per_band": by_band_gain(n5, rows)}
        curve = {}
        try:
            marks = n3.read_stage_runs(root / cell / "eval", [f"{cell}_e{m}" for m in cfg["training"]["marks"]], curve_sits)
            for mark in cfg["training"]["marks"]:
                curve[str(mark)] = n3.weighted_mean([r for v in n3._by_situation(marks, f"{cell}_e{mark}", curve_list).values() for r in v])
        except FileNotFoundError as error:
            return {**report, "status": "incomplete", "missing": str(error)}
        curve["600"] = n3.weighted_mean([r for s in curve_list for r in rows[s]])
        row["curve"] = curve
        row["curve_rising"] = curve[str(cfg["training"]["marks"][-1])] - curve["600"] >= float(cfg["rules"]["curve_rising_s"])
        curve_rows[cell] = curve
        report["cells"][cell] = row
    report["curve_fixed_s"] = n3.weighted_mean([fixed[s] for s in curve_list])
    report["curve_n5_s"] = n3.weighted_mean([r for s in curve_list for r in n5[s]])
    report["reading"] = {}
    for algo in ("DQN", "PPO"):
        base = by_cell[f"{algo}_64"]
        gains = {tag: {"gain": n3.paired_gain(base, by_cell[f"{algo}_{tag}"]), "models": model_gains(base, by_cell[f"{algo}_{tag}"])} for tag in ("256", "64x3")}
        delays = {tag: report["cells"][f"{algo}_{tag}"]["summary"]["mean_delay_s"] for tag in NETWORKS}
        chosen = size_reading(gains, delays, cfg["rules"])
        name = f"{algo}_{chosen}"
        row = report["cells"][name]
        report["reading"][algo] = {"sizes_vs_64": gains, "chosen_size": chosen, "chosen_cell": name, "delay_s": row["summary"]["mean_delay_s"], "percent_vs_fixed": row["percent_vs_fixed"], "beats_fixed": row["summary"]["mean_delay_s"] < fixed_mean, "beats_n5": beats(row["gain_vs_n5"], cfg["rules"]["beats_n5_gain_s"]), "curve_rising": row["curve_rising"], "reaches_target_here": -row["percent_vs_fixed"] >= float(cfg["rules"]["target_reduction_pct"])}
    a, b = report["reading"]["DQN"]["chosen_cell"], report["reading"]["PPO"]["chosen_cell"]
    report["reading"]["ppo_minus_dqn"] = n3.paired_gain(by_cell[a], by_cell[b])
    training = {}
    for cell in cfg["cells"]:
        rows = {}
        for seed in cfg["training"]["seeds"]:
            status = root / cell / "train" / f"seed_{seed}" / "status.json"
            if status.exists():
                s = w.read(status)
                rows[str(seed)] = {k: s.get(k) for k in ("episodes", "decisions", "discarded", "recent_delay_s", "cpu_hours")} | {"last": {k: s.get("last", {}).get(k) for k in ("approx_kl", "explained_variance", "entropy", "mean_loss", "replay")}}
        training[cell] = rows
    report["training"] = training
    return {**report, "status": "done"}


gain_text = n3.gain_text


def render(report: dict) -> str:
    lines = ["# TUNE1 결과 — 무리 세계, 처음부터 DQN·PPO와 은닉층(검증 날짜 개발 비교, VU 84상황)", ""]
    if report["status"] != "done":
        return "\n".join([*lines, f"아직 끝나지 않았습니다: {report.get('missing', report['status'])}"]) + "\n"
    fx, n5 = report["fixed"], report["n5"]
    lines += [f"고정신호 {fx['mean_delay_s']:.2f}초(동 좌회전 {fx['left_east_s']:.1f}, 남 좌회전 {fx['left_south_s']:.1f}, 300초 초과 판당 {fx['long_per_run']:.1f}). N5 {n5['summary']['mean_delay_s']:.2f}초({n5['percent_vs_fixed']:+.1f}%).", "",
              "| 칸 | 지연 | 고정신호 대비 | N5 대비 [t95] | 모델별 지연 | 동 좌회전 | 남 좌회전 | 300초 초과(판당) | 미확인 충돌 | 판당 초 |", "|---|---:|---:|---|---|---:|---:|---:|---:|---:|"]
    for cell, row in report["cells"].items():
        m = row["summary"]
        lines.append(f"| {cell} | {m['mean_delay_s']:.2f} | {row['percent_vs_fixed']:+.1f}% | {gain_text(row['gain_vs_n5'])} | {', '.join(f'{x:.2f}' for x in m['per_model_s'])} | {m['left_east_s']:.1f} | {m['left_south_s']:.1f} | {m['long_per_run']:.2f} | {m['collisions_unexplained']} | {m['seconds_per_run']:.0f} |")
    lines += ["", f"## 학습 곡선(VU1 12상황, 고정신호 {report['curve_fixed_s']:.2f}초, N5 {report['curve_n5_s']:.2f}초)", "", "| 칸 | 150판 | 300판 | 450판 | 600판 |", "|---|---:|---:|---:|---:|"]
    for cell, row in report["cells"].items():
        c = row["curve"]
        lines.append(f"| {cell} | {c['150']:.2f} | {c['300']:.2f} | {c['450']:.2f} | {c['600']:.2f} |")
    lines += ["", "## 읽기", ""]
    for algo in ("DQN", "PPO"):
        r = report["reading"][algo]
        sizes = ", ".join(f"{k} {gain_text(v['gain'])}(모델 {', '.join(f'{x:+.2f}' for x in v['models'])})" for k, v in r["sizes_vs_64"].items())
        lines.append(f"- **{algo}:** 64-64 대비 {sizes} → 고른 은닉층 **{r['chosen_size']}**, {r['delay_s']:.2f}초({r['percent_vs_fixed']:+.1f}%). 고정신호보다 짧음: {r['beats_fixed']}, N5보다 짧음(기준 충족): {r['beats_n5']}, 곡선 아직 오름: {r['curve_rising']}, 이 상황에서 −35% 도달: {r['reaches_target_here']}.")
    lines += [f"- PPO − DQN(고른 칸끼리): {gain_text(report['reading']['ppo_minus_dqn'])}초(양수면 PPO가 짧다).", "- 검증 날짜의 개발 비교이고 모델 3개 값이다. 처음부터 강화학습한 값이다. −35% 판단은 TUNE3의 VJ 상황에서 한다. 무리 도착은 모형이다."]
    return "\n".join(lines) + "\n"


def save_report(out: Path, report: dict) -> None:
    s2e._write(Path(out) / "judge.json", w.plain(report))
    (Path(out) / "summary.md").write_text(render(report), encoding="utf-8")


# ----------------------------------------------------------------------------- 점검(맥·Colab에서 같은 코드)
SMALL_TRAINING = {"episodes": 4, "marks": [2], "parallel_episodes": 2}
SMALL = {"DQN": {"learning_starts": 10, "target_every_updates": 50}, "PPO": {"episodes_per_update": 2, "critic_warmup_updates": 1}}
SAME_KEYS = ("executed_sha16", "route_sha256", "actual_total_delay")
MAC_DELAY_TOLERANCE = 1e-5


def same_as_reference(saved: dict, got: dict, tolerance: float = MAC_DELAY_TOLERANCE) -> bool:
    """맥 기록과 비교: 신호·차량 파일은 정확히 같아야 한다. 총지연은 플랫폼 차이(N5 한 판 0.203차량·초, 네 런타임 모두 같은 값, 282절 정정)만 허용한다."""
    if saved["executed_sha16"] != got["executed_sha16"] or saved["route_sha256"] != got["route_sha256"]:
        return False
    return abs(float(saved["actual_total_delay"]) - float(got["actual_total_delay"])) <= tolerance * abs(float(saved["actual_total_delay"]))


def check(cell: str, prep: Path, out: Path, work: Path, cfg: dict) -> dict:
    """(1) 작은 학습이 끝까지 돈다(계획기 0, 보상 720, 품질 오류 0) (2) 저장 모델이 평가 경로로 돈다 (3) 탐욕 학습 정책과 평가 경로의 신호가 같다 (4) N5 한 판이 맥 기록과 같다."""
    out, work = Path(out), Path(work)
    shutil.rmtree(work, ignore_errors=True)
    nn.prepare_caches(WORLD)
    spec = cfg["cells"][cell]
    small = work / "train"
    status = train(cell, 42, small, work / "episodes", cfg, overrides=SMALL[spec["algo"]], training=SMALL_TRAINING)
    model = small / "seed_42" / "final"
    s = next(x for x in cfg["situations_screen"] if ac.situation_id(x) == cfg["curve_situations"][0])
    fp_ = "check"
    eval_out = work / "eval"
    job = cell_jobs(eval_out, work / "eval_work", fp_, cell, [model], [s])[0]
    done = eval_worker(job)
    metrics = w.read(Path(job["target"]) / "metrics.json") if done["ok"] else {}
    net = load_net(model)
    greedy = TrainPolicy(net, "DQN", {**DQN, "epsilon_start": 0.0, "epsilon_end": 0.0}, 0, 1)
    with nn.planner_free(greedy, "NN") as guard:
        driven = pt.run_episode(s, work / "greedy", greedy, max_seconds=nn.MAX_EPISODE_S)
    n5_job = n3.nn_jobs(work / "n5_out", work / "n5", fp_, N5, {42: Path(prep) / "start" / "seed_42"}, [s])[0]
    n5_done = r2.job_worker(n5_job)
    n5_metrics = w.read(Path(n5_job["target"]) / "metrics.json") if n5_done["ok"] else {}
    report = {"cell": cell, "train": {k: status[k] for k in ("finished", "episodes", "decisions", "discarded", "recent_delay_s", "cpu_hours")}, "last": status["last"], "train_ok": bool(status["finished"] and status["decisions"] > 0 and status["discarded"] == 0),
              "eval_ok": bool(done["ok"]) and not ref.quality_errors(metrics), "eval_error": done.get("error"), "eval_seconds": done["seconds"], "eval_delay_s": metrics.get("mean_delay_s"), "eval_planned": (metrics.get("information") or {}).get("planned"),
              "same_as_greedy": bool(metrics) and all(metrics.get(k) == driven.metrics.get(k) for k in SAME_KEYS), "planner_calls": guard["tail_calls"] + int(metrics.get("tail_planner_calls") or 0), "n5": {k: n5_metrics.get(k) for k in SAME_KEYS}, "n5_delay_s": n5_metrics.get("mean_delay_s"), "n5_error": n5_done.get("error")}
    if REFERENCE.exists():
        saved = w.read(REFERENCE)["n5"]
        report["same_as_mac"] = same_as_reference(saved, report["n5"])
        report["mac_delay_difference"] = abs(float(saved["actual_total_delay"]) - float(report["n5"]["actual_total_delay"]))
    report["ok"] = bool(report["train_ok"] and report["eval_ok"] and report["same_as_greedy"] and report["planner_calls"] == 0 and n5_done["ok"] and report.get("same_as_mac", True))
    s2e._write(out / "check.json", w.plain(report))
    shutil.rmtree(work, ignore_errors=True)
    return report


# ----------------------------------------------------------------------------- 명령
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("make-config")
    p.add_argument("--prep", required=True)
    p.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("check")
    p.add_argument("--cell", required=True)
    p.add_argument("--prep", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--work", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--cell", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("train")
    p.add_argument("--cell", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--work", required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--cell", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    p = sub.add_parser("baseline")
    p.add_argument("--prep", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    p = sub.add_parser("judge")
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    w.configure()
    if args.cmd == "make-config":
        print(json.dumps(make_config(Path(args.prep), args.workers), ensure_ascii=False, indent=1))
        return 0
    cfg = config()
    if args.cmd == "check":
        report = check(args.cell, Path(args.prep), Path(args.out), Path(args.work), cfg)
        print(json.dumps(w.plain(report), ensure_ascii=False, indent=1))
        return 0 if report["ok"] else 1
    if args.cmd == "prepare":
        fp_ = ref.register(Path(args.out) / args.cell / "train", extra_files=SOURCES)
        for seed in cfg["training"]["seeds"]:
            (Path(args.out) / args.cell / "train" / f"seed_{seed}").mkdir(parents=True, exist_ok=True)
        nn.prepare_caches(WORLD)
        print("준비 끝. 지문", fp_)
        return 0
    if args.cmd == "train":
        ref.register(Path(args.out) / args.cell / "train", extra_files=SOURCES)
        result = train(args.cell, args.seed, Path(args.out) / args.cell / "train", Path(args.work), cfg)
        print(json.dumps(w.plain(result), ensure_ascii=False))
        return 0 if result["finished"] else 1
    if args.cmd == "evaluate":
        status = evaluate(args.cell, Path(args.out), Path(args.work), args.workers, cfg)
        return 1 if status["failed"] else 0
    if args.cmd == "baseline":
        status = baseline(Path(args.prep), Path(args.out), Path(args.work), args.workers, cfg)
        return 1 if status["failed"] else 0
    report = collect_report(Path(args.root), cfg)
    save_report(Path(args.out), report)
    print(render(report))
    return 0 if report["status"] == "done" else 3


if __name__ == "__main__":
    raise SystemExit(main())
