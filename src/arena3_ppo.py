"""ARENA3 PPO — 교수 제안(Stable-Baselines3 계열) 비교 실험(220·221절, 설정 config/arena3_ppo_v1.json).

G1·S2와 같은 에피소드 실행기·선택 지점 SMDP·정확 보상·관측 119칸·6모드 행동·학습 상황을 쓰고, 학습 규칙만 PPO로 바꾼다.
SB3 라이브러리를 직접 쓰지 않는 이유: 보상이 에피소드가 끝난 뒤 tripinfo로 정산되는데(run_arena_episode),
SB3 환경의 step()은 매 단계 보상을 바로 돌려줘야 한다. 갱신 규칙과 기본값은 SB3 PPO를 따른다
(클립 0.2, GAE λ 0.95, 에포크 10, 미니배치 64, 학습률 3e-4, 미니배치마다 이득 정규화, 엔트로피 0, 행위자·평가자 64-64 분리).

- P1: M4 출발(G1과 같은 출발·같은 학습 상황). 처음 탐욕 정책은 M4이고, 후보 2개일 때 M4 확률 0.75(G1 탐험률 0.5와 같음).
- P2: 같은 난수의 S2A_pre에서 출발. 처음 탐욕 정책 = S2A_pre.
- 선택 사이 시간이 다르므로 할인은 0.99^(경과 5초 단계 수), GAE 계수는 (0.99·0.95)^(경과 단계 수).
- 처음 두 번의 갱신은 평가자만 몬테카를로 반환값으로 맞춘다(행위자는 그대로).
- 예산은 경과시간이 아니라 에피소드 수(난수당 448판, G1 수준). CPU가 적은 런타임에서도 경험 양이 같다(216절).

명령(Colab, 난수마다 따로; 끊기면 같은 명령으로 이어서):
  python src/arena3_ppo.py smoke    --variant P2 --seed 42 --out <Drive>/arena3/ppo_P2/smoke --work /content/ppo_smoke --init-root <Drive>/arena3/stage2/S2A_train
  python src/arena3_ppo.py train    --variant P2 --seed 42 --out <Drive>/arena3/ppo_P2/train --work /content/ppo --init-root <Drive>/arena3/stage2/S2A_train
  python src/arena3_ppo.py evaluate --variant P2 --train <Drive>/arena3/ppo_P2/train --out <Drive>/arena3/ppo_P2/eval --work /content/ppo_eval --workers 8
  python src/arena3_ppo.py judge    --baseline <Drive>/arena3/referee/baseline --ppo P1=<…>/ppo_P1/eval --ppo P2=<…>/ppo_P2/eval --reference G1=<…>/referee/G1 … --out <Drive>/arena3/ppo_judge
  python src/arena3_ppo.py confirm  --judge <Drive>/arena3/ppo_judge --baseline <Drive>/arena3/stage2/confirm/baseline --out <Drive>/arena3/ppo_judge/confirm --work /content/ppo_confirm --workers 8
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import math
import multiprocessing
import os
import platform
import random
import shutil
import statistics
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import arena3_common as ac  # noqa: E402
import arena3_g1 as g1  # noqa: E402
import arena3_referee as ref  # noqa: E402
import arena3_stage2_eval as s2e  # noqa: E402
import explore_weight as w  # noqa: E402

CONFIG = ROOT / "config/arena3_ppo_v1.json"
STAGE2_CONFIG = ROOT / "config/arena3_stage2_v1.json"
KIND = "arena3-ppo-v1"
VARIANTS = ("P1", "P2")
SOURCES = ("src/arena3_ppo.py", "src/arena3_g1.py", "src/arena3_stage2_train.py", "src/arena3_stage2_data.py",
           "src/arena3_stage2_eval.py", "config/arena3_ppo_v1.json", "config/arena3_stage2_v1.json")
OBS, N_MODES = ac.OBSERVATION_SIZE, ac.N_MODES


# ----------------------------------------------------------------------------- 설정
def config() -> dict:
    cfg = w.read(CONFIG)
    if cfg.get("experiment") != "ARENA3_PPO_V1":
        raise ValueError("PPO 설정 파일이 아닙니다.")
    if cfg["base_lock_sha256"] != w.file_hash(ac.CONFIG) or cfg["stage2_config_sha256"] != w.file_hash(STAGE2_CONFIG):
        raise ValueError("PPO 설정이 가리키는 ARENA3 잠금·2단계 설정이 지금 파일과 다릅니다.")
    return cfg


def effective_config(overrides: dict | None = None) -> dict:
    """점검(smoke)만 묶음 크기·예열 수를 줄여 쓴다. 본 학습은 overrides=None."""
    cfg = copy.deepcopy(config())
    for key, value in (overrides or {}).items():
        if key not in cfg:
            raise KeyError(f"설정에 없는 항목입니다: {key}")
        cfg[key] = value
    return cfg


def settings(variant: str, cfg: dict | None = None) -> dict:
    if variant not in VARIANTS:
        raise ValueError(f"알 수 없는 PPO 방식: {variant}")
    return {"kind": KIND, "variant": variant, "ppo_config": cfg if cfg is not None else config(), "g1": g1.settings(),
            "arena3_lock_sha256": w.file_hash(ac.CONFIG),
            "source_sha256": {p: w.file_hash(ROOT / p) for p in SOURCES}}


def _write(path: Path, value: dict) -> None:
    s2e._write(Path(path), value)


def _prepare_map_cache() -> None:
    """작업자 여러 개가 동시에 지도 캐시를 처음 만들다 부딪히지 않게(218절) 부모 프로세스에서 한 번 만든다."""
    from focused_eval import CACHE, dqn_template_network
    dqn_template_network(CACHE)


# ----------------------------------------------------------------------------- 행위자·평가자
class Actor(nn.Module):
    """행동 확률 = softmax((Q − Q[기준 모드]) / 온도 + 기준 편향·[기준 모드]), 유효 모드만.

    Q는 G1·S2와 같은 DuelingNet이다. 상대값만 쓰므로 DuelingNet의 값 머리는 상쇄되어 학습되지 않는다.
    """

    def __init__(self, temperature: float, anchor_bias: float, net: g1.DuelingNet | None = None):
        super().__init__()
        self.net = net if net is not None else g1.DuelingNet(tuple(g1.gcfg()["network"]))
        self.temperature = float(temperature)
        self.anchor_bias = nn.Parameter(torch.tensor(float(anchor_bias), dtype=torch.float32))

    def logits(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.reshape(-1, OBS)
        q = self.net(obs)
        anchor = g1.anchor_mode(obs)
        relative = q - q.gather(1, anchor.view(-1, 1))
        bonus = F.one_hot(anchor, N_MODES).to(relative.dtype) * self.anchor_bias
        mask = torch.from_numpy(g1.mode_mask(obs.detach().cpu().numpy()))
        return (relative / self.temperature + bonus).masked_fill(~mask, float("-inf"))


class Critic(nn.Module):
    """SB3 PPO 기본 평가자: 64-64 tanh, 직교 초기화(은닉 √2, 출력 1)."""

    def __init__(self, hidden=(64, 64)):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(OBS, hidden[0]), nn.Tanh(), nn.Linear(hidden[0], hidden[1]), nn.Tanh())
        self.out = nn.Linear(hidden[1], 1)
        for layer in self.body:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, math.sqrt(2.0))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.out.weight, 1.0)
        nn.init.zeros_(self.out.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.out(self.body(obs.reshape(-1, OBS))).squeeze(1)


def log_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(logits, dim=1)                # 막힌 모드는 −inf


def entropy(logits: torch.Tensor) -> torch.Tensor:
    """막힌 모드(−inf)에서 0·(−inf)가 생기지 않게 유효 모드만 더한다."""
    logp = torch.log_softmax(logits, dim=1)
    finite = torch.isfinite(logp)
    safe = torch.where(finite, logp, torch.zeros_like(logp))
    p = safe.exp() * finite
    return -(p * safe).sum(dim=1)


def greedy_mode(actor: Actor, obs) -> int:
    """평가용 탐욕 선택(가장 큰 값, 같으면 M4 쪽). P2의 처음 행위자는 S2A_pre의 g1.greedy_mode와 같은 선택을 낸다."""
    obs = np.asarray(obs, dtype=np.float32)
    with torch.no_grad():
        t = torch.from_numpy(obs).view(1, -1)
        z = actor.logits(t)[0].numpy().astype(np.float64)
        anchor = int(g1.anchor_mode(t)[0])
    if np.isnan(z).any() or not np.isfinite(z[anchor]):
        raise ValueError("행위자 출력이 유한하지 않습니다.")
    z[anchor] += 1e-6 / actor.temperature                # g1.greedy_mode의 Q 기준 1e-6과 같은 동률 처리
    return int(np.argmax(z))


class SampledPolicy:
    """학습용: 행위자 확률로 뽑고, 선택 지점마다 관측·모드·그때의 로그 확률을 남긴다."""
    name = "PPO"

    def __init__(self, actor: Actor, rng_seed: int):
        self.actor, self.rng = actor, random.Random(int(rng_seed))
        self.taken: list[dict] = []

    def decide(self, snap, ctx) -> int:
        obs = np.asarray(snap.observation, dtype=np.float32)
        with torch.no_grad():
            logp = log_probs(self.actor.logits(torch.from_numpy(obs).view(1, -1)))[0].numpy().astype(np.float64)
        valid = [m for m in range(N_MODES) if np.isfinite(logp[m])]
        if valid != [m for m in range(N_MODES) if snap.mask[m]]:
            raise ValueError("행위자의 유효 모드가 실행기의 모드 마스크와 다릅니다.")
        mode = self.rng.choices(valid, weights=[math.exp(logp[m]) for m in valid])[0]
        self.taken.append({"step": int(ctx.step), "observation": obs.tolist(), "mode": int(mode),
                           "log_prob": float(logp[mode])})
        return snap.green(mode)


class GreedyPolicy:
    def __init__(self, actor: Actor, name: str):
        self.actor, self.name = actor, name

    def decide(self, snap, ctx) -> int:
        return snap.green(greedy_mode(self.actor, snap.observation))


def _actor_bytes(actor: Actor) -> bytes:
    buf = io.BytesIO()
    torch.save(actor.state_dict(), buf)
    return buf.getvalue()


def _actor_from_bytes(raw: bytes, temperature: float) -> Actor:
    actor = Actor(temperature, 0.0)
    actor.load_state_dict(torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True))
    actor.eval()
    return actor


# ----------------------------------------------------------------------------- 에피소드 작업자
def episode_worker(job: dict) -> dict:
    """G1 작업자와 같은 실행·품질 검사, 정책만 SampledPolicy."""
    started = time.monotonic()
    work = Path(job["work"])
    try:
        w.configure()
        torch.set_num_threads(1)
        shutil.rmtree(work, ignore_errors=True)
        policy = SampledPolicy(_actor_from_bytes(job["actor"], job["temperature"]), job["rng_seed"])
        result = ac.run_arena_episode(job["situation"], work, policy, max_seconds=3600.0, trace=False)
        m = result.metrics
        if [c["step"] for c in result.choices] != [t["step"] for t in policy.taken]:
            raise RuntimeError("실행기의 선택 지점과 정책 호출 지점이 다릅니다.")
        errors = [k for k in g1.QUALITY if m.get(k, 0)]
        if not m["cost_check"].get("passed"):
            errors.append("cost_check")
        status = "ok" if not errors and not m["collisions_unexplained"] else ("discard" if not errors else "error")
        keep = {k: m.get(k) for k in ("mean_delay_s", "vehicles", "actual_total_delay", "choice_points", "long_waits",
                                       "left_east", "left_south", "collisions_unexplained", "route_sha256", "seconds")}
        return {"ok": status != "error", "status": status, "errors": errors, "metrics": keep, "choices": policy.taken,
                "rewards": result.rewards, "seconds": time.monotonic() - started,
                "collision_details": m.get("collision_details") if status == "discard" else None}
    except BaseException as error:
        return {"ok": False, "status": "error", "errors": [f"{type(error).__name__}: {error}"],
                "seconds": time.monotonic() - started}
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ----------------------------------------------------------------------------- 반환값·이득(SMDP)
def gae(rewards, discounts, factors, dones, values, next_values) -> np.ndarray:
    """선택 지점 k의 이득: δ_k = r_k + d_k·V(s_{k+1}) − V(s_k), A_k = δ_k + c_k·A_{k+1}.
    d_k = γ^n_k(할인), c_k = (γλ)^n_k(GAE 계수), n_k = 다음 선택까지 5초 단계 수. 끝 전이는 다음 값을 쓰지 않는다."""
    adv = np.zeros(len(rewards), dtype=np.float64)
    last = 0.0
    for k in range(len(rewards) - 1, -1, -1):
        live = 0.0 if dones[k] else 1.0
        delta = float(rewards[k]) + float(discounts[k]) * live * float(next_values[k]) - float(values[k])
        last = delta + float(factors[k]) * live * last
        adv[k] = last
    return adv


def episode_batch(result: dict, critic: Critic, ppo: dict, lam_per_5s: float) -> dict | None:
    """에피소드 하나 → 선택 지점별 관측·모드·옛 로그 확률·이득·반환값. 선택 지점이 없으면 None."""
    choices = result["choices"]
    if not choices:
        return None
    gamma = float(ppo["gamma_per_5s"])
    transitions = g1.smdp_transitions(choices, result["rewards"], gamma, 1)
    steps = [int(c["step"]) for c in choices] + [len(result["rewards"])]
    gaps = np.diff(np.asarray(steps, dtype=np.float64))
    discounts = np.asarray([t["discount"] for t in transitions], np.float64)
    if not np.allclose(discounts, gamma ** gaps, rtol=1e-9, atol=0.0):
        raise ValueError("SMDP 할인과 선택 간격이 맞지 않습니다.")
    obs = np.stack([t["obs"] for t in transitions]).astype(np.float32)
    nxt = np.stack([t["next_obs"] for t in transitions]).astype(np.float32)
    with torch.no_grad():
        values = critic(torch.from_numpy(obs)).numpy().astype(np.float64)
        next_values = critic(torch.from_numpy(nxt)).numpy().astype(np.float64)
    rewards = np.asarray([t["reward"] for t in transitions], np.float64)
    dones = np.asarray([t["done"] for t in transitions], bool)
    adv = gae(rewards, discounts, (gamma * float(lam_per_5s)) ** gaps, dones, values, next_values)
    return {"obs": obs, "mode": np.asarray([c["mode"] for c in choices], np.int64),
            "log_prob": np.asarray([c["log_prob"] for c in choices], np.float64),
            "advantages": adv, "returns": adv + values, "values": values}


def concat(parts: list[dict]) -> dict:
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def non_anchor(choices: list[dict]) -> int:
    """M4와 다른 녹색을 고른 선택 수(기준 모드가 아닌 유효 모드 = M4와 다른 녹색)."""
    if not choices:
        return 0
    anchors = g1.anchor_mode(torch.from_numpy(np.asarray([c["observation"] for c in choices], np.float32))).numpy()
    return int(sum(int(c["mode"]) != int(a) for c, a in zip(choices, anchors)))


# ----------------------------------------------------------------------------- 학습기
class Learner:
    def __init__(self, actor: Actor, critic: Critic, seed: int, ppo: dict):
        self.actor, self.critic, self.seed = actor, critic, int(seed)
        self.actor_opt = torch.optim.Adam(actor.parameters(), lr=float(ppo["learning_rate"]), eps=float(ppo["adam_eps"]))
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr=float(ppo["learning_rate"]), eps=float(ppo["adam_eps"]))
        self.rng = np.random.default_rng(self.seed + 300_000)


def ppo_update(learner: Learner, batch: dict, ppo: dict, update_actor: bool) -> dict:
    """SB3 PPO.train()과 같은 순서: 에포크마다 섞고, 미니배치마다 이득 정규화 → 클립 목적 → 평가자 MSE.
    행위자·평가자는 옵티마이저와 기울기 자르기를 따로 쓴다(설정 ppo.optimizers)."""
    n = len(batch["obs"])
    obs = torch.from_numpy(batch["obs"])
    mode = torch.from_numpy(batch["mode"])
    old = torch.from_numpy(batch["log_prob"]).float()
    adv_all = torch.from_numpy(batch["advantages"]).float()
    ret = torch.from_numpy(batch["returns"]).float()
    clip, size = float(ppo["clip_range"]), int(ppo["batch_size"])
    log = {"policy_loss": [], "value_loss": [], "approx_kl": [], "clip_fraction": [], "entropy": []}
    stopped = False
    for _ in range(int(ppo["n_epochs"])):
        order = learner.rng.permutation(n)
        for start in range(0, n, size):
            ix = torch.from_numpy(order[start:start + size])
            if update_actor:
                logits = learner.actor.logits(obs[ix])
                logp = log_probs(logits).gather(1, mode[ix].view(-1, 1)).squeeze(1)
                log_ratio = logp - old[ix]
                ratio = torch.exp(log_ratio)
                adv = adv_all[ix]
                if ppo["normalize_advantage"] and len(ix) > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg = -torch.min(ratio * adv, torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv).mean()
                ent = entropy(logits).mean()
                loss = pg - float(ppo["ent_coef"]) * ent
                if not bool(torch.isfinite(loss)):
                    raise ValueError("PPO 행위자 손실이 유한하지 않습니다.")
                with torch.no_grad():
                    kl = float(((ratio - 1.0) - log_ratio).mean())
                if ppo["target_kl"] is not None and kl > 1.5 * float(ppo["target_kl"]):
                    stopped = True
                    break
                learner.actor_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(learner.actor.parameters(), float(ppo["max_grad_norm"]))
                learner.actor_opt.step()
                log["policy_loss"].append(float(pg.detach()))
                log["approx_kl"].append(kl)
                log["clip_fraction"].append(float(((ratio.detach() - 1.0).abs() > clip).float().mean()))
                log["entropy"].append(float(ent.detach()))
            value = learner.critic(obs[ix])
            vloss = float(ppo["vf_coef"]) * F.mse_loss(value, ret[ix])
            if not bool(torch.isfinite(vloss)):
                raise ValueError("PPO 평가자 손실이 유한하지 않습니다.")
            learner.critic_opt.zero_grad()
            vloss.backward()
            nn.utils.clip_grad_norm_(learner.critic.parameters(), float(ppo["max_grad_norm"]))
            learner.critic_opt.step()
            log["value_loss"].append(float(vloss.detach()))
        if stopped:
            break
    var = float(np.var(batch["returns"]))
    explained = 1.0 - float(np.var(batch["returns"] - batch["values"])) / var if var > 0 else None
    out = {"decisions": int(n), "actor_updated": bool(update_actor), "stopped_by_kl": stopped,
           "explained_variance": explained, "advantage_std": float(np.std(batch["advantages"]))}
    out.update({k: (float(np.mean(v)) if v else None) for k, v in log.items()})
    return out


# ----------------------------------------------------------------------------- 저장
def save_model(folder: Path, learner: Learner, state: dict, variant: str, stage: str, cfg: dict) -> str:
    """심판이 읽는 모델 폴더: model.pt(행위자·평가자) + manifest.json(해시·설정)."""
    folder = Path(folder)
    folder.parent.mkdir(parents=True, exist_ok=True)
    tmp = folder.with_name(f".{folder.name}.{uuid.uuid4().hex}.tmp")
    tmp.mkdir()
    meta = {"kind": KIND, "settings": settings(variant, cfg), "seed": learner.seed, "variant": variant, "stage": stage,
            "episodes": state["episodes"], "updates": state["updates"], "choices": state["choices"],
            "init_identity": state["init_identity"]}
    torch.save({**meta, "actor": learner.actor.state_dict(), "critic": learner.critic.state_dict()}, tmp / "model.pt")
    sha = w.file_hash(tmp / "model.pt")
    _write(tmp / "manifest.json", {**meta, "model_sha256": sha})
    if folder.exists():
        shutil.rmtree(folder)
    os.replace(tmp, folder)
    return sha


def model_identity(folder: Path) -> str:
    folder = Path(folder)
    manifest = w.read(folder / "manifest.json")
    if manifest.get("kind") != KIND or manifest.get("settings") != settings(manifest.get("variant")):
        raise ValueError(f"PPO 모델의 종류·설정·코드가 지금과 다릅니다: {folder}")
    sha = w.file_hash(folder / "model.pt")
    if sha != manifest["model_sha256"]:
        raise ValueError(f"PPO 모델 파일 해시가 다릅니다: {folder}")
    return sha


def load_actor(folder: Path) -> Actor:
    model_identity(folder)
    payload = torch.load(Path(folder) / "model.pt", map_location="cpu", weights_only=False)
    actor = Actor(payload["settings"]["ppo_config"]["actor"]["temperature"], 0.0)
    actor.load_state_dict(payload["actor"])
    actor.eval()
    return actor


def save_checkpoint(seed_dir: Path, learner: Learner, state: dict, variant: str, cfg: dict) -> None:
    payload = {"kind": KIND, "settings": settings(variant, cfg), "seed": learner.seed, "variant": variant,
               "state": state, "actor": learner.actor.state_dict(), "critic": learner.critic.state_dict(),
               "actor_opt": learner.actor_opt.state_dict(), "critic_opt": learner.critic_opt.state_dict(),
               "rng": learner.rng.bit_generator.state, "torch_rng": torch.get_rng_state()}
    tmp = seed_dir / "checkpoint.pt.tmp"
    torch.save(payload, tmp)
    sha = w.file_hash(tmp)
    os.replace(tmp, seed_dir / "checkpoint.pt")
    _write(seed_dir / "checkpoint.json", {"sha256": sha, "episodes": state["episodes"], "updates": state["updates"],
                                          "updated_unix": time.time()})


def load_checkpoint(seed_dir: Path, variant: str, seed: int, cfg: dict) -> tuple[Learner, dict] | None:
    path = seed_dir / "checkpoint.pt"
    if not path.exists():
        return None
    meta = w.read(seed_dir / "checkpoint.json")
    if w.file_hash(path) != meta["sha256"]:
        raise ValueError("체크포인트 해시가 다릅니다.")
    p = torch.load(path, map_location="cpu", weights_only=False)
    if p.get("kind") != KIND or p.get("settings") != settings(variant, cfg) or p.get("seed") != seed or p.get("variant") != variant:
        raise ValueError("체크포인트의 종류·설정·코드·난수가 다릅니다. 기존 결과를 보존하고 새 폴더를 쓰세요.")
    if p["state"]["episodes"] != meta["episodes"]:
        raise ValueError("체크포인트 기록이 서로 다릅니다.")
    actor = Actor(cfg["actor"]["temperature"], 0.0)
    actor.load_state_dict(p["actor"])
    critic = Critic(tuple(cfg["critic"]["network"]))
    critic.load_state_dict(p["critic"])
    learner = Learner(actor, critic, seed, cfg["ppo"])
    learner.actor_opt.load_state_dict(p["actor_opt"])
    learner.critic_opt.load_state_dict(p["critic_opt"])
    learner.rng.bit_generator.state = p["rng"]
    torch.set_rng_state(p["torch_rng"])
    return learner, p["state"]


# ----------------------------------------------------------------------------- 출발 모델
def initial_actor(variant: str, seed: int, cfg: dict, init_root: Path | None) -> tuple[Actor, dict | None]:
    spec = cfg["variants"][variant]
    torch.manual_seed(int(seed))
    if variant == "P1":
        return Actor(cfg["actor"]["temperature"], spec["anchor_bias"]), None
    if init_root is None:
        raise ValueError("P2는 --init-root(2단계 S2A_train 폴더)가 필요합니다.")
    import arena3_stage2_train as s2
    folder = Path(init_root) / f"seed_{seed}" / "pretrained"
    manifest = w.read(folder / "manifest.json")
    if manifest.get("variant") != "advantage" or manifest.get("stage") != "pretrained" or manifest.get("seed") != seed:
        raise ValueError("P2의 출발 모델은 같은 난수의 S2A_pre(이득 회귀 사전학습)여야 합니다.")
    identity = s2.model_identity(folder)
    actor = Actor(cfg["actor"]["temperature"], spec["anchor_bias"], net=s2.load_model(folder))
    return actor, {"model": "S2A_pre", "folder": str(folder.resolve()), "sha256": identity}


# ----------------------------------------------------------------------------- 학습
def train(variant: str, seed: int, out: Path, work: Path, *, init_root: Path | None = None, executor=None,
          worker=episode_worker, max_updates: int | None = None, overrides: dict | None = None) -> dict:
    cfg = effective_config(overrides)
    if sys.platform == "darwin" and worker is episode_worker:
        raise RuntimeError("맥에서는 SUMO를 실행하지 않습니다. 실제 학습은 Colab에서 실행하세요.")
    if variant not in VARIANTS or seed not in cfg["seeds"]:
        raise ValueError(f"PPO 방식은 {VARIANTS}, 난수는 {cfg['seeds']} 중 하나여야 합니다.")
    seed_dir = Path(out) / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    spend = g1.SpendLog(seed_dir / "spend.jsonl")
    loaded = load_checkpoint(seed_dir, variant, seed, cfg)
    if loaded is None:
        actor, init_identity = initial_actor(variant, seed, cfg, init_root)
        torch.manual_seed(int(seed) + 1)
        critic = Critic(tuple(cfg["critic"]["network"]))
        learner = Learner(actor, critic, seed, cfg["ppo"])
        state = {"episodes": 0, "updates": 0, "choices": 0, "discarded": [], "history": [], "updates_log": [],
                 "init_identity": init_identity}
        save_model(seed_dir / "initial", learner, state, variant, "initial", cfg)
        save_checkpoint(seed_dir, learner, state, variant, cfg)
    else:
        learner, state = loaded
    budget = int(cfg["episodes_per_seed"])
    temperature = float(cfg["actor"]["temperature"])
    train_dates = set(ac.config()["dates"]["train"])

    def status(finished: bool = False) -> dict:
        last = state["updates_log"][-1] if state["updates_log"] else {}
        result = {"variant": variant, "seed": seed, "finished": finished, "episodes": state["episodes"],
                  "budget_episodes": budget, "updates": state["updates"], "choices": state["choices"],
                  "discarded": len(state["discarded"]), "cpu_hours": spend.total() / 3600,
                  "last_update": last, "init_identity": state["init_identity"], "updated_unix": time.time()}
        _write(seed_dir / "status.json", result)
        return result

    status()
    done = 0
    if state["episodes"] < budget and (max_updates is None or max_updates > 0):
        if worker is episode_worker:
            _prepare_map_cache()
        pool = executor or ProcessPoolExecutor(max_workers=int(cfg["parallel_episodes"]),
                                               mp_context=multiprocessing.get_context("spawn"),
                                               initializer=g1._worker_init)
        with pool:
            while state["episodes"] < budget and (max_updates is None or done < max_updates):
                raw = _actor_bytes(learner.actor)
                jobs = []
                for b in range(min(int(cfg["episodes_per_update"]), budget - state["episodes"])):
                    idx = state["episodes"] + b
                    s = g1.scenario_for(seed, idx)              # G1과 같은 학습 상황
                    if s["date"] not in train_dates:
                        raise ValueError("학습 날짜가 아닌 상황이 뽑혔습니다.")
                    jobs.append({"situation": s, "work": str(Path(work) / variant / f"seed_{seed}" / f"e{idx}"),
                                 "actor": raw, "temperature": temperature,
                                 "rng_seed": ac.seed32("PPO_SAMPLE", variant, seed, idx)})
                results = list(pool.map(worker, jobs))
                bad = [(job, r) for job, r in zip(jobs, results) if not r["ok"]]
                if bad:
                    spend.add(sum(float(r.get("seconds") or 0.0) for r in results), f"failed update e{state['episodes']}")
                    job, r = bad[0]
                    raise RuntimeError(f"PPO {variant} seed {seed} 에피소드 {ac.situation_id(job['situation'])} 실패: {r['errors']}")
                learn_started = time.monotonic()
                warm = state["updates"] < int(cfg["critic_warmup_updates"])
                lam = 1.0 if warm else float(cfg["ppo"]["gae_lambda_per_5s"])
                parts = []
                for job, r in zip(jobs, results):
                    idx = state["episodes"]
                    state["episodes"] += 1
                    sid = ac.situation_id(job["situation"])
                    if r["status"] == "discard":
                        state["discarded"].append({"episode": idx, "situation": sid, "collision": r["collision_details"]})
                        if len(state["discarded"]) > int(cfg["max_discarded"]):
                            raise RuntimeError(f"버린 에피소드가 {cfg['max_discarded']}개를 넘었습니다.")
                        continue
                    part = episode_batch(r, learner.critic, cfg["ppo"], lam)
                    if part is not None:
                        parts.append(part)
                    state["choices"] += len(r["choices"])
                    state["history"].append({"episode": idx, "situation": sid, **r["metrics"], "decisions": len(r["choices"]),
                                             "non_m4_decisions": non_anchor(r["choices"]), "update": state["updates"]})
                info = ppo_update(learner, concat(parts), cfg["ppo"], update_actor=not warm) if parts else \
                    {"decisions": 0, "actor_updated": False}
                state["updates"] += 1
                state["updates_log"].append({"update": state["updates"], "warmup": warm, "episodes": state["episodes"],
                                             "learn_seconds": time.monotonic() - learn_started, **info})
                spend.add(sum(float(r["seconds"]) for r in results) + time.monotonic() - learn_started,
                          f"update {state['updates']}")
                save_checkpoint(seed_dir, learner, state, variant, cfg)
                done += 1
                s_now = status()
                recent = [h["mean_delay_s"] for h in state["history"][-len(jobs):] if h.get("mean_delay_s") is not None]
                print(f"PPO {variant} seed {seed}: 갱신 {state['updates']}{'(평가자 예열)' if warm else ''}, "
                      f"에피소드 {state['episodes']}/{budget}, 선택 {state['choices']}, 최근 지연 "
                      f"{statistics.fmean(recent) if recent else None}, KL {info.get('approx_kl')}, "
                      f"설명 분산 {info.get('explained_variance')}, 작업시간 {s_now['cpu_hours']:.2f}h", flush=True)
    finished = state["episodes"] >= budget
    if finished:
        final = seed_dir / "final"
        if not (final / "manifest.json").exists():
            save_model(final, learner, state, variant, "final", cfg)
    return status(finished)


# ----------------------------------------------------------------------------- 평가
def declaration(name: str, folders: list[Path], part: str) -> dict:
    seeds_cfg = config()["seeds"]
    if name not in VARIANTS or len(folders) != len(seeds_cfg):
        raise ValueError("PPO 학습기는 P1/P2의 세 모델이어야 합니다.")
    identities, seeds, inits = [], [], []
    for folder in folders:
        identities.append(model_identity(folder))
        manifest = w.read(Path(folder) / "manifest.json")
        if manifest.get("variant") != name or manifest.get("stage") != "final":
            raise ValueError("평가 이름과 학습 방식·단계가 다릅니다(최종 모델만 평가).")
        seeds.append(manifest.get("seed"))
        inits.append(manifest.get("init_identity"))
    if seeds != seeds_cfg:
        raise ValueError("평가 모델의 순서는 난수 42·43·44여야 합니다.")
    return {"name": name, "part": part, "model_dirs": [str(Path(p).resolve()) for p in folders],
            "model_identities": identities, "seeds": seeds, "init_identities": inits}


def model_dirs(train_root: Path) -> list[Path]:
    return [Path(train_root) / f"seed_{s}" / "final" for s in config()["seeds"]]


def eval_worker(job: dict) -> dict:
    started = time.monotonic()
    work = Path(job["work"])
    try:
        if platform.system() == "Darwin":
            raise RuntimeError("실제 SUMO 평가는 Colab에서만 실행합니다. 맥에서는 judge 명령만 사용하세요.")
        w.configure()
        torch.set_num_threads(1)
        shutil.rmtree(work, ignore_errors=True)
        if job["kind"] != "ppo":
            raise ValueError(f"알 수 없는 작업 종류: {job['kind']}")
        if model_identity(Path(job["model_dir"])) != job["model_identity"]:
            raise ValueError("작업 시작 후 모델이 바뀌었습니다.")
        policy = GreedyPolicy(load_actor(Path(job["model_dir"])), name=job["policy"])
        result = ac.run_arena_episode(job["situation"], work, policy, max_seconds=1800.0)
        expected = job["situation"].get("route_sha256")
        if expected and result.metrics["route_sha256"] != expected:
            raise ValueError("평가 차량 파일의 SHA가 잠금과 다릅니다.")
        ac.write_run(Path(job["target"]), result,
                     {"policy": job["policy"], "fingerprint": job["fingerprint"],
                      "model_identity": job.get("model_identity"), "job_seconds": time.monotonic() - started})
        return {"ok": True, "seconds": time.monotonic() - started, "name": f"{job['policy']}/{job['sid']}"}
    except BaseException as error:
        return {"ok": False, "seconds": time.monotonic() - started, "name": f"{job['policy']}/{job['sid']}",
                "error": f"{type(error).__name__}: {error}"}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def evaluate(name: str, folders: list[Path], out: Path, work: Path, workers: int, part: str = "eval", *,
             executor=None, worker=eval_worker) -> dict:
    situations = s2e.situations_for(part)
    out = Path(out)
    fp = ref.register(out, extra_files=SOURCES)
    decl = declaration(name, [Path(p) for p in folders], part)
    s2e._freeze(out / "models.json", decl)
    jobs = []
    for i, (folder, identity) in enumerate(zip(folders, decl["model_identities"])):
        jobs += ref.jobs_for(out, Path(work), fp, f"{name}_m{i}", situations, model_dir=folder,
                             model_identity=identity, kind="ppo")
    if worker is eval_worker and executor is None:
        _prepare_map_cache()
    status = ref.run_jobs(jobs, max(1, workers), ref.Ledger(out, name), worker=worker, executor=executor)
    _write(out / "status" / f"{name}_{part}.json", {**status, "total": len(jobs), "part": part, "updated_unix": time.time()})
    return status


# ----------------------------------------------------------------------------- 판정
def paired_difference(a: dict, b: dict, t975: float) -> dict:
    """같은 상황·같은 차량에서 날짜별 M4 대비 이득(판정과 같은 차량 가중 정의)의 차 a − b."""
    dates = sorted(a["per_date_gain_s"])
    if dates != sorted(b["per_date_gain_s"]):
        raise ValueError("짝 비교할 두 후보의 날짜가 다릅니다.")
    diffs = {d: float(a["per_date_gain_s"][d]) - float(b["per_date_gain_s"][d]) for d in dates}
    mean, low, high = ref.t_interval(list(diffs.values()), t975)
    return {"mean_s": mean, "t95_s": [low, high], "positive_dates": sum(v > 0 for v in diffs.values()),
            "dates": len(dates), "per_date_s": diffs}


def _ppo_runs(spec: str, situations: list[dict], part: str, base_fp: str):
    name, raw = spec.split("=", 1)
    folder = Path(raw)
    if name not in VARIANTS:
        raise ValueError(f"PPO 후보 이름은 {VARIANTS} 중 하나여야 합니다: {name}")
    if not (folder / "models.json").exists():
        return name, None, None
    decl = w.read(folder / "models.json")
    if decl.get("name") != name or decl.get("part") != part:
        raise ValueError("모델 선언의 이름/배치가 요청과 다릅니다.")
    if decl != declaration(name, [Path(p) for p in decl["model_dirs"]], part):
        raise ValueError("평가에 사용한 PPO 모델이 바뀌었습니다.")
    if s2e.folder_record(folder)["evaluation_fingerprint"] != base_fp:
        raise ValueError(f"{name} 평가 지문이 기준선과 다릅니다.")
    runs = [s2e.verified_runs(folder, f"{name}_m{i}", situations, model_identity=x, complete=False)
            for i, x in enumerate(decl["model_identities"])]
    return name, decl, runs


def collect(baseline: Path, ppo_specs: list[str], reference_specs: list[str], part: str = "eval") -> dict:
    cfg = ac.config()
    situations = s2e.situations_for(part)
    base_record = s2e.verify_baseline(baseline)
    m4 = s2e.verified_runs(baseline, "M4", situations)
    p0 = s2e.verified_runs(baseline, "P0", situations)
    for policy, rows in (("M4", m4), ("P0", p0)):
        problems = {m["situation"]: ref.quality_errors(m) for m in rows if ref.quality_errors(m)}
        if problems:
            raise ValueError(f"기준선 {policy}의 품질 문제가 남아 있습니다: {problems}")
    results, pending, models = {}, {}, {}
    for spec in ppo_specs:
        name, decl, runs = _ppo_runs(spec, situations, part, base_record["evaluation_fingerprint"])
        if decl is None:
            pending[name] = "아직 평가 결과 없음"
        elif any(len(r) != len(situations) for r in runs):
            pending[name] = {"counts": [len(r) for r in runs], "expected": len(situations)}
        else:
            results[name] = ref.judge(runs, m4, p0, cfg, situations)
            models[name] = decl
    for spec in reference_specs:                         # G1·S2A_pre·S2A·S2B 등 이미 끝난 후보(같은 심판)
        name, folder, runs = s2e.candidate_models(spec, situations, part)
        want = 1 if name in ("W4", "BPRIME") else cfg["judge"]["learner_models"]
        if not runs or len(runs) != want or any(len(r) != len(situations) for r in runs):
            pending[name] = {"counts": [len(r) for r in runs] if runs else [], "expected": len(situations)}
            continue
        if s2e.folder_record(folder)["evaluation_fingerprint"] != base_record["evaluation_fingerprint"]:
            raise ValueError(f"{name} 평가 지문이 기준선과 다릅니다.")
        results[name] = ref.judge(runs, m4, p0, cfg, situations)
    t975 = float(cfg["judge"]["t975"])
    paired = {f"{a}-{b}": paired_difference(results[a], results[b], t975)
              for a, b in config()["rules"]["paired_comparisons"] if a in results and b in results}
    return {"results": results, "pending": pending, "models": models, "paired": paired, "part": part,
            "evaluation_fingerprint": base_record["evaluation_fingerprint"],
            "sensitivity_m4_vs_p0": ref.judge([p0], m4, None, cfg, situations),
            "lock_sha256": w.file_hash(ac.CONFIG), "ppo_config_sha256": w.file_hash(CONFIG), "updated_unix": time.time()}


def render(report: dict, title: str) -> str:
    text = ref.render(report["results"], report["sensitivity_m4_vs_p0"]).replace("# ARENA3 심판 결과", f"# ARENA3 PPO — {title}")
    if report["paired"]:
        text += "\n## 짝 비교(같은 상황, 날짜 7개, 초/대, +면 앞쪽이 짧음)\n\n| 비교 | 평균 | t95 | 앞쪽이 나은 날짜 |\n|---|---|---|---|\n"
        for key, p in report["paired"].items():
            text += f"| {key.replace('-', ' − ')} | {p['mean_s']:+.3f} | [{p['t95_s'][0]:+.3f}, {p['t95_s'][1]:+.3f}] | {p['positive_dates']}/{p['dates']} |\n"
    if report["pending"]:
        text += "\n아직 평가 중: " + json.dumps(report["pending"], ensure_ascii=False) + "\n"
    text += "\nP1 − G1은 같은 출발·같은 학습 상황의 PPO 대 DQN, P2 − S2A_pre는 PPO 미세조정의 몫이다. 반복 사용 날짜의 개발 비교.\n"
    return text


def save_report(out: Path, report: dict, title: str) -> None:
    out = Path(out)
    _write(out / "judge.json", report)
    (out / "summary.md").write_text(render(report, title))


# ----------------------------------------------------------------------------- C1·C2 확인
def choose_winner(results: dict) -> str | None:
    eligible = [(n, r) for n, r in results.items() if n in VARIANTS and r.get("status") == "pass"]
    return min(eligible, key=lambda x: (-float(x[1]["gain_date_mean_s"]), x[0]))[0] if eligible else None


def confirm(judge_dir: Path, baseline: Path, out: Path, work: Path, workers: int, *, executor=None,
            worker=eval_worker) -> dict:
    """개발 판정에서 통과한 P1/P2 중 날짜 평균이 큰 하나만 C1·C2에서 한 번 잰다. 고른 뒤에는 다시 고르지 않는다."""
    out = Path(out)
    selection_path = out / "selection.json"
    if selection_path.exists():
        selected = w.read(selection_path)
        if selected["winner"]:
            current = declaration(selected["winner"], [Path(p) for p in selected["model_dirs"]], "eval")
            if current["model_identities"] != selected["model_identities"]:
                raise ValueError("확인 전에 고정한 모델이 바뀌었습니다.")
    else:
        report = w.read(Path(judge_dir) / "judge.json")
        if any(n in VARIANTS for n in report["pending"]):
            raise ValueError("P1·P2 개발 평가가 끝나기 전에 승자를 고를 수 없습니다.")
        winner = choose_winner(report["results"])
        selected = {"winner": winner, "judged": sorted(n for n in VARIANTS if n in report["results"]),
                    "lock_sha256": w.file_hash(ac.CONFIG), "evaluation_fingerprint": report["evaluation_fingerprint"],
                    "development_report_sha256": w.digest(report), "model_dirs": [], "model_identities": [],
                    "selection_unix": time.time()}
        if winner:
            decl = report["models"][winner]
            current = declaration(winner, [Path(p) for p in decl["model_dirs"]], "eval")
            if current["model_identities"] != decl["model_identities"]:
                raise ValueError("승자 평가 모델과 지금 모델이 다릅니다.")
            selected.update(model_dirs=decl["model_dirs"], model_identities=decl["model_identities"])
        _write(selection_path, selected)
    if not selected["winner"]:
        result = {"status": "no_passing_ppo", "selection": selected}
        _write(out / "confirmation.json", result)
        return result
    s2e.verify_baseline(baseline, selected["evaluation_fingerprint"])
    name = selected["winner"]
    status = evaluate(name, [Path(p) for p in selected["model_dirs"]], out / name, Path(work) / name, workers,
                      "confirmation", executor=executor, worker=worker)
    if status["failed"]:
        return {"status": "candidate_failed", "failed": status["failed"]}
    report = collect(baseline, [f"{name}={out / name}"], [], part="confirmation")
    save_report(out, report, "C1·C2 확인")
    result = {"status": report["results"][name]["status"], "selection": selected, "result": report["results"][name]}
    _write(out / "confirmation.json", result)
    return result


# ----------------------------------------------------------------------------- 명령
SMOKE_OVERRIDES = {"episodes_per_seed": 4, "episodes_per_update": 2, "critic_warmup_updates": 1}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for command in ("train", "smoke"):
        p = sub.add_parser(command)
        p.add_argument("--variant", choices=VARIANTS, required=True)
        p.add_argument("--seed", type=int, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--work", type=Path, required=True)
        p.add_argument("--init-root", type=Path, default=None, help="P2: 2단계 S2A_train 폴더")
    p = sub.add_parser("evaluate")
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("judge")
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--ppo", action="append", default=[], help="P1=<평가 폴더> 또는 P2=<평가 폴더>")
    p.add_argument("--reference", action="append", default=[], help="G1=…, S2A_pre=… 등 이미 끝난 후보")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("confirm")
    p.add_argument("--judge", type=Path, required=True, help="개발 판정 폴더(judge.json)")
    p.add_argument("--baseline", type=Path, required=True, help="C1·C2 기준선(2단계 confirm/baseline)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    w.configure()
    torch.set_num_threads(1)
    if args.cmd in ("train", "smoke", "evaluate", "confirm") and platform.system() == "Darwin":
        parser.error("맥에서는 SUMO를 실행하지 않습니다. 이 명령은 Colab 노트북에서 실행하세요.")
    if args.cmd in ("train", "smoke"):
        ref.register(args.out, extra_files=SOURCES)
        overrides = SMOKE_OVERRIDES if args.cmd == "smoke" else None
        result = train(args.variant, args.seed, args.out, args.work, init_root=args.init_root, overrides=overrides,
                       max_updates=2 if args.cmd == "smoke" else None)
        if args.cmd == "smoke":
            log = w.read(args.out / f"seed_{args.seed}" / "status.json")
            ok = (result["episodes"] == SMOKE_OVERRIDES["episodes_per_seed"] and result["updates"] == 2
                  and result["choices"] > 0 and not result["discarded"] and log["last_update"].get("actor_updated"))
            if not ok:
                raise RuntimeError(f"PPO 점검이 기대와 다릅니다: {result}")
            _write(args.out / "smoke.json", {"passed": True, **result})
        print(json.dumps(w.plain(result), ensure_ascii=False))
        return 0
    if args.cmd == "evaluate":
        status = evaluate(args.variant, model_dirs(args.train), args.out, args.work, args.workers)
        return 1 if status["failed"] else 0
    if args.cmd == "judge":
        report = collect(args.baseline, args.ppo, args.reference)
        save_report(args.out, report, "개발 평가")
        print(render(report, "개발 평가"))
        return 0 if not any(n in VARIANTS for n in report["pending"]) else 2
    result = confirm(args.judge, args.baseline, args.out, args.work, args.workers)
    print(json.dumps(w.plain(result), ensure_ascii=False, indent=2))
    return 1 if result["status"] == "candidate_failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
