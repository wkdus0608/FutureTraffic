"""ARENA3 G1 — GPT 안: 선택 지점끼리만 잇는 SMDP 듀얼링 DQN(M4 출발, 208절).

N3(explore_night_train)와 다른 점만 바꿨다.
- 학습 단위: 5초 단계가 아니라 '선택 지점 → 다음 선택 지점'. 재생 자료는 선택 지점만 담는다.
  두 지점 사이의 실제 지연 증가(WEIGHT1과 같은 정확 정산, 꼬리는 마지막 단계에 접음)를 5초당 0.99로 할인해 합치고,
  다음 지점까지의 할인은 0.99^(경과 단계 수). 마지막 선택 지점의 전이는 에피소드 끝까지 합쳐 done.
- 탐험률: 선택 결정 수 기준 0.5 → 0.1(첫 3,000번), 이후 0.1.
- 모드 6개(W4 포함)와 공통 관측 119칸. 출발점은 M4: 이점 머리를 0으로 시작하고 기준 모드를 M4의 대표 모드로 둔다.
- 학습 상황: 학습 8일 × 06:00~21:45의 15분 칸 × 에피소드마다 새 도착 배치. 난수마다 20 CPU-시간(3개 합 60).
- 같은 정책 가중치로 에피소드 2개를 동시에 모은 뒤 차례로 배운다(정책 지연은 한 묶음).

명령(Colab, 난수마다 따로 실행하고 끊기면 같은 명령으로 이어서):
  python src/arena3_g1.py train --seed 42 --out <Drive>/arena3/G1_train --work /content/g1
학습이 끝나면 <out>/seed_<난수>/final 이 심판 평가(arena3_referee.py learner)의 모델 폴더다.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import multiprocessing
import os
import random
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import arena3_common as ac  # noqa: E402
import explore_weight as w  # noqa: E402

KIND = "arena3-g1-smdp-dueling-v1"
OBS, N_MODES = ac.OBSERVATION_SIZE, ac.N_MODES
POLICY = "G1"


def gcfg() -> dict:
    return ac.config()["g1"]


def settings() -> dict:
    return {"kind": KIND, **gcfg(), "observation_size": OBS, "modes": [list(m) for m in ac.MODES6]}


# ----------------------------------------------------------------------------- 관측에서 모드 정보
def mode_mask(obs) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    mask = np.concatenate([obs[..., ac.BASE_MASK], obs[..., ac.W4_MASK:ac.W4_MASK + 1]], axis=-1) > 0.5
    if not np.all(mask.any(axis=-1)):
        raise ValueError("유효한 모드가 없는 관측입니다.")
    return mask


def mode_greens(obs: torch.Tensor) -> torch.Tensor:
    obs = obs.reshape(-1, OBS)
    base = obs[:, ac.BASE_ONEHOT].reshape(-1, 5, 4)
    w4 = obs[:, ac.W4_ONEHOT].reshape(-1, 1, 4)
    return torch.cat([base, w4], dim=1).argmax(dim=2)


def anchor_mode(obs: torch.Tensor) -> torch.Tensor:
    """M4와 같은 첫 녹색을 내는 가장 작은 유효 모드(N3의 baseline_mode를 6모드로)."""
    obs = obs.reshape(-1, OBS)
    greens = mode_greens(obs)
    valid = torch.cat([obs[:, ac.BASE_MASK], obs[:, ac.W4_MASK:ac.W4_MASK + 1]], dim=1) > 0.5
    same = (greens == greens[:, ac.M4:ac.M4 + 1]) & valid
    order = torch.arange(N_MODES, 0, -1, dtype=torch.float32, device=obs.device)
    return (same.float() * order).argmax(dim=1)


class DuelingNet(nn.Module):
    def __init__(self, hidden=(64, 64)):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(OBS, hidden[0]), nn.ReLU(), nn.Linear(hidden[0], hidden[1]), nn.ReLU())
        self.value = nn.Linear(hidden[1], 1)
        self.advantage = nn.Linear(hidden[1], N_MODES)
        for layer in (self.value, self.advantage):        # 학습 전 탐욕 정책 = M4
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        hidden = self.trunk(obs)
        adv = self.advantage(hidden)
        return self.value(hidden) + adv - adv.gather(1, anchor_mode(obs).view(-1, 1))


def greedy_mode(net: DuelingNet, obs) -> int:
    obs = np.asarray(obs, dtype=np.float32)
    mask = mode_mask(obs)
    with torch.no_grad():
        t = torch.from_numpy(obs).view(1, -1)
        q = net(t)[0].numpy().astype(np.float64)
        anchor = int(anchor_mode(t)[0])
    if not np.all(np.isfinite(q)):
        raise ValueError("Q 값이 유한하지 않습니다.")
    q = np.where(mask, q, -np.inf)
    q[anchor] += 1e-6                                     # 정확히 같으면 M4 쪽
    return int(q.argmax())


def epsilon_at(choices: int, cfg: dict) -> float:
    frac = min(1.0, max(0.0, choices / cfg["epsilon_decay_choices"]))
    return cfg["epsilon_start"] + frac * (cfg["epsilon_end"] - cfg["epsilon_start"])


# ----------------------------------------------------------------------------- SMDP 전이
def smdp_transitions(choices: list[dict], rewards: list[float], gamma: float, n: int) -> list[dict]:
    """choices: [{step, observation, mode}] (단계 순), rewards: 제어 720단계 보상(꼬리 접힘).
    지점 k의 한 칸 보상 r_k = Σ_{i=s_k}^{s_{k+1}-1} γ^{i-s_k} rewards[i], 할인 d_k = γ^{s_{k+1}-s_k} (마지막은 끝까지, done)."""
    T = len(rewards)
    steps = [int(c["step"]) for c in choices]
    if steps != sorted(steps) or len(set(steps)) != len(steps) or (steps and (steps[0] < 0 or steps[-1] >= T)):
        raise ValueError("선택 지점 단계가 올바르지 않습니다.")
    one = []
    for k, s in enumerate(steps):
        end = steps[k + 1] if k + 1 < len(steps) else T
        r = sum(gamma ** (i - s) * float(rewards[i]) for i in range(s, end))
        one.append((r, gamma ** (end - s), k + 1 == len(steps)))
    out = []
    for k, c in enumerate(choices):
        total, disc, done, j = 0.0, 1.0, False, k
        while j < len(choices) and j < k + n:
            r, d, last = one[j]
            total += disc * r
            disc *= d
            j += 1
            if last:
                done = True
                break
        nxt = choices[j]["observation"] if not done else c["observation"]
        out.append({"obs": np.asarray(c["observation"], np.float32), "mode": int(c["mode"]), "reward": total,
                    "discount": disc, "next_obs": np.asarray(nxt, np.float32), "done": done})
    return out


class Replay:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, OBS), np.float32)
        self.next_obs = np.zeros((capacity, OBS), np.float32)
        self.mode = np.zeros(capacity, np.int64)
        self.reward = np.zeros(capacity, np.float32)
        self.discount = np.zeros(capacity, np.float32)
        self.done = np.zeros(capacity, np.bool_)
        self.position = self.size = 0

    def add(self, transitions: list[dict]) -> None:
        for t in transitions:
            if t["obs"].shape != (OBS,) or not mode_mask(t["obs"])[t["mode"]]:
                raise ValueError("재생 자료의 관측 또는 모드가 올바르지 않습니다.")
            p = self.position
            self.obs[p], self.next_obs[p], self.mode[p] = t["obs"], t["next_obs"], t["mode"]
            self.reward[p], self.discount[p], self.done[p] = t["reward"], t["discount"], t["done"]
            self.position = (p + 1) % self.capacity
            self.size = min(self.capacity, self.size + 1)

    def sample(self, batch: int, rng: np.random.Generator) -> dict:
        ids = rng.integers(0, self.size, size=batch)
        return {k: torch.from_numpy(getattr(self, k)[ids]) for k in ("obs", "next_obs", "mode", "reward", "discount", "done")}

    def state(self) -> dict:
        n = self.size
        return {"capacity": self.capacity, "position": self.position, "size": n,
                **{k: getattr(self, k).copy() for k in ("obs", "next_obs", "mode", "reward", "discount", "done")}}

    @classmethod
    def from_state(cls, s: dict) -> "Replay":
        r = cls(s["capacity"])
        for k in ("obs", "next_obs", "mode", "reward", "discount", "done"):
            getattr(r, k)[:] = s[k]
        r.position, r.size = int(s["position"]), int(s["size"])
        return r


class Learner:
    def __init__(self, seed: int):
        cfg = gcfg()
        torch.manual_seed(seed)
        self.seed = int(seed)
        self.online = DuelingNet(tuple(cfg["network"]))
        self.target = copy.deepcopy(self.online)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=cfg["learning_rate"])
        self.replay = Replay(cfg["replay_size"])
        self.sample_rng = np.random.default_rng(seed + 100_000)
        self.updates = self.syncs = 0

    def gradient_step(self) -> float:
        cfg = gcfg()
        d = self.replay.sample(cfg["batch_size"], self.sample_rng)
        with torch.no_grad():
            nxt = d["next_obs"]
            mask = torch.from_numpy(mode_mask(nxt.numpy()))
            chosen = self.online(nxt).masked_fill(~mask, -torch.inf).argmax(dim=1, keepdim=True)
            value = self.target(nxt).gather(1, chosen).squeeze(1)
            value = torch.where(d["done"], torch.zeros_like(value), value)
            target = d["reward"] + d["discount"] * value
        pred = self.online(d["obs"]).gather(1, d["mode"].view(-1, 1)).squeeze(1)
        loss = F.smooth_l1_loss(pred, target)
        if not bool(torch.isfinite(loss)):
            raise ValueError("손실이 유한하지 않습니다.")
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.updates += 1
        if self.updates % cfg["target_every_updates"] == 0:
            self.target.load_state_dict(self.online.state_dict())
            self.syncs += 1
        return float(loss.detach())

    def learn(self, transitions: list[dict]) -> dict:
        cfg = gcfg()
        self.replay.add(transitions)
        losses = []
        if self.replay.size >= cfg["learning_starts"]:
            for _ in range(cfg["updates_per_transition"] * len(transitions)):
                losses.append(self.gradient_step())
        return {"updates": len(losses), "mean_loss": float(np.mean(losses)) if losses else None}


# ----------------------------------------------------------------------------- 정책
class EpsilonPolicy:
    name = POLICY

    def __init__(self, net: DuelingNet, base_choices: int, rng_seed: int, cfg: dict):
        self.net, self.base, self.cfg = net, int(base_choices), cfg
        self.rng = random.Random(rng_seed)
        self.taken: list[dict] = []

    def decide(self, snap, ctx) -> int:
        eps = epsilon_at(self.base + len(self.taken), self.cfg)
        valid = [m for m in range(N_MODES) if snap.mask[m]]
        mode = self.rng.choice(valid) if self.rng.random() < eps else greedy_mode(self.net, snap.observation)
        self.taken.append({"step": int(ctx.step), "observation": snap.observation.tolist(), "mode": int(mode),
                           "epsilon": eps})
        return snap.green(mode)


class GreedyPolicy:
    def __init__(self, net: DuelingNet, name: str = POLICY):
        self.net, self.name = net, name

    def decide(self, snap, ctx) -> int:
        return snap.green(greedy_mode(self.net, snap.observation))


def _state_bytes(net: DuelingNet) -> bytes:
    buf = io.BytesIO()
    torch.save(net.state_dict(), buf)
    return buf.getvalue()


def _net_from_bytes(raw: bytes) -> DuelingNet:
    net = DuelingNet(tuple(gcfg()["network"]))
    net.load_state_dict(torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True))
    net.eval()
    return net


# ----------------------------------------------------------------------------- 에피소드 작업자
def _worker_init():
    try:
        os.setsid()
    except OSError:
        pass
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)


QUALITY = ("missing", "unfinished", "changed_by_rules", "mode_action_errors", "p0_mismatch", "collisions_at_204820")


def episode_worker(job: dict) -> dict:
    started = time.monotonic()
    work = Path(job["work"])
    try:
        w.configure()
        torch.set_num_threads(1)
        shutil.rmtree(work, ignore_errors=True)
        policy = EpsilonPolicy(_net_from_bytes(job["net"]), job["base_choices"], job["rng_seed"], gcfg())
        result = ac.run_arena_episode(job["situation"], work, policy, max_seconds=3600.0, trace=False)
        m = result.metrics
        if [c["step"] for c in result.choices] != [t["step"] for t in policy.taken]:
            raise RuntimeError("실행기의 선택 지점과 정책 호출 지점이 다릅니다.")
        errors = [k for k in QUALITY if m.get(k, 0)]
        if not m["cost_check"].get("passed"):
            errors.append("cost_check")
        status = "ok" if not errors and not m["collisions_unexplained"] else (
            "discard" if not errors else "error")
        keep = {k: m.get(k) for k in ("mean_delay_s", "vehicles", "actual_total_delay", "choice_points", "long_waits",
                                       "left_east", "left_south", "collisions_unexplained", "route_sha256", "seconds")}
        return {"ok": status != "error", "status": status, "errors": errors, "metrics": keep,
                "choices": policy.taken, "rewards": result.rewards, "seconds": time.monotonic() - started,
                "collision_details": m.get("collision_details") if status == "discard" else None}
    except BaseException as error:
        return {"ok": False, "status": "error", "errors": [f"{type(error).__name__}: {error}"],
                "seconds": time.monotonic() - started}
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ----------------------------------------------------------------------------- 저장
def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_model(folder: Path, learner: Learner, meta: dict) -> str:
    """심판이 읽는 모델 폴더: model.pt(온라인 신경망) + manifest.json(해시·설정)."""
    folder = Path(folder)
    tmp = folder.with_name("." + folder.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    torch.save({"kind": KIND, "settings": settings(), "online": learner.online.state_dict(), **meta}, tmp / "model.pt")
    w.write(tmp / "manifest.json", {"kind": KIND, "settings": settings(), "model_sha256": _sha(tmp / "model.pt"), **meta})
    if folder.exists():
        shutil.rmtree(folder)
    os.replace(tmp, folder)
    return _sha(folder / "model.pt")


def model_identity(folder: Path) -> str:
    folder = Path(folder)
    manifest = w.read(folder / "manifest.json")
    if manifest.get("kind") != KIND or manifest.get("settings") != settings():
        raise ValueError(f"G1 모델의 종류 또는 학습 설정이 잠금과 다릅니다: {folder}")
    sha = _sha(folder / "model.pt")
    if sha != manifest["model_sha256"]:
        raise ValueError(f"G1 모델 파일 해시가 다릅니다: {folder}")
    return sha


def load_model(folder: Path) -> DuelingNet:
    model_identity(folder)
    payload = torch.load(Path(folder) / "model.pt", map_location="cpu", weights_only=False)
    net = DuelingNet(tuple(gcfg()["network"]))
    net.load_state_dict(payload["online"])
    net.eval()
    return net


def save_checkpoint(seed_dir: Path, learner: Learner, state: dict) -> None:
    payload = {"kind": KIND, "settings": settings(), "seed": learner.seed, "online": learner.online.state_dict(),
               "target": learner.target.state_dict(), "optimizer": learner.optimizer.state_dict(),
               "replay": learner.replay.state(), "sample_rng": learner.sample_rng.bit_generator.state,
               "torch_rng": torch.get_rng_state(), "updates": learner.updates, "syncs": learner.syncs, "state": state}
    tmp = seed_dir / "checkpoint.pt.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, seed_dir / "checkpoint.pt")
    w.write(seed_dir / "checkpoint.json", {"sha256": _sha(seed_dir / "checkpoint.pt"), "episodes": state["episodes"],
                                           "updated_unix": time.time()})


def load_checkpoint(seed_dir: Path, seed: int) -> tuple[Learner, dict] | None:
    path = seed_dir / "checkpoint.pt"
    if not path.exists():
        return None
    if _sha(path) != w.read(seed_dir / "checkpoint.json")["sha256"]:
        raise ValueError("체크포인트 해시가 다릅니다.")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["kind"] != KIND or payload["settings"] != settings() or payload["seed"] != seed:
        raise ValueError("체크포인트의 종류·설정·난수가 다릅니다.")
    learner = Learner(seed)
    learner.online.load_state_dict(payload["online"])
    learner.target.load_state_dict(payload["target"])
    learner.optimizer.load_state_dict(payload["optimizer"])
    learner.replay = Replay.from_state(payload["replay"])
    learner.sample_rng.bit_generator.state = payload["sample_rng"]
    torch.set_rng_state(payload["torch_rng"])
    learner.updates, learner.syncs = int(payload["updates"]), int(payload["syncs"])
    return learner, payload["state"]


class SpendLog:
    """끊겨도 버려진 계산까지 세는 덧붙이기 장부(체크포인트보다 먼저 쓴다)."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def total(self) -> float:
        if not self.path.exists():
            return 0.0
        return sum(float(json.loads(line)["seconds"]) for line in self.path.read_text().splitlines() if line.strip())

    def add(self, seconds: float, note: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"seconds": float(seconds), "note": note, "unix": time.time()}) + "\n")
            f.flush()
            os.fsync(f.fileno())


def scenario_for(seed: int, episode: int) -> dict:
    """에피소드 번호마다 정해지는 학습 상황(학습 8일 × 15분 칸 × 새 배치)."""
    cfg = ac.config()
    rng = random.Random(ac.seed32("G1_SCENARIO", seed, episode))
    date = rng.choice(cfg["dates"]["train"])
    slot = rng.choice(ac.TRAIN_SLOTS)
    return ac.make_situation(date, slot, f"G1s{seed}e{episode}")


def train(seed: int, out: Path, work: Path, *, executor=None, worker=episode_worker, max_batches: int | None = None) -> dict:
    cfg = gcfg()
    if seed not in cfg["seeds"]:
        raise ValueError(f"학습 난수는 {cfg['seeds']} 중 하나여야 합니다.")
    seed_dir = Path(out) / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    spend = SpendLog(seed_dir / "spend.jsonl")
    budget = cfg["cpu_hours_per_seed"] * 3600.0
    loaded = load_checkpoint(seed_dir, seed)
    if loaded is None:
        learner = Learner(seed)
        state = {"episodes": 0, "choices": 0, "control_steps": 0, "discarded": [], "history": [], "milestones": []}
    else:
        learner, state = loaded
    train_dates = set(ac.config()["dates"]["train"])
    pool = executor or ProcessPoolExecutor(max_workers=cfg["parallel_episodes"],
                                           mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init)
    batches = 0
    with pool:
        while spend.total() < budget:
            if max_batches is not None and batches >= max_batches:
                break
            raw = _state_bytes(learner.online)
            jobs = []
            for b in range(cfg["parallel_episodes"]):
                idx = state["episodes"] + b
                s = scenario_for(seed, idx)
                if s["date"] not in train_dates:
                    raise ValueError("학습 날짜가 아닌 상황이 뽑혔습니다.")
                jobs.append({"situation": s, "work": str(Path(work) / f"seed_{seed}" / f"e{idx}"), "net": raw,
                             "base_choices": state["choices"], "rng_seed": ac.seed32("G1_EPSILON", seed, idx)})
            started = time.monotonic()
            results = list(pool.map(worker, jobs))
            failed = [(job, r) for job, r in zip(jobs, results) if not r["ok"]]
            if failed:
                spend.add(sum(float(r["seconds"]) for r in results), f"failed batch e{state['episodes']}")
                job, r = failed[0]
                raise RuntimeError(f"G1 seed {seed} 에피소드 {ac.situation_id(job['situation'])} 실패: {r['errors']}")
            learn_started = time.monotonic()
            for job, r in zip(jobs, results):
                idx = state["episodes"]
                state["episodes"] += 1
                state["control_steps"] += 720
                if r["status"] == "discard":
                    state["discarded"].append({"episode": idx, "situation": ac.situation_id(job["situation"]),
                                               "collision": r["collision_details"]})
                    if len(state["discarded"]) > cfg["max_discarded"]:
                        raise RuntimeError(f"버린 에피소드가 {cfg['max_discarded']}개를 넘었습니다.")
                    continue
                # 보상은 WEIGHT1 fold_tail_rewards 그대로(지연 증가 ÷ 1000, 꼬리는 마지막 단계에 접힘).
                transitions = smdp_transitions(r["choices"], r["rewards"], cfg["gamma_per_5s"], cfg["n_choice_steps"])
                state["choices"] += len(r["choices"])
                info = learner.learn(transitions)
                state["history"].append({"episode": idx, "situation": ac.situation_id(job["situation"]), **r["metrics"],
                                         "transitions": len(transitions), "epsilon_first": r["choices"][0]["epsilon"]
                                         if r["choices"] else None, **info, "replay": learner.replay.size,
                                         "updates_total": learner.updates})
            learn_seconds = time.monotonic() - learn_started
            spend.add(sum(float(r["seconds"]) for r in results) + learn_seconds, f"batch e{state['episodes']}")
            for steps in cfg["milestone_control_steps"]:
                if state["control_steps"] >= steps and steps not in state["milestones"]:
                    save_model(seed_dir / "milestones" / f"steps_{steps}", learner,
                               {"seed": seed, "episodes": state["episodes"], "control_steps": state["control_steps"]})
                    state["milestones"].append(steps)
            batches += 1
            save_checkpoint(seed_dir, learner, state)
            last = state["history"][-1] if state["history"] else {}
            print(f"G1 seed {seed}: 에피소드 {state['episodes']}, 선택 결정 {state['choices']}, 갱신 {learner.updates}, "
                  f"최근 지연 {last.get('mean_delay_s')}, 사용 {spend.total() / 3600:.2f}/{budget / 3600:.0f} CPU-시간 "
                  f"(묶음 {time.monotonic() - started:.0f}초)", flush=True)
    finished = spend.total() >= budget
    if finished:
        save_model(seed_dir / "final", learner, {"seed": seed, "episodes": state["episodes"],
                                                 "control_steps": state["control_steps"], "choices": state["choices"]})
    summary = {"seed": seed, "finished": finished, "episodes": state["episodes"], "choices": state["choices"],
               "updates": learner.updates, "discarded": len(state["discarded"]), "cpu_hours": spend.total() / 3600}
    w.write(seed_dir / "status.json", {**summary, "updated_unix": time.time()})
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--max-batches", type=int, default=None, help="점검용: 묶음 수 제한(본 학습에는 쓰지 않음)")
    args = parser.parse_args()
    w.configure()
    torch.set_num_threads(1)
    import arena3_referee as ref
    ref.register(Path(args.out), extra_files=("src/arena3_g1.py",))
    print(json.dumps(train(args.seed, args.out, args.work, max_batches=args.max_batches), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
