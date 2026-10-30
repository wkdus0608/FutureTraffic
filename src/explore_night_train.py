"""WEIGHT1_NIGHT learner: M4-start dueling mode DQN with choice-point replay.

The environment, observation (110 values), the five weight modes, the reward and the
episode runner are exactly those of WEIGHT1.  Only the learner changes:

* Q(s,m) = V(s) + A(s,m) - A(s,b(s)), where b(s) is the smallest valid mode whose first
  green equals mode 4's (M4).  V and A start at zero, so the untrained greedy policy is
  exactly M4 and learning can only move away from M4 through the advantage head A.
* A quarter of every minibatch is drawn from decision points (more than one valid mode);
  only about 2% of control steps are such points, so uniform replay hardly trains A.
* Exploration is epsilon-greedy among valid modes, 0.5 -> 0.2 over the first 20,000 steps
  and then constant, because only decision points contain a choice to explore.
* A training episode whose only quality failure is an unexplained collision is discarded
  and logged instead of stopping the whole night (it never enters replay).

The locked WEIGHT1 modules (core, episode, train) are imported or left untouched.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import random
import shutil
import time
import uuid
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import explore_weight_train as base

OBSERVATION_SIZE, N_MODES = base.OBSERVATION_SIZE, base.N_MODES
MODE_MASK = base.MODE_MASK
ONEHOT = slice(84, 104)          # five modes x four greens, written by explore_weight_core.observation_vector
CONTROL_STEPS = base.CONTROL_STEPS
GAMMA, N_STEPS = 0.99, 10
LEARNING_RATE, REPLAY_SIZE, BATCH_SIZE = 0.001, 100_000, 64
LEARNING_STARTS, TRAIN_EVERY, TARGET_EVERY = 1_000, 4, 5_000
EXPLORATION_STEPS, EPS_START, EPS_END = 20_000, 0.5, 0.2
CHOICE_SHARE = 0.25
MAX_DISCARDED = 5
CHECKPOINT_KIND = "weight1-night-dueling-v1"
BASELINE_MODE = 4
SETTINGS = {
    "observation_size": OBSERVATION_SIZE, "mode_mask": [104, 109], "modes": N_MODES,
    "network": [64, 64], "dueling_baseline": "M4 representative", "gamma": GAMMA, "n_steps": N_STEPS,
    "learning_rate": LEARNING_RATE, "buffer_size": REPLAY_SIZE, "batch_size": BATCH_SIZE,
    "learning_starts": LEARNING_STARTS, "train_freq": TRAIN_EVERY, "target_update_interval": TARGET_EVERY,
    "exploration_steps": EXPLORATION_STEPS, "exploration_initial_eps": EPS_START,
    "exploration_final_eps": EPS_END, "choice_share": CHOICE_SHARE,
    "collection": "complete reconciled episode, then replay insertion and scheduled updates",
}


def code_fingerprint() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


def exploration_at(control_steps: int) -> float:
    fraction = min(1.0, max(0.0, control_steps / EXPLORATION_STEPS))
    return EPS_START + fraction * (EPS_END - EPS_START)


def baseline_mode(observation: torch.Tensor) -> torch.Tensor:
    """b(s): smallest valid mode proposing the same first green as mode 4, for (B,110) input."""
    obs = observation.reshape(-1, OBSERVATION_SIZE)
    greens = obs[:, ONEHOT].reshape(-1, N_MODES, 4).argmax(dim=2)
    valid = obs[:, MODE_MASK] > 0.5
    same = (greens == greens[:, BASELINE_MODE:BASELINE_MODE + 1]) & valid
    order = torch.arange(N_MODES, 0, -1, dtype=torch.float32, device=obs.device)   # 5,4,3,2,1: earlier mode wins
    return (same.float() * order).argmax(dim=1)


class DuelingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(OBSERVATION_SIZE, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.value = nn.Linear(64, 1)
        self.advantage = nn.Linear(64, N_MODES)
        for layer in (self.value, self.advantage):        # untrained greedy policy == M4
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        hidden = self.trunk(observation)
        advantage = self.advantage(hidden)
        anchor = baseline_mode(observation)
        return self.value(hidden) + advantage - advantage.gather(1, anchor.view(-1, 1))


class ChoiceReplay(base.EpisodeReplay):
    """n-step episode replay whose minibatches reserve a share for decision points."""

    def __init__(self, capacity: int = REPLAY_SIZE, gamma: float = GAMMA, n_steps: int = N_STEPS,
                 choice_share: float = CHOICE_SHARE):
        super().__init__(capacity, gamma, n_steps)
        if not 0.0 <= choice_share < 1.0:
            raise ValueError("choice_share must be in [0, 1)")
        self.choice_share = float(choice_share)
        self.choice = np.zeros(capacity, np.bool_)

    def _mark(self, positions: np.ndarray) -> None:
        masks = self.observations[positions][:, MODE_MASK] > 0.5
        self.choice[positions] = masks.sum(axis=1) > 1

    def add_episode(self, transitions: list[dict]) -> None:
        start, count = self.position, len(transitions)
        super().add_episode(transitions)
        self._mark((start + np.arange(count)) % self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, torch.Tensor]:
        if self.size == 0:
            raise ValueError("Cannot sample empty replay")
        pool = np.flatnonzero(self.choice[:self.size])
        wanted = int(round(batch_size * self.choice_share)) if len(pool) else 0
        ids = rng.integers(0, self.size, size=batch_size - wanted)
        if wanted:
            ids = np.concatenate([ids, pool[rng.integers(0, len(pool), size=wanted)]])
        return {name: torch.from_numpy(getattr(self, name)[ids]) for name in
                ("observations", "next_observations", "actions", "rewards", "discounts", "dones")}

    @classmethod
    def from_state_dict(cls, data: dict) -> "ChoiceReplay":
        replay = super().from_state_dict(data)
        if replay.size:
            replay._mark(np.arange(replay.size))
        return replay


class NightDQN:
    def __init__(self, seed: int, replay_capacity: int = REPLAY_SIZE):
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        self.seed = int(seed)
        self.online = DuelingNet()
        self.target = copy.deepcopy(self.online)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=LEARNING_RATE)
        self.replay = ChoiceReplay(replay_capacity)
        self.sample_rng = np.random.default_rng(seed + 100_000)
        self.num_timesteps = self.n_updates = self.target_syncs = self.episodes = 0
        self.exploration_rate = exploration_at(0)

    def q_values(self, observation) -> np.ndarray:
        obs = np.asarray(observation, np.float32)
        if obs.shape != (OBSERVATION_SIZE,) or not np.all(np.isfinite(obs)):
            raise ValueError("Invalid mode observation")
        with torch.no_grad():
            return self.online(torch.from_numpy(obs).view(1, -1))[0].numpy()

    def gradient_step(self) -> float:
        data = self.replay.sample(BATCH_SIZE, self.sample_rng)
        with torch.no_grad():
            nxt = data["next_observations"]
            target = base.double_targets(data["rewards"], data["discounts"], data["dones"],
                                         self.online(nxt), self.target(nxt), nxt[:, MODE_MASK] > 0.5)
        prediction = self.online(data["observations"]).gather(1, data["actions"].long().view(-1, 1)).squeeze(1)
        loss = F.smooth_l1_loss(prediction, target)
        if not bool(torch.isfinite(loss)):
            raise ValueError("Non-finite mode DQN loss")
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.n_updates += 1
        return float(loss.detach())

    def learn_episode(self, transitions: list[dict]) -> dict:
        self.replay.add_episode(transitions)
        started, losses = time.monotonic(), []
        end = self.num_timesteps + len(transitions)
        for step in range(self.num_timesteps + 1, end + 1):
            if step % TARGET_EVERY == 0:
                self.target.load_state_dict(self.online.state_dict())
                self.target_syncs += 1
            if step > LEARNING_STARTS and step % TRAIN_EVERY == 0:
                losses.append(self.gradient_step())
        self.num_timesteps, self.episodes = end, self.episodes + 1
        self.exploration_rate = exploration_at(end)
        return {"gradient_seconds": time.monotonic() - started,
                "mean_loss": float(np.mean(losses)) if losses else None,
                "updates_this_episode": len(losses), "updates": self.n_updates,
                "target_syncs": self.target_syncs, "control_steps": self.num_timesteps,
                "exploration_rate": self.exploration_rate, "replay_size": self.replay.size,
                "choice_points_in_replay": int(self.replay.choice[:self.replay.size].sum())}


def build_model(seed: int) -> NightDQN:
    return NightDQN(seed)


def predict_mode(model: NightDQN, observation, epsilon: float = 0.0, rng=None) -> int:
    mask = base.mode_mask(observation)
    if mask.ndim != 1:
        raise ValueError("predict_mode expects one observation")
    options = np.flatnonzero(mask).tolist()
    rng = rng if rng is not None else random
    if float(epsilon) > 0 and rng.random() < float(epsilon):
        return int(rng.choice(options))
    values = model.q_values(observation)
    if values.shape != (N_MODES,) or not np.all(np.isfinite(values)):
        raise ValueError("Non-finite mode Q values")
    anchor = int(baseline_mode(torch.from_numpy(np.asarray(observation, np.float32)))[0])
    values = np.where(mask, values, -np.inf)
    values[anchor] += 1e-6          # exact ties keep the M4 representative
    return int(values.argmax())


_sha = base._sha
_atomic_json = base._atomic_json


def _save_checkpoint(out: Path, model: NightDQN, fingerprint: str, state: dict) -> Path:
    parent = out / "checkpoints"
    parent.mkdir(parents=True, exist_ok=True)
    token = f"ep_{model.episodes:06d}_{uuid.uuid4().hex[:10]}"
    stage, final = parent / (".tmp_" + token), parent / token
    stage.mkdir()
    payload = {"kind": CHECKPOINT_KIND, "settings": SETTINGS, "fingerprint": fingerprint,
               "seed": model.seed, "episodes": model.episodes, "control_steps": model.num_timesteps,
               "updates": model.n_updates, "target_syncs": model.target_syncs,
               "online": model.online.state_dict(), "target": model.target.state_dict(),
               "optimizer": model.optimizer.state_dict(), "sample_rng": model.sample_rng.bit_generator.state,
               "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
               "python_rng": random.getstate(), "trainer_state": state}
    torch.save(payload, stage / "model.pt")
    with (stage / "replay.pkl").open("wb") as f:
        pickle.dump(model.replay.state_dict(), f, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = {"kind": CHECKPOINT_KIND, "fingerprint": fingerprint, "settings": SETTINGS,
                "seed": model.seed, "episodes": model.episodes, "control_steps": model.num_timesteps,
                "updates": model.n_updates, "target_syncs": model.target_syncs,
                "files": {n: _sha(stage / n) for n in ("model.pt", "replay.pkl")}}
    (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(stage, final)
    previous = None
    if (out / "latest.json").exists():
        previous = json.loads((out / "latest.json").read_text()).get("checkpoint")
    _atomic_json(out / "latest.json", {"checkpoint": str(final.relative_to(out)),
                                      "manifest_sha256": _sha(final / "manifest.json")})
    keep = {final.name, Path(previous).name if previous else ""}
    for candidate in parent.iterdir():
        if not candidate.is_dir() or candidate.name in keep or not candidate.name.startswith("ep_"):
            continue
        try:
            old = json.loads((candidate / "manifest.json").read_text())
        except (OSError, ValueError):
            continue
        if old.get("kind") == CHECKPOINT_KIND and old.get("fingerprint") == fingerprint:
            shutil.rmtree(candidate)
    return final


def validate_checkpoint(out: Path, fingerprint: str | None = None) -> dict:
    out = Path(out)
    pointer = None
    if (out / "latest.json").exists():
        pointer = json.loads((out / "latest.json").read_text())
        path = out / pointer["checkpoint"]
        if path.resolve().parent != (out / "checkpoints").resolve():
            raise ValueError("Checkpoint pointer escapes its owned directory")
    else:
        path = out
    manifest_path = path / "manifest.json"
    if pointer and _sha(manifest_path) != pointer["manifest_sha256"]:
        raise ValueError("Checkpoint manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("kind") != CHECKPOINT_KIND or manifest.get("settings") != SETTINGS:
        raise ValueError("Checkpoint kind or frozen learner settings mismatch")
    if fingerprint is not None and manifest.get("fingerprint") != fingerprint:
        raise ValueError("Checkpoint experiment fingerprint mismatch")
    for name, expected in manifest["files"].items():
        if name not in ("model.pt", "replay.pkl") or _sha(path / name) != expected:
            raise ValueError(f"Checkpoint file hash mismatch: {name}")
    if set(manifest["files"]) != {"model.pt", "replay.pkl"}:
        raise ValueError("Checkpoint files incomplete")
    return {"path": str(path), **manifest}


def _load_checkpoint(out: Path, fingerprint: str | None = None, *, replay: bool = True):
    identity = validate_checkpoint(out, fingerprint)
    path = Path(identity["path"])
    payload = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
    for name in ("kind", "settings", "fingerprint", "seed", "episodes", "control_steps", "updates", "target_syncs"):
        if payload[name] != identity[name]:
            raise ValueError(f"Checkpoint model/manifest mismatch: {name}")
    model = build_model(int(payload["seed"]))
    model.online.load_state_dict(payload["online"])
    model.target.load_state_dict(payload["target"])
    model.optimizer.load_state_dict(payload["optimizer"])
    model.num_timesteps, model.n_updates = int(payload["control_steps"]), int(payload["updates"])
    model.episodes, model.target_syncs = int(payload["episodes"]), int(payload["target_syncs"])
    model.exploration_rate = exploration_at(model.num_timesteps)
    model.sample_rng.bit_generator.state = payload["sample_rng"]
    if replay:
        with (path / "replay.pkl").open("rb") as f:
            model.replay = ChoiceReplay.from_state_dict(pickle.load(f))
    torch.set_rng_state(payload["torch_rng"])
    np.random.set_state(payload["numpy_rng"])
    random.setstate(payload["python_rng"])
    return model, payload["trainer_state"], identity


def load_model(out: Path, fingerprint: str | None = None) -> NightDQN:
    return _load_checkpoint(out, fingerprint, replay=False)[0]


def checkpoint_identity(out: Path, fingerprint: str | None = None) -> str:
    return validate_checkpoint(out, fingerprint)["files"]["model.pt"]


class EpisodeDiscard(Exception):
    """The episode's only quality failure is an unexplained collision; it never enters replay."""


def check_episode(result, expected_steps: int = CONTROL_STEPS) -> None:
    metrics = result.metrics
    failures = [key for key in ("missing", "unfinished", "mode_errors", "mode_action_errors", "changed_by_rules",
                                "outside_allowed", "rule_errors", "unexplained", "collisions_at_204820",
                                "collisions_unexplained") if float(metrics.get(key, 0) or 0) != 0]
    for key in ("cost_check", "cost_check_passed", "reward_check"):
        check = metrics.get(key)
        if check is False or (isinstance(check, dict) and check.get("passed") is False):
            raise ValueError(f"Training episode failed reconciliation: {key}")
    if not (metrics.get("cost_check_passed") is True
            or isinstance(metrics.get("cost_check"), dict) and metrics["cost_check"].get("passed") is True):
        raise ValueError("Training episode has no positive exact-cost reconciliation")
    if len(result.transitions) != expected_steps:
        raise ValueError(f"Expected {expected_steps} control transitions, got {len(result.transitions)}")
    if failures == ["collisions_unexplained"]:
        raise EpisodeDiscard(f"collisions_unexplained={metrics['collisions_unexplained']}")
    if failures:
        raise ValueError(f"Training episode failed quality check: {failures}")


def _milestone(out: Path, checkpoint: Path, episode: int) -> Path:
    destination = out / "stages" / f"episode_{episode:06d}"
    if destination.exists():
        source_manifest = json.loads((checkpoint / "manifest.json").read_text())
        if validate_checkpoint(destination)["files"] != source_manifest["files"]:
            raise ValueError("Existing immutable stage has different model/replay hashes")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.with_name(".tmp_" + destination.name + uuid.uuid4().hex[:8])
    try:
        shutil.copytree(checkpoint, stage, copy_function=os.link)
    except OSError:
        if stage.exists():
            shutil.rmtree(stage)
        shutil.copytree(checkpoint, stage)
    os.replace(stage, destination)
    return destination


def train_seed(out: Path, scenarios: list[tuple[str, str]], seed: int, target_episodes: int,
               run_episode: Callable, *, fingerprint: str, budget_check: Callable | None = None) -> dict:
    """Collect, reconcile, learn, atomically save.  ``out`` is the exact seed directory."""
    started = time.monotonic()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scenarios = [tuple(s) for s in scenarios]
    if not scenarios or len(set(scenarios)) != len(scenarios):
        raise ValueError("Training scenarios must be nonempty and unique")
    if any(d in {"20220811", "20220823", "20220928", "20220813"} for d, _ in scenarios):
        raise ValueError("Development/confirmation dates cannot enter training")
    scenario_sha = hashlib.sha256(json.dumps(scenarios).encode()).hexdigest()
    if (out / "latest.json").exists():
        model, state, identity = _load_checkpoint(out, fingerprint)
        if model.seed != seed or state["scenario_sha256"] != scenario_sha:
            raise ValueError("Resume seed or scenario list differs")
        scenario_rng, exploration_rng = random.Random(), random.Random()
        scenario_rng.setstate(state["scenario_rng"])
        exploration_rng.setstate(state["exploration_rng"])
        history, discarded = list(state["history"]), list(state.get("discarded", []))
    else:
        if (out / "run.json").exists():
            raise ValueError("Existing run has no committed checkpoint; inspect before reusing")
        model, scenario_rng, exploration_rng = build_model(seed), random.Random(seed), random.Random(seed + 200_000)
        history, discarded, identity = [], [], None
    if target_episodes < model.episodes:
        raise ValueError("Cannot move a completed training run backwards")
    run_episode_seconds, checkpoint_seconds, gradient_seconds = 0.0, 0.0, 0.0
    checkpoint = Path(identity["path"]) if identity else None
    while model.episodes < target_episodes:
        if budget_check is not None:
            budget_check()
        episode_index = model.episodes
        date, period = scenario_rng.choice(scenarios)
        base_step, policy_calls, chosen_modes = model.num_timesteps, 0, []

        def policy(observation, snapshot=None):
            nonlocal policy_calls
            action = predict_mode(model, observation, exploration_at(base_step + policy_calls), exploration_rng)
            policy_calls += 1
            chosen_modes.append(action)
            return action

        work = out / "work" / f"episode_{episode_index + 1:06d}"
        result = run_episode(date, period, seed + episode_index, work, policy,
                             collect=True, train_seed=seed, trace=True)
        run_episode_seconds += float(result.seconds)
        try:
            check_episode(result)
        except EpisodeDiscard as why:
            discarded.append({"after_episode": model.episodes, "date": date, "period": period,
                              "sumo_seed": seed + episode_index, "reason": str(why),
                              "details": result.metrics.get("collision_details")})
            with (out / "discarded.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(discarded[-1], ensure_ascii=False, default=str) + "\n")
            print(f"NIGHT seed {seed}: discarded {date} {period} ({why}); {len(discarded)}/{MAX_DISCARDED}", flush=True)
            if len(discarded) > MAX_DISCARDED:
                raise ValueError(f"More than {MAX_DISCARDED} training episodes were discarded")
            continue
        if policy_calls != CONTROL_STEPS:
            raise ValueError(f"Policy called {policy_calls} times, expected {CONTROL_STEPS}")
        if [int(t["mode"]) for t in result.transitions] != chosen_modes:
            raise ValueError("Reconciled replay modes differ from policy-selected modes")
        learning = model.learn_episode(result.transitions)
        gradient_seconds += learning["gradient_seconds"]
        history.append({"episode": model.episodes, "date": date, "period": period,
                        "sumo_seed": seed + episode_index, "episode_seconds": float(result.seconds),
                        "metrics": result.metrics, **learning})
        state = {"scenario_sha256": scenario_sha, "scenarios": scenarios,
                 "scenario_rng": scenario_rng.getstate(), "exploration_rng": exploration_rng.getstate(),
                 "history": history, "discarded": discarded}
        save_started = time.monotonic()
        checkpoint = _save_checkpoint(out, model, fingerprint, state)
        checkpoint_seconds += time.monotonic() - save_started
        _atomic_json(out / "run.json", {"kind": CHECKPOINT_KIND, "fingerprint": fingerprint, "seed": seed,
                     "settings": SETTINGS, "episodes": model.episodes, "control_steps": model.num_timesteps,
                     "updates": model.n_updates, "target_syncs": model.target_syncs,
                     "discarded": len(discarded), "checkpoint": str(checkpoint), "history": history})
        print(f"NIGHT seed {seed}: episode {model.episodes}/{target_episodes}, steps {model.num_timesteps}, "
              f"delay {result.metrics.get('mean_delay_s')}, epsilon {model.exploration_rate:.4f}", flush=True)
    if checkpoint is None:
        raise ValueError("No complete training episode exists")
    stage = _milestone(out, checkpoint, model.episodes)
    elapsed = time.monotonic() - started
    return {"seed": seed, "episodes": model.episodes, "control_steps": model.num_timesteps,
            "updates": model.n_updates, "target_syncs": model.target_syncs, "discarded": len(discarded),
            "exploration_rate": model.exploration_rate, "checkpoint": str(checkpoint),
            "stage": str(stage), "model_sha256": _sha(checkpoint / "model.pt"),
            "seconds": elapsed, "episode_seconds": run_episode_seconds,
            "compute_seconds": max(0.0, elapsed - run_episode_seconds),
            "gradient_seconds": gradient_seconds, "checkpoint_seconds": checkpoint_seconds,
            "fingerprint": fingerprint}
