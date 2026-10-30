"""WEIGHT1: canonical weight-mode Double DQN and episode-boundary checkpoints.

The replay action is a weight mode (0..4), never an executed green (0..3).
Episodes are reconciled against tripinfo before their transitions enter replay.
All control-time schedules are absolute, so pilot -> full continuation does not
restart exploration or the target-network schedule. This is a new experiment;
the locked focused training/environment modules are not patched.
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

OBSERVATION_SIZE, N_MODES = 110, 5
MODE_MASK = slice(104, 109)
CONTROL_STEPS = 720
PILOT_EPISODES, FULL_EPISODES = 42, 139
GAMMA, N_STEPS = 0.99, 10
LEARNING_RATE, REPLAY_SIZE, BATCH_SIZE = 0.001, 100_000, 64
LEARNING_STARTS, TRAIN_EVERY, TARGET_EVERY = 1_000, 4, 5_000
EXPLORATION_STEPS, EPS_START, EPS_END = 20_000, 1.0, 0.05
CHECKPOINT_KIND = "weight1-mode-dqn-v1"
SETTINGS = {
    "observation_size": OBSERVATION_SIZE, "mode_mask": [104, 109], "modes": N_MODES,
    "network": [64, 64], "gamma": GAMMA, "n_steps": N_STEPS,
    "learning_rate": LEARNING_RATE, "buffer_size": REPLAY_SIZE, "batch_size": BATCH_SIZE,
    "learning_starts": LEARNING_STARTS, "train_freq": TRAIN_EVERY,
    "target_update_interval": TARGET_EVERY, "exploration_steps": EXPLORATION_STEPS,
    "exploration_initial_eps": EPS_START, "exploration_final_eps": EPS_END,
    "collection": "complete reconciled episode, then replay insertion and scheduled updates",
}


def code_fingerprint() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


def exploration_at(control_steps: int) -> float:
    fraction = min(1.0, max(0.0, control_steps / EXPLORATION_STEPS))
    return EPS_START + fraction * (EPS_END - EPS_START)


def mode_mask(observation) -> np.ndarray:
    obs = np.asarray(observation, dtype=np.float32)
    if obs.shape[-1] != OBSERVATION_SIZE or not np.all(np.isfinite(obs)):
        raise ValueError("WEIGHT1 observation must be finite and have 110 fields")
    mask = obs[..., MODE_MASK] > 0.5
    if not np.all(mask.any(axis=-1)):
        raise ValueError("A live WEIGHT1 observation has no canonical weight mode")
    return mask


def double_targets(rewards, discounts, dones, online_next, target_next, masks):
    """Double DQN: online selects a legal mode, target evaluates that mode."""
    mask = masks.bool()
    done = dones.bool().reshape(-1)
    if bool((~mask.any(dim=1) & ~done).any()):
        raise ValueError("Nonterminal next observation has no canonical mode")
    # A terminal observation may be an all-zero placeholder. Its value is zero.
    safe = mask.clone()
    safe[~safe.any(dim=1), 0] = True
    selected = online_next.masked_fill(~safe, -torch.inf).argmax(dim=1, keepdim=True)
    values = target_next.gather(1, selected).squeeze(1)
    values = torch.where(done, torch.zeros_like(values), values)
    return rewards.reshape(-1) + discounts.reshape(-1) * values


class EpisodeReplay:
    """A ring of precomputed n-step transitions, made within one episode only."""

    def __init__(self, capacity: int = REPLAY_SIZE, gamma: float = GAMMA, n_steps: int = N_STEPS):
        if capacity <= 0 or n_steps <= 0:
            raise ValueError("Replay capacity and n_steps must be positive")
        self.capacity, self.gamma, self.n_steps = int(capacity), float(gamma), int(n_steps)
        self.observations = np.empty((capacity, OBSERVATION_SIZE), np.float32)
        self.next_observations = np.empty_like(self.observations)
        self.actions = np.empty(capacity, np.int64)
        self.rewards = np.empty(capacity, np.float32)
        self.discounts = np.empty(capacity, np.float32)
        self.dones = np.empty(capacity, np.bool_)
        self.position = self.size = 0

    def add_episode(self, transitions: list[dict]) -> None:
        if not transitions or not bool(transitions[-1]["done"]):
            raise ValueError("Only complete, reconciled episodes may enter replay")
        if any(bool(t["done"]) for t in transitions[:-1]):
            raise ValueError("Episode has a premature terminal transition")
        checked = []
        for index, tr in enumerate(transitions):
            obs, nxt = np.asarray(tr["obs"], np.float32), np.asarray(tr["next_obs"], np.float32)
            if obs.shape != (OBSERVATION_SIZE,) or nxt.shape != (OBSERVATION_SIZE,):
                raise ValueError("Replay observation shape is not (110,)")
            mask = mode_mask(obs)
            action = int(tr["mode"])
            if not 0 <= action < N_MODES or not mask[action]:
                raise ValueError("Replay action is not a canonical weight mode")
            if not np.isfinite(float(tr["reward"])) or not np.all(np.isfinite(nxt)):
                raise ValueError("Replay contains a non-finite reward or next observation")
            if not tr["done"]:
                mode_mask(nxt)
            if index + 1 < len(transitions) and not np.array_equal(nxt, np.asarray(transitions[index + 1]["obs"], np.float32)):
                raise ValueError("Adjacent episode observations do not join")
            checked.append((obs, nxt, action, float(tr["reward"]), bool(tr["done"])))
        for start, (obs, _, action, _, _) in enumerate(checked):
            reward, discount, last = 0.0, 1.0, start
            for last in range(start, min(start + self.n_steps, len(checked))):
                reward += discount * checked[last][3]
                discount *= self.gamma
                if checked[last][4]:
                    break
            pos = self.position
            self.observations[pos], self.next_observations[pos] = obs, checked[last][1]
            self.actions[pos], self.rewards[pos] = action, reward
            self.discounts[pos], self.dones[pos] = discount, checked[last][4]
            self.position = (pos + 1) % self.capacity
            self.size = min(self.capacity, self.size + 1)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, torch.Tensor]:
        if self.size == 0:
            raise ValueError("Cannot sample empty replay")
        ids = rng.integers(0, self.size, size=batch_size)
        return {name: torch.from_numpy(getattr(self, name)[ids]) for name in
                ("observations", "next_observations", "actions", "rewards", "discounts", "dones")}

    def state_dict(self) -> dict:
        return {"capacity": self.capacity, "gamma": self.gamma, "n_steps": self.n_steps,
                "position": self.position, "size": self.size,
                **{name: getattr(self, name)[:self.size].copy() for name in
                   ("observations", "next_observations", "actions", "rewards", "discounts", "dones")}}

    @classmethod
    def from_state_dict(cls, data: dict) -> "EpisodeReplay":
        replay = cls(data["capacity"], data["gamma"], data["n_steps"])
        replay.position, replay.size = int(data["position"]), int(data["size"])
        if not (0 <= replay.position < replay.capacity and 0 <= replay.size <= replay.capacity):
            raise ValueError("Invalid replay cursor")
        for name in ("observations", "next_observations", "actions", "rewards", "discounts", "dones"):
            getattr(replay, name)[:replay.size] = data[name]
        return replay


class ModeDQN:
    def __init__(self, seed: int, replay_capacity: int = REPLAY_SIZE):
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        self.seed = int(seed)
        self.online = nn.Sequential(nn.Linear(OBSERVATION_SIZE, 64), nn.ReLU(),
                                    nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, N_MODES))
        # Equal initial Q values make deterministic inference choose the lowest
        # canonical mode, including M0. Exploration is still scheduled normally.
        nn.init.zeros_(self.online[-1].weight)
        nn.init.zeros_(self.online[-1].bias)
        self.target = copy.deepcopy(self.online)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=LEARNING_RATE)
        self.replay = EpisodeReplay(replay_capacity)
        self.sample_rng = np.random.default_rng(seed + 100_000)
        self.num_timesteps = self.n_updates = self.target_syncs = self.episodes = 0
        self.exploration_rate = exploration_at(0)

    def q_values(self, observation) -> np.ndarray:
        obs = np.asarray(observation, np.float32)
        if obs.shape != (OBSERVATION_SIZE,) or not np.all(np.isfinite(obs)):
            raise ValueError("Invalid mode observation")
        with torch.no_grad():
            return self.online(torch.from_numpy(obs)).numpy()

    def gradient_step(self) -> float:
        data = self.replay.sample(BATCH_SIZE, self.sample_rng)
        with torch.no_grad():
            nxt = data["next_observations"]
            target = double_targets(data["rewards"], data["discounts"], data["dones"],
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
            # Match collection-before-gradient target replacement at each step.
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
                "exploration_rate": self.exploration_rate, "replay_size": self.replay.size}


def build_model(seed: int) -> ModeDQN:
    return ModeDQN(seed)


def predict_mode(model: ModeDQN, observation, epsilon: float = 0.0, rng=None) -> int:
    mask = mode_mask(observation)
    if mask.ndim != 1:
        raise ValueError("predict_mode expects one observation")
    options = np.flatnonzero(mask).tolist()
    rng = rng if rng is not None else random
    if float(epsilon) > 0 and rng.random() < float(epsilon):
        return int(rng.choice(options))
    values = model.q_values(observation)
    if values.shape != (N_MODES,) or not np.all(np.isfinite(values)):
        raise ValueError("Non-finite mode Q values")
    return int(np.where(mask, values, -np.inf).argmax())


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp_" + uuid.uuid4().hex)
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _save_checkpoint(out: Path, model: ModeDQN, fingerprint: str, state: dict) -> Path:
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
    # Keep the current and immediately preceding owned generation. Milestone
    # checkpoints below are hard-linked and survive generation cleanup.
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
            model.replay = EpisodeReplay.from_state_dict(pickle.load(f))
    torch.set_rng_state(payload["torch_rng"])
    np.random.set_state(payload["numpy_rng"])
    random.setstate(payload["python_rng"])
    return model, payload["trainer_state"], identity


def load_model(out: Path, fingerprint: str | None = None) -> ModeDQN:
    """Load the last committed model; no checkpoint or state is modified."""
    return _load_checkpoint(out, fingerprint, replay=False)[0]


def checkpoint_identity(out: Path, fingerprint: str | None = None) -> str:
    return validate_checkpoint(out, fingerprint)["files"]["model.pt"]


def _check_episode(result, expected_steps: int = CONTROL_STEPS) -> None:
    metrics = result.metrics
    for key in ("missing", "unfinished", "mode_errors", "mode_action_errors", "changed_by_rules", "outside_allowed",
                "rule_errors", "unexplained", "collisions_at_204820", "collisions_unexplained"):
        if float(metrics.get(key, 0) or 0) != 0:
            raise ValueError(f"Training episode failed quality check: {key}={metrics[key]}")
    collisions = metrics.get("collisions")
    if isinstance(collisions, dict) and float(collisions.get("unexplained", 0) or 0):
        raise ValueError("Training episode has unexplained collisions")
    for key in ("cost_check", "cost_check_passed", "reward_check"):
        check = metrics.get(key)
        if check is False or (isinstance(check, dict) and check.get("passed") is False):
            raise ValueError(f"Training episode failed reconciliation: {key}")
    if not (metrics.get("cost_check_passed") is True
            or isinstance(metrics.get("cost_check"), dict) and metrics["cost_check"].get("passed") is True):
        raise ValueError("Training episode has no positive exact-cost reconciliation")
    if len(result.transitions) != expected_steps:
        raise ValueError(f"Expected {expected_steps} control transitions, got {len(result.transitions)}")


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
    """Collect, reconcile, learn, atomically save; continuation uses absolute steps.

    ``out`` is the exact seed directory, not its parent. ``budget_check()`` is
    called before each episode and may raise the orchestrator's BudgetExceeded.
    The caller accounts elapsed wall time including checkpoint and gradients;
    result episode_seconds / compute_seconds permit disjoint budget accounting.
    """
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
        history = list(state["history"])
    else:
        if (out / "run.json").exists():
            raise ValueError("Existing run has no committed checkpoint; inspect before reusing")
        model, scenario_rng, exploration_rng = build_model(seed), random.Random(seed), random.Random(seed + 200_000)
        history, identity = [], None
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
        _check_episode(result)
        if policy_calls != CONTROL_STEPS:
            raise ValueError(f"Policy called {policy_calls} times, expected {CONTROL_STEPS}")
        if [int(t["mode"]) for t in result.transitions] != chosen_modes:
            raise ValueError("Reconciled replay modes differ from policy-selected modes")
        episode_elapsed = float(result.seconds)
        run_episode_seconds += episode_elapsed
        learning = model.learn_episode(result.transitions)
        gradient_seconds += learning["gradient_seconds"]
        history.append({"episode": model.episodes, "date": date, "period": period,
                        "sumo_seed": seed + episode_index, "episode_seconds": episode_elapsed,
                        "metrics": result.metrics, **learning})
        state = {"scenario_sha256": scenario_sha, "scenarios": scenarios,
                 "scenario_rng": scenario_rng.getstate(), "exploration_rng": exploration_rng.getstate(),
                 "history": history}
        save_started = time.monotonic()
        checkpoint = _save_checkpoint(out, model, fingerprint, state)
        checkpoint_seconds += time.monotonic() - save_started
        _atomic_json(out / "run.json", {"kind": CHECKPOINT_KIND, "fingerprint": fingerprint, "seed": seed,
                     "settings": SETTINGS, "episodes": model.episodes, "control_steps": model.num_timesteps,
                     "updates": model.n_updates, "target_syncs": model.target_syncs,
                     "checkpoint": str(checkpoint), "history": history})
        print(f"WEIGHT1 seed {seed}: episode {model.episodes}/{target_episodes}, "
              f"steps {model.num_timesteps}, delay {result.metrics.get('mean_delay_s')}, "
              f"epsilon {model.exploration_rate:.4f}", flush=True)
    if checkpoint is None:
        raise ValueError("No complete training episode exists")
    stage = _milestone(out, checkpoint, model.episodes)
    elapsed = time.monotonic() - started
    return {"seed": seed, "episodes": model.episodes, "control_steps": model.num_timesteps,
            "updates": model.n_updates, "target_syncs": model.target_syncs,
            "exploration_rate": model.exploration_rate, "checkpoint": str(checkpoint),
            "stage": str(stage), "model_sha256": _sha(checkpoint / "model.pt"),
            "seconds": elapsed, "episode_seconds": run_episode_seconds,
            "compute_seconds": max(0.0, elapsed - run_episode_seconds),
            "gradient_seconds": gradient_seconds, "checkpoint_seconds": checkpoint_seconds,
            "fingerprint": fingerprint}
