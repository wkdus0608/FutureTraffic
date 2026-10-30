"""[추가 탐색] 이동별 정보 DQN(MOV)과 '고를 때만 묻는' DQN(CHO) — 단독 모형, 지문 밖(진행 기록 114절).

세 비교(114.1절): A = H1(87절 모델 그대로), B = MOV, C = CHO. 한 번에 하나씩 바꾼다.
- B(MOV): 관측에 이동별 기다리는 차 8칸을 더한다(64 → 72칸).
  - 이동: 어느 녹색에서든 보호 녹색(G)을 받는 이동 8개(진입로 4개 × 직진·좌회전). 예측 제어 P0가 점수에 쓰는 이동과 같다.
  - 값: (감지 구간에서 멈춘 차 + 도로에 못 들어온 차) ÷ 그 진입로 감지 구간에 설 수 있는 차 수(최대 1). 세는 법은 단순 제어
    (focused_eval.QueueFirst.waiting)·P0와 같고, 차가 어느 이동인지는 차의 경로로 안다.
  - 자리: 보행 칸 뒤, 끝 4칸(그대로 실행될 행동 표시) 앞. 학습은 H1과 같다(focused_train.py, n = 10, Double).
- C(CHO): B와 같은 관측(B 결과와 상관없이). 규칙 확인은 5초마다 하되, 허용 녹색이 하나뿐인 5초는 묻지 않고 그 녹색으로
  진행한다(ChoiceOnlyEnv). 경험 하나 = 고른 순간 → 다음 고를 순간. 그 사이 보상은 Σ γ^i r_i로, 지난 5초 칸 수 k와 함께 남긴다
  (다음 값의 할인 γ^k). H1의 '단계' 설정을 제어 시뮬레이션 시간으로 옮긴다(114.3절, 아래 상수).

잠근 파일·focused_train.py·평가 코드는 고치지 않는다. 실행할 때만 focused_rl.ZoneObservation을 바꿔 끼운다(학습 환경과 평가의
run_day가 함수 안에서 이 이름을 가져온다). 평가는 기존 평가 코드 그대로다(predict만 쓰므로 HoldAwareDQN으로 불러도 같다).
모델 폴더의 explore.json에 관측 코드 지문을 남기고, 평가는 같은 관측 코드의 모델만 받는다.

사용(Colab 36번 칸):
  FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240 python src/explore_choice.py check --out <Drive>/explore_choice/check.json
  FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240 FOCUSED_DQN_NSTEPS=10 FOCUSED_DQN_DOUBLE=1 python src/explore_choice.py train-mov \
      --seed 42 --timesteps 100000 --out <Drive>/explore_choice/models_mov/pet1 --resume
  FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240 python src/explore_choice.py train-cho --seed 42 --out <Drive>/explore_choice/models_cho/pet1 --resume
  FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240 python src/explore_choice.py eval --dates 20220928 20220813 --policies dqn \
      --models <seed 폴더 5개> --arrivals random platoon --out <Drive>/explore_choice/single/eval --tmp /content/choice_tmp
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import explore_dqn  # noqa: E402  원래 학습기를 바꿔 끼우기 전에 가져온다(Double 학습기가 원래 HoldAwareDQN을 이어받게)

N_MOVEMENTS = 8
STEP_S = 5
# H1(87절: 10만 단계, n = 10, focused_train.DQN_SETTINGS)의 '단계' 설정 → 제어 시뮬레이션 시간(114.3절)
H1_TIMESTEPS, H1_N_STEPS = 100_000, 10
BUDGET_S = H1_TIMESTEPS * STEP_S                    # 총 경험 50만 초(에피소드 약 139개)
WINDOW_S = H1_N_STEPS * STEP_S                      # n-step 묶음: 합계 50초 이상이 될 때까지
RANDOM_S = 1_000 * STEP_S                           # learning_starts 1,000단계
EXPLORE_S = int(0.2 * H1_TIMESTEPS) * STEP_S        # exploration_fraction 0.2 → 10만 초 동안 1.0 → 0.05
TRAIN_EVERY_S = 4 * STEP_S                          # train_freq 4단계 → 경사 하강 20초마다 1번
TARGET_EVERY_S = 5_000 * STEP_S                     # target_update_interval 5,000단계 → 25,000초
EPS_START, EPS_END = 1.0, 0.05
GAMMA = 0.99                                        # 5초당
TOTAL_UPDATES = (BUDGET_S - RANDOM_S) // TRAIN_EVERY_S   # 24,750번(H1과 같음)
TOTAL_SYNCS = BUDGET_S // TARGET_EVERY_S                 # 20번
CHECKPOINT_S = 50_000
CHO_SETTINGS = {"learning_rate": 1e-3, "buffer_size": 100_000, "batch_size": 64, "gamma": GAMMA}
DQN_ENV_VARS = ("FOCUSED_DQN_NSTEPS", "FOCUSED_DQN_GAMMA", "FOCUSED_DQN_TARGET_UPDATE", "FOCUSED_DQN_DOUBLE")


def schedule() -> dict:
    return {"step_s": STEP_S, "budget_s": BUDGET_S, "window_s": WINDOW_S, "random_s": RANDOM_S, "explore_s": EXPLORE_S,
            "eps": [EPS_START, EPS_END], "train_every_s": TRAIN_EVERY_S, "target_every_s": TARGET_EVERY_S, "gamma": GAMMA,
            "total_updates": TOTAL_UPDATES, "total_syncs": TOTAL_SYNCS}


# ------------------------------------------------------------------ B: 이동별 관측
def protected_movements(greens: list[str]) -> list[tuple[str, str]]:
    """어느 녹색에서든 보호 녹색(G)을 받는 이동(P0가 점수에 쓰는 이동과 같음), 정렬."""
    from focused_rl import MOVEMENT_LINKS
    return sorted(m for m, links in MOVEMENT_LINKS.items() if any(g[l] == "G" for g in greens for l in links))


def waiting_by_movement(sumo, movement_of: dict) -> Counter:
    """단순 제어(QueueFirst.waiting)와 같은 셈: 감지 구간에서 멈춘 차(0.1m/s 미만) + 도로에 못 들어온 차, 이동별."""
    from focused_rl import ALL_ZONE_EDGES
    n = Counter()
    for edge in ALL_ZONE_EDGES:
        for v in sumo.edge.getLastStepVehicleIDs(edge):
            if sumo.vehicle.getSpeed(v) < 0.1:
                n[movement_of.get(v)] += 1
    for v in sumo.simulation.getPendingVehicles():
        n[movement_of.get(v)] += 1
    return n


def movement_features(counts: Counter, movements: list[tuple[str, str]], capacity: dict) -> list[float]:
    """이동마다 기다리는 차 ÷ 그 진입로(이동의 첫 도로) 감지 구간에 설 수 있는 차 수, 최대 1."""
    return [min(1.0, counts[m] / capacity[m[0]]) for m in movements]


def insert_before_mask(base, extra, n_actions: int):
    """기존 관측 끝 n칸(그대로 실행될 행동 표시)은 맨 끝에 둔다. HoldAwareDQN이 끝 n칸을 표시로, 앞 n칸을 현시로 읽는다."""
    return np.concatenate([base[:-n_actions], np.asarray(extra, dtype=np.float32), base[-n_actions:]]).astype(np.float32)


def movement_observation_class(base):
    from gymnasium import spaces

    class MovementObservation(base):
        movement_info = True

        def __init__(self, ts):
            super().__init__(ts)
            self._movement_of = self._movements = None    # 신호 객체가 에피소드마다 새로 만들어지므로 에피소드마다 한 번 읽는다

        def _setup(self) -> None:
            from focused_eval import QueueFirst
            self._movement_of = QueueFirst.movements(Path(self.ts.env._route))
            self._movements = protected_movements([p.state for p in self.ts.green_phases])
            if len(self._movements) != N_MOVEMENTS:
                raise RuntimeError(f"보호 녹색 이동이 {len(self._movements)}개다({N_MOVEMENTS}개여야 함).")

        def __call__(self):
            obs = super().__call__()                     # 진입로 감지 구간 크기(_capacity)도 여기서 정해진다
            if self._movement_of is None:
                self._setup()
            extra = movement_features(waiting_by_movement(self.ts.sumo, self._movement_of), self._movements, self._capacity)
            return insert_before_mask(obs, extra, self.ts.num_green_phases)

        def observation_space(self):
            size = super().observation_space().shape[0] + N_MOVEMENTS
            return spaces.Box(low=np.zeros(size, dtype=np.float32), high=np.ones(size, dtype=np.float32))

    return MovementObservation


def install_observation() -> None:
    """focused_rl의 관측 클래스를 바꿔 끼운다(여러 번 불러도 한 번만)."""
    import focused_rl
    if not getattr(focused_rl.ZoneObservation, "movement_info", False):
        focused_rl.ZoneObservation = movement_observation_class(focused_rl.ZoneObservation)


def _fingerprint(parts, extra: str = "") -> str:
    src = "".join(inspect.getsource(p) for p in parts) + extra
    return hashlib.sha256(src.encode()).hexdigest()[:12]


def observation_fingerprint() -> str:
    return _fingerprint([protected_movements, waiting_by_movement, movement_features, insert_before_mask,
                         movement_observation_class], f"N_MOVEMENTS={N_MOVEMENTS}")


# ------------------------------------------------------------------ C: 고를 때만 묻기
def allowed_of(obs, n: int) -> list[int]:
    return [i for i in range(n) if obs[-n + i] > 0.5]


class ChoiceOnlyEnv(gym.Wrapper):
    """허용 녹색이 둘 이상인 순간에만 학습기에 묻는다(114.3절). 하나뿐인 5초는 그 녹색으로 진행하고 보상을 할인해 더한다.

    step의 보상 = Σ_{i<k} γ^i r_i(r_0은 고른 녹색의 5초), info["elapsed_units"] = k(5초 칸 수),
    info["executed_action"] = 고른 순간 실제로 켜진 녹색. reset 뒤 첫 고를 순간까지의 진행은 어느 경험에도 들어가지 않지만
    제어 시간(control_units)에는 센다. 안쪽 환경(Monitor 포함)은 5초마다 원래 보상을 그대로 받는다."""

    def __init__(self, env, gamma: float = GAMMA):
        super().__init__(env)
        self.gamma = gamma
        self.n = env.action_space.n
        self.control_units = 0          # 지난 제어 5초 칸(묻지 않은 칸 포함). 학습기가 시간 일정에 쓴다
        self.stats = Counter()          # choices·forced·reset_forced·changed_by_rules

    def _advance(self, obs, info):
        """허용 녹색이 하나뿐인 동안 진행한다. (obs, Σ γ^i r_i, 칸 수, terminated, truncated, info)"""
        ret, k, terminated, truncated = 0.0, 0, False, False
        while not (terminated or truncated):
            allowed = allowed_of(obs, self.n)
            if len(allowed) != 1:
                break
            obs, r, terminated, truncated, info = self.env.step(allowed[0])
            if info.get("executed_action", allowed[0]) != allowed[0]:
                self.stats["changed_by_rules"] += 1
            ret += self.gamma ** k * float(r)
            k += 1
        return obs, ret, k, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, _, k, terminated, truncated, _ = self._advance(obs, info)
        if terminated or truncated:
            raise RuntimeError("에피소드가 첫 고를 순간 전에 끝났다.")
        self.control_units += k
        self.stats["reset_forced"] += k
        return obs, {**info, "reset_units": k}

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        executed = info.get("executed_action", int(action))
        tail, k = 0.0, 0
        if not (terminated or truncated):
            obs, tail, k, terminated, truncated, info = self._advance(obs, info)
        self.stats["choices"] += 1
        self.stats["forced"] += k
        self.control_units += 1 + k
        info = {**info, "executed_action": executed, "elapsed_units": 1 + k}
        return obs, float(r) + self.gamma * tail, terminated, truncated, info


class TimedNStepBuffer(ReplayBuffer):
    """경험마다 지난 5초 칸 수(units)를 아는 n-step 저장소(114.3절).

    묶음 = 이 경험부터 이어진 경험을 합계 window_units칸 이상이 될 때까지(최소 1개, 에피소드 끝·시간 끝에서 끊음).
    반환 = Σ_j γ^(앞 경험들의 칸 합) R_j(R_j는 감싸기가 이미 할인해 더한 경험 보상), 다음 값의 할인 = γ^(묶음 칸 합).
    칸이 모두 1이면 stable-baselines3의 NStepReplayBuffer(n = window_units)와 같다(시험으로 확인)."""

    def __init__(self, *args, gamma: float = GAMMA, window_units: int = WINDOW_S // STEP_S, **kwargs):
        super().__init__(*args, **kwargs)
        if self.optimize_memory_usage:
            raise NotImplementedError("TimedNStepBuffer는 optimize_memory_usage를 쓰지 않는다.")
        self.gamma, self.window_units = gamma, int(window_units)
        self.units = np.ones((self.buffer_size, self.n_envs), dtype=np.int32)

    def add(self, obs, next_obs, action, reward, done, infos) -> None:
        self.units[self.pos] = np.array([int(info.get("elapsed_units", 1)) for info in infos])
        super().add(obs, next_obs, action, reward, done, infos)

    def windows(self, batch_inds: np.ndarray, env_indices: np.ndarray) -> dict:
        """묶음 계산. 쓰는 자리(self.pos)를 넘지 않게 마지막 저장 경험을 잠시 '시간 끝'으로 본다(NStepReplayBuffer와 같음)."""
        last_valid = self.pos - 1
        original = self.timeouts[last_valid].copy()
        self.timeouts[last_valid] = np.logical_or(original, np.logical_not(self.dones[last_valid]))
        try:
            steps = np.arange(self.window_units).reshape(1, -1)          # 경험 하나는 1칸 이상이라 묶음은 많아야 window_units개
            idx = (batch_inds[:, None] + steps) % self.buffer_size
            e = env_indices[:, None]
            units = self.units[idx, e].astype(np.int64)
            cum = np.cumsum(units, axis=1)
            stop = np.logical_or(self.dones[idx, e], self.timeouts[idx, e])
            end = np.logical_or(stop, cum >= self.window_units).argmax(axis=1)
            last = (batch_inds + end) % self.buffer_size
            return {"idx": idx, "units": units, "cum": cum, "end": end, "mask": steps <= end[:, None], "last": last,
                    "next_dones": self.dones[last, env_indices][:, None].astype(np.float32),
                    "next_timeouts": self.timeouts[last, env_indices][:, None].astype(np.float32)}
        finally:
            self.timeouts[last_valid] = original

    def _get_samples(self, batch_inds: np.ndarray, env=None) -> ReplayBufferSamples:
        env_indices = np.random.randint(0, self.n_envs, size=batch_inds.shape)
        w = self.windows(batch_inds, env_indices)
        rewards = self._normalize_reward(self.rewards[w["idx"], env_indices[:, None]], env)
        before = w["cum"] - w["units"]                                    # 이 경험이 시작할 때까지 지난 칸
        returns = (rewards * self.gamma ** before * w["mask"]).sum(axis=1, keepdims=True).astype(np.float32)
        total = np.take_along_axis(w["cum"], w["end"][:, None], axis=1)
        discounts = (self.gamma ** total).astype(np.float32)
        next_obs = self._normalize_obs(self.next_observations[w["last"], env_indices], env)
        obs = self._normalize_obs(self.observations[batch_inds, env_indices], env)
        return ReplayBufferSamples(observations=self.to_torch(obs), actions=self.to_torch(self.actions[batch_inds, env_indices]),
                                   next_observations=self.to_torch(next_obs),
                                   dones=self.to_torch(w["next_dones"] * (1.0 - w["next_timeouts"])),
                                   rewards=self.to_torch(returns), discounts=self.to_torch(discounts))

    def window_stats(self) -> dict:
        """저장된 경험 전체의 묶음 길이(경험 수·초)와 경험 하나의 길이."""
        n = self.buffer_size if self.full else self.pos
        if n == 0:
            return {"transitions": 0}
        inds = np.arange(n)
        w = self.windows(inds, np.zeros(n, dtype=np.int64))
        secs = np.take_along_axis(w["cum"], w["end"][:, None], axis=1)[:, 0] * STEP_S
        return {"transitions": int(n), "mean_window_transitions": round(float((w["end"] + 1).mean()), 3),
                "mean_window_s": round(float(secs.mean()), 2), "max_window_s": int(secs.max()),
                "mean_transition_s": round(float(self.units[:n, 0].mean() * STEP_S), 2)}


def exploration_at(control_s: float) -> float:
    return EPS_START + (EPS_END - EPS_START) * min(1.0, control_s / EXPLORE_S)


def owed_updates(control_s: float) -> int:
    """H1과 같은 때까지의 경사 하강 횟수: 처음 RANDOM_S 뒤 TRAIN_EVERY_S마다 1번, 많아야 TOTAL_UPDATES."""
    return int(min(TOTAL_UPDATES, max(0, (control_s - RANDOM_S) // TRAIN_EVERY_S)))


def owed_syncs(control_s: float) -> int:
    return int(min(TOTAL_SYNCS, control_s // TARGET_EVERY_S))


def choice_dqn_class():
    from stable_baselines3.common.utils import polyak_update

    Base = explore_dqn.double_hold_aware_dqn_class()

    class ChoiceDQN(Base):
        """Double HoldAwareDQN에 시간 일정(114.3절)을 얹는다. 일정은 '단계'가 아니라 제어 시뮬레이션 시간(control_s)으로 센다.
        환경은 ChoiceOnlyEnv(Monitor(학습 환경))이고, 저장소는 TimedNStepBuffer다."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.control_s = 0.0          # 저장된다(이어 학습)
            self.target_syncs = 0
            self._seen_units = 0

        def _excluded_save_params(self) -> list[str]:
            # 저장소 종류를 남기지 않아야 평가가 이 파일 없이 원래 학습기로 불러진다(이어 학습은 load에서 다시 정함).
            return [*super()._excluded_save_params(), "replay_buffer_class", "replay_buffer_kwargs", "_seen_units"]

        def learn(self, *args, **kwargs):
            self._seen_units = 0                         # 새로 만든 감싸기의 칸 수는 0부터 센다
            return super().learn(*args, **kwargs)

        def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, dones, infos) -> None:
            super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)
            units = int(self.env.get_attr("control_units")[0])
            self.control_s += STEP_S * (units - self._seen_units)
            self._seen_units = units

        def _on_step(self) -> None:
            self._n_calls += 1
            want = owed_syncs(self.control_s)
            if want > self.target_syncs:
                polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), 1.0)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
                self.target_syncs = want
            self.exploration_rate = exploration_at(self.control_s)
            self.logger.record("rollout/exploration_rate", self.exploration_rate)

        def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
            if self.control_s < RANDOM_S:                # 학습 전 무작위 탐색(그대로 실행될 행동 안에서)
                allowed = np.asarray(self._last_obs)[:, -self.action_space.n:] > 0.5
                actions = np.array([self._random_allowed(row) for row in allowed])
                return actions, actions
            return super()._sample_action(0, action_noise, n_envs)

        def train(self, gradient_steps: int, batch_size: int = 100) -> None:
            owed = owed_updates(self.control_s) - self._n_updates
            if owed > 0:
                super().train(owed, batch_size)

    return ChoiceDQN


def budget_callback():
    """제어 시간이 BUDGET_S에 이르면 멈춘다(그 뒤 첫 경험은 저장하지 않음)."""
    from stable_baselines3.common.callbacks import BaseCallback

    class Budget(BaseCallback):
        def _on_step(self) -> bool:
            return self.model.control_s < BUDGET_S

    return Budget()


def checkpoint_callback(folder: Path):
    """제어 CHECKPOINT_S초마다 모델·저장소를 같은 이름으로 덮어써 저장한다."""
    from stable_baselines3.common.callbacks import BaseCallback

    class Checkpoint(BaseCallback):
        def _on_training_start(self) -> None:
            self.done = int(self.model.control_s // CHECKPOINT_S)

        def _on_step(self) -> bool:
            k = int(self.model.control_s // CHECKPOINT_S)
            if k > self.done:
                folder.mkdir(parents=True, exist_ok=True)
                self.model.save(folder / "model")
                self.model.save_replay_buffer(folder / "replay_buffer")
                (folder / "checkpoint.json").write_text(json.dumps({"control_s": self.model.control_s,
                                                                    "transitions": self.model.num_timesteps,
                                                                    "saved": time.strftime("%Y-%m-%d %H:%M:%S")}))
                self.done = k
            return True

    return Checkpoint()


def learner_fingerprint() -> str:
    return _fingerprint([ChoiceOnlyEnv, TimedNStepBuffer, exploration_at, owed_updates, owed_syncs, choice_dqn_class,
                         budget_callback], json.dumps(schedule(), sort_keys=True))


# ------------------------------------------------------------------ 설정 확인·기록
def _check_env(want: dict) -> None:
    for name in DQN_ENV_VARS:
        if os.environ.get(name, "") != want.get(name, ""):
            raise SystemExit(f"{name}={want.get(name, '(비움)')}로 돌린다(지금 {os.environ.get(name, '(비움)')}).")
    if os.environ.get("FOCUSED_REWARD", "queue") != "queue":
        raise SystemExit("보상은 기본(queue)이어야 한다.")


MOV_ENV = {"FOCUSED_DQN_NSTEPS": "10", "FOCUSED_DQN_DOUBLE": "1"}      # H1과 같은 학습(87절)
CHO_ENV: dict = {}                                                       # C는 이 파일의 시간 일정을 쓴다


def explore_record(experiment: str) -> dict:
    rec = {"experiment": experiment, "observation": "movement", "observation_code": observation_fingerprint(),
           "extra_features": N_MOVEMENTS, "double_dqn": True, "reward": "queue", "note": "진행 기록 114절, src/explore_choice.py"}
    if experiment == "MOV":
        rec.update(n_steps=H1_N_STEPS, baseline="H1(87절, n = 10 + Double)")
    else:
        rec.update(learner_code=learner_fingerprint(), schedule=schedule(), baseline="MOV(같은 관측, 5초마다 묻기)")
    return rec


def _write_record(seed_dir: Path, experiment: str) -> dict:
    record = explore_record(experiment)
    rec_file = seed_dir / "explore.json"
    if rec_file.exists():
        saved = json.loads(rec_file.read_text())
        keys = ("experiment", "observation_code", "learner_code")
        if tuple(saved.get(k) for k in keys) != tuple(record.get(k) for k in keys):
            raise SystemExit(f"{rec_file}: 이어 학습할 모델의 코드({[saved.get(k) for k in keys]})가 지금"
                             f"({[record.get(k) for k in keys]})과 다르다. 처음부터 다시 학습한다.")
    seed_dir.mkdir(parents=True, exist_ok=True)
    rec_file.write_text(json.dumps(record, ensure_ascii=False, indent=1))
    return record


# ------------------------------------------------------------------ 명령
def train_mov(rest: list[str]) -> None:
    """B: H1과 같은 학습 명령(focused_train)에 관측만 바꿔 끼운다."""
    _check_env(MOV_ENV)
    value = lambda name, default=None: rest[rest.index(name) + 1] if name in rest else default
    if value("--out") is None:
        raise SystemExit("--out(모델 폴더)이 필요하다.")
    if value("--timesteps") != str(H1_TIMESTEPS):
        raise SystemExit(f"H1과 같게 --timesteps {H1_TIMESTEPS}로 돌린다.")
    _write_record(Path(value("--out")) / f"seed_{value('--seed', '42')}", "MOV")
    install_observation()
    import focused_train
    sys.argv = [sys.argv[0], *rest]
    focused_train.main()


def train_cho(args) -> None:
    """C: 학습 환경을 ChoiceOnlyEnv로 감싸고 시간 일정으로 학습한다. 기록 형식은 focused_train과 같다(평가·학습곡선 도구가 읽음)."""
    import torch
    torch.set_num_threads(1)
    from stable_baselines3.common.monitor import Monitor

    import focused_train as ft
    from focused_data import COUNTS, TARGET_TLS, environment_fingerprint, load_counts, variant_signature
    from focused_rl import CROSSWALK_MIN_GREEN, CROSSWALKS, SignalConstraints
    from focused_scenario import ARRIVAL_WINDOW_S, ENTRY_BRANCHES, driver_vtype_attributes

    _check_env(CHO_ENV)
    out = Path(args.out) / f"seed_{args.seed}"
    checkpoint = out / "checkpoint"
    resume = args.resume and (checkpoint / "checkpoint.json").exists()
    env_fp = environment_fingerprint()
    if resume and json.loads((out / "run.json").read_text())["environment_fingerprint"] != env_fp:
        raise SystemExit("체크포인트의 환경 지문이 지금과 다르다. 처음부터 다시 학습한다.")
    record = _write_record(out, "CHO")
    install_observation()
    ChoiceDQN = choice_dqn_class()
    counts = load_counts(COUNTS)
    scenarios = ft.training_scenarios(counts, Path(args.split_file))
    constraints = SignalConstraints()
    workdir = Path(tempfile.mkdtemp(prefix=f"choice_dqn_{args.seed}_"))
    env = ft.WarmStartTrainingEnv(scenarios, counts, args.seed + (1000 if resume else 0), workdir,
                                  min_green=args.min_green, max_green=args.max_green, yellow=args.yellow, constraints=constraints)
    monitored = Monitor(env, filename=str(out / "monitor.csv"), override_existing=not resume)
    wrapped = ChoiceOnlyEnv(monitored, GAMMA)
    buffer = {"replay_buffer_class": TimedNStepBuffer,
              "replay_buffer_kwargs": {"gamma": GAMMA, "window_units": WINDOW_S // STEP_S}}
    if resume:
        model = ChoiceDQN.load(checkpoint / "model", env=wrapped, **buffer)
        model.load_replay_buffer(checkpoint / "replay_buffer")
        print(f"이어서 학습: 제어 {model.control_s:.0f}초부터", flush=True)
    else:
        model = ChoiceDQN(policy="MlpPolicy", env=wrapped, seed=args.seed, verbose=0, learning_starts=0, train_freq=1,
                          gradient_steps=1, target_update_interval=10 ** 9, exploration_initial_eps=EPS_START,
                          exploration_final_eps=EPS_END, **CHO_SETTINGS, **buffer)
        (out / "run.json").write_text(json.dumps({"environment_fingerprint": env_fp, "seed": args.seed,
                                                  "started": time.strftime("%Y-%m-%d %H:%M:%S")}))
    started = time.time()
    model.learn(total_timesteps=10 ** 9, reset_num_timesteps=not resume, progress_bar=False,
                callback=[budget_callback(), checkpoint_callback(checkpoint)])
    model.save(out / f"dqn_{TARGET_TLS}")
    wrapped.close()
    stored = model.replay_buffer.size()
    total_units = wrapped.control_units
    choice = {"control_s": model.control_s, "budget_s": BUDGET_S, "gradient_updates": model._n_updates,
              "target_syncs": model.target_syncs, "stored_transitions": stored,
              "replay_per_transition": round(model._n_updates * CHO_SETTINGS["batch_size"] / max(1, stored), 2),
              "this_run": {**dict(wrapped.stats), "control_units": total_units,
                           "choice_share": round(wrapped.stats["choices"] / max(1, total_units), 4)},
              "windows": model.replay_buffer.window_stats()}
    config = {
        "environment_fingerprint": env_fp, "variant": variant_signature(), "train_code_fingerprint": record["learner_code"],
        "seed": args.seed, "timesteps": int(model.num_timesteps), "resumed": resume,
        "split_file": str(args.split_file), "train_dates": sorted({d for d, _ in scenarios}),
        "episode": {"warmup_s": ft.WARMUP_S, "control_s": ft.CONTROL_S,
                    "start": "실제 운영 신호로 30분 돌린 뒤, 시각 P 이후 첫 녹색 시작에 넘겨받음"},
        "scenarios_available": len(scenarios), "episodes_this_run": len(env.history), "scenarios_used_this_run": env.history,
        "arrival_window_s": ARRIVAL_WINDOW_S, "driver_model_vtype_attributes": driver_vtype_attributes(),
        "entry_branches": ENTRY_BRANCHES, "seconds_this_run": round(time.time() - started, 1),
        "dqn": {**CHO_SETTINGS, **schedule(), "algorithm": "ChoiceDQN: Double HoldAwareDQN, 고를 때만 묻기, 시간 일정(114.3절)"},
        "explore": {"double_dqn": True, "reward": "queue", "experiment": "CHO"}, "choice": choice,
        "signal_constraints": {
            "vehicle_max_cumulative_wait_s": constraints.max_wait, "phase_min_green_s": constraints.phase_min,
            "crosswalks": {n: {"min_green_s": CROSSWALK_MIN_GREEN[n], "max_wait_s": c["max_wait"],
                               "phases": constraints.serving.get(n, [])} for n, c in CROSSWALKS.items()},
            "decisions": constraints.decisions, "rule_decided": dict(constraints.rule_decided),
            "ped_interventions": constraints.ped_interventions, "vehicle_interventions": constraints.vehicle_interventions,
            "max_green_switches": constraints.max_green_switches},
        "environment": {"delta_time": STEP_S, "yellow_time": args.yellow, "min_green": args.min_green,
                        "max_green": args.max_green, "time_to_teleport": -1},
    }
    (out / "training_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
    print(json.dumps({k: config[k] for k in ("seed", "timesteps", "resumed", "seconds_this_run", "choice")}, ensure_ascii=False), flush=True)


def evaluate(rest: list[str]) -> None:
    """B·C 모델을 기존 평가(focused_conflicts measure, PET 포함)로 돌린다. 바꿔 끼운 관측을 하위 실행에 넘기려고 fork로 돌린다."""
    import multiprocessing
    import focused_conflicts
    if not sys.platform.startswith("linux"):
        raise SystemExit("바꿔 끼운 관측을 하위 실행에 넘기려면 fork가 필요하다(Colab·리눅스에서 실행).")
    multiprocessing.set_start_method("fork", force=True)
    listed = lambda name: (lambda xs: xs[:next((i for i, a in enumerate(xs) if a.startswith("--")), len(xs))])(
        rest[rest.index(name) + 1:] if name in rest else [])
    if listed("--policies") != ["dqn"]:
        raise SystemExit("이 평가는 --policies dqn만 받는다(다른 정책은 기존 결과를 다시 쓴다).")
    models = listed("--models")
    if not models:
        raise SystemExit("--models가 필요하다.")
    for m in models:
        rec = Path(m) / "explore.json"
        saved = json.loads(rec.read_text()) if rec.exists() else {}
        if saved.get("observation_code") != observation_fingerprint():
            raise SystemExit(f"{m}: 이 관측 코드({observation_fingerprint()})로 학습한 모델이 아니다({saved.get('observation_code')}).")
        if not (Path(m) / "training_config.json").exists():
            raise SystemExit(f"{m}: 학습이 끝나지 않았다.")
    install_observation()
    sys.argv = [sys.argv[0], "measure", *rest]
    focused_conflicts.main()


class _Tap(gym.Wrapper):
    """점검용: 5초마다의 원래 보상을 적는다."""

    def __init__(self, env):
        super().__init__(env)
        self.rewards: list[float] = []

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        self.rewards.append(float(r))
        return obs, r, terminated, truncated, info


def check(args) -> None:
    """학습 전 점검(SUMO, 학습 날짜 두 에피소드). 하나라도 어긋나면 멈춘다(학습하지 않는다).
    1번 에피소드(5초마다, 무작위 허용 녹색): 관측 72칸·0~1, 끝 표시 = 운영 규칙의 허용 녹색, 8칸 = 단순 제어가 세는 이동별 기다리는 차 ÷
    진입로 감지 구간 크기, 8칸이 한 번 이상 0보다 큼, 규칙이 바꾼 선택 0.
    2번 에피소드(고를 때만 묻기): 받는 상태마다 허용 녹색 2개 이상, 경험 보상 = 안쪽 5초 보상의 할인 합, 칸 수 합 = 720, 규칙이 바꾼 선택 0."""
    import random

    import focused_train as ft
    from focused_data import COUNTS, TARGET_TLS, load_counts
    from focused_eval import QueueFirst
    from focused_rl import SignalConstraints

    _check_env(CHO_ENV)
    install_observation()
    counts = load_counts(COUNTS)
    first, second = [(args.date, p) for p in args.periods]
    constraints = SignalConstraints()
    env = ft.WarmStartTrainingEnv([first], counts, 42, Path(tempfile.mkdtemp(prefix="choice_check_")), constraints=constraints)
    n = env.action_space.n
    rng = random.Random(0)
    problems: list[str] = []
    report = {"date": args.date, "periods": args.periods, "observation_code": observation_fingerprint(),
              "learner_code": learner_fingerprint()}
    try:
        # 1. 이동별 관측(5초마다)
        obs, _ = env.reset()
        size = env.observation_space.shape[0]
        sim = env.current
        ts = sim.traffic_signals[TARGET_TLS]
        greens = [p.state for p in ts.green_phases]
        movements = protected_movements(greens)
        queue_first = QueueFirst(QueueFirst.movements(env.route), greens)
        steps, changed, positive, done = 0, 0, 0, False
        while not done:
            if obs.shape != (size,) or obs.min() < 0 or obs.max() > 1:
                problems.append(f"1-{steps}: 관측 모양 {obs.shape} 또는 범위 {obs.min():.2f}~{obs.max():.2f}")
            allowed = constraints.allowed(sim)
            if allowed_of(obs, n) != allowed:
                problems.append(f"1-{steps}: 끝 표시 {allowed_of(obs, n)}가 허용 녹색 {allowed}와 다름")
            cap = sim.traffic_signals[TARGET_TLS].observation_fn._capacity
            want = movement_features(queue_first.waiting(sim.sumo), movements, cap)
            got = obs[-n - N_MOVEMENTS:-n]
            if np.abs(got - np.asarray(want, dtype=np.float32)).max() > 1e-6:
                problems.append(f"1-{steps}: 8칸 {np.round(got, 3).tolist()} ≠ 단순 제어 셈 {np.round(want, 3).tolist()}")
            positive += bool(got.max() > 0)
            a = rng.choice(allowed)
            obs, _, terminated, truncated, info = env.step(a)
            changed += info["executed_action"] != a
            steps, done = steps + 1, terminated or truncated
        report["movement"] = {"observation_size": size, "steps": steps, "steps_with_waiting": positive,
                              "movements": ["->".join(m) for m in movements], "changed_by_rules": changed}
        if size != 64 + N_MOVEMENTS:
            problems.append(f"관측이 {size}칸({64 + N_MOVEMENTS}칸이어야 함)")
        if positive == 0:
            problems.append("8칸이 한 번도 0보다 크지 않음(이동 대응이 어긋났을 수 있음)")
        if changed:
            problems.append(f"1: 규칙이 바꾼 선택 {changed}번")
        # 2. 고를 때만 묻기
        env.scenarios = [second]
        tap = _Tap(env)
        wrapped = ChoiceOnlyEnv(tap, GAMMA)
        obs, info = wrapped.reset()
        reset_units = info["reset_units"]
        units, choices, done = reset_units, 0, False
        while not done:
            allowed = allowed_of(obs, n)
            if len(allowed) < 2:
                problems.append(f"2-{choices}: 고를 순간인데 허용 녹색 {allowed}")
            a = rng.choice(allowed)
            start = len(tap.rewards)
            obs, r, terminated, truncated, info = wrapped.step(a)
            inner = tap.rewards[start:]
            k = info["elapsed_units"]
            if len(inner) != k or abs(r - sum(GAMMA ** i * x for i, x in enumerate(inner))) > 1e-6:
                problems.append(f"2-{choices}: 경험 보상 {r:.6f}·칸 {k}가 안쪽 보상 {len(inner)}칸의 할인 합과 다름")
            if info["executed_action"] != a:
                problems.append(f"2-{choices}: 고른 {a} ≠ 실행 {info['executed_action']}")
            units, choices, done = units + k, choices + 1, terminated or truncated
        want_units = ft.CONTROL_S // STEP_S
        if units != want_units or wrapped.control_units != want_units:
            problems.append(f"2: 칸 수 합 {units}·{wrapped.control_units}(={want_units}이어야 함)")
        if wrapped.stats["changed_by_rules"]:
            problems.append(f"2: 규칙이 바꾼 선택 {wrapped.stats['changed_by_rules']}번")
        report["choice"] = {"units": units, "reset_units": reset_units, "choices": choices,
                            "choice_share": round(choices / max(1, units), 4),
                            "mean_transition_s": round(STEP_S * (units - reset_units) / max(1, choices), 2),
                            "stats": dict(wrapped.stats)}
    finally:
        env.close()
    report.update(problems=problems[:20], problem_count=len(problems), ok=not problems)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if problems:
        raise SystemExit("점검 실패: 학습하지 않는다.")
    print("점검 통과", flush=True)


def main() -> None:
    from focused_scenario import AIHUB_YELLOW_S
    if len(sys.argv) >= 2 and sys.argv[1] in ("train-mov", "eval"):
        {"train-mov": train_mov, "eval": evaluate}[sys.argv[1]](sys.argv[2:])
        return
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--date", default="20220811")
    c.add_argument("--periods", nargs=2, default=["0800", "1730"])
    c.add_argument("--out")
    t = sub.add_parser("train-cho")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--out", required=True)
    t.add_argument("--resume", action="store_true")
    t.add_argument("--split-file", default=str(PROJECT / "config/focused_split.json"))
    t.add_argument("--min-green", type=int, default=15)
    t.add_argument("--max-green", type=int, default=120)
    t.add_argument("--yellow", type=int, default=AIHUB_YELLOW_S)
    args = ap.parse_args()
    check(args) if args.cmd == "check" else train_cho(args)


if __name__ == "__main__":
    import explore_choice                     # 저장소 클래스가 '__main__'이 아니라 이 모듈 이름으로 저장되게
    explore_choice.main()
