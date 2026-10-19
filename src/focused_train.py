"""석천사거리(204820) DQN 학습.

에피소드 하나
- 학습 날짜의 한 시각 P를 무작위로 고른다(00:30~23:00, 15분 단위).
- P 30분 전부터 실제 운영 신호로 돌려 대기열이 있는 상태를 만든다(평가와 같은 '붐비는 도로'에서 배우게).
- P 이후 처음 녹색이 켜지는 순간부터 60분 동안 DQN이 5초마다 녹색 현시를 고른다(운영 규칙 적용).
- 차량은 그 90분 동안 기록대로 계속 들어온다.
학습 날짜는 config/focused_split.json의 train만 쓴다. 시험·검증 날짜는 쓰지 않는다.

체크포인트(모델 + 경험 저장소)를 주기적으로 덮어써 저장하고, --resume이면 마지막 체크포인트에서
이어서 학습한다(Colab 세션이 끊길 때 대비). 결과 폴더에는 환경 지문을 적어 두고, 평가는 지문이 같은
모델만 받는다.

사용:
  .venv/bin/python src/focused_train.py --seed 42 --timesteps 100000 --out results/focused_204820/models
  .venv/bin/python src/focused_train.py --seed 42 --timesteps 100000 --out <같은 폴더> --resume
"""
from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
import time
from pathlib import Path

import gymnasium as gym

from focused_data import (COUNTS, PROJECT, TARGET_TLS, environment_fingerprint, fingerprint, load_counts,
                          variant_signature)
from focused_scenario import (AIHUB_YELLOW_S, ARRIVAL_WINDOW_S, ENTRY_BRANCHES, dqn_template_network,
                              driver_vtype_attributes, period_start_s, real_signal_states, window_route_file)

CACHE = PROJECT / "results/focused_204820/cache"
OUT = PROJECT / "results/focused_204820/models"
SPLIT_FILE = PROJECT / "config/focused_split.json"
WARMUP_S = 1800      # DQN에게 넘기기 전 실제 신호로 돌리는 시간
CONTROL_S = 3600     # DQN이 신호를 고르는 시간
DQN_SETTINGS = {"learning_rate": 1e-3, "buffer_size": 100_000, "learning_starts": 1_000, "batch_size": 64,
                "train_freq": 4, "target_update_interval": 5_000, "exploration_fraction": 0.2,
                "exploration_initial_eps": 1.0, "exploration_final_eps": 0.05}
# 학습 설정 비교 실험(10-18, 진행 기록 75절): 목표망 교체 간격만 바꿔 학습한다. 비우면 기본값(5,000). 학습 기록의 dqn에 남는다.
if os.environ.get("FOCUSED_DQN_TARGET_UPDATE"):
    DQN_SETTINGS["target_update_interval"] = int(os.environ["FOCUSED_DQN_TARGET_UPDATE"])


def training_scenarios(counts: dict, split_file: Path = SPLIT_FILE) -> list[tuple[str, str]]:
    """(날짜, 시각) 후보: 학습 날짜에서 준비 30분과 제어 60분이 하루 안에 들어가는 15분 시각."""
    train_dates = set(json.loads(Path(split_file).read_text())["train"])
    return sorted((d, p) for d, p in counts
                  if d in train_dates and WARMUP_S <= period_start_s(p) <= 86400 - CONTROL_S)


def warm_start(date: str, period: str, counts: dict, route_file: Path) -> dict:
    """에피소드 준비: 90분 차량 파일을 쓰고, 준비 구간의 1초마다 실제 신호와 넘겨받을 녹색을 정한다."""
    from focused_rl import CROSSWALKS
    origin = period_start_s(period) - WARMUP_S
    window_route_file(date, origin, origin + WARMUP_S + CONTROL_S, route_file, origin_s=origin, counts=counts)
    states = real_signal_states(date, origin, WARMUP_S + 600)
    # P 이후 처음 녹색이 막 켜지는 순간에 넘겨받는다(황색 도중이나 녹색 중간에 끊지 않게).
    handover = next(i for i in range(WARMUP_S, len(states)) if "y" not in states[i] and states[i] != states[i - 1])
    served = {name: max((t + 1 for t in range(handover) if all(states[t][l] == "G" for l in c["links"])), default=0.0)
              for name, c in CROSSWALKS.items()}
    return {"date": date, "period": period, "warmup_states": states[:handover], "handover_state": states[handover],
            "handover_s": handover, "last_served": served}


class WarmStartTrainingEnv(gym.Env):
    """에피소드마다 학습 날짜의 한 시각을 골라, 대기열이 있는 상태에서 DQN에게 넘겨주는 환경."""

    def __init__(self, scenarios: list[tuple[str, str]], counts, seed: int, workdir: Path,
                 min_green: int = 15, max_green: int = 120, yellow: int = AIHUB_YELLOW_S, constraints=None):
        super().__init__()
        from focused_rl import ZoneObservation, zone_queue_reward
        from focused_sumo import make_environment
        self.scenarios, self.counts = scenarios, counts
        self.rng = random.Random(seed)
        self.seed_base, self.episode = seed, 0
        self.route = workdir / "train_routes.rou.xml"
        self.constraints = constraints
        self.history: list[tuple[str, str]] = []
        first = self._prepare()
        self.current = make_environment(seed=seed, net_file=dqn_template_network(CACHE), route_file=self.route,
                                        simulation_duration_sec=first["handover_s"] + CONTROL_S,
                                        observation_class=ZoneObservation, reward_fn=zone_queue_reward,
                                        min_green=min_green, max_green=max_green, yellow_time=yellow)
        if self.constraints is not None:
            self.constraints.bind(self.current)
        self._apply(first)
        self.observation_space = self.current.observation_space
        self.action_space = self.current.action_space
        self.started = False

    def _prepare(self) -> dict:
        date, period = self.rng.choice(self.scenarios)
        self.history.append((date, period))
        return warm_start(date, period, self.counts, self.route)

    def _apply(self, start: dict) -> None:
        env = self.current
        env.warmup_states, env.warmup_handover = start["warmup_states"], start["handover_state"]
        env.sim_max_time = start["handover_s"] + CONTROL_S
        self.last_served = start["last_served"]

    def reset(self, *, seed=None, options=None):
        if self.started:
            # 환경을 새로 만들지 않고 같은 경로 파일에 다음 에피소드 차량을 쓴 뒤 다시 시작한다.
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
        # 학습기(HoldAwareDQN)가 재생 버퍼에 저장할 '실제로 실행된 행동' = 이번 5초 동안 켜진 현시.
        info = dict(info or {})
        info["executed_action"] = int(self.current.traffic_signals[TARGET_TLS].green_phase)
        return obs, reward, terminated, truncated, info

    def close(self):
        self.current.close()


def _checkpoint_callback(folder: Path, every: int):
    """every 단계마다 모델과 경험 저장소를 같은 이름으로 덮어써 저장한다(Drive 용량을 아끼려고 마지막 것만)."""
    from stable_baselines3.common.callbacks import BaseCallback

    class LatestCheckpoint(BaseCallback):
        def _on_step(self) -> bool:
            if every and self.num_timesteps % every == 0:
                folder.mkdir(parents=True, exist_ok=True)
                self.model.save(folder / "model")
                self.model.save_replay_buffer(folder / "replay_buffer")
                (folder / "checkpoint.json").write_text(json.dumps({"timesteps": self.num_timesteps,
                                                                    "saved": time.strftime("%Y-%m-%d %H:%M:%S")}))
            return True

    return LatestCheckpoint()


def train(args) -> None:
    import torch
    torch.set_num_threads(1)  # 여러 학습을 동시에 돌리므로 프로세스당 1개 코어만 쓴다.
    from stable_baselines3.common.monitor import Monitor

    from focused_rl import CROSSWALK_MIN_GREEN, CROSSWALKS, SignalConstraints, hold_aware_dqn_class
    DQN = hold_aware_dqn_class()

    counts = load_counts(COUNTS)
    scenarios = training_scenarios(counts, Path(args.split_file))
    out = Path(args.out) / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "checkpoint"
    resume = args.resume and (checkpoint / "checkpoint.json").exists()
    env_fp = environment_fingerprint()
    if resume:
        saved = json.loads((out / "run.json").read_text())
        if saved["environment_fingerprint"] != env_fp:
            raise SystemExit(f"체크포인트의 환경 지문 {saved['environment_fingerprint']}이 지금 {env_fp}과 다릅니다. "
                             "환경이 바뀌었으니 처음부터 다시 학습하세요.")
    constraints = SignalConstraints()
    workdir = Path(tempfile.mkdtemp(prefix=f"focused_dqn_{args.seed}_"))
    env = WarmStartTrainingEnv(scenarios, counts, args.seed + (1000 if resume else 0), workdir,
                               min_green=args.min_green, max_green=args.max_green, yellow=args.yellow,
                               constraints=constraints)
    monitored = Monitor(env, filename=str(out / "monitor.csv"), override_existing=not resume)
    if resume:
        model = DQN.load(checkpoint / "model", env=monitored)
        model.load_replay_buffer(checkpoint / "replay_buffer")
        print(f"이어서 학습: {model.num_timesteps}단계부터")
    else:
        # 64-64
        model = DQN(policy="MlpPolicy", env=monitored, seed=args.seed, verbose=0, **DQN_SETTINGS)
        (out / "run.json").write_text(json.dumps({"environment_fingerprint": env_fp, "seed": args.seed,
                                                  "started": time.strftime("%Y-%m-%d %H:%M:%S")}))
    remaining = max(0, args.timesteps - model.num_timesteps)
    started = time.time()
    model.learn(total_timesteps=remaining, reset_num_timesteps=not resume, progress_bar=False,
                callback=_checkpoint_callback(checkpoint, args.checkpoint_freq))
    model.save(out / f"dqn_{TARGET_TLS}")
    monitored.close()
    config = {
        "environment_fingerprint": env_fp, "variant": variant_signature(),
        "train_code_fingerprint": fingerprint(["src/focused_train.py"]),
        "seed": args.seed, "timesteps": args.timesteps, "resumed": resume,
        "split_file": str(args.split_file), "train_dates": sorted({d for d, _ in scenarios}),
        "episode": {"warmup_s": WARMUP_S, "control_s": CONTROL_S,
                    "start": "실제 운영 신호로 30분 돌린 뒤, 시각 P 이후 첫 녹색 시작에 넘겨받음"},
        "scenarios_available": len(scenarios), "episodes_this_run": len(env.history),
        "scenarios_used_this_run": env.history,
        "arrival_window_s": ARRIVAL_WINDOW_S, "driver_model_vtype_attributes": driver_vtype_attributes(),
        "entry_branches": ENTRY_BRANCHES, "seconds_this_run": round(time.time() - started, 1),
        "dqn": {**DQN_SETTINGS, "algorithm": "HoldAwareDQN: 실제 실행 행동 저장, 미래 가치와 행동 선택 모두 그대로 실행될 행동 안에서"},
        "signal_constraints": {
            "vehicle_max_cumulative_wait_s": constraints.max_wait, "phase_min_green_s": constraints.phase_min,
            "crosswalks": {n: {"min_green_s": CROSSWALK_MIN_GREEN[n], "max_wait_s": c["max_wait"],
                               "phases": constraints.serving.get(n, [])} for n, c in CROSSWALKS.items()},
            "decisions": constraints.decisions, "rule_decided": dict(constraints.rule_decided),
            # DQN은 그대로 실행될 행동 안에서만 고르므로 아래 셋(규칙이 DQN 선택을 바꾼 수)은 0이어야 한다.
            "ped_interventions": constraints.ped_interventions,
            "vehicle_interventions": constraints.vehicle_interventions,
            "max_green_switches": constraints.max_green_switches},
        "environment": {"delta_time": 5, "yellow_time": args.yellow, "min_green": args.min_green,
                        "max_green": args.max_green, "time_to_teleport": -1},
    }
    (out / "training_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in config.items() if k != "scenarios_used_this_run"}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--split-file", default=str(SPLIT_FILE), help="train 날짜만 학습한다")
    parser.add_argument("--min-green", type=int, default=15, help="보행자 횡단이 없는 현시의 최소 녹색(초)")
    parser.add_argument("--max-green", type=int, default=120)
    parser.add_argument("--yellow", type=int, default=AIHUB_YELLOW_S, help="황색시간(초). 실제 운영 신호는 대부분 4초")
    parser.add_argument("--checkpoint-freq", type=int, default=10_000, help="이 단계마다 체크포인트를 덮어써 저장")
    parser.add_argument("--resume", action="store_true", help="out 폴더의 마지막 체크포인트에서 이어서 학습")
    parser.add_argument("--out", default=str(OUT))
    train(parser.parse_args())


if __name__ == "__main__":
    main()
