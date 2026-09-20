from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gymnasium as gym

from rl_environment import (
    ACTION_INTERVAL_SEC,
    ENVIRONMENT_VERSION,
    PROJECT_ROOT,
    REWARD_FUNCTION,
    SCENARIO_DIR,
    TARGET_TLS,
    YELLOW_TIME_SEC,
    make_environment,
)


class CyclingScenarioEnv(gym.Env):
    """에피소드가 끝날 때마다 다음 날짜·시간대 시나리오로 교체한다.

    DQN 하나가 학습용 시나리오 전체를 경험하도록 하기 위한 얇은 포장이다.
    도로망은 같고 차량 경로만 달라진다는 전제에서 관측·행동 공간은 동일하다.
    """

    def __init__(self, scenario_dirs: list[Path], seed: int):
        super().__init__()
        self.scenario_dirs = scenario_dirs
        self.base_seed = seed
        self.episode = 0
        self.scenario_index = 0
        self.current_env = make_environment(
            seed=seed, scenario_dir=self.scenario_dirs[self.scenario_index]
        )
        self.observation_space = self.current_env.observation_space
        self.action_space = self.current_env.action_space
        self.metadata = self.current_env.metadata
        self.started = False

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if self.started:
            self.current_env.close()
            self.episode += 1
            self.scenario_index = self.episode % len(self.scenario_dirs)
            self.current_env = make_environment(
                seed=self.base_seed + self.episode,
                scenario_dir=self.scenario_dirs[self.scenario_index],
            )
            if self.current_env.observation_space != self.observation_space:
                raise RuntimeError("시나리오 사이의 관측 공간이 다릅니다.")
            if self.current_env.action_space != self.action_space:
                raise RuntimeError("시나리오 사이의 행동 공간이 다릅니다.")
        self.started = True
        episode_seed = self.base_seed + self.episode if seed is None else seed
        return self.current_env.reset(seed=episode_seed, options=options)

    def step(self, action):
        return self.current_env.step(action)

    def close(self) -> None:
        self.current_env.close()


def load_scenario_dirs(args: argparse.Namespace) -> list[Path]:
    """직접 지정한 시나리오 또는 splits.json의 학습 목록을 읽는다."""
    if args.split_file:
        split_file = args.split_file.expanduser().resolve()
        data = json.loads(split_file.read_text(encoding="utf-8"))
        scenario_dirs = [
            (split_file.parent / name).resolve()
            for name in data["splits"][args.split]
        ]
    else:
        scenario_dirs = [path.expanduser().resolve() for path in args.scenario_dirs]
    if not scenario_dirs:
        raise ValueError("학습할 시나리오가 없습니다.")
    return scenario_dirs


def build_environment(scenario_dirs: list[Path], seed: int):
    """시나리오가 하나면 기본 환경, 여러 개면 순환 환경을 만든다."""
    if len(scenario_dirs) == 1:
        return make_environment(seed=seed, scenario_dir=scenario_dirs[0])
    return CyclingScenarioEnv(scenario_dirs, seed)


def validate_resume_model(
    model_path: Path,
    scenario_dirs: list[Path],
) -> dict[str, Any]:
    """재개할 모델이 현재 황색시간·보상으로 학습됐는지 확인한다."""
    config_path = model_path.parent / "training_config.json"
    if not config_path.exists():
        raise RuntimeError(
            f"재개 정보가 없습니다: {config_path}\n"
            "이전 황색 2초 모델은 이어서 학습하지 말고 새로 학습하세요."
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("environment_version") != ENVIRONMENT_VERSION:
        raise RuntimeError(
            f"환경 버전이 달라 이어서 학습할 수 없습니다: "
            f"{config.get('environment_version')} != {ENVIRONMENT_VERSION}"
        )
    # Colab을 다시 연결하면 상위 경로가 달라질 수 있으므로 절대경로 전체가 아니라
    # 날짜·시간대 시나리오 폴더 이름을 비교한다.
    previous_scenarios = [
        Path(path).name for path in config.get("scenario_dirs", [])
    ]
    current_scenarios = [path.name for path in scenario_dirs]
    if previous_scenarios != current_scenarios:
        raise RuntimeError(
            "재개 모델과 현재 학습 데이터가 다릅니다. 샘플 모델을 전체 날짜 "
            "학습에 이어 붙이지 말고 같은 시나리오 목록을 사용하세요."
        )
    return config


def train_one_seed(
    args: argparse.Namespace,
    seed: int,
    scenario_dirs: list[Path],
) -> Path:
    """seed 하나의 DQN을 새로 만들거나 체크포인트에서 이어서 학습한다."""
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor

    output_root = args.output_dir.expanduser().resolve()
    output_dir = output_root / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = (
        args.checkpoint_dir.expanduser().resolve()
        if args.checkpoint_dir
        else output_root / "checkpoints"
    )
    checkpoint_dir = checkpoint_root / f"seed_{seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = build_environment(scenario_dirs, seed)
    monitored_env = Monitor(env, filename=str(output_dir / "monitor.csv"))
    tensorboard_dir = output_dir / "tensorboard"

    resume_model = args.resume_model.expanduser().resolve() if args.resume_model else None
    resume_replay = (
        args.resume_replay_buffer.expanduser().resolve()
        if args.resume_replay_buffer else None
    )
    if resume_model:
        previous_config = validate_resume_model(resume_model, scenario_dirs)
        if int(previous_config["seed"]) != seed:
            raise RuntimeError("재개 모델의 seed와 --seed 값이 다릅니다.")
        model = DQN.load(
            resume_model,
            env=monitored_env,
            tensorboard_log=str(tensorboard_dir),
        )
        if resume_replay:
            model.load_replay_buffer(resume_replay)
        reset_num_timesteps = False
        print(f"이어 학습 모델: {resume_model}")
        print(f"기존 학습 단계: {model.num_timesteps:,}")
    else:
        # 아래 값들은 DQN이 무엇을 얼마나 오래 기억하고, 언제부터 학습하며,
        # 무작위 탐험을 얼마나 할지를 정한다. 최종 실험에서는 이 값을 먼저
        # 고정한 뒤 테스트 날짜를 보지 않아야 공정하다.
        model = DQN(
            # MlpPolicy: 45개 숫자 관측값을 완전연결 신경망에 넣어 행동 4개의
            # 가치를 계산한다. 이미지 입력이 아니므로 CNN은 필요하지 않다.
            policy="MlpPolicy",
            env=monitored_env,
            # 한 번의 신경망 업데이트에서 파라미터를 바꾸는 크기이다.
            learning_rate=1e-3,
            # 과거 경험 (상태, 행동, 보상, 다음 상태)을 최대 10만 개 저장한다.
            buffer_size=100_000,
            # 처음 1,000단계는 경험만 모으고 그 뒤부터 신경망을 업데이트한다.
            learning_starts=1_000,
            # 저장된 경험 중 64개를 무작위로 꺼내 한 번 학습한다.
            batch_size=64,
            # SUMO 행동 4번마다 DQN을 한 번 업데이트한다.
            train_freq=4,
            # 학습 목표를 계산하는 별도 신경망을 5,000단계마다 동기화한다.
            target_update_interval=5_000,
            # 초반 20% 동안 무작위 행동 비율을 100%에서 5%까지 줄인다.
            exploration_fraction=0.2,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            tensorboard_log=str(tensorboard_dir),
            seed=seed,
            verbose=1,
        )
        reset_num_timesteps = True

    config: dict[str, Any] = {
        "environment_version": ENVIRONMENT_VERSION,
        "seed": seed,
        "target_tls": TARGET_TLS,
        "reward_function": REWARD_FUNCTION,
        "yellow_time_sec": YELLOW_TIME_SEC,
        "action_interval_sec": ACTION_INTERVAL_SEC,
        "additional_timesteps_requested": args.timesteps,
        "scenario_dirs": [str(path) for path in scenario_dirs],
        "resume_model": str(resume_model) if resume_model else None,
        "resume_replay_buffer": str(resume_replay) if resume_replay else None,
        "model_checkpoint_frequency": args.model_checkpoint_freq,
        "replay_checkpoint_frequency": args.replay_checkpoint_freq,
    }
    # 체크포인트 폴더에도 설정을 저장해야 Colab이 끊긴 뒤 호환성을 검사할 수 있다.
    for directory in (output_dir, checkpoint_dir):
        (directory / "training_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    model_checkpoints = CheckpointCallback(
        save_freq=args.model_checkpoint_freq,
        save_path=str(checkpoint_dir),
        name_prefix=f"dqn_{TARGET_TLS}_seed{seed}",
        save_replay_buffer=False,
    )
    # Stable-Baselines3는 replay buffer 저장 시 같은 단계의 모델도 함께 저장한다.
    state_checkpoints = CheckpointCallback(
        save_freq=args.replay_checkpoint_freq,
        save_path=str(checkpoint_dir),
        name_prefix=f"dqn_state_{TARGET_TLS}_seed{seed}",
        save_replay_buffer=True,
    )
    callbacks = CallbackList([model_checkpoints, state_checkpoints])

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            # 이어 학습에서는 기존 단계 번호를 유지해야 50,000, 100,000처럼
            # 누적된 시점의 체크포인트 이름과 탐험 스케줄이 올바르게 이어진다.
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=not args.no_progress_bar,
        )
        model_path = output_dir / f"dqn_{TARGET_TLS}"
        replay_path = output_dir / "replay_buffer.pkl"
        model.save(model_path)
        model.save_replay_buffer(replay_path)
        config["final_num_timesteps"] = model.num_timesteps
        (output_dir / "training_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"학습 모델 저장: {model_path}.zip")
        print(f"replay buffer 저장: {replay_path}")
        return model_path.with_suffix(".zip")
    finally:
        monitored_env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stable-Baselines3 DQN으로 교차로 신호를 학습합니다."
    )
    parser.add_argument(
        "--timesteps", type=int, default=100_000,
        help="이번 실행에서 추가로 학습할 단계 수",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds", nargs="+", type=int,
        help="최종 실험용 독립 학습 seed 목록. 지정하면 --seed 대신 사용합니다.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "results" / "dqn_20220810_0500",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        help="Google Drive 경로를 지정하면 런타임 종료 후에도 체크포인트가 남습니다.",
    )
    parser.add_argument("--model-checkpoint-freq", type=int, default=10_000)
    parser.add_argument("--replay-checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--resume-model", type=Path)
    parser.add_argument("--resume-replay-buffer", type=Path)
    parser.add_argument("--scenario-dirs", nargs="+", type=Path, default=[SCENARIO_DIR])
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--no-progress-bar", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = args.seeds or [args.seed]
    if (args.resume_model or args.resume_replay_buffer) and len(seeds) != 1:
        raise ValueError("이어 학습은 seed 하나씩 실행하세요.")
    if args.resume_replay_buffer and not args.resume_model:
        raise ValueError("replay buffer를 불러오려면 --resume-model도 필요합니다.")
    if args.model_checkpoint_freq <= 0 or args.replay_checkpoint_freq <= 0:
        raise ValueError("체크포인트 간격은 1 이상이어야 합니다.")

    scenario_dirs = load_scenario_dirs(args)
    for seed in seeds:
        print(f"\n=== 학습 seed {seed} 시작 ===")
        train_one_seed(args, seed, scenario_dirs)


if __name__ == "__main__":
    main()
