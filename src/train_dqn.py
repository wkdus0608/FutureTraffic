from __future__ import annotations

import argparse
from pathlib import Path

from rl_environment import PROJECT_ROOT, TARGET_TLS, make_environment


def main() -> None:
    parser = argparse.ArgumentParser(description="Stable-Baselines3 DQN으로 신호를 학습합니다.")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / "results" / "dqn_20220810_0500"
    output_dir.mkdir(parents=True, exist_ok=True)

    env = make_environment(
        seed=args.seed,
        output_prefix=output_dir / "training_metrics",
    )

    from stable_baselines3 import DQN
    from stable_baselines3.common.monitor import Monitor

    monitored_env = Monitor(env, filename=str(output_dir / "monitor.csv"))
    model = DQN(
        policy="MlpPolicy",
        env=monitored_env,
        learning_rate=1e-3,
        buffer_size=100_000,
        learning_starts=1_000,
        batch_size=64,
        train_freq=4,
        target_update_interval=5_000,
        exploration_fraction=0.2,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        tensorboard_log=str(output_dir / "tensorboard"),
        seed=args.seed,
        verbose=1,
    )

    try:
        model.learn(total_timesteps=args.timesteps, progress_bar=True)
        model_path = output_dir / f"dqn_{TARGET_TLS}"
        model.save(model_path)
        print(f"학습 모델 저장: {model_path}.zip")
    finally:
        monitored_env.close()


if __name__ == "__main__":
    main()
