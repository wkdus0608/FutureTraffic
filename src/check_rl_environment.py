from __future__ import annotations

import argparse

from rl_environment import TARGET_TLS, make_environment


def main() -> None:
    parser = argparse.ArgumentParser(description="SUMO-RL 환경을 짧게 실행해 연결을 확인합니다.")
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    env = make_environment(seed=42)
    try:
        obs, info = env.reset(seed=42)
        print(f"제어 교차로: {TARGET_TLS}")
        print(f"관측값 개수: {obs.shape[0]}")
        print(f"선택 가능한 녹색 신호: {env.action_space.n}개")

        for step in range(1, args.steps + 1):
            action = env.action_space.sample()
            _, reward, terminated, truncated, info = env.step(action)
            print(
                f"{step:02d}단계 | 신호 {int(action)} | "
                f"보상 {float(reward):.3f} | 시간 {info['step']:.0f}초"
            )
            if terminated or truncated:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
