from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from rl_environment import (
    REWARD_FUNCTION,
    SCENARIO_DIR,
    TARGET_TLS,
    YELLOW_TIME_SEC,
    make_environment,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="SUMO-RL 환경을 짧게 실행해 연결을 확인합니다.")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--scenario-dir", type=Path, default=SCENARIO_DIR)
    args = parser.parse_args()

    scenario_dir = args.scenario_dir.expanduser().resolve()
    network_root = ET.parse(scenario_dir / "network.net.xml").getroot()
    logic = network_root.find(f"tlLogic[@id='{TARGET_TLS}']")
    if logic is None:
        raise RuntimeError(f"도로망에서 고정신호 {TARGET_TLS}를 찾지 못했습니다.")
    fixed_yellow_durations = {
        int(float(phase.attrib["duration"]))
        for phase in logic.findall("phase")
        if "y" in phase.attrib["state"]
    }

    env = make_environment(seed=42, scenario_dir=scenario_dir)
    try:
        obs, info = env.reset(seed=42)
        print(f"제어 교차로: {TARGET_TLS}")
        print(f"관측값 개수: {obs.shape[0]}")
        print(f"선택 가능한 녹색 신호: {env.action_space.n}개")
        print(f"고정신호 황색시간: {sorted(fixed_yellow_durations)}초")
        print(f"DQN 황색신호 시간: {env.traffic_signals[TARGET_TLS].yellow_time}초")
        print(f"보상 함수: {REWARD_FUNCTION}")
        assert fixed_yellow_durations == {YELLOW_TIME_SEC}
        assert env.traffic_signals[TARGET_TLS].yellow_time == YELLOW_TIME_SEC

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
