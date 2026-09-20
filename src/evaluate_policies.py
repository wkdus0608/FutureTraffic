from __future__ import annotations

import argparse
import csv
import json
import statistics
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from rl_environment import (
    ACTION_INTERVAL_SEC,
    PROJECT_ROOT,
    QUEUE_MEASUREMENT_INTERVAL_SEC,
    TARGET_TLS,
    make_environment,
)


POLICIES = ("fixed", "random", "dqn")


def mean(values: list[float]) -> float:
    """빈 목록에서도 오류가 나지 않는 산술평균 함수."""
    return statistics.fmean(values) if values else 0.0


def sample_std(values: list[float]) -> float:
    """여러 seed 결과의 표본 표준편차를 계산한다."""
    return statistics.stdev(values) if len(values) > 1 else 0.0


def parse_tripinfo(path: Path) -> dict[str, float | int]:
    """SUMO가 차량별로 기록한 운행 결과를 읽는다.

    이 값들은 5초마다 표본을 뽑아 만든 값이 아니다. SUMO가 각 차량의 전체 운행을
    추적해서 저장한 값이므로, ``global_mean_waiting_time_sec``은 이번 대기행렬
    측정 간격 문제와 별도로 계산되는 핵심 지표이다.
    """
    trips = ET.parse(path).getroot().findall("tripinfo")
    completed = [trip for trip in trips if float(trip.attrib.get("arrival", "-1")) >= 0]
    waits = [float(trip.attrib["waitingTime"]) for trip in trips]
    durations = [float(trip.attrib["duration"]) for trip in trips]
    losses = [float(trip.attrib["timeLoss"]) for trip in trips]
    return {
        "recorded_vehicles": len(trips),
        "completed_tripinfo_vehicles": len(completed),
        "unfinished_tripinfo_vehicles": len(trips) - len(completed),
        "global_mean_waiting_time_sec": mean(waits),
        "global_mean_travel_time_sec": mean(durations),
        "global_mean_time_loss_sec": mean(losses),
    }


def parse_statistics(path: Path) -> dict[str, int]:
    """SUMO 전체 실행에서 충돌·텔레포트·투입 차량 수를 읽는다."""
    root = ET.parse(path).getroot()
    vehicles = root.find("vehicles")
    safety = root.find("safety")
    teleports = root.find("teleports")
    if vehicles is None or safety is None or teleports is None:
        raise ValueError(f"SUMO 통계 파일의 형식을 확인할 수 없습니다: {path}")
    return {
        "loaded_vehicles": int(vehicles.attrib["loaded"]),
        "inserted_vehicles": int(vehicles.attrib["inserted"]),
        "running_vehicles_at_end": int(vehicles.attrib["running"]),
        "waiting_vehicles_at_end": int(vehicles.attrib["waiting"]),
        "collision_count": int(safety.attrib["collisions"]),
        "emergency_stop_count": int(safety.attrib["emergencyStops"]),
        "emergency_braking_count": int(safety.attrib["emergencyBraking"]),
        "teleport_count_from_statistics": int(teleports.attrib["total"]),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """딕셔너리 목록을 열 이름이 있는 CSV 파일로 저장한다."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_sumo_output_command(run_dir: Path) -> tuple[str, Path, Path]:
    """SUMO에 차량별 결과와 안전성 통계를 저장하라고 전달할 옵션을 만든다."""
    tripinfo = (run_dir / "tripinfo.xml").resolve()
    summary = (run_dir / "summary.xml").resolve()
    statistics_file = (run_dir / "statistics.xml").resolve()
    for path in (tripinfo, summary, statistics_file):
        if " " in str(path):
            raise ValueError(f"평가 출력 경로에 공백을 사용할 수 없습니다: {path}")
    command = " ".join(
        [
            "--tripinfo-output",
            str(tripinfo),
            "--tripinfo-output.write-unfinished",
            "true",
            "--summary-output",
            str(summary),
            "--statistic-output",
            str(statistics_file),
            "--duration-log.statistics",
            "true",
            "--no-step-log",
            "true",
        ]
    )
    return command, tripinfo, statistics_file


def run_policy(
    policy: str,
    seed: int,
    output_dir: Path,
    model: Any | None,
) -> dict[str, Any]:
    """정책 하나를 seed 하나로 30분간 실행하고 평가 지표를 반환한다.

    처리 순서
    ---------
    1. fixed, random, dqn 중 하나가 5초마다 사용할 행동을 정한다.
    2. ``env.step(action)``이 SUMO를 5초 진행한다.
    3. 환경 내부 기록기가 그 5초를 1초씩 빠짐없이 기록한다.
    4. 종료 후 차량별 결과, 대기행렬, 안전성 지표를 하나로 합친다.
    """
    run_dir = output_dir / "runs" / f"{policy}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    sumo_command, tripinfo_path, statistics_path = make_sumo_output_command(run_dir)

    env = make_environment(
        seed=seed,
        fixed_ts=policy == "fixed",
        additional_sumo_cmd=sumo_command,
        # 행동은 여전히 5초마다 선택하지만, 평가용 대기행렬은 매초 저장한다.
        record_per_second_queue=True,
    )
    rng = np.random.default_rng(seed)
    # decision_rows: 5초마다 한 줄. 어떤 행동을 골랐고 어떤 보상을 받았는지 확인한다.
    decision_rows: list[dict[str, Any]] = []
    total_reward = 0.0

    try:
        observation, _ = env.reset(seed=seed)
        terminated = truncated = False
        decision = 0

        while not (terminated or truncated):
            # 세 정책은 행동을 고르는 방법만 다르고 나머지 SUMO 조건은 같다.
            if policy == "fixed":
                # None을 전달하면 도로망에 저장된 고정신호 계획을 그대로 따른다.
                action = None
            elif policy == "random":
                # 비교 기준: 가능한 네 신호 중 하나를 무작위로 선택한다.
                action = int(rng.integers(env.action_space.n))
            else:
                if model is None:
                    raise RuntimeError("DQN 모델이 로드되지 않았습니다.")
                # 학습된 DQN은 현재 관측값 45개를 보고 녹색 신호 하나를 고른다.
                # deterministic=True는 평가 중 탐험용 무작위 행동을 사용하지 않는다는 뜻이다.
                action, _ = model.predict(observation, deterministic=True)
                action = int(action)

            # 한 번 호출할 때 SUMO 시간이 ACTION_INTERVAL_SEC(현재 5초) 흐른다.
            observation, reward, terminated, truncated, info = env.step(action)
            reward_value = float(np.asarray(reward).item())
            total_reward += reward_value
            decision += 1
            decision_rows.append(
                {
                    "policy": policy,
                    "seed": seed,
                    "decision": decision,
                    "simulation_time_sec": float(info["step"]),
                    "action": "fixed" if action is None else action,
                    "reward": reward_value,
                    "local_queue_vehicles": int(info[f"{TARGET_TLS}_stopped"]),
                    "local_accumulated_waiting_time_sec": float(
                        info[f"{TARGET_TLS}_accumulated_waiting_time"]
                    ),
                    "local_normalized_speed": float(info[f"{TARGET_TLS}_average_speed"]),
                    "system_running_vehicles": int(info["system_total_running"]),
                    "system_backlogged_vehicles": int(info["system_total_backlogged"]),
                    "system_stopped_vehicles": int(info["system_total_stopped"]),
                    "system_arrived_vehicles": int(info["system_total_arrived"]),
                    "system_departed_vehicles": int(info["system_total_departed"]),
                    "system_teleported_vehicles": int(info["system_total_teleported"]),
                    "system_mean_speed_mps": float(info["system_mean_speed"]),
                }
            )
    finally:
        env.close()

    # env.step()이 끝날 때만 저장한 5초 단위 자료이다.
    # 행동·보상 분석용으로 남기지만, 대기행렬 평균 계산에는 사용하지 않는다.
    write_csv(run_dir / "decision_timeseries_5s.csv", decision_rows)

    # 환경이 SUMO의 매 1초마다 모은 평가 전용 자료이다.
    # policy와 seed를 붙이면 여러 실행의 CSV를 나중에 합쳐도 구분할 수 있다.
    queue_rows = [
        {"policy": policy, "seed": seed, **record}
        for record in env.per_second_queue_records
    ]
    write_csv(run_dir / "queue_timeseries_1s.csv", queue_rows)

    # 평균 대기행렬 = 매초 정지 차량 수의 평균
    # 차량·초(vehicle-seconds) = 매초 정지 차량 수를 전부 더한 값
    # 예: 차량 3대가 10초 동안 서 있으면 30 차량·초이다.
    queues = [float(row["local_queue_vehicles"]) for row in queue_rows]

    # 아래 보조 지표들은 여전히 행동 직후(5초 간격) 정보이다.
    # 대기행렬 핵심 지표와 혼동하지 않도록 이름과 계산 근거를 분리한다.
    local_waits = [
        float(row["local_accumulated_waiting_time_sec"]) for row in decision_rows
    ]
    local_speeds = [float(row["local_normalized_speed"]) for row in decision_rows]
    system_stopped = [float(row["system_stopped_vehicles"]) for row in decision_rows]
    system_speeds = [float(row["system_mean_speed_mps"]) for row in decision_rows]
    final = decision_rows[-1]

    result: dict[str, Any] = {
        "policy": policy,
        "seed": seed,
        "target_tls": TARGET_TLS,
        "decision_count": len(decision_rows),
        "queue_sample_count": len(queue_rows),
        "action_interval_sec": ACTION_INTERVAL_SEC,
        "queue_measurement_interval_sec": QUEUE_MEASUREMENT_INTERVAL_SEC,
        "simulation_time_sec": final["simulation_time_sec"],
        "total_reward": total_reward,
        "local_mean_queue_vehicles": mean(queues),
        "local_max_queue_vehicles": max(queues, default=0.0),
        "local_queue_vehicle_seconds": (
            sum(queues) * QUEUE_MEASUREMENT_INTERVAL_SEC
        ),
        "local_zero_queue_fraction": (
            sum(queue == 0 for queue in queues) / len(queues) if queues else 0.0
        ),
        "local_mean_accumulated_waiting_time_sec": mean(local_waits),
        "local_mean_normalized_speed": mean(local_speeds),
        "system_mean_stopped_vehicles": mean(system_stopped),
        "system_mean_speed_mps": mean(system_speeds),
        "arrived_vehicles": final["system_arrived_vehicles"],
        "departed_vehicles": final["system_departed_vehicles"],
        "backlogged_vehicles": final["system_backlogged_vehicles"],
        "teleported_vehicles": final["system_teleported_vehicles"],
    }
    result.update(parse_tripinfo(tripinfo_path))
    result.update(parse_statistics(statistics_path))
    result["not_arrived_vehicles"] = int(result["loaded_vehicles"]) - int(
        result["arrived_vehicles"]
    )
    result["completion_rate"] = (
        float(result["arrived_vehicles"]) / float(result["loaded_vehicles"])
        if result["loaded_vehicles"]
        else 0.0
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def summarize_by_policy(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """같은 정책을 여러 seed로 실행한 결과의 평균과 표준편차를 만든다."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[str(result["policy"])].append(result)

    metrics = [
        "global_mean_waiting_time_sec",
        "global_mean_travel_time_sec",
        "global_mean_time_loss_sec",
        "local_mean_queue_vehicles",
        "local_max_queue_vehicles",
        "local_queue_vehicle_seconds",
        "local_mean_accumulated_waiting_time_sec",
        "local_mean_normalized_speed",
        "arrived_vehicles",
        "teleported_vehicles",
        "collision_count",
        "not_arrived_vehicles",
        "total_reward",
    ]
    summaries: list[dict[str, Any]] = []
    for policy in POLICIES:
        runs = grouped.get(policy, [])
        if not runs:
            continue
        row: dict[str, Any] = {"policy": policy, "runs": len(runs)}
        for metric in metrics:
            values = [float(run[metric]) for run in runs]
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_std"] = sample_std(values)
        summaries.append(row)
    return summaries


def write_report(path: Path, results: list[dict[str, Any]]) -> None:
    """사람이 바로 읽을 수 있는 핵심 비교표를 Markdown으로 저장한다."""
    names = {"fixed": "고정신호", "random": "무작위", "dqn": "DQN"}
    lines = [
        "# 신호제어 평가 결과",
        "",
        f"- 제어 교차로: `{TARGET_TLS}`",
        f"- 신호 선택 간격: {ACTION_INTERVAL_SEC}초",
        f"- 대기행렬 측정 간격: {QUEUE_MEASUREMENT_INTERVAL_SEC}초",
        "",
        "| 방법 | seed | 전체 평균 대기(초) | 교차로 평균 대기행렬(대) | 최대 대기행렬(대) | 도착(대) | 미도착(대) | 충돌 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {names[str(result['policy'])]} | {result['seed']} | "
            f"{result['global_mean_waiting_time_sec']:.2f} | "
            f"{result['local_mean_queue_vehicles']:.2f} | "
            f"{result['local_max_queue_vehicles']:.0f} | "
            f"{result['arrived_vehicles']} | {result['not_arrived_vehicles']} | "
            f"{result['collision_count']} |"
        )
    lines.extend(
        [
            "",
            "> 학습 결론은 여러 seed와 학습에 사용하지 않은 교통량을 평가한 후 내린다.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """터미널이나 Colab에서 전달한 평가 옵션을 읽는다."""
    parser = argparse.ArgumentParser(
        description="고정신호·무작위 신호·DQN을 동일한 SUMO 조건에서 평가합니다."
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=POLICIES,
        default=list(POLICIES),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "results" / "dqn_20220810_0500" / "dqn_204820.zip",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "evaluation_20220810_0500",
    )
    return parser.parse_args()


def main() -> None:
    """모델 로드 → 정책별 반복 평가 → CSV와 보고서 저장을 순서대로 수행한다."""
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = None
    if "dqn" in args.policies:
        model_path = args.model.expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(
                f"DQN 모델이 없습니다: {model_path}\n"
                "먼저 train_dqn.py를 실행하세요."
            )
        from stable_baselines3 import DQN

        model = DQN.load(model_path)

    results: list[dict[str, Any]] = []
    for policy in args.policies:
        for seed in args.seeds:
            print(f"[{policy}] seed={seed} 평가 시작")
            result = run_policy(policy, seed, output_dir, model)
            results.append(result)
            print(
                f"  평균 대기 {result['global_mean_waiting_time_sec']:.2f}초 | "
                f"교차로 평균 대기행렬 {result['local_mean_queue_vehicles']:.2f}대 | "
                f"도착 {result['arrived_vehicles']}대"
            )

    write_csv(output_dir / "runs.csv", results)
    summaries = summarize_by_policy(results)
    write_csv(output_dir / "summary_by_policy.csv", summaries)
    write_report(output_dir / "report.md", results)
    (output_dir / "evaluation_config.json").write_text(
        json.dumps(
            {
                "policies": args.policies,
                "seeds": args.seeds,
                "model": str(args.model.expanduser().resolve()),
                "target_tls": TARGET_TLS,
                "action_interval_sec": ACTION_INTERVAL_SEC,
                "queue_sampling_interval_sec": QUEUE_MEASUREMENT_INTERVAL_SEC,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"평가 완료: {output_dir}")


if __name__ == "__main__":
    main()
