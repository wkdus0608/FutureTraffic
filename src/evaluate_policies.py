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

from rl_environment import PROJECT_ROOT, TARGET_TLS, make_environment


POLICIES = ("fixed", "random", "dqn")
DELTA_TIME = 5


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def parse_tripinfo(path: Path) -> dict[str, float | int]:
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
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_sumo_output_command(run_dir: Path) -> tuple[str, Path, Path]:
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
    run_dir = output_dir / "runs" / f"{policy}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    sumo_command, tripinfo_path, statistics_path = make_sumo_output_command(run_dir)

    env = make_environment(
        seed=seed,
        fixed_ts=policy == "fixed",
        additional_sumo_cmd=sumo_command,
    )
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    total_reward = 0.0

    try:
        observation, _ = env.reset(seed=seed)
        terminated = truncated = False
        decision = 0

        while not (terminated or truncated):
            if policy == "fixed":
                action = None
            elif policy == "random":
                action = int(rng.integers(env.action_space.n))
            else:
                if model is None:
                    raise RuntimeError("DQN 모델이 로드되지 않았습니다.")
                action, _ = model.predict(observation, deterministic=True)
                action = int(action)

            observation, reward, terminated, truncated, info = env.step(action)
            reward_value = float(np.asarray(reward).item())
            total_reward += reward_value
            decision += 1
            rows.append(
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

    write_csv(run_dir / "timeseries.csv", rows)
    queues = [float(row["local_queue_vehicles"]) for row in rows]
    local_waits = [float(row["local_accumulated_waiting_time_sec"]) for row in rows]
    local_speeds = [float(row["local_normalized_speed"]) for row in rows]
    system_stopped = [float(row["system_stopped_vehicles"]) for row in rows]
    system_speeds = [float(row["system_mean_speed_mps"]) for row in rows]
    final = rows[-1]

    result: dict[str, Any] = {
        "policy": policy,
        "seed": seed,
        "target_tls": TARGET_TLS,
        "decision_count": len(rows),
        "simulation_time_sec": final["simulation_time_sec"],
        "total_reward": total_reward,
        "local_mean_queue_vehicles": mean(queues),
        "local_max_queue_vehicles": max(queues, default=0.0),
        "local_queue_vehicle_seconds": sum(queues) * DELTA_TIME,
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
    names = {"fixed": "고정신호", "random": "무작위", "dqn": "DQN"}
    lines = [
        "# 신호제어 평가 결과",
        "",
        f"- 제어 교차로: `{TARGET_TLS}`",
        f"- 대기행렬 측정 간격: {DELTA_TIME}초",
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
                "sampling_interval_sec": DELTA_TIME,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"평가 완료: {output_dir}")


if __name__ == "__main__":
    main()
