from __future__ import annotations

import argparse
import csv
import json
import statistics
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rl_environment import (
    ACTION_INTERVAL_SEC,
    ENVIRONMENT_VERSION,
    PROJECT_ROOT,
    QUEUE_MEASUREMENT_INTERVAL_SEC,
    REWARD_FUNCTION,
    SCENARIO_DIR,
    TARGET_TLS,
    YELLOW_TIME_SEC,
    make_environment,
)


POLICIES = ("fixed", "random", "dqn")


@dataclass
class ModelSpec:
    """평가할 DQN 모델과 그 모델을 학습한 seed를 함께 보관한다."""

    path: Path
    training_seed: int
    model: Any


def mean(values: list[float]) -> float:
    """빈 목록에서도 오류가 나지 않는 산술평균 함수."""
    return statistics.fmean(values) if values else 0.0


def sample_std(values: list[float]) -> float:
    """여러 독립 실행 결과의 표본 표준편차를 계산한다."""
    return statistics.stdev(values) if len(values) > 1 else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """딕셔너리 목록을 열 이름이 있는 CSV 파일로 저장한다."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_tripinfo(path: Path) -> dict[str, float | int]:
    """SUMO의 차량별 기록에서 *도착한 차량*의 평균만 계산한다.

    미도착 차량의 duration은 완전한 통행시간이 아니므로 평균에 섞지 않는다.
    미완료 수와 완료율은 별도로 보고해 성능이 좋아 보이는 착시를 막는다.
    """
    trips = ET.parse(path).getroot().findall("tripinfo")
    completed = [trip for trip in trips if float(trip.attrib.get("arrival", "-1")) >= 0]

    def completed_values(attribute: str) -> list[float]:
        return [float(trip.attrib.get(attribute, "0")) for trip in completed]

    waits = completed_values("waitingTime")
    durations = completed_values("duration")
    losses = completed_values("timeLoss")
    depart_delays = completed_values("departDelay")
    total_delays = [loss + delay for loss, delay in zip(losses, depart_delays)]
    return {
        "recorded_tripinfo_vehicles": len(trips),
        "completed_tripinfo_vehicles": len(completed),
        "unfinished_tripinfo_vehicles": len(trips) - len(completed),
        "global_mean_waiting_time_sec": mean(waits),
        "global_mean_travel_time_sec": mean(durations),
        "global_mean_time_loss_sec": mean(losses),
        "global_mean_depart_delay_sec": mean(depart_delays),
        # 도로 진입 전 지연까지 포함한 전체 지연이다.
        "global_mean_total_delay_sec": mean(total_delays),
    }


def parse_statistics(path: Path) -> dict[str, int]:
    """SUMO 전체 실행에서 투입 차량 수와 안전성 통계를 읽는다."""
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
            "--tripinfo-output", str(tripinfo),
            "--tripinfo-output.write-unfinished", "true",
            "--summary-output", str(summary),
            "--statistic-output", str(statistics_file),
            "--duration-log.statistics", "true",
            "--no-step-log", "true",
        ]
    )
    return command, tripinfo, statistics_file


def run_policy(
    policy: str,
    evaluation_seed: int,
    output_dir: Path,
    scenario_dir: Path,
    model_spec: ModelSpec | None = None,
) -> dict[str, Any]:
    """정책 하나를 시나리오 하나에서 실행하고 지표를 반환한다.

    evaluation_seed는 SUMO 실행 seed이고 training_seed는 DQN 학습 seed이다.
    둘을 분리해야 같은 모델 반복평가와 독립 모델 반복학습을 구분할 수 있다.
    """
    scenario_id = scenario_dir.name
    training_seed = model_spec.training_seed if model_spec else None
    run_name = f"{policy}_eval{evaluation_seed}"
    if policy == "dqn":
        if model_spec is None:
            raise RuntimeError("DQN 정책에는 학습된 모델이 필요합니다.")
        run_name = f"dqn_train{training_seed}_eval{evaluation_seed}"

    run_dir = output_dir / "runs" / scenario_id / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    sumo_command, tripinfo_path, statistics_path = make_sumo_output_command(run_dir)
    env = make_environment(
        seed=evaluation_seed,
        fixed_ts=policy == "fixed",
        additional_sumo_cmd=sumo_command,
        record_per_second_queue=True,
        scenario_dir=scenario_dir,
    )
    rng = np.random.default_rng(evaluation_seed)
    decision_rows: list[dict[str, Any]] = []
    total_reward = 0.0

    try:
        observation, _ = env.reset(seed=evaluation_seed)
        terminated = truncated = False
        decision = 0
        while not (terminated or truncated):
            # 행동 선택만 다르고 도로·차량·황색시간은 세 정책 모두 같다.
            if policy == "fixed":
                # None이면 도로망에 저장된 원래 고정신호 프로그램을 그대로 쓴다.
                action = None
            elif policy == "random":
                # 학습 없이 네 녹색 현시 중 하나를 무작위로 고르는 낮은 기준선이다.
                action = int(rng.integers(env.action_space.n))
            else:
                # 평가에서는 탐험을 끄고 DQN이 가장 좋다고 판단한 행동만 고른다.
                action, _ = model_spec.model.predict(observation, deterministic=True)
                action = int(action)

            observation, reward, terminated, truncated, info = env.step(action)
            reward_value = float(np.asarray(reward).item())
            total_reward += reward_value
            decision += 1
            decision_rows.append(
                {
                    "policy": policy,
                    "scenario_id": scenario_id,
                    "training_seed": "" if training_seed is None else training_seed,
                    "evaluation_seed": evaluation_seed,
                    "decision": decision,
                    "simulation_time_sec": float(info["step"]),
                    "action": "fixed" if action is None else action,
                    "reward": reward_value,
                    "local_queue_vehicles": int(info[f"{TARGET_TLS}_stopped"]),
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

    if not decision_rows:
        raise RuntimeError(f"평가 기록이 생성되지 않았습니다: {run_name}")

    # 5초 자료는 행동·보상 분석용, 1초 자료는 최종 대기행렬 평가용이다.
    write_csv(run_dir / "decision_timeseries_5s.csv", decision_rows)
    queue_rows = [
        {
            "policy": policy,
            "scenario_id": scenario_id,
            "training_seed": "" if training_seed is None else training_seed,
            "evaluation_seed": evaluation_seed,
            **record,
        }
        for record in env.per_second_queue_records
    ]
    write_csv(run_dir / "queue_timeseries_1s.csv", queue_rows)

    # 0초 기다린 차량도 남기고, 실제 진출 차로에서 확인됐는지도 함께 저장한다.
    local_vehicle_rows = [
        {
            "vehicle_id": vehicle_id,
            "local_waiting_time_sec": env.local_vehicle_wait_seconds.get(vehicle_id, 0),
            "passed_intersection": vehicle_id in env.local_passed_vehicle_ids,
        }
        for vehicle_id in sorted(env.local_vehicle_ids)
    ]
    write_csv(run_dir / "local_vehicle_waiting.csv", local_vehicle_rows)

    queues = [float(row["local_queue_vehicles"]) for row in queue_rows]
    queue_vehicle_seconds = sum(queues) * QUEUE_MEASUREMENT_INTERVAL_SEC
    id_based_wait_seconds = float(sum(env.local_vehicle_wait_seconds.values()))
    if queue_vehicle_seconds != id_based_wait_seconds:
        raise RuntimeError(
            "대기행렬 합계와 차량 ID별 정지시간 합계가 일치하지 않습니다: "
            f"{queue_vehicle_seconds} != {id_based_wait_seconds}"
        )

    local_vehicle_count = len(env.local_vehicle_ids)
    local_passed_vehicle_count = len(env.local_passed_vehicle_ids)
    local_remaining_vehicle_count = len(
        env.local_vehicle_ids - env.local_passed_vehicle_ids
    )
    local_wait_values = [
        float(env.local_vehicle_wait_seconds.get(vehicle_id, 0))
        for vehicle_id in env.local_vehicle_ids
    ]
    final = decision_rows[-1]
    result: dict[str, Any] = {
        "policy": policy,
        "scenario_id": scenario_id,
        "training_seed": "" if training_seed is None else training_seed,
        "evaluation_seed": evaluation_seed,
        "model_path": "" if model_spec is None else str(model_spec.path),
        "target_tls": TARGET_TLS,
        "environment_version": ENVIRONMENT_VERSION,
        "reward_function": REWARD_FUNCTION,
        "yellow_time_sec": YELLOW_TIME_SEC,
        "decision_count": len(decision_rows),
        "queue_sample_count": len(queue_rows),
        "action_interval_sec": ACTION_INTERVAL_SEC,
        "queue_measurement_interval_sec": QUEUE_MEASUREMENT_INTERVAL_SEC,
        "simulation_time_sec": final["simulation_time_sec"],
        # 총보상은 학습 진단용이며 최종 우열은 교통 지표로 판단한다.
        "total_reward": total_reward,
        "local_observed_vehicles": local_vehicle_count,
        "local_passed_vehicles": local_passed_vehicle_count,
        "local_remaining_observed_vehicles": local_remaining_vehicle_count,
        # 요청한 교차로 차량당 대기는 정지 차량·초를 실제 통과 차량 수로 나눈다.
        # 통과 차량이 0대인 비정상 실행에서는 평균을 0으로 꾸미지 않고 null로 둔다.
        "local_mean_waiting_time_per_vehicle_sec": (
            queue_vehicle_seconds / local_passed_vehicle_count
            if local_passed_vehicle_count else None
        ),
        "local_max_waiting_time_per_vehicle_sec": max(local_wait_values, default=0.0),
        "local_mean_queue_vehicles": mean(queues),
        "local_max_queue_vehicles": max(queues, default=0.0),
        "local_queue_vehicle_seconds": queue_vehicle_seconds,
        "local_zero_queue_fraction": (
            sum(queue == 0 for queue in queues) / len(queues) if queues else 0.0
        ),
        "arrived_vehicles": int(final["system_arrived_vehicles"]),
        "departed_vehicles": int(final["system_departed_vehicles"]),
        "backlogged_vehicles": int(final["system_backlogged_vehicles"]),
        "teleported_vehicles": int(final["system_teleported_vehicles"]),
    }
    result.update(parse_tripinfo(tripinfo_path))
    result.update(parse_statistics(statistics_path))
    result["not_departed_vehicles"] = int(result["loaded_vehicles"]) - int(
        result["departed_vehicles"]
    )
    result["not_arrived_vehicles"] = int(result["loaded_vehicles"]) - int(
        result["arrived_vehicles"]
    )
    result["completion_rate"] = (
        float(result["arrived_vehicles"]) / float(result["loaded_vehicles"])
        if result["loaded_vehicles"] else 0.0
    )
    result["is_complete_run"] = (
        result["not_arrived_vehicles"] == 0
        and result["teleported_vehicles"] == 0
        and result["teleport_count_from_statistics"] == 0
        and result["collision_count"] == 0
    )
    result["comparison_status"] = (
        "complete" if result["is_complete_run"]
        else "incomplete_do_not_rank_by_mean_only"
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


SUMMARY_METRICS = [
    "global_mean_waiting_time_sec",
    "global_mean_travel_time_sec",
    "global_mean_total_delay_sec",
    "local_mean_waiting_time_per_vehicle_sec",
    "local_mean_queue_vehicles",
    "local_max_queue_vehicles",
    "local_queue_vehicle_seconds",
    "completion_rate",
    "not_arrived_vehicles",
    "teleport_count_from_statistics",
    "collision_count",
    "total_reward",
]


def numeric_metric(run: dict[str, Any], metric: str) -> float:
    """집계할 수 없는 null 지표가 있으면 원인을 드러내며 중단한다."""
    value = run[metric]
    if value is None:
        raise ValueError(
            f"{run['policy']} 실행의 {metric}을 계산할 수 없습니다. "
            "교차로 통과 차량 수를 확인하세요."
        )
    return float(value)


def summarize_by_policy(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """DQN은 학습 seed를 독립 단위로 삼아 평균·표준편차를 만든다."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[str(result["policy"])].append(result)

    summaries: list[dict[str, Any]] = []
    for policy in POLICIES:
        runs = grouped.get(policy, [])
        if not runs:
            continue
        if policy == "dqn":
            by_training_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for run in runs:
                by_training_seed[int(run["training_seed"])].append(run)
            units = [
                {
                    metric: mean([numeric_metric(run, metric) for run in seed_runs])
                    for metric in SUMMARY_METRICS
                }
                for seed_runs in by_training_seed.values()
            ]
        else:
            units = runs

        row: dict[str, Any] = {
            "policy": policy,
            "raw_runs": len(runs),
            "independent_units": len(units),
            "complete_runs": sum(bool(run["is_complete_run"]) for run in runs),
            "incomplete_runs": sum(not bool(run["is_complete_run"]) for run in runs),
        }
        for metric in SUMMARY_METRICS:
            values = [float(unit[metric]) for unit in units]
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_std"] = sample_std(values)
        summaries.append(row)
    return summaries


def write_report(path: Path, results: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> None:
    """핵심 결과와 불완전 실행 경고를 Markdown 보고서로 저장한다."""
    names = {"fixed": "고정신호", "random": "무작위", "dqn": "DQN"}
    lines = [
        "# 신호제어 평가 결과", "",
        f"- 제어 교차로: `{TARGET_TLS}`",
        f"- 보상 함수: `{REWARD_FUNCTION}` (현재 정지 차량 수의 음수)",
        f"- 신호 선택 간격: {ACTION_INTERVAL_SEC}초",
        f"- 황색신호: {YELLOW_TIME_SEC}초",
        f"- 대기행렬 측정 간격: {QUEUE_MEASUREMENT_INTERVAL_SEC}초", "",
        "## 정책별 요약", "",
        "| 방법 | 독립 단위 | 전체 대기(초) | 통행시간(초) | 교차로 차량당 대기(초) | 평균/최대 대기행렬(대) | 완료율 | 미도착/텔레포트/충돌 | 불완전 실행 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {names[str(row['policy'])]} | {row['independent_units']} | "
            f"{row['global_mean_waiting_time_sec_mean']:.2f} ± {row['global_mean_waiting_time_sec_std']:.2f} | "
            f"{row['global_mean_travel_time_sec_mean']:.2f} ± {row['global_mean_travel_time_sec_std']:.2f} | "
            f"{row['local_mean_waiting_time_per_vehicle_sec_mean']:.2f} ± {row['local_mean_waiting_time_per_vehicle_sec_std']:.2f} | "
            f"{row['local_mean_queue_vehicles_mean']:.2f} ± {row['local_mean_queue_vehicles_std']:.2f}/"
            f"{row['local_max_queue_vehicles_mean']:.2f} ± {row['local_max_queue_vehicles_std']:.2f} | "
            f"{row['completion_rate_mean'] * 100:.2f}% | "
            f"{row['not_arrived_vehicles_mean']:.2f}/"
            f"{row['teleport_count_from_statistics_mean']:.2f}/"
            f"{row['collision_count_mean']:.2f} | {row['incomplete_runs']} |"
        )

    lines.extend([
        "", "## 개별 실행", "",
        "| 방법 | 시나리오 | 학습 seed | 평가 seed | 전체 대기(초) | 통행시간(초) | 교차로 차량당 대기(초) | 평균/최대 대기행렬 | 도착/전체 | 미도착/텔레포트/충돌 | 상태 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for result in results:
        training_seed = result["training_seed"] or "-"
        lines.append(
            f"| {names[str(result['policy'])]} | {result['scenario_id']} | "
            f"{training_seed} | {result['evaluation_seed']} | "
            f"{result['global_mean_waiting_time_sec']:.2f} | "
            f"{result['global_mean_travel_time_sec']:.2f} | "
            f"{result['local_mean_waiting_time_per_vehicle_sec']:.2f} | "
            f"{result['local_mean_queue_vehicles']:.2f}/{result['local_max_queue_vehicles']:.0f} | "
            f"{result['arrived_vehicles']}/{result['loaded_vehicles']} | "
            f"{result['not_arrived_vehicles']}/"
            f"{result['teleport_count_from_statistics']}/"
            f"{result['collision_count']} | "
            f"{result['comparison_status']} |"
        )
    if any(not bool(result["is_complete_run"]) for result in results):
        lines.extend([
            "",
            "> **주의:** 미도착·텔레포트·충돌이 있는 실행은 완료 차량 평균만으로 순위를 정하지 않습니다.",
        ])
    lines.extend([
        "",
        "> 최종 결론은 서로 다른 학습 seed 5개와 학습에 사용하지 않은 날짜를 평가한 후 내립니다.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def load_scenario_dirs(args: argparse.Namespace) -> list[Path]:
    """직접 지정한 폴더 또는 날짜 분할 JSON에서 시나리오를 읽는다."""
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
        raise ValueError("평가할 시나리오가 없습니다.")
    for scenario_dir in scenario_dirs:
        if not scenario_dir.exists():
            raise FileNotFoundError(f"시나리오 폴더가 없습니다: {scenario_dir}")
    return scenario_dirs


def load_models(paths: list[Path]) -> list[ModelSpec]:
    """환경 버전이 일치하는 새 모델만 불러온다."""
    from stable_baselines3 import DQN

    specs: list[ModelSpec] = []
    for supplied_path in paths:
        path = supplied_path.expanduser().resolve()
        if path.suffix != ".zip":
            path = path.with_suffix(".zip")
        if not path.exists():
            raise FileNotFoundError(f"DQN 모델이 없습니다: {path}")
        config_path = path.parent / "training_config.json"
        if not config_path.exists():
            raise RuntimeError(
                f"{path}는 새 환경 정보가 없는 예비 모델입니다. "
                "황색 3초·queue 보상으로 다시 학습하세요."
            )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("environment_version") != ENVIRONMENT_VERSION:
            raise RuntimeError(
                f"모델 환경 버전이 다릅니다: {config.get('environment_version')} "
                f"!= {ENVIRONMENT_VERSION}"
            )
        specs.append(ModelSpec(path, int(config["seed"]), DQN.load(path)))
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="고정신호·무작위 신호·DQN을 동일한 SUMO 조건에서 평가합니다."
    )
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42],
        help="SUMO 평가 seed. 학습 seed와는 별개입니다.",
    )
    parser.add_argument(
        "--models", nargs="+", type=Path,
        default=[PROJECT_ROOT / "results" / "dqn_20220810_0500" / "seed_42" / f"dqn_{TARGET_TLS}.zip"],
        help="서로 다른 학습 seed로 만든 DQN 모델 경로들",
    )
    parser.add_argument("--scenario-dirs", nargs="+", type=Path, default=[SCENARIO_DIR])
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "results" / "evaluation_20220810_0500",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_dirs = load_scenario_dirs(args)
    model_specs = load_models(args.models) if "dqn" in args.policies else []

    results: list[dict[str, Any]] = []
    for scenario_dir in scenario_dirs:
        for policy in args.policies:
            specs: list[ModelSpec | None] = model_specs if policy == "dqn" else [None]
            for model_spec in specs:
                for evaluation_seed in args.seeds:
                    training_text = f", train={model_spec.training_seed}" if model_spec else ""
                    print(f"[{scenario_dir.name}/{policy}] eval={evaluation_seed}{training_text} 평가 시작")
                    result = run_policy(
                        policy, evaluation_seed, output_dir, scenario_dir, model_spec
                    )
                    results.append(result)
                    print(
                        f"  전체 대기 {result['global_mean_waiting_time_sec']:.2f}초 | "
                        f"교차로 차량당 대기 {result['local_mean_waiting_time_per_vehicle_sec']:.2f}초 | "
                        f"완료율 {result['completion_rate'] * 100:.2f}%"
                    )

    write_csv(output_dir / "runs.csv", results)
    summaries = summarize_by_policy(results)
    write_csv(output_dir / "summary_by_policy.csv", summaries)
    write_report(output_dir / "report.md", results, summaries)
    (output_dir / "evaluation_config.json").write_text(
        json.dumps(
            {
                "policies": args.policies,
                "evaluation_seeds": args.seeds,
                "models": [str(spec.path) for spec in model_specs],
                "scenarios": [str(path) for path in scenario_dirs],
                "target_tls": TARGET_TLS,
                "environment_version": ENVIRONMENT_VERSION,
                "reward_function": REWARD_FUNCTION,
                "yellow_time_sec": YELLOW_TIME_SEC,
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
