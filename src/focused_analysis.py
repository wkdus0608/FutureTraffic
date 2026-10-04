"""석천사거리 실험의 정책 행동 분석: 녹색 길이·전환 빈도·방향별 대기·이동별 지연.

평가 폴더의 per_second.csv(매초 신호 상태)와 vehicles.csv(차량별 지연)를 읽는다.
DQN이 비현실적으로 짧은 녹색을 반복하거나 특정 방향을 오래 세워 두는지 확인한다.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from focused_data import PROJECT
from focused_scenario import FocusedNetwork


def green_runs(states: list[str]) -> list[tuple[str, int]]:
    """연속된 같은 녹색 상태(황색 제외)의 길이 목록."""
    runs, current, length = [], None, 0
    for s in states:
        if "y" in s:
            if current is not None:
                runs.append((current, length))
            current, length = None, 0
            continue
        if s == current:
            length += 1
        else:
            if current is not None:
                runs.append((current, length))
            current, length = s, 1
    if current is not None:
        runs.append((current, length))
    return runs


def movement_max_red(states: list[str], fn: FocusedNetwork, vehicles_by_movement: Counter) -> dict:
    """차량이 있는 이동별로 '녹색이 안 켜진 가장 긴 연속 시간'(초)."""
    out = {}
    for (a, b), n in vehicles_by_movement.items():
        idx = [i for i, (x, y, _) in fn.links.items() if (x, y) == (a, b)]
        longest = run = 0
        for s in states:
            if any(s[i] in "Gg" for i in idx):
                run = 0
            else:
                run += 1
                longest = max(longest, run)
        out[f"{a}->{b}"] = longest
    return out


def analyse(eval_dir: Path) -> dict:
    fn = FocusedNetwork()
    by_policy = defaultdict(lambda: {"green": [], "switches_per_hour": [], "max_red": defaultdict(list),
                                     "loss": defaultdict(list)})
    for run in sorted((eval_dir / "runs").glob("*/*")):
        per_second, vehicles = run / "per_second.csv", run / "vehicles.csv"
        if not per_second.exists() or not vehicles.exists():
            continue
        policy = json.loads((run / "metrics.json").read_text())["policy"]
        rows = list(csv.DictReader(per_second.open()))
        states = [r["tls_state"] for r in rows[:900]]  # 수요가 들어오는 앞 15분
        runs = green_runs(states)
        by_policy[policy]["green"] += [n for _, n in runs]
        by_policy[policy]["switches_per_hour"].append(len(runs) * 3600 / max(1, len(states)))
        veh_rows = list(csv.DictReader(vehicles.open()))
        counts = Counter((v["from_edge"], v["to_edge"]) for v in veh_rows if v["from_edge"])
        for movement, red in movement_max_red(states, fn, counts).items():
            by_policy[policy]["max_red"][movement].append(red)
        for v in veh_rows:
            if v["from_edge"] and v["arrival"] not in ("", "-1", "-1.00"):
                key = f"{v['from_edge']}->{v['to_edge']}"
                by_policy[policy]["loss"][key].append(float(v["time_loss"]) + float(v["depart_delay"]))
    summary = {}
    for policy, d in by_policy.items():
        greens = sorted(d["green"])
        summary[policy] = {
            "runs": len(d["switches_per_hour"]),
            "green_seconds_p10_median_p90": [greens[len(greens) // 10], statistics.median(greens), greens[9 * len(greens) // 10]] if greens else None,
            "share_green_under_10s": round(sum(g < 10 for g in greens) / len(greens), 3) if greens else None,
            "phase_changes_per_hour": round(statistics.fmean(d["switches_per_hour"]), 1),
            "worst_max_red_seconds": {m: max(v) for m, v in d["max_red"].items()},
            "mean_delay_by_movement": {m: round(statistics.fmean(v), 1) for m, v in d["loss"].items()},
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=PROJECT / "results/focused_204820/eval_20220810")
    args = parser.parse_args()
    summary = analyse(args.eval_dir)
    (args.eval_dir / "behaviour.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    for policy, s in summary.items():
        print(policy, {k: v for k, v in s.items() if k not in ("worst_max_red_seconds", "mean_delay_by_movement")})
        print("   가장 오래 빨간불(초):", dict(sorted(s["worst_max_red_seconds"].items(), key=lambda x: -x[1])[:4]))


if __name__ == "__main__":
    main()
