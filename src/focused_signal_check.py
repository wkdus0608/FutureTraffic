"""AI Hub 신호 자료 해석 판정: 관측 교통량을 처리할 수 있는 신호인가? (19일 전체)

실제로 지나간 차량은 실제 신호가 준 녹색 시간 안에 지나갔어야 한다.
이동별 필요 녹색(대수 ÷ (차로 수 × 포화교통류율))과 해석별 가용 녹색을 비교한다.
비율이 1을 크게 넘으면 그 해석으로는 관측 교통량을 처리할 수 없다.
또한 녹색 상태 4개와 순환 순서가 모든 날짜·시간대에서 같은지 확인한다.
"""
from __future__ import annotations

import json
from collections import Counter

from focused_data import PROJECT, load_counts, read_signal_phases, signal_xml_files
from focused_scenario import FocusedNetwork, aihub_states

SATURATION = 0.5  # 차로당 0.5대/초 = 1,800대/시


def cyclic(seq):
    start = min(range(len(seq)), key=lambda i: seq[i])
    return tuple(seq[start:] + seq[:start])


def main() -> None:
    fn = FocusedNetwork()
    counts = load_counts(PROJECT / "results/focused_204820/counts_204820.csv")
    info = aihub_states()
    result = {"saturation_veh_per_s_per_lane": SATURATION, "dates": {}}
    orders = Counter()
    state_sets = Counter()
    for date in sorted({d for d, _ in counts}):
        files = signal_xml_files(date)
        day = {}
        for period, path in sorted(files.items()):
            phases = [(d, s) for d, s in read_signal_phases(path) if d > 0]
            greens = [s for d, s in phases if "y" not in s]
            cycle = []
            for g in greens:
                if g not in cycle:
                    cycle.append(g)
            orders[cyclic(cycle)] += 1
            state_sets[frozenset(cycle)] += 1
            volume = Counter()
            for r in counts[(date, period)]:
                volume[(r["from_edge"], r["to_edge"])] += r["count"]
            entry = {"vehicles": sum(volume.values())}
            for name, mapping in (("asis", lambda s: s), ("corrected", lambda s: info["previous"][s])):
                green = Counter()
                for d, s in phases:
                    if "y" not in s:
                        green[mapping(s)] += d
                worst = 0.0
                for (a, b), v in volume.items():
                    idx = [i for i, (x, y, _) in fn.links.items() if (x, y) == (a, b)]
                    g = sum(sec for s, sec in green.items() if any(s[i] in "Gg" for i in idx))
                    worst = max(worst, (v / (len(idx) * SATURATION)) / g if g else float("inf"))
                entry[name] = round(worst, 3)
            day[period] = entry
        result["dates"][date] = day
    allp = [e for day in result["dates"].values() for e in day.values()]
    for name in ("asis", "corrected"):
        result[f"periods_over_1.0_{name}"] = sum(e[name] > 1.0 for e in allp)
        result[f"periods_over_1.2_{name}"] = sum(e[name] > 1.2 for e in allp)
        result[f"periods_over_1.5_{name}"] = sum(e[name] > 1.5 for e in allp)
    result["periods_total"] = len(allp)
    result["green_cycle_orders"] = len(orders)
    result["green_state_sets"] = len(state_sets)
    out = PROJECT / "results/focused_204820/signal_capacity_check.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print({k: v for k, v in result.items() if k != "dates"})
    # 날짜별 요약
    for date, day in result["dates"].items():
        print(date, "asis>1:", sum(e["asis"] > 1 for e in day.values()), "corrected>1:", sum(e["corrected"] > 1 for e in day.values()),
              "corrected>1.2:", sum(e["corrected"] > 1.2 for e in day.values()))


if __name__ == "__main__":
    main()
