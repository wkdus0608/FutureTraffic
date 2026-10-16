"""v6 환경 변형(전환 규칙 없음·1초·1.5초, 북쪽 도로 연장)의 결과를 한 표로 비교한다. 계산만 하고 SUMO를 돌리지 않는다.

입력: --set 이름=폴더 (여러 번). 폴더 안에 <날짜>/<실행>/metrics.json(과 statistics.xml)이 있어야 한다.
      focused_conflicts.py measure 결과(<out>/<평가 지문>) 또는 focused_eval.py run 결과(<out>/<평가 지문>)를 그대로 쓴다.
출력: 이름 × 정책 × 도착 방식별 한 줄
      차량당 평균 지연, 300초 넘게 기다린 차, 도착 못 한 차, 평균 통행 시간, 통행 완료, 좌회전 두 곳의 지연,
      보행 대기 한도 초과, 확인 안 된 충돌.
지연·통행 시간은 차량 수로 가중한 평균이다(날짜가 달라도 같은 날짜 묶음끼리만 비교한다 — 날짜 목록을 함께 적는다).

사용: .venv/bin/python src/focused_v6_report.py --set v5=<폴더> --set none=<폴더> --set pet1=<폴더> --out <md 파일>
      .venv/bin/python src/focused_v6_report.py --root <Drive>/v6_check --rule-sets pet1 pet1_5 none --dates 20220928 20220813
"""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

LEFT_EAST = "477706392_02->477706883_02 (l)"      # 길주로 동쪽 접근로 좌회전
LEFT_SOUTH = "477706883_01->477707013_01 (l)"     # 석천로 남쪽 접근로 좌회전
POLICY_ORDER = ["fixed", "fixedc", "actuated", "greedy", "dqn"]
LABEL = {"fixed": "고정신호(원자료)", "fixedc": "고정신호(보정)", "actuated": "감응식", "greedy": "단순 제어", "dqn": "DQN"}


def load_runs(root: Path) -> list[dict]:
    runs = []
    for m in sorted(root.glob("*/*/metrics.json")):
        row = json.loads(m.read_text())
        stats = m.parent / "statistics.xml"
        row["_trip"] = None
        if stats.exists():
            t = ET.parse(stats).getroot().find("vehicleTripStatistics")
            if t is not None:
                row["_trip"] = (int(t.get("count")), float(t.get("duration")))
        runs.append(row)
    return runs


def pooled(runs: list[dict]) -> dict:
    n = sum(r["all"]["vehicles"] for r in runs)
    out = {"runs": len(runs), "days": len({r["date"] for r in runs}), "vehicles": n,
           "mean_delay_s": sum(r["all"]["mean_delay_s"] * r["all"]["vehicles"] for r in runs) / n,
           "long_waits": sum(r["all"]["long_waits"] for r in runs),
           "not_arrived": sum(r["all"]["not_arrived"] for r in runs),
           "unexplained": sum(len(r["collisions_unexplained"]) for r in runs),
           "ped_over": sum(c["waits_over_limit"] for r in runs for c in r["pedestrian"].values())}
    trips = [r["_trip"] for r in runs if r["_trip"]]
    if len(trips) == len(runs):
        count = sum(c for c, _ in trips)
        out["travel_s"] = sum(c * d for c, d in trips) / count
        out["completed"] = count
    for key, name in ((LEFT_EAST, "left_east"), (LEFT_SOUTH, "left_south")):
        v = [(r["all"]["by_movement"][key]["vehicles"], r["all"]["by_movement"][key]["mean_delay_s"])
             for r in runs if key in r["all"]["by_movement"]]
        out[name] = sum(c * d for c, d in v) / sum(c for c, _ in v) if v else None
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", action="append", default=[], help="이름=폴더")
    p.add_argument("--root", default=None, help="measure의 --out 폴더. --rule-sets와 함께 쓰면 규칙마다 평가 지문 폴더를 찾는다")
    p.add_argument("--rule-sets", nargs="*", default=[], help="예: pet1 pet1_5 none")
    p.add_argument("--dates", nargs="*", default=None, help="이 날짜만(예: 검증 날짜). 정책마다 날짜가 다르면 비교가 틀어진다")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    specs = list(args.set)
    if args.root:
        import os
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from focused_eval import evaluation_fingerprint
        for rule_set in args.rule_sets:
            os.environ["FOCUSED_RULESET"] = rule_set
            specs.append(f"{rule_set}={Path(args.root) / evaluation_fingerprint()}")
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    dates: dict[str, set] = defaultdict(set)
    for spec in specs:
        name, folder = spec.split("=", 1)
        for r in load_runs(Path(folder)):
            if args.dates and r["date"] not in args.dates:
                continue
            groups[(name, r["policy"], r["arrivals"])].append(r)
            dates[name].add(r["date"])
    fmt = lambda x, d=1: "—" if x is None else f"{x:.{d}f}"
    lines = ["| 환경 | 정책 | 도착 | 날짜 | 실행 | 차량당 지연(초) | 300초 초과(대) | 도착 못 함 | 평균 통행 시간(초) | 통행 완료(대) | "
             "길주로 동쪽 좌회전(초) | 석천로 남쪽 좌회전(초) | 보행 한도 초과 | 확인 안 된 충돌 |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (name, policy, arrivals), runs in sorted(groups.items(), key=lambda kv: (kv[0][2], kv[0][0],
                                                                                POLICY_ORDER.index(kv[0][1]))):
        s = pooled(runs)
        seeds = len({r.get("train_seed") for r in runs if r.get("train_seed") is not None})
        label = LABEL.get(policy, policy) + (f"(seed {seeds}개)" if seeds else "")
        lines.append(f"| {name} | {label} | {arrivals} | {s['days']} | {s['runs']} | {s['mean_delay_s']:.2f} | "
                     f"{s['long_waits']} | {s['not_arrived']} | {fmt(s.get('travel_s'))} | "
                     f"{'—' if 'completed' not in s else f'{s['completed']:,}'} | {fmt(s['left_east'])} | {fmt(s['left_south'])} | "
                     f"{s['ped_over']} | {s['unexplained']} |")
    lines += ["", "날짜 목록: " + "; ".join(f"{n}: {len(d)}일" for n, d in sorted(dates.items()))]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
