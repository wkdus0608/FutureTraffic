"""민감도: 동쪽 접근로 차량이 이웃 신호(204822)에서 무리 지어 오는 경우.

본 실험은 정지선 도착을 15분 안에서 무작위로 뽑는다. 실제로는 동쪽 이웃 신호 204822가 석천사거리와 같은 주기로
돌아(1,824개 시간대의 97%), 동쪽 차량이 204822의 녹색에 맞춰 무리로 온다. 이 변형은 동쪽 접근로 차량의
출발 시각만 204822의 실제 녹색 시간에 맞춘다. 차량 수·번호·차종·경로·시간대는 그대로이고, 다른 접근로는 바꾸지 않는다.
학습한 모델과 잠근 환경 파일은 그대로 쓴다(평가 조건만 바꾸는 민감도).

1. 204822에서 석천사거리 쪽 도로(477706439_01)로 나가는 이동별 1분 통과 기록을 원본에서 뽑는다.
2. 204822 신호 기록을 하루 시간축으로 잇는다(빈 시간은 마지막 현시의 다음 순서 현시로, 204820과 같은 규칙).
3. 이동마다 1분 통과량과 '1분 안의 녹색 초'의 상관이 가장 큰 신호 상태를 그 이동이 지나가는 녹색으로 본다.
   학습·검증 날짜만 쓴다. 상관이 0.5보다 작으면 뚜렷한 녹색이 없다고 보고 지금처럼 무작위로 둔다.
   (원본 상태 이름은 204820처럼 한 현시 밀려 있을 수 있으나, 시간 창만 쓰므로 영향이 없다.)
4. 동쪽 차량마다 204822 이동을 그 15분의 이동별 통과량 비율로 정하고, 그 이동이 지나가는 녹색 시간 안에서
   출발 시각을 뽑는다(정지선 도착이 원래 15분 안에 들도록 자유 주행 시간만큼 앞당김).

사용:
  .venv/bin/python src/focused_platoon.py extract     # data/derived/east_upstream_204822.csv
  .venv/bin/python src/focused_platoon.py identify    # config/east_platoon.json (근거 포함)
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import random
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from focused_data import DERIVED, PROJECT, environment_fingerprint, fingerprint, iter_turn_files, read_signal_phases, signal_xml_files
from focused_scenario import UPSTREAM_LIMIT_M, FocusedNetwork, day_route_file, period_start_s

EAST_APPROACH = "477706392_02"
UPSTREAM_TLS = "204822"
LINK_EDGE = "477706439_01"            # 204822 → 204885. 이 도로로 나간 차가 동쪽 접근로 경로의 시작 도로로 들어온다
UPSTREAM_COUNTS = DERIVED / "east_upstream_204822.csv"
PLATOON_CONFIG = PROJECT / "config/east_platoon.json"
SPLIT_FILE = PROJECT / "config/focused_split.json"
MIN_CORRELATION = 0.5
DAY0_UNIX = 1660057200                # 2022-08-10 00:00 (KST)
PLATOON_FILES = ["src/focused_platoon.py", "config/east_platoon.json", "data/derived/east_upstream_204822.csv"]


def extract_upstream_counts(output: Path = UPSTREAM_COUNTS) -> dict:
    """204822에서 LINK_EDGE로 나간 이동별 1분 통과 기록(19일 모두)을 표로 저장한다."""
    counts = defaultdict(int)
    for name, data in iter_turn_files():
        date = Path(name).stem[6:14]
        for interval in ET.fromstring(data).iter("interval"):
            minute = (int(float(interval.get("begin"))) - DAY0_UNIX) % 86400 // 60
            for rel in interval.iter("edgeRelation"):
                if rel.get("to") == LINK_EDGE:
                    counts[(date, minute, rel.get("from"))] += int(float(rel.get("count")))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "minute", "from_edge", "count"])
        for (date, minute, origin), n in sorted(counts.items()):
            w.writerow([date, minute, origin, n])
    return {"rows": len(counts), "vehicles": sum(counts.values())}


def load_upstream_counts(path: Path = UPSTREAM_COUNTS) -> dict[tuple[str, int, str], int]:
    with path.open(encoding="utf-8") as f:
        return {(r["date"], int(r["minute"]), r["from_edge"]): int(r["count"]) for r in csv.DictReader(f)}


def upstream_states(date: str) -> list[str]:
    """204822 신호를 1초마다의 원본 상태로 하루 이어 붙인다. 빈 시간은 마지막 현시의 다음 순서 현시로 채운다."""
    files = signal_xml_files(date)
    phases = {key: [(d, s) for d, s in read_signal_phases(files[key], UPSTREAM_TLS) if d > 0] for key in sorted(files)}
    seq = [p for key in sorted(phases) for p in phases[key]]
    following = {}
    for (a, b), _ in Counter((x[1], y[1]) for x, y in zip(seq, seq[1:]) if x[1] != y[1]).most_common():
        following.setdefault(a, b)
    states: list[str] = []
    for key in sorted(phases):
        part = [s for d, s in phases[key] for _ in range(int(round(d)))][:900]
        if part and len(part) < 900:
            part += [following.get(part[-1], part[-1])] * (900 - len(part))
        states += part or [""] * 900
    return states


def _corr(x: list[float], y: list[float]) -> float:
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sx = sum((a - mx) ** 2 for a in x) ** 0.5
    sy = sum((b - my) ** 2 for b in y) ** 0.5
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else 0.0


def identify(output: Path = PLATOON_CONFIG) -> dict:
    """이동별로 지나가는 녹색 상태를 학습·검증 날짜의 1분 기록으로 찾고, 근거와 함께 저장한다."""
    split = json.loads(SPLIT_FILE.read_text())
    dates = sorted(split["train"] + split["validation"])
    counts = load_upstream_counts()
    green_seconds: dict[tuple[str, int], Counter] = {}
    totals = Counter()
    for date in dates:
        states = upstream_states(date)
        for m in range(1440):
            window = Counter(states[m * 60:(m + 1) * 60])
            green_seconds[(date, m)] = window
            totals.update({s: n for s, n in window.items() if s and "y" not in s and "G" in s})
    candidates = [s for s, n in totals.most_common() if n >= 0.05 * sum(totals.values())]
    keys = sorted(green_seconds)
    movements = {}
    for origin in sorted({o for d, _, o in counts if d in dates}):
        flow = [counts.get((d, m, origin), 0) for d, m in keys]
        corr = {s: round(_corr([green_seconds[k][s] for k in keys], flow), 3) for s in candidates}
        best = max(corr, key=corr.get)
        movements[origin] = {"vehicles": sum(flow), "correlation": corr,
                             "release_state": best if corr[best] >= MIN_CORRELATION else None}
    config = {
        "description": "동쪽 이웃 신호 204822에서 석천사거리 쪽 도로(477706439_01)로 나가는 이동별로, 차가 실제로 지나가는 "
                       "녹색 상태(원본 이름). 1분 통과량과 1분 안의 녹색 초의 상관이 가장 큰 상태이며, 상관이 "
                       f"{MIN_CORRELATION}보다 작으면 null(무작위 그대로).",
        "dates_used": dates, "minutes_used": len(keys), "candidate_states": candidates,
        "green_seconds_total": {s: totals[s] for s in candidates}, "movements": movements,
    }
    output.write_text(json.dumps(config, ensure_ascii=False, indent=1))
    return config


def platoon_route_file(date: str, cache_dir: Path) -> tuple[Path, dict[str, str]]:
    """하루 차량 파일의 무리 도착 변형. 동쪽 접근로 차량의 출발 시각만 204822의 녹색에 맞춘다."""
    base, period_of = day_route_file(date, cache_dir)
    path = cache_dir / f"day_{date}_{environment_fingerprint()}_platoon_{fingerprint(PLATOON_FILES)}.rou.xml"
    if path.exists():
        return path, period_of
    fn = FocusedNetwork()
    chain = fn._chain(EAST_APPROACH, True, UPSTREAM_LIMIT_M)
    start = chain[0]
    if LINK_EDGE not in fn.pred[start]:
        raise ValueError(f"{LINK_EDGE}가 동쪽 경로의 시작 도로 {start} 앞에 있지 않습니다.")
    link_tt = fn.length[LINK_EDGE] / fn.speed[LINK_EDGE]
    movements = json.loads(PLATOON_CONFIG.read_text())["movements"]
    release = {o: m["release_state"] for o, m in movements.items()}
    main_origin = max(movements, key=lambda o: movements[o]["vehicles"])     # 통과 기록이 없는 15분이면 가장 많은 이동
    counts = load_upstream_counts()
    states = upstream_states(date)
    windows = {s: [t for t, x in enumerate(states) if x == s] for s in set(release.values()) if s}   # 정렬된 초 목록
    root = ET.parse(base).getroot()
    routes = {r.get("id"): r.get("edges").split() for r in root.iter("route")}
    rng = random.Random(int(date) * 31 + 7)
    lead_of, share_of, changed = {}, {}, Counter()
    for v in root.findall("vehicle"):
        edges = routes[v.get("route")]
        if edges[0] != start:
            continue
        rid = v.get("route")
        if rid not in lead_of:
            lead_of[rid] = fn.travel_time_to_stopline(edges, EAST_APPROACH)
        p0 = period_start_s(period_of[v.get("id")])
        if p0 not in share_of:
            share_of[p0] = [sum(counts.get((date, m, o), 0) for m in range(p0 // 60, p0 // 60 + 15)) for o in release]
        weights = share_of[p0]
        origin = rng.choices(list(release), weights=weights)[0] if sum(weights) else main_origin
        lo, hi = p0 - lead_of[rid] - link_tt, p0 + 900 - lead_of[rid] - link_tt
        state = release.get(origin)
        seconds = windows.get(state, []) if state else []
        a, b = bisect.bisect_left(seconds, lo), bisect.bisect_left(seconds, hi)
        if b > a:
            t = seconds[rng.randrange(a, b)] + rng.random()       # 그 녹색 1초 안에서 고르게
            v.set("depart", f"{max(0.0, t + link_tt):.2f}")
            changed["platoon"] += 1
        else:
            v.set("depart", f"{max(0.0, rng.uniform(lo, hi) + link_tt):.2f}")
            changed["uniform" if state is None else "no_green_window"] += 1
    vehicles = sorted(root.findall("vehicle"), key=lambda v: float(v.get("depart")))
    for v in root.findall("vehicle"):
        root.remove(v)
    root.extend(vehicles)
    tmp = path.with_suffix(".tmp.xml")
    ET.ElementTree(root).write(tmp, encoding="utf-8", xml_declaration=True)
    tmp.replace(path)
    path.with_suffix(".summary.json").write_text(json.dumps(dict(changed)))
    return path, period_of


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("extract", "identify"))
    args = parser.parse_args()
    if args.command == "extract":
        print(extract_upstream_counts())
    else:
        config = identify()
        for origin, m in config["movements"].items():
            idx = {s: i for i, s in enumerate(config["candidate_states"])}
            corr = ", ".join(f"{idx[s]}:{v:+.2f}" for s, v in m["correlation"].items())
            print(f"{origin} {m['vehicles']:,}대 상관 {corr} → {idx.get(m['release_state'], '무작위')}")


if __name__ == "__main__":
    main()
