"""12개 신호의 '원본 신호 칸 → 지도 연결 칸' 대응과 '한 현시 밀림'을 1분 통과 기록과 지도로 확인한다(SUMO 실행 없음, 지문 밖).

칸 대응(구조): 원본 기록과 지도의 칸 수가 같은 9곳은 그대로(번호가 같음). 다른 3곳은 남는 원본 칸이 지도에 신호 연결로 없는
회전(분리 우회전 차로·지도에 없는 회전)의 자리에 있으므로 그 칸을 뺀다(아래 STRUCTURAL).
'지금 규칙' = 구조 대응 + 한 현시 밀림(석천사거리처럼 원본 녹색을 순서상 바로 앞 녹색으로, 황색은 앞 녹색의 황색으로).

확인 네 가지(학습·검증 15일만 씀):
1. 상관: 이동마다 1분 통과량과 그 이동 칸의 '1분 안 녹색 초'의 상관. 15분 이동 평균을 빼서 밤낮 교통량과 시간대별 신호 계획의
   영향을 없애고 주기 단위 변동만 본다. 밀림 없음과 한 현시 밀림을 비교한다.
2. 지나가는 상태: 이동마다 통과량이 가장 잘 맞는 원본 녹색 상태를 찾는다. 한 현시 밀림이 맞다면 그 상태는 '자기 칸이 녹색인
   원본 상태의 다음 상태'여야 한다.
3. 부딪힘(SUMO가 교차로마다 정한 '서로 부딪히는 연결' 표):
   (가) 지금 규칙으로 만든 실제 신호에 부딪히는 두 연결이 함께 우선 녹색(G)인 상태가 있는가
   (나) 통과 기록만 믿고 '같은 원본 상태에 지나가는 이동끼리 같은 녹색'으로 읽으면 부딪히는 짝이 생기는가(생기면 그 읽기는 불가능)
4. 용량: 읽기마다 '15분 통과 대수 > 그 15분 녹색 동안 지날 수 있는 최대 대수(차로당 시간당 2,200대로 넉넉하게)'인 구간 비율.
   넘는 구간이 많으면 그 읽기로는 실제로 지나간 차를 다 보낼 수 없으므로 틀린 읽기다.
   읽기: 지금 규칙 / 밀림 없음(구조 대응, 원본 그대로) / 통과 기록(이동마다 가장 잘 맞는 원본 상태의 시간에 녹색)

사용: .venv/bin/python src/network_signal_columns.py      (맥 약 1분)
결과: results/network_inventory/signal_columns_<날짜>.json, .md
"""
from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from focused_data import NETWORK, PROJECT, iter_turn_files, signal_xml_files

OUT = PROJECT / "results/network_inventory"
SPLIT = PROJECT / "config/focused_split.json"
DAY0_UNIX = 1660057200                     # 2022-08-10 00:00 (KST)
DETREND_MIN = 15                           # 이 길이(분)의 이동 평균을 빼고 상관을 본다
MIN_VEHICLES = 3000                        # 15일 통과가 이보다 적은 이동은 상태·용량 확인에서 뺀다
GREEN_SHARE = 0.02                         # 하루의 2% 넘게 켜지는 녹색만 '주요 녹색 상태'로 본다
SATURATION_PER_S = 2200 / 3600             # 차로당 녹색 1초에 지날 수 있는 최대 대수(넉넉한 값)
BUSY_15MIN = 20                            # 15분에 이보다 적게 지난 구간은 용량 확인에서 뺀다
OVER_LIMIT = 0.05                          # 넘는 15분이 이보다 많으면 '용량 부족'으로 표시
# 원본 칸 수가 지도와 다른 곳: 빼는 원본 칸(0부터)과 그 이유. 나머지 원본 칸은 순서대로 지도 칸 0, 1, 2…에 대응한다.
STRUCTURAL = {
    "cluster_204796_205202": {"drop": [24, 25], "why": "진입로 4(477765078_01)의 원본 칸 20~25 가운데 25는 녹색 A·C에 양보 녹색(분리 "
                              "우회전 477765078_01 → 477753313_01, 지도는 신호 밖 분리 차로), 20~24는 신호가 똑같은 녹색 5칸인데 지도는 "
                              "직진 3 + 좌회전 1(4칸)이라 하나를 뺀다(같은 신호라 어느 칸을 빼도 결과가 같다)"},
    "201330": {"drop": [11], "why": "진입로 2(477643151_02)의 원본 칸 11은 모든 녹색에서 양보 녹색 = 우회전(지도는 신호 밖 분리 차로 "
               "477643151_02 → 477605685_01)"},
    "201306": {"drop": [8, 16], "why": "원본 칸 8은 진입로 3(477610076_02)의 항상 양보 녹색 = 지도에 없는 우회전 "
               "477610076_02 → 477649580_01_01, 칸 16은 진입로 4(477651439_02)의 마지막 녹색에만 켜지는 칸 = 지도에 없는 좌회전 "
               "477651439_02 → 477649578_02_02"},
}
READINGS = ("rule", "no_shift", "passages")
READING_NAMES = {"rule": "지금 규칙", "no_shift": "밀림 없음", "passages": "통과 기록"}


def map_links(network: Path = NETWORK) -> dict[str, dict[int, tuple[str, str, str]]]:
    root = ET.parse(network).getroot()
    out = defaultdict(dict)
    for c in root.iter("connection"):
        if c.get("tl"):
            out[c.get("tl")][int(c.get("linkIndex"))] = (c.get("from"), c.get("to"), c.get("dir"))
    return dict(out)


def junction_foes(network: Path = NETWORK) -> dict[str, np.ndarray]:
    """신호마다 (칸 × 칸) '서로 부딪힘' 표. 12곳 모두 신호 칸 번호가 교차로 연결 순서(진입 차로 순서)와 같아야 한다(확인함)."""
    root = ET.parse(network).getroot()
    conns = defaultdict(list)
    for c in root.iter("connection"):
        if c.get("tl"):
            conns[c.get("tl")].append(c)
    junctions = {j.get("id"): j for j in root.iter("junction")}
    out = {}
    for tls, cs in conns.items():
        jids = {c.get("via")[1:].rsplit("_", 2)[0] for c in cs}
        if len(jids) != 1:
            raise ValueError(f"{tls}: 신호 하나가 교차로 {sorted(jids)}를 맡아 칸 번호를 요청 번호로 바로 쓸 수 없습니다.")
        j = junctions[jids.pop()]
        order = []
        for lane in j.get("incLanes").split():
            edge, k = lane.rsplit("_", 1)
            order += [int(c.get("linkIndex")) for c in cs if c.get("from") == edge and c.get("fromLane") == k]
        if order != list(range(len(cs))):
            raise ValueError(f"{tls}: 신호 칸 번호가 교차로 연결 순서와 다릅니다.")
        req = {int(r.get("index")): r.get("foes") for r in j.iter("request")}
        n = len(req)
        m = np.zeros((n, n), dtype=bool)
        for i, foes in req.items():
            for k in range(n):
                m[i, k] = foes[n - 1 - k] == "1"
        out[tls] = m | m.T
    return out


def day_states(date: str) -> dict[str, list[str]]:
    """그날 12개 신호를 1초마다의 원본 상태로 잇는다. 빈 시간은 마지막 현시의 다음 순서 현시로(석천사거리와 같은 규칙)."""
    files = signal_xml_files(date)
    per_tls = defaultdict(dict)                  # tls → {HHMM: [(d, s)]}
    for key, path in sorted(files.items()):
        for logic in ET.parse(path).getroot().iter("tlLogic"):
            per_tls[logic.get("id")][key] = [(float(p.get("duration")), p.get("state"))
                                             for p in logic.iter("phase") if float(p.get("duration")) > 0]
    out = {}
    for tls, phases in per_tls.items():
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
        out[tls] = states[:86400] + [states[-1] if states else ""] * max(0, 86400 - len(states))
    return out


def is_yellow(s: str) -> bool:
    return "y" in s


def is_green(s: str) -> bool:
    return "G" in s and not is_yellow(s)


def most_common_pairs(pairs: Counter) -> dict:
    """{(가, 나): 횟수} → 가마다 가장 많이 나온 나."""
    out = {}
    for (a, b), _ in pairs.most_common():
        out.setdefault(a, b)
    return out


def shift_map(all_states: list[str]) -> dict[str, str]:
    """'한 현시 밀림' 보정: 녹색 → 순서상 바로 앞 녹색, 황색 → 그 앞 녹색의 황색.
    녹색끼리 황색 없이 바로 바뀌는 곳이 있으면(204920: 매 주기 한 번) 그 녹색의 황색이 기록에 없으므로, 그 녹색에서 다음 녹색에
    없는 칸만 황색으로 바꿔 만든다. (단순히 '두 단계 앞 상태'로 하면 그런 곳에서 녹색이 황색으로 바뀌는 잘못이 생긴다.)"""
    seq = [s for i, s in enumerate(all_states) if s and (i == 0 or s != all_states[i - 1])]
    prev_green, next_green, yellow_after, green_before = Counter(), Counter(), Counter(), Counter()
    last = None
    for a, b in zip(seq, seq[1:]):
        if is_green(a) and is_yellow(b):
            yellow_after[(a, b)] += 1
            green_before[(b, a)] += 1
    for s in seq:
        if is_green(s):
            if last is not None and last != s:
                prev_green[(s, last)] += 1
                next_green[(last, s)] += 1
            last = s
    prev_green, next_green, yellow_after, green_before = map(most_common_pairs, (prev_green, next_green, yellow_after, green_before))
    out = {}
    for s in set(seq):
        if is_green(s):
            out[s] = prev_green.get(s, s)
        elif is_yellow(s) and s in green_before:
            p = prev_green.get(green_before[s])
            if p is None:
                out[s] = s
            elif p in yellow_after:
                out[s] = yellow_after[p]
            else:
                nxt = next_green.get(p, green_before[s])
                out[s] = "".join("y" if p[c] in "Gg" and nxt[c] not in "Gg" else p[c] for c in range(len(p)))
        else:
            out[s] = s
    return out


def green_order(all_states: list[str]) -> tuple[list[str], dict[str, str]]:
    """주요 녹색 상태(하루의 2% 넘게)를 원본 순환 순서대로, 그리고 각 상태의 다음 녹색."""
    total = Counter(all_states)
    greens = [s for s, n in total.most_common() if s and "y" not in s and "G" in s and n > GREEN_SHARE * len(all_states)]
    seq = [s for i, s in enumerate(all_states) if s in greens and (i == 0 or s != all_states[i - 1])]
    succ = {}
    for (a, b), _ in Counter(zip(seq, seq[1:])).most_common():
        succ.setdefault(a, b)
    order = [greens[0]]
    while len(order) < len(greens) and succ.get(order[-1]) not in order and succ.get(order[-1]) is not None:
        order.append(succ[order[-1]])
    return order + [s for s in greens if s not in order], succ


def passages_per_minute(dates: set[str]) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    out = defaultdict(lambda: defaultdict(lambda: np.zeros(1440)))
    for name, data in iter_turn_files():
        date = re.search(r"/(\d{8})/", name).group(1)
        if date not in dates:
            continue
        for interval in ET.fromstring(data).iter("interval"):
            minute = (int(float(interval.get("begin"))) - DAY0_UNIX) % 86400 // 60
            for rel in interval.iter("edgeRelation"):
                out[(rel.get("from"), rel.get("to"))][date][minute] += int(float(rel.get("count")))
    return out


def detrend(a: np.ndarray, window: int = DETREND_MIN) -> np.ndarray:
    """첫 축을 따라 가운데 맞춘 이동 평균(가장자리는 끝값으로 채움)을 뺀다. 2차원이면 열마다."""
    a = np.asarray(a, dtype=float)
    pad = np.pad(a, [(window // 2, window // 2)] + [(0, 0)] * (a.ndim - 1), mode="edge")
    c = np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(pad, axis=0)])
    return a - (c[window:] - c[:-window]) / window


def corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def column_map(tls: str, n_map: int, n_rec: int, drop: list[int] | None = None) -> list[int]:
    """지도 칸 i → 원본 칸. drop을 뺀 원본 칸을 순서대로 대응시킨다."""
    drop = STRUCTURAL.get(tls, {}).get("drop", []) if drop is None else drop
    keep = [c for c in range(n_rec) if c not in drop]
    if len(keep) != n_map:
        raise ValueError(f"{tls}: 원본 {n_rec}칸에서 {drop}을 빼면 {len(keep)}칸인데 지도는 {n_map}칸입니다.")
    return keep


class Signal:
    """한 신호의 15일 원본 상태를 상태 번호 배열로 바꿔 두고, 칸·상태별 녹색 초를 빨리 센다."""

    def __init__(self, tls: str, states: dict[str, list[str]], dates: list[str]):
        self.tls, self.dates = tls, dates
        all_states = [s for d in dates for s in states[d][tls]]
        self.shift = shift_map(all_states)
        self.order, self.succ = green_order(all_states)
        self.uniq = sorted(set(all_states))
        pos = {s: i for i, s in enumerate(self.uniq)}
        self.idx = {d: np.fromiter((pos[s] for s in states[d][tls]), dtype=np.int32, count=86400) for d in dates}
        self.n_rec = Counter(len(s) for s in self.order).most_common(1)[0][0]

        def table(view, chars):
            t = np.zeros((len(self.uniq), self.n_rec), dtype=bool)
            for i, s in enumerate(self.uniq):
                v = view(s)
                for c in range(min(self.n_rec, len(v))):
                    t[i, c] = v[c] in chars
            return t
        self.green = {"none": table(lambda s: s, "Gg"), "one_phase": table(lambda s: self.shift.get(s, s), "Gg")}
        # 원본 칸마다 1분 안 녹색(G·g) 초, 날마다 15분 이동 평균을 빼고 15일을 이음: (1440 × 날짜, 원본 칸)
        self.col_minutes = {name: np.concatenate([detrend(t[self.idx[d]].reshape(1440, 60, self.n_rec).sum(1)) for d in dates])
                            for name, t in self.green.items()}

    def state_minutes(self, d: str, s: str) -> np.ndarray:
        return (self.idx[d] == self.uniq.index(s)).reshape(1440, 60).sum(1).astype(float)


def movements_of(links: dict[int, tuple[str, str, str]]) -> dict[tuple[str, str, str], list[int]]:
    mv = defaultdict(list)
    for i, (f, t, d) in links.items():
        mv[(f, t, d)].append(i)
    return mv


def flow_series(flows, dates: list[str]) -> dict[tuple[str, str], np.ndarray]:
    """이동마다 1분 통과량, 날마다 15분 이동 평균을 빼고 15일을 이음."""
    return {k: np.concatenate([detrend(v[d]) for d in dates]) for k, v in flows.items()}


def correlation_rows(sig: Signal, links, mapping, flows, ys) -> dict:
    """이동마다 상관(밀림 없음/한 현시 밀림): 1분 통과량 vs 그 이동 칸들의 평균 1분 녹색 초, 둘 다 15분 이동 평균을 뺌."""
    rows = {}
    for (f, t, d), idx in movements_of(links).items():
        if (f, t) not in flows:
            continue
        cols = sorted({mapping[i] for i in idx})
        vehicles = int(sum(flows[(f, t)][day].sum() for day in sig.dates))
        res = {name: corr(sig.col_minutes[name][:, cols].mean(1), ys[(f, t)]) for name in ("none", "one_phase")}
        rows[f"{f}->{t}"] = {"links": idx, "record_columns": cols, "vehicles": vehicles, "dir": d, **res}
    return rows


def release_rows(sig: Signal, links, mapping, flows, ys, foes: np.ndarray) -> tuple[dict, list]:
    """이동마다 지나가는 원본 상태, 한 현시 밀림이면 와야 할 상태, 세 가지 읽기의 용량. 그리고 (나) 통과 기록대로 읽을 때 부딪히는 짝."""
    name = {s: "ABCDEFGH"[i] for i, s in enumerate(sig.order)}
    occ = {s: np.concatenate([detrend(sig.state_minutes(day, s)) for day in sig.dates]) for s in sig.order}
    rows, by_release = {}, defaultdict(list)
    for (f, t, d), idx in movements_of(links).items():
        if d in "rt" or (f, t) not in flows:
            continue
        vehicles = int(sum(flows[(f, t)][day].sum() for day in sig.dates))
        if vehicles < MIN_VEHICLES:
            continue
        cols = sorted({mapping[i] for i in idx})
        c = {s: corr(occ[s], ys[(f, t)]) for s in sig.order}
        best = max(c, key=lambda s: -9 if c[s] is None else c[s])
        own = [s for s in sig.order if all(s[k] == "G" for k in cols)]
        expected = [sig.succ[s] for s in own if s in sig.succ]
        by_release[best].append((f, t, d, idx))
        cap = {}
        for reading in READINGS:
            over, busy, ratio = 0, 0, []
            for day in sig.dates:
                if reading == "passages":
                    g = sig.idx[day] == sig.uniq.index(best)
                else:
                    table = sig.green["one_phase" if reading == "rule" else "none"]
                    g = table[:, cols].all(1)[sig.idx[day]]
                capacity = g.reshape(96, 900).sum(1) * len(idx) * SATURATION_PER_S
                q = flows[(f, t)][day].reshape(96, 15).sum(1)
                b = q >= BUSY_15MIN
                over += int((q[b] > capacity[b]).sum())
                busy += int(b.sum())
                ratio += (q[b] / np.maximum(capacity[b], 1e-9)).tolist()
            cap[reading] = {"over_share": over / busy if busy else 0.0,
                            "p95_ratio": float(np.percentile(ratio, 95)) if ratio else 0.0}
        rows[f"{f}->{t}"] = {
            "dir": d, "links": idx, "record_columns": cols, "vehicles": vehicles,
            "own": [name[s] for s in own], "expected_if_shift": [name[s] for s in expected],
            "release": name[best], "release_corr": c[best],
            "corr_by_state": {name[s]: c[s] for s in sig.order},
            "consistent": (best in expected) if expected else None, "capacity": cap,
        }
    clashes = []
    for s, ms in by_release.items():
        for a, b in combinations(ms, 2):
            if foes[np.ix_(a[3], b[3])].any():
                clashes.append({"state": name[s], "a": f"{a[0]}->{a[1]}({a[2]})", "b": f"{b[0]}->{b[1]}({b[2]})"})
    return rows, clashes


def rule_conflicts(sig: Signal, mapping, foes: np.ndarray, n_map: int) -> list[dict]:
    """(가) 지금 규칙으로 만든 실제 상태마다, 부딪히는 두 연결이 함께 G인지."""
    seconds = Counter()
    for day in sig.dates:
        seconds.update(dict(zip(*np.unique(sig.idx[day], return_counts=True))))
    actual = Counter()
    for u, n in seconds.items():
        s = sig.uniq[u]
        if s:
            actual[sig.shift.get(s, s)] += int(n)
    out = []
    for s, n in actual.items():
        g = [i for i in range(n_map) if s[mapping[i]] == "G"]
        pairs = [(a, b) for a, b in combinations(g, 2) if foes[a, b]]
        if pairs:
            out.append({"state": s, "seconds": n, "pairs": pairs})
    return out


def weighted(rows: dict, key: str) -> float | None:
    v = [(r["vehicles"], r[key]) for r in rows.values() if r[key] is not None]
    return sum(n * c for n, c in v) / sum(n for n, _ in v) if v else None


def main() -> None:
    t0 = time.time()
    split = json.loads(SPLIT.read_text())
    dates = sorted(split["train"] + split["validation"])
    links = map_links()
    foes = junction_foes()
    print(f"신호 기록 읽기({len(dates)}일)…", flush=True)
    states = {d: day_states(d) for d in dates}
    flows = passages_per_minute(set(dates))
    ys = flow_series(flows, dates)
    print(f"  {time.time() - t0:.0f}초", flush=True)
    result = {"dates": dates, "method": {"detrend_minutes": DETREND_MIN, "min_vehicles": MIN_VEHICLES,
                                         "saturation_per_lane_hour": 2200, "busy_15min": BUSY_15MIN,
                                         "over_limit": OVER_LIMIT}, "tls": {}}
    fmt = lambda x: "—" if x is None else f"{x:.3f}"
    lines = ["## 교차로별 요약(학습·검증 15일, 15분 이동 평균을 뺀 1분 상관)", "",
             "| 신호 | 칸(지도/원본) | 칸 대응 | 상관: 밀림 없음 | 상관: 한 현시 밀림 | 지나가는 상태가 밀림과 맞는 이동 | "
             "(가) 지금 규칙의 G-G 부딪힘 | (나) 통과 기록대로 읽으면 부딪히는 짝 | 지금 규칙으로 용량 부족한 이동 | 다른 칸 대응 중 최고 상관 |",
             "|---|---|---|---:|---:|---|---|---|---|---|"]
    detail = ["", "## 이동별(우회전·유턴·15일 3천 대 미만 제외)", "",
              "A·B·C·D는 원본 기록의 녹색 상태(원본 순환 순서). '밀림이면'은 한 현시 밀림이 맞을 때 차가 지나가야 할 원본 상태.", "",
              "| 신호 | 이동 | 회전 | 15일 통과 | 지도 칸 → 원본 칸 | 자기 칸이 녹색인 원본 상태 | 밀림이면 | 실제로 지나가는 상태(상관) | 맞음 | "
              "넘는 15분: 지금 규칙 / 밀림 없음 / 통과 기록 |",
              "|---|---|---|---:|---|---|---|---|:---:|---|"]
    states_md = ["", "## 원본 녹색 상태", "", "| 신호 | 이름 | 상태 | 하루 비율 |", "|---|---|---|---:|"]
    conflicts_md = ["", "## 부딪힘 자세히", ""]
    for tls in sorted(links):
        sig = Signal(tls, states, dates)
        n_map = max(links[tls]) + 1
        mapping = column_map(tls, n_map, sig.n_rec)
        rows = correlation_rows(sig, links[tls], mapping, flows, ys)
        w0, w1 = weighted(rows, "none"), weighted(rows, "one_phase")
        rel, clashes = release_rows(sig, links[tls], mapping, flows, ys, foes[tls])
        rconf = rule_conflicts(sig, mapping, foes[tls], n_map)
        alt = None
        if sig.n_rec != n_map:          # 다른 대응: 뺄 칸을 모두 바꿔 보며 가장 잘 맞는 것(한 현시 밀림 가정)
            trials = []
            for drop in combinations(range(sig.n_rec), sig.n_rec - n_map):
                m = column_map(tls, n_map, sig.n_rec, list(drop))
                if m != mapping:
                    trials.append((weighted(correlation_rows(sig, links[tls], m, flows, ys), "one_phase") or -1, list(drop)))
            trials.sort(reverse=True)
            alt = {"best_other_drop": trials[0][1], "best_other_corr": round(trials[0][0], 3), "tried": len(trials)}
        judged = [r for r in rel.values() if r["consistent"] is not None]
        ok = [r for r in judged if r["consistent"]]
        short = {k: r for k, r in rel.items() if r["capacity"]["rule"]["over_share"] > OVER_LIMIT}
        total_seconds = 86400 * len(dates)
        result["tls"][tls] = {
            "map_columns": n_map, "record_columns": sig.n_rec, "mapping": mapping, "structural": STRUCTURAL.get(tls),
            "green_states": {"ABCDEFGH"[i]: s for i, s in enumerate(sig.order)},
            "corr_none": w0, "corr_one_phase": w1, "alternatives": alt, "movements": rows, "release": rel,
            "consistent": f"{len(ok)}/{len(judged)}",
            "consistent_vehicle_share": sum(r["vehicles"] for r in ok) / max(1, sum(r["vehicles"] for r in judged)),
            "rule_conflicts": rconf, "passage_reading_clashes": clashes, "rule_capacity_short": sorted(short),
        }
        alt_txt = "—" if not alt else f"{fmt(alt['best_other_corr'])}(뺄 칸 {alt['best_other_drop']}, {alt['tried']}가지 중)"
        share = result["tls"][tls]["consistent_vehicle_share"]
        rc = "없음" if not rconf else "; ".join(f"{len(c['pairs'])}쌍(하루 {c['seconds'] / total_seconds:.0%})" for c in rconf)
        cl = "없음" if not clashes else f"**{len(clashes)}개**"
        sh = "없음" if not short else ", ".join(f"**{k}**({rel[k]['capacity']['rule']['over_share']:.0%})" for k in sorted(short))
        lines.append(f"| {tls} | {n_map}/{sig.n_rec} | {'그대로' if n_map == sig.n_rec else '원본 ' + str(STRUCTURAL[tls]['drop']) + ' 뺌'} | "
                     f"{fmt(w0)} | {fmt(w1)} | {len(ok)}/{len(judged)}(통과량 {share:.0%}) | {rc} | {cl} | {sh} | {alt_txt} |")
        for i, s in enumerate(sig.order):
            n = sum(int((sig.idx[d] == sig.uniq.index(s)).sum()) for d in dates)
            states_md.append(f"| {tls} | {'ABCDEFGH'[i]} | `{s}` | {n / total_seconds:.0%} |")
        for mv, r in sorted(rel.items(), key=lambda kv: -kv[1]["vehicles"]):
            cap = " / ".join(f"{r['capacity'][k]['over_share']:.0%}" for k in READINGS)
            mark = {True: "○", False: "**✕**", None: "—"}[r["consistent"]]
            detail.append(f"| {tls} | {mv} | {r['dir']} | {r['vehicles']:,} | {r['links']} → {r['record_columns']} | "
                          f"{','.join(r['own']) or '없음'} | {','.join(r['expected_if_shift']) or '—'} | "
                          f"{r['release']}({r['release_corr']:+.2f}) | {mark} | {cap} |")
        for c in rconf:
            conflicts_md.append(f"- {tls} (가) 상태 `{c['state']}`(하루 {c['seconds'] / total_seconds:.0%}): 부딪히는 G 칸 짝 {c['pairs']}")
        for c in clashes:
            conflicts_md.append(f"- {tls} (나) 원본 상태 {c['state']}에 함께 지나가는 {c['a']} × {c['b']}")
        print(lines[-1], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d")
    (OUT / f"signal_columns_{stamp}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    (OUT / f"signal_columns_{stamp}.md").write_text("\n".join(lines + states_md + conflicts_md + detail) + "\n")
    print("\n".join(conflicts_md))
    print(f"\n저장: {OUT}/signal_columns_{stamp}.json, .md ({time.time() - t0:.0f}초)")


if __name__ == "__main__":
    main()
