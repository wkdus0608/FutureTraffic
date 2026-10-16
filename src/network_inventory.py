"""12개 신호 교차로의 자료·지도 오류 목록. 원본 자료와 지도만 읽고 SUMO는 돌리지 않는다(환경·평가 지문 밖).

1. 이동(진입 도로 → 진출 도로)마다 19일 회전교통량(통과 횟수)을 모으고, 지도에서 그 이동이 어떻게 표현되는지 나눈다.
   - 직접 연결: 지도에 그 교차로의 연결이 있다.
   - 번호 대응: 기록의 도로 번호가 지도에 없고, 같은 도로 계열의 다른 번호로 대응한다(이미 쓰는 대응 + 예전 대응표).
   - 다른 길: 직접 연결은 없지만 지도에 분리된 회전 차로가 따로 그려져 있다(예전 대응표의 경로를 지금 지도에서 다시 확인).
   - 이을 길 없음: 지도에 그 이동을 나타낼 연결이 없다(유턴·기타).
   - 지도에 없는 번호: 도로 번호가 지도에 없고 대응도 없다.
2. 이동의 모양(직진·좌회전·우회전·유턴)을 지도 좌표로 계산한다(진입 도로 끝 방향과 진출 도로 시작 방향의 각도).
3. 신호: 지도의 연결 칸 수와 AI Hub 신호 기록(19일 × 96개 15분)의 글자 수를 비교하고, 한 번도 녹색을 받지 않는 칸을 센다.
4. 교차로마다 AI Hub 설명서의 '보완 지점'(측정 지점이 아님: 201300·201329·201330)을 표시한다.

사용: .venv/bin/python src/network_inventory.py
결과: results/network_inventory/inventory_<날짜>.json, inventory_<날짜>.md
"""
from __future__ import annotations

import json
import math
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict, deque
from pathlib import Path

from focused_data import ID_ALIAS, NETWORK, PROJECT, SIGNALS, iter_turn_files

OUT = PROJECT / "results/network_inventory"
ARCHIVE_MOVEMENTS = PROJECT.parent / "자주프_보관_20261005/config/network_movements.json"
SUPPLEMENT_NODES = {"201300", "201329", "201330"}     # AI Hub 설명서: 측정 지점이 아닌 보완 지점(예전 점검 확인)


class Map:
    def __init__(self, path: Path = NETWORK):
        root = ET.parse(path).getroot()
        self.edges, self.shape, self.length = {}, {}, {}
        for e in root.iter("edge"):
            if e.get("function") == "internal" or e.get("id", "").startswith(":"):
                continue
            lanes = e.findall("lane")
            mid = lanes[len(lanes) // 2]
            self.edges[e.get("id")] = (e.get("from"), e.get("to"))
            self.shape[e.get("id")] = [tuple(map(float, p.split(","))) for p in mid.get("shape").split()]
            self.length[e.get("id")] = float(mid.get("length"))
        self.succ = defaultdict(set)
        self.pred = defaultdict(set)
        self.connections = defaultdict(list)          # (from, to) → [(dir, tl, linkIndex)]
        self.tls_links = defaultdict(set)
        self.tls_nodes = defaultdict(set)
        for c in root.iter("connection"):
            f, t = c.get("from"), c.get("to")
            if f.startswith(":") or t.startswith(":"):
                continue
            self.succ[f].add(t)
            self.pred[t].add(f)
            self.connections[(f, t)].append((c.get("dir"), c.get("tl"), c.get("linkIndex")))
            if c.get("tl"):
                self.tls_links[c.get("tl")].add(int(c.get("linkIndex")))
                self.tls_nodes[c.get("tl")].add(self.edges[f][1])
        self.node_tls = {n: t for t, ns in self.tls_nodes.items() for n in ns}

    def junction(self, f: str, t: str) -> str | None:
        if f in self.edges:
            return self.edges[f][1]
        if t in self.edges:
            return self.edges[t][0]
        return None

    def turn_angle(self, f: str, t: str) -> float | None:
        """진입 도로 끝 방향에서 진출 도로 시작 방향까지 돈 각도(도, 왼쪽 +)."""
        if f not in self.shape or t not in self.shape:
            return None
        (a1, a2), (b1, b2) = self.shape[f][-2:], self.shape[t][:2]
        h1 = math.atan2(a2[1] - a1[1], a2[0] - a1[0])
        h2 = math.atan2(b2[1] - b1[1], b2[0] - b1[0])
        return (math.degrees(h2 - h1) + 180) % 360 - 180

    def shortest_path(self, f: str, t: str, max_edges: int = 8) -> list[str] | None:
        """f에서 t로 가는 가장 짧은(도로 수) 길. 없으면 None."""
        seen, queue = {f: None}, deque([f])
        while queue:
            e = queue.popleft()
            if e == t:
                path = []
                while e is not None:
                    path.append(e)
                    e = seen[e]
                return path[::-1]
            depth = 0
            x = e
            while seen[x] is not None:
                depth += 1
                x = seen[x]
            if depth >= max_edges:
                continue
            for n in sorted(self.succ[e]):
                if n not in seen:
                    seen[n] = e
                    queue.append(n)
        return None


def shape_kind(angle: float | None, f: str, t: str) -> str:
    base = lambda e: re.sub(r"_0[12](_0[12])*$", "", e)
    if angle is None:
        return "알 수 없음"
    if abs(angle) > 150 or (base(f) == base(t) and f != t and abs(angle) > 120):
        return "유턴"
    if abs(angle) <= 30:
        return "직진"
    return "좌회전" if angle > 0 else "우회전"


def read_counts() -> tuple[dict, dict, dict]:
    total, by_date, by_class = Counter(), defaultdict(Counter), defaultdict(Counter)
    for name, data in iter_turn_files():
        m = re.search(r"/(\d{8})/\d{14}(\w)\.xml$", name) or re.search(r"/(\d{8})/.*?(\w)\.xml$", name)
        date, cls = m.group(1), m.group(2)
        for er in ET.fromstring(data).iter("edgeRelation"):
            key = (er.get("from"), er.get("to"))
            n = int(float(er.get("count")))
            total[key] += n
            by_date[key][date] += n
            by_class[key][cls] += n
    return total, by_date, by_class


def archive_overrides() -> dict:
    if not ARCHIVE_MOVEMENTS.exists():
        return {}
    data = json.loads(ARCHIVE_MOVEMENTS.read_text())
    return {(o["source_from"], o["source_to"]): o for o in data.get("overrides", [])}


def classify(mp: Map, f: str, t: str, overrides: dict) -> dict:
    o = overrides.get((f, t))
    alias = {**ID_ALIAS, **{o["source_to"]: o["path"][-1] for o in overrides.values() if o["resolution"] == "alias"}}
    out = {"archive": None if o is None else {"resolution": o["resolution"], "status": o["status"],
                                               "path": o.get("path"), "evidence": o.get("evidence", "")[:300]}}
    f2, t2 = alias.get(f, f), alias.get(t, t)
    missing = [e for e in (f, t) if e not in mp.edges]
    if missing:
        if all(e in mp.edges for e in (f2, t2)) and (f2, t2) in mp.connections:
            return {**out, "category": "번호 대응", "mapped": [f2, t2], "missing_ids": missing}
        return {**out, "category": "지도에 없는 번호", "missing_ids": missing}
    if (f, t) in mp.connections:
        return {**out, "category": "직접 연결", "map_dir": sorted({d for d, _, _ in mp.connections[(f, t)]})}
    if o is not None and o["resolution"] == "bypass" and o.get("path"):
        path = o["path"]
        contiguous = all(b in mp.succ[a] for a, b in zip(path, path[1:])) and all(e in mp.edges for e in path)
        shared_up = bool(mp.pred[path[0]] & mp.pred[f]) or path[0] in mp.pred[f] or bool(mp.pred[path[0]] & {f})
        shared_down = bool(mp.succ[path[-1]] & mp.succ[t]) or t in mp.succ[path[-1]] or path[-1] in mp.succ[t]
        return {**out, "category": "다른 길(분리 회전 차로)", "path": path,
                "path_length_m": round(sum(mp.length[e] for e in path if e in mp.length), 1),
                "path_contiguous": contiguous, "shares_upstream": shared_up, "shares_downstream": shared_down}
    sp = mp.shortest_path(f, t)
    other = None
    if sp:
        nodes = {mp.edges[e][1] for e in sp[:-1]}
        other = {"edges": len(sp), "length_m": round(sum(mp.length[e] for e in sp), 1),
                 "passes_other_signals": sorted({mp.node_tls[n] for n in nodes if n in mp.node_tls} - {mp.node_tls.get(mp.junction(f, t))})}
    return {**out, "category": "이을 길 없음", "detour": other}


def read_signals(mp: Map) -> dict:
    lengths, files, ever_green = defaultdict(Counter), Counter(), defaultdict(set)
    for day in sorted(p for p in SIGNALS.iterdir() if p.is_dir()):
        for xml in sorted(day.glob("*s.xml")):
            for logic in ET.parse(xml).getroot().iter("tlLogic"):
                tid = logic.get("id")
                files[tid] += 1
                for ph in logic.iter("phase"):
                    s = ph.get("state")
                    lengths[tid][len(s)] += 1
                    ever_green[(tid, len(s))].update(i for i, ch in enumerate(s) if ch in "Gg")
    out = {}
    for tid in sorted(set(mp.tls_links) | set(lengths)):
        n_map = max(mp.tls_links[tid]) + 1 if mp.tls_links.get(tid) else None
        lens = dict(lengths.get(tid, {}))
        main = max(lens, key=lens.get) if lens else None
        never = sorted(set(range(main)) - ever_green[(tid, main)]) if main else None
        out[tid] = {"map_links": n_map, "record_lengths": lens, "record_files": files.get(tid, 0),
                    "match": main == n_map and len(lens) == 1, "never_green_columns": never}
    return out


def main() -> None:
    t0 = time.time()
    mp = Map()
    overrides = archive_overrides()
    total, by_date, by_class = read_counts()
    dates = sorted({d for c in by_date.values() for d in c})
    print(f"회전 기록 {len(total)}개 이동, 통과 {sum(total.values()):,}회, 날짜 {len(dates)}일 ({time.time() - t0:.0f}초)", flush=True)
    movements = []
    for (f, t), n in sorted(total.items(), key=lambda kv: -kv[1]):
        node = mp.junction(f, t)
        tls = mp.node_tls.get(node, node)
        angle = mp.turn_angle(*(lambda c: (c.get("mapped", [f, t])[0], c.get("mapped", [f, t])[1]))(classify(mp, f, t, overrides)))
        c = classify(mp, f, t, overrides)
        daily = [by_date[(f, t)].get(d, 0) for d in dates]
        movements.append({"from": f, "to": t, "tls": tls, "node": node, "count_19d": n,
                          "daily_mean": round(n / len(dates), 1), "daily_min": min(daily), "daily_max": max(daily),
                          "days_with_count": sum(1 for x in daily if x), "by_class": dict(by_class[(f, t)]),
                          "angle_deg": None if angle is None else round(angle, 1),
                          "shape": shape_kind(angle, f, t), **c})
    tls_total = Counter()
    for m in movements:
        tls_total[m["tls"]] += m["count_19d"]
    for m in movements:
        m["share_of_tls"] = round(100 * m["count_19d"] / tls_total[m["tls"]], 2)
    signals = read_signals(mp)
    summary = {}
    for tid in sorted(tls_total, key=lambda k: -tls_total[k]):
        ms = [m for m in movements if m["tls"] == tid]
        cat = Counter()
        for m in ms:
            cat[m["category"]] += m["count_19d"]
        summary[tid] = {"daily_mean": round(tls_total[tid] / len(dates)), "movements": len(ms),
                        "share_by_category": {k: round(100 * v / tls_total[tid], 2) for k, v in cat.items()},
                        "supplement_node": tid in SUPPLEMENT_NODES, "signal": signals.get(tid)}
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d")
    result = {"dates": dates, "network": str(NETWORK), "movements": movements, "summary": summary, "signals": signals,
              "archive_registry": str(ARCHIVE_MOVEMENTS) if ARCHIVE_MOVEMENTS.exists() else None}
    (OUT / f"inventory_{stamp}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    order = ["직접 연결", "번호 대응", "다른 길(분리 회전 차로)", "이을 길 없음", "지도에 없는 번호"]
    lines = ["## 교차로별 요약(19일)", "",
             "| 교차로 | 하루 통과 | 이동 수 | " + " | ".join(order) + " | 신호 칸(지도/기록) | 보완 지점 |",
             "|---|---:|---:|" + "---:|" * len(order) + "---|:---:|"]
    for tid, s in summary.items():
        sig = s["signal"] or {}
        rec = "/".join(str(k) for k in sorted(sig.get("record_lengths", {}))) or "—"
        flag = "" if sig.get("match") else " ⚠"
        lines.append(f"| {tid} | {s['daily_mean']:,} | {s['movements']} | "
                     + " | ".join(f"{s['share_by_category'].get(k, 0):.1f}%" if s['share_by_category'].get(k) else "—" for k in order)
                     + f" | {sig.get('map_links')}/{rec}{flag} | {'○' if s['supplement_node'] else ''} |")
    lines += ["", "## 직접 연결이 아닌 이동", "",
              "| 교차로 | 이동 | 모양(각도) | 분류 | 19일 합계 | 하루 평균(최소~최대) | 교차로 안 비율 | 확인 내용 |",
              "|---|---|---|---|---:|---:|---:|---|"]
    for m in movements:
        if m["category"] == "직접 연결":
            continue
        if m["category"] == "번호 대응":
            note = f"지도에 없는 {', '.join(m['missing_ids'])} → {m['mapped'][0]} → {m['mapped'][1]} 연결 있음"
        elif m["category"].startswith("다른 길"):
            note = (f"경로 {' → '.join(m['path'])} ({m['path_length_m']}m), 이어짐 {m['path_contiguous']}, "
                    f"상류 공유 {m['shares_upstream']}, 하류 공유 {m['shares_downstream']}")
        elif m["category"] == "이을 길 없음":
            d = m["detour"]
            note = ("지도에 다른 길 없음" if not d else
                    f"가장 짧은 다른 길 {d['edges']}개 도로 {d['length_m']}m" + (f", 다른 신호 {d['passes_other_signals']} 지남" if d["passes_other_signals"] else ""))
            if m["archive"]:
                note += f" | 예전 판단: {m['archive']['resolution']}/{m['archive']['status']}"
        else:
            note = f"지도에 없는 번호 {m['missing_ids']}"
        lines.append(f"| {m['tls']} | {m['from']} → {m['to']} | {m['shape']}({m['angle_deg']}°) | {m['category']} | "
                     f"{m['count_19d']:,} | {m['daily_mean']:,.0f}({m['daily_min']:,}~{m['daily_max']:,}) | {m['share_of_tls']}% | {note} |")
    lines += ["", "## 신호 기록과 지도의 칸 수", "", "| 신호 | 지도 칸 수 | 기록 글자 수(단계 수) | 기록 파일 | 같음 | 한 번도 녹색이 아닌 칸 |",
              "|---|---:|---|---:|:---:|---|"]
    for tid, s in signals.items():
        lines.append(f"| {tid} | {s['map_links']} | {', '.join(f'{k}({v:,})' for k, v in sorted(s['record_lengths'].items()))} | "
                     f"{s['record_files']} | {'○' if s['match'] else '✕'} | {s['never_green_columns']} |")
    (OUT / f"inventory_{stamp}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n저장: {OUT}/inventory_{stamp}.json, .md ({time.time() - t0:.0f}초)")


if __name__ == "__main__":
    main()
