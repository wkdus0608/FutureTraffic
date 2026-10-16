"""12개 신호 교차로 지도(지문 밖): AI Hub 원본 지도 + 붙은 비신호 교차로 10쌍 합침 + v6 북쪽 도로 연장 240m.

왜 합치나: 원본에는 신호 없는 교차로끼리 0.2~9.3m짜리 도로로 이어진 곳이 있다(대부분 AI Hub가 지도를 고치며 넣은
SN2022… 교차로). SUMO에서 이런 곳은 차가 두 교차로 안에 걸쳐 서로를 막는다(SUMO 설명서: 가까운 교차로는 합칠 것).
08-11 12개 지도 시험 주행에서 막힘·충돌·하루 교착의 시작이 이런 곳(201330 옆 0.2m, cluster 옆 1.7m, 204922·605257 0.29m,
201300 옆 4.6m, 201286 옆 8.8m 등)에 모였다.
신호 교차로와 그 연결은 건드리지 않는다(아래 검사).

만드는 순서(순서가 중요): 원본에 netconvert로 합치기 → focused_data와 같은 함수로 북쪽 연장.
netconvert는 차로 모양을 교차로 위치로 다시 계산하므로, 연장을 먼저 하면 연장이 지워진다(287.1m → 47.1m).

검사(만들 때마다): 신호 연결 249개·신호 프로그램이 원본과 같음, 없어진 도로 = 합친 짝 사이의 10m 미만 도로뿐,
석천사거리 단독 모형이 쓰는 도로는 모두 남음, 남은 도로의 차로 수·허용 차종·속도가 같음, 북쪽 연장 도로가 원본 + 240m.
사용: .venv/bin/python src/network_map.py   (만들고 검사, 몇 초)
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import focused_data as fd

CACHE = fd.PROJECT / "results/network_demand/map"
# 합칠 비신호 교차로 짝(묶음): (교차로들, 이어 주는 도로, 그 길이 m, 위치). 기준: 신호 없는 교차로끼리 10m 미만 도로로
# 이어진 곳(netconvert가 교차로를 합칠 때 쓰는 기본 거리 10m). 단, 석천사거리 단독 모형이 쓰는 도로(진입·진출 구간)에 닿는
# 짝은 합치지 않는다(DQN 감지 구간과 단독 모형 경로를 그대로 두려고): 477706926_0102(남쪽 0.2m), 477770958_0102(서쪽 0.2m),
# 477752894_020202(동쪽 6.5m), 477767118_0101(7.7m), CL20220622172647A_01(9.8m).
JOINS = [
    (("SN20220924161517A", "SN20220924161525A"), "477604248_0201_01/02", 0.20, "201292 동쪽"),
    (("201669", "202475"), "477643150_01/02", 0.20, "201330 옆"),
    (("201618", "202477"), "477643153_01/02", 0.20, "201330 옆"),
    (("SN20220622164058B", "SN20220622164054B"), "477648264_0102_01/02", 0.20, "201305 옆"),
    (("SN20220920230831C", "SN20220920230824C"), "477649571_0102_01/02", 0.20, "201301 남쪽"),
    (("204922", "605257", "605256"), "477771616_01/02 0.29m, 477771614_01/02 9.29m", 0.29, "204920 동쪽(10/5 교착 지점)"),
    (("SN20220920231902C", "SN20220920231858C"), "477604827_0102_01/02", 0.67, "201301 북쪽"),
    (("205835", "204795"), "477748651_01", 1.70, "cluster 동쪽"),
    (("SN20220924161746A", "SN20220924161800A"), "477649508_0201_01/02", 3.91, "201292 서쪽"),
    (("201670", "SN20220924184144A"), "477610004_01_01/02", 4.63, "201300 북쪽"),
    (("634204", "634264"), "477649546_01/02", 5.41, "201305 동쪽"),
    (("201278", "CTN20220928231319G"), "CL20220928231319G_01/02", 5.88, "201292–201293 사이"),
    (("SN20220622164105B", "SN20220622164203B"), "477648264_020201_01/02", 5.95, "201305 북쪽"),
    (("205830", "205833"), "477748647_01", 6.50, "cluster 서쪽"),
    (("203039", "SN20220920172609F"), "477610003_02_01/02", 7.68, "201301 남쪽"),
    (("202119", "203822"), "477655706_01/02", 7.91, "201300 옆"),
    (("201290", "203946"), "477658274_01/02", 8.18, "201293 동쪽"),
    (("201286", "203737"), "477654165_01/02", 8.77, "204822–201293 사이"),
]
MAX_JOIN_EDGE_M = 10.0


def _digest() -> str:
    h = hashlib.sha256(fd.NETWORK.read_bytes())
    h.update(json.dumps([JOINS, fd.variant()["north_extension_edge"], fd.variant()["north_extension_m"]]).encode())
    h.update(Path(__file__).read_bytes())
    return h.hexdigest()[:12]


def network_12() -> Path:
    """12개 지도 파일(캐시). 없으면 만들고 검사한다."""
    path = CACHE / f"net12_{_digest()}.net.xml"
    if path.exists():
        return path
    CACHE.mkdir(parents=True, exist_ok=True)
    nodes = CACHE / "join.nod.xml"
    root = ET.Element("nodes")
    for group, *_ in JOINS:
        ET.SubElement(root, "join", nodes=" ".join(group))
    ET.ElementTree(root).write(nodes, encoding="utf-8", xml_declaration=True)
    joined = path.with_suffix(f".{os.getpid()}.joined.xml")
    from focused_sumo import configure_sumo
    configure_sumo()
    import sumolib
    log = path.with_suffix(".netconvert.log")
    with log.open("w") as f:
        subprocess.run([sumolib.checkBinary("netconvert"), "-s", str(fd.NETWORK), "-n", str(nodes), "-o", str(joined)],
                       check=True, stdout=f, stderr=subprocess.STDOUT)
    v = fd.variant()
    text = joined.read_text(encoding="utf-8")
    if v["north_extension_m"] > 0:
        text = fd._extended_network_text(text, v["north_extension_edge"], v["north_extension_m"])
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    joined.unlink()
    report = verify(fd.NETWORK, tmp)
    tmp.replace(path)
    path.with_suffix(".check.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    return path


def _load(path: Path):
    r = ET.parse(path).getroot()
    edges = {}
    for e in r.iter("edge"):
        if e.get("function") != "internal":
            edges[e.get("id")] = [(l.get("index"), l.get("allow"), l.get("disallow"), l.get("speed"), float(l.get("length")))
                                  for l in e.iter("lane")]
    tls_links = {(c.get("from"), c.get("to"), c.get("fromLane"), c.get("toLane")): (c.get("tl"), c.get("linkIndex"), c.get("dir"))
                 for c in r.iter("connection") if c.get("tl")}
    programs = {t.get("id"): [(p.get("duration"), p.get("state")) for p in t.iter("phase")] for t in r.iter("tlLogic")}
    return edges, tls_links, programs


def join_edges(original: Path) -> set[str]:
    """합친 묶음 안의 교차로끼리 잇는 도로(합치면 없어져야 할 도로)."""
    groups = {n: i for i, (group, *_) in enumerate(JOINS) for n in group}
    out = set()
    for e in ET.parse(original).getroot().iter("edge"):
        if e.get("function") != "internal" and e.get("from") in groups and groups.get(e.get("to")) == groups[e.get("from")]:
            out.add(e.get("id"))
    return out


def single_model_edges() -> set[str]:
    """석천사거리 단독 모형이 쓰는 도로(08-11 08:00·18:00 차량 경로). 12개 지도에서도 그대로 있어야 한다."""
    import tempfile
    from focused_scenario import window_route_file
    used = set()
    with tempfile.TemporaryDirectory() as tmp:
        for start in (8 * 3600, 18 * 3600):
            out = Path(tmp) / f"w{start}.rou.xml"
            window_route_file("20220811", start, start + 900, out)
            for r in ET.parse(out).getroot().iter("route"):
                used.update(r.get("edges").split())
    return used


def verify(original: Path, built: Path) -> dict:
    ea, ta, pa = _load(original)
    eb, tb, pb = _load(built)
    removed = sorted(set(ea) - set(eb))
    expected = join_edges(original)
    problems = []
    if ta != tb:
        problems.append("신호 연결이 원본과 다름")
    if pa != pb:
        problems.append("신호 프로그램이 원본과 다름")
    if set(eb) - set(ea):
        problems.append(f"새 도로 {sorted(set(eb) - set(ea))}")
    if set(removed) != expected:
        problems.append(f"없어진 도로 {sorted(set(removed) ^ expected)}가 합친 짝 사이 도로와 다름")
    if any(max(l[4] for l in ea[e]) >= MAX_JOIN_EDGE_M for e in removed):
        problems.append(f"{MAX_JOIN_EDGE_M:g}m 넘는 도로가 없어짐")
    lost = sorted(single_model_edges() - set(eb))
    if lost:
        problems.append(f"석천사거리 단독 모형이 쓰는 도로가 없어짐: {lost}")
    lanes = [e for e in set(ea) & set(eb) if [l[:4] for l in ea[e]] != [l[:4] for l in eb[e]]]
    if lanes:
        problems.append(f"차로 수·허용·속도가 바뀐 도로 {lanes}")
    v = fd.variant()
    ext = v["north_extension_edge"]
    if v["north_extension_m"] > 0:
        want = _load(fd.extended_network(v["north_extension_m"], ext))[0][ext]
        got = eb[ext]
        if max(abs(a[4] - b[4]) for a, b in zip(want, got)) > 1.0:
            problems.append(f"북쪽 연장 길이 {[l[4] for l in got]} ≠ v6 {[l[4] for l in want]}")
    if problems:
        raise ValueError("12개 지도 검사 실패: " + "; ".join(problems))
    changed = {e: (round(ea[e][0][4], 1), round(eb[e][0][4], 1)) for e in set(ea) & set(eb)
               if abs(ea[e][0][4] - eb[e][0][4]) > 0.5}
    return {"removed_edges": removed, "tls_links": len(tb), "length_changed_edges": dict(sorted(changed.items())),
            "north_extension": [l[4] for l in eb[ext]]}


def main() -> None:
    path = network_12()
    report = json.loads(path.with_suffix(".check.json").read_text())
    print("12개 지도:", path)
    print(f"  없어진 짧은 도로 {len(report['removed_edges'])}개, 신호 연결 {report['tls_links']}개 원본과 같음, "
          f"북쪽 연장 차로 길이 {report['north_extension']}, 길이가 0.5m 넘게 바뀐 도로 {len(report['length_changed_edges'])}개")


if __name__ == "__main__":
    sys.exit(main())
