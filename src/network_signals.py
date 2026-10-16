"""12개 신호 교차로 지도의 하루 실제 신호(원본 신호 기록 재생)를 SUMO 추가 파일로 만든다 — 지문 밖.

규칙(docs/12개교차로_오류목록_20261015.md 4절, network_signal_columns로 확인):
- 원본 칸 → 지도 칸은 구조 대응(column_map), 상태는 '한 현시 밀림' 보정(shift_map: 녹색은 앞 녹색, 황색은 앞 녹색의 황색).
- 황색 보정(녹색·양보 녹색에서 적색이 되는 연결은 반드시 황색을 거치게 한다):
  - 황색 현시 동안 g로 남았다가 다음에 적색이 되는 연결은 황색으로(석천사거리 단독 모형의 유턴 황색과 같은 규칙)
  - 녹색끼리 황색 없이 바뀌어 적색이 되는 연결이 있으면(204920: 매 주기 한 번), 끝나는 녹색의 마지막 4초를 그 연결만
    황색으로 바꾼다(v6 전적색처럼 끝나는 녹색에서 떼어 와 하루 시간표를 지킴, 녹색은 적어도 1초 남김)
- 부딪히는 우선 녹색: 같은 현시에 G인 두 연결이 SUMO 교차로 표에서 서로 부딪히면(foes), 표에서 양보하게 되어 있는 쪽
  (response)을 양보 녹색 g로 바꾼다(SUMO가 신호를 스스로 만들 때와 같은 방식). 원본 기록에서 같이 켜지지만 지도에서는
  차로가 겹치는 곳이다: 204920 동쪽 직좌(19·20번)와 서쪽 직진(1~6번), cluster 좌회전 17번과 5번.
  둘 다 양보하게 되어 있지 않으면 바꾸지 않고 센다.
- 석천사거리(204820)는 단독 모형과 같은 실제 신호(focused_scenario.aihub_signal_program)다. 정책 비교에서는 이 신호만 바꾼다.

사용: .venv/bin/python src/network_signals.py --date 20220811
결과: results/network_demand/signals/real_<날짜>.add.xml
"""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import focused_data as fd
import network_signal_columns as nsc
from focused_scenario import aihub_signal_program

OUT = fd.PROJECT / "results/network_demand/signals"
TARGET_TLS = "204820"
YELLOW_S = 4
PROGRAM_ID = "real"


def compress(states: list[str]) -> list[tuple[int, str]]:
    phases: list[tuple[int, str]] = []
    for s in states:
        if phases and phases[-1][1] == s:
            phases[-1] = (phases[-1][0] + 1, s)
        else:
            phases.append((1, s))
    return phases


def add_yellows(phases: list[tuple[int, str]]) -> tuple[list[tuple[int, str]], Counter]:
    """녹색·양보 녹색 → 적색인 연결이 황색을 거치게 한다. 돌려주는 Counter: {'유턴 황색': 바꾼 현시 수, '황색 넣음': 넣은 수}."""
    out, note = [], Counter()
    for i, (d, s) in enumerate(phases):
        nxt = phases[i + 1][1] if i + 1 < len(phases) else None
        ends = {k for k, (a, b) in enumerate(zip(s, nxt)) if a in "Gg" and b == "r"} if nxt else set()
        if not ends:
            out.append((d, s))
            continue
        yellow = "".join("y" if k in ends else c for k, c in enumerate(s))
        if "y" in s:                                     # 황색 현시에 남은 녹색·양보 녹색
            out.append((d, yellow))
            note["유턴 황색"] += 1
        elif d > 1:                                      # 녹색에서 바로 적색: 끝 4초(녹색 1초는 남김)를 그 연결만 황색으로
            y = min(YELLOW_S, d - 1)
            out += [(d - y, s), (y, yellow)]
            note["황색 넣음"] += 1
        else:
            out.append((d, s))
            note["황색 못 넣음(녹색 1초)"] += 1
    return out, note


def right_of_way(tls: str) -> tuple[list[list[bool]], list[list[bool]]]:
    """12개 지도의 교차로 표: foes[i][j](부딪힘), yields[i][j](i가 j에게 양보). 신호 칸 번호 = 요청 번호(12곳 확인)."""
    import xml.etree.ElementTree as ET2
    from network_map import network_12
    if tls not in _ROW:
        root = ET2.parse(network_12()).getroot()
        jid = next(c.get("via")[1:].rsplit("_", 2)[0] for c in root.iter("connection") if c.get("tl") == tls)
        j = next(x for x in root.iter("junction") if x.get("id") == jid)
        req = {int(r.get("index")): r for r in j.iter("request")}
        n = len(req)
        bit = lambda text, k: text[n - 1 - k] == "1"
        _ROW[tls] = ([[bit(req[i].get("foes"), k) for k in range(n)] for i in range(n)],
                     [[bit(req[i].get("response"), k) for k in range(n)] for i in range(n)])
    return _ROW[tls]


_ROW: dict = {}


def yield_conflicts(state: str, foes, yields) -> tuple[str, int, int]:
    """같이 G인 부딪히는 두 연결 중 양보하게 되어 있는 쪽을 g로. 돌려줌: (새 상태, 바꾼 수, 못 정한 짝 수)."""
    s = list(state)
    changed = unresolved = 0
    g = [i for i, c in enumerate(s) if c == "G"]
    for a in g:
        for b in g:
            if a < b and s[a] == "G" and s[b] == "G" and (foes[a][b] or foes[b][a]):
                if yields[a][b]:
                    s[a] = "g"; changed += 1
                elif yields[b][a]:
                    s[b] = "g"; changed += 1
                else:
                    unresolved += 1
    return "".join(s), changed, unresolved


def neighbor_program(date: str, tls: str, states: list[str]) -> tuple[list[tuple[int, str]], dict]:
    """원본 1초 상태(86,400개) → 지도 칸 순서의 실제 신호 현시 목록."""
    known = [s for s in states if s]
    if not known:
        raise ValueError(f"{date} {tls}: 신호 기록이 없습니다.")
    shift = nsc.shift_map(states)
    n_rec = Counter(len(s) for s in known).most_common(1)[0][0]
    links = nsc.map_links()[tls]
    n_map = max(links) + 1
    mapping = nsc.column_map(tls, n_map, n_rec)
    first = known[0]
    actual = [shift.get(s or first, s or first) for s in states]
    foes, yields = right_of_way(tls)
    fixed, notes = {}, Counter()
    for a in set(actual):
        st, changed, unresolved = yield_conflicts("".join(a[c] for c in mapping), foes, yields)
        fixed[a] = st
        if changed:
            notes["양보 녹색으로 바꾼 상태"] += 1
        if unresolved:
            notes["못 정한 부딪힘 짝"] += unresolved
    per_second = [fixed[a] for a in actual]
    phases, note = add_yellows(compress(per_second))
    return phases, {"links": n_map, "record_columns": n_rec, "phases": len(phases), **note, **notes}


def day_programs(date: str) -> tuple[dict[str, list[tuple[int, str]]], dict]:
    states = nsc.day_states(date)
    programs, info = {}, {}
    for tls in sorted(states):
        if tls == TARGET_TLS:
            programs[tls] = aihub_signal_program(date, 0, 86400)
            info[tls] = {"links": len(programs[tls][0][1]), "phases": len(programs[tls]), "source": "단독 모형과 같은 신호"}
        else:
            programs[tls], info[tls] = neighbor_program(date, tls, states[tls])
        total = sum(d for d, _ in programs[tls])
        if total != 86400:
            raise ValueError(f"{date} {tls}: 하루 신호 길이 {total}초 ≠ 86400초")
    return programs, info


def write_additional(programs: dict[str, list[tuple[int, str]]], output: Path) -> Path:
    root = ET.Element("additional")
    for tls, phases in sorted(programs.items()):
        logic = ET.SubElement(root, "tlLogic", id=tls, type="static", programID=PROGRAM_ID, offset="0")
        for d, s in phases:
            ET.SubElement(logic, "phase", duration=str(d), state=s)
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    return output


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--date", required=True)
    args = ap.parse_args()
    programs, info = day_programs(args.date)
    path = write_additional(programs, OUT / f"real_{args.date}.add.xml")
    (OUT / f"real_{args.date}.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
    for tls, x in info.items():
        print(tls, x)
    print("저장:", path)


if __name__ == "__main__":
    main()
