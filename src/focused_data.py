"""석천사거리(204820) 실험에 쓰는 자료를 읽는다. 자료 설명: data/README.md

data/raw (AI Hub 원본, 읽기만 한다)
- network/1200012n.net.xml: 부천 도로망 지도(도로·차로·교차로·신호 연결)
- network/exp.rou.xml: 차종 정의(승용·버스·트럭·이륜)
- turn_counts/: 19일 × 96개 15분 × 4차종 = 7,296개 회전교통량 XML이 압축된 구간 그대로.
  꺼낼 때마다 CRC32로 검사한다.
- signals/<날짜>/: 15분 단위 신호 XML(19일 × 96개)
data/derived (원본에서 이 파일의 함수로 다시 만들 수 있는 표)
- counts_204820.csv: 204820 진입 도로에서 시작하는 회전 기록만 모은 표
- branch_inflow_477748652_01.csv: 서쪽 접근로 4차로 가지로 들어가는 상류 회전 기록
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import struct
import xml.etree.ElementTree as ET
import zlib
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "data/raw"
DERIVED = PROJECT / "data/derived"
TARGET_TLS = "204820"
# 원본 회전량의 진출 도로 477706911_02는 지도에 없다. 204820에서 그 방향으로 나가는
# 도로는 477706911_02_02 하나뿐이므로 그 도로로 대응시킨다(조건부 대응, 기록 유지).
ID_ALIAS = {"477706911_02": "477706911_02_02"}
VEHICLE_CODES = {"p": "passenger", "b": "bus", "t": "truck", "m": "motorcycle"}

NETWORK = RAW / "network/1200012n.net.xml"
VTYPES = RAW / "network/exp.rou.xml"
BLOCK = RAW / "turn_counts/all_turn_xml_compressed_block.bin"
BLOCK_META = RAW / "turn_counts/turn_block_metadata.json"
ENTRIES = RAW / "turn_counts/turn_entries.json"   # Drive 원본 압축의 중앙 목록 중 회전량 XML 7,296개 항목
SIGNALS = RAW / "signals"
COUNTS = DERIVED / "counts_204820.csv"


ENV_V6 = PROJECT / "config/environment_v6.json"
TRANSITION_RULES = PROJECT / "config/transition_rules.json"
NETWORK_CACHE = PROJECT / "results/focused_204820/cache"


def variant() -> dict:
    """이 실행의 환경 변형: 전환 규칙 묶음(전적색 길이)과 북쪽 진입로 연장 길이.

    환경 변수 FOCUSED_RULESET(none·pet1·pet1_5), FOCUSED_NORTH_EXT(m)로 고른다. 정하지 않으면
    config/environment_v6.json의 기본값이다. 규칙 none + 연장 0이면 v5와 같은 환경이다.
    """
    cfg = json.loads(ENV_V6.read_text())
    sets = json.loads(TRANSITION_RULES.read_text())["rule_sets"]
    name = os.environ.get("FOCUSED_RULESET") or cfg["default_rule_set"]
    if name not in sets:
        raise SystemExit(f"전환 규칙 묶음 '{name}'이 없습니다({', '.join(sets)}). FOCUSED_RULESET을 확인하세요.")
    extension = float(os.environ.get("FOCUSED_NORTH_EXT", cfg["north_extension_m"]))
    all_red = {tuple(int(x) for x in k.split(">")): int(v) for k, v in sets[name].items()}
    return {"rule_set": name, "all_red": all_red, "north_extension_m": extension,
            "north_extension_edge": cfg["north_extension_edge"]}


def variant_signature() -> str:
    v = variant()
    rules = ",".join(f"{i}>{j}:{r}" for (i, j), r in sorted(v["all_red"].items()))
    return f"rule_set={v['rule_set']};all_red={rules};north_ext={v['north_extension_m']:g}"


def _extended_network_text(text: str, edge: str, length_m: float) -> str:
    """edge(들어오는 도로)의 시작점을 진행 반대 방향으로 length_m 늘린다. 차로 모양의 첫 점과 길이만 바꾼다."""
    pattern = re.compile(r'(<lane id="%s_(\d+)"[^>]*?length=")([\d.]+)("[^>]*?shape=")([^"]+)(")' % re.escape(edge))
    found = 0

    def replace(m):
        nonlocal found
        found += 1
        pts = [tuple(float(c) for c in p.split(",")) for p in m.group(5).split()]
        (x0, y0), (x1, y1) = pts[0], pts[1]
        d = math.hypot(x1 - x0, y1 - y0)
        pts[0] = (x0 - (x1 - x0) / d * length_m, y0 - (y1 - y0) / d * length_m)
        shape = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        return f"{m.group(1)}{float(m.group(3)) + length_m:.2f}{m.group(4)}{shape}{m.group(6)}"

    out = pattern.sub(replace, text)
    if not found:
        raise ValueError(f"연장할 도로 {edge}의 차로를 지도에서 찾지 못했습니다.")
    return out


def extended_network(length_m: float, edge: str) -> Path:
    """원본 지도에 북쪽 진입로 연장을 반영한 사본(캐시). 내용으로 이름을 정해 여러 프로세스가 같이 써도 안전하다."""
    raw = NETWORK.read_bytes()
    key = hashlib.sha256(raw + f"|{edge}|{length_m:g}".encode()).hexdigest()[:12]
    path = NETWORK_CACHE / f"net_north_ext_{length_m:g}m_{key}.net.xml"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(_extended_network_text(raw.decode("utf-8"), edge, length_m), encoding="utf-8")
        tmp.replace(path)
    return path


def find_network() -> Path:
    if not NETWORK.exists():
        raise FileNotFoundError(f"지도 파일이 없습니다: {NETWORK}")
    v = variant()
    if v["north_extension_m"] > 0:
        return extended_network(v["north_extension_m"], v["north_extension_edge"])
    return NETWORK


# ------------------------------------------------------------------ 환경 지문
# 가상 도로(차량·운전자·신호·DQN이 보는 것·규칙)를 정하는 파일. 하나라도 바뀌면 지문이 바뀌고,
# 지문이 다른 캐시·결과·모델은 다시 쓰지 않는다(예전 결과가 새 결과처럼 쓰이는 것을 막는다).
ENV_FILES = [
    "src/focused_data.py", "src/focused_scenario.py", "src/focused_sumo.py", "src/focused_rl.py",
    "config/driver_model.json", "config/focused_split.json", "config/vehicle_wait_rule.json",
    "config/environment_v6.json", "config/transition_rules.json",
    "data/raw/manifest.json", "data/derived/counts_204820.csv", "data/derived/branch_inflow_477748652_01.csv",
]


def fingerprint(paths: list[str]) -> str:
    """파일 이름과 내용으로 만든 짧은 지문(12자리)."""
    h = hashlib.sha256()
    for rel in paths:
        h.update(rel.encode())
        h.update(hashlib.sha256((PROJECT / rel).read_bytes()).digest())
    h.update(variant_signature().encode())     # 같은 파일이라도 전환 규칙·북쪽 연장이 다르면 다른 환경이다
    return h.hexdigest()[:12]


def environment_fingerprint() -> str:
    return fingerprint(ENV_FILES)


def target_movements(network: Path) -> dict[tuple[str, str], str]:
    """204820이 제어하는 (진입 도로, 진출 도로) 쌍과 회전 방향."""
    root = ET.parse(network).getroot()
    return {
        (c.get("from"), c.get("to")): c.get("dir")
        for c in root.findall("connection")
        if c.get("tl") == TARGET_TLS
    }


def iter_turn_files():
    """압축 구간에서 교차로_all 회전량 XML을 하나씩 검증하며 꺼낸다."""
    meta = json.loads(BLOCK_META.read_text())
    block = BLOCK.read_bytes()
    items = [
        i for i in json.loads(ENTRIES.read_text())
        if "교통량데이터(회전교통량기반)/교차로_all/" in i["name"] and i["name"].endswith(".xml")
    ]
    if len(items) != meta["files"]:
        raise ValueError(f"파일 수 불일치: {len(items)} != {meta['files']}")
    for item in items:
        position = item["offset"] - meta["offset"]
        head = struct.unpack_from("<4s5H3I2H", block, position)
        if head[0] != b"PK\x03\x04":
            raise ValueError(f"로컬 헤더 오류: {item['name']}")
        start = position + 30 + head[-2] + head[-1]
        compressed = block[start:start + item["compressed_size"]]
        if item["method"] == 8:
            decoder = zlib.decompressobj(-15)
            data = decoder.decompress(compressed, item["size"] + 1)
        else:
            data = compressed
        if len(data) != item["size"] or zlib.crc32(data) != item["crc32"]:
            raise ValueError(f"CRC 불일치: {item['name']}")
        yield item["name"], data


def extract_target_counts(output_csv: Path) -> dict:
    """204820 기록을 (날짜, 시각, 차종, 분, 진입, 진출, 대수) 표로 저장한다."""
    movements = target_movements(find_network())
    incoming = {a for a, _ in movements}
    rows = []
    alias_count = 0
    unknown = defaultdict(int)
    for name, data in iter_turn_files():
        stem = Path(name).stem  # 예: 90001220220810050000p
        date, hhmm, code = stem[6:14], stem[14:18], stem[-1]
        for interval in ET.fromstring(data).iter("interval"):
            begin = float(interval.get("begin"))
            for relation in interval.iter("edgeRelation"):
                origin, destination = relation.get("from"), relation.get("to")
                if origin not in incoming:
                    continue
                count = int(float(relation.get("count")))
                if destination in ID_ALIAS:
                    destination = ID_ALIAS[destination]
                    alias_count += count
                if (origin, destination) not in movements:
                    unknown[(origin, destination)] += count
                    continue
                rows.append({
                    "date": date, "period": hhmm, "vehicle_type": VEHICLE_CODES[code],
                    "begin_unix": int(begin), "from_edge": origin, "to_edge": destination,
                    "count": count,
                })
    rows.sort(key=lambda r: (r["date"], r["period"], r["begin_unix"], r["vehicle_type"], r["from_edge"], r["to_edge"]))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"rows": len(rows), "total_count": sum(r["count"] for r in rows),
            "alias_mapped_count": alias_count,
            "unmapped_pairs": {f"{a}->{b}": v for (a, b), v in unknown.items()},
            "dates": sorted({r["date"] for r in rows})}


def load_counts(csv_path: Path):
    """저장한 표를 {(날짜, 시각): [행...]}으로 읽는다."""
    scenarios = defaultdict(list)
    with csv_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["count"] = int(row["count"])
            row["begin_unix"] = int(row["begin_unix"])
            scenarios[(row["date"], row["period"])].append(row)
    return scenarios


# 길주로 서쪽 접근로(477707013_02)는 두 갈래에서 들어온다. 4차로 가지(477767666_02_01)로 오는 차량은
# 상류 송내 신호(cluster_204796_205202)를 지나 477748652_01로 들어가는 회전 기록으로 셀 수 있다.
# 19일 모두 이 접근로 진입량의 약 46%(15분 시간대 10~90% 범위 34~54%)이고, 나머지는
# 2차로 가지(477744390_01, 왼쪽 4·5차로로 합류)로 들어온다. docs/가상도로_정상화_20261004.md 참고.
BRANCH_INFLOW_EDGE = "477748652_01"
BRANCH_INFLOW = DERIVED / "branch_inflow_477748652_01.csv"


def extract_branch_inflow(output_csv: Path = BRANCH_INFLOW) -> dict:
    """477748652_01로 들어가는 모든 회전 기록을 (날짜, 시각)별로 더해 저장한다."""
    totals = defaultdict(int)
    for name, data in iter_turn_files():
        stem = Path(name).stem
        date, hhmm = stem[6:14], stem[14:18]
        for relation in ET.fromstring(data).iter("edgeRelation"):
            if relation.get("to") == BRANCH_INFLOW_EDGE:
                totals[(date, hhmm)] += int(float(relation.get("count")))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "period", "count"])
        for (date, period), count in sorted(totals.items()):
            writer.writerow([date, period, count])
    return {"rows": len(totals), "total": sum(totals.values())}


def load_branch_inflow(csv_path: Path = BRANCH_INFLOW) -> dict[tuple[str, str], int]:
    with csv_path.open(encoding="utf-8") as handle:
        return {(r["date"], r["period"]): int(r["count"]) for r in csv.DictReader(handle)}


def signal_xml_files(date: str = "20220810") -> dict[str, Path]:
    """해당 날짜의 신호 XML을 {시각(HHMM): 경로}로 돌려준다. 그 날짜 자료가 없으면 빈 사전."""
    folder = SIGNALS / date
    return {path.name[14:18]: path for path in sorted(folder.glob("*s.xml"))} if folder.is_dir() else {}


def read_signal_phases(path: Path, tls: str = TARGET_TLS) -> list[tuple[float, str]]:
    for logic in ET.parse(path).getroot().findall("tlLogic"):
        if logic.get("id") == tls:
            return [(float(p.get("duration")), p.get("state")) for p in logic.findall("phase")]
    raise KeyError(f"{path.name}에 {tls} 신호가 없습니다.")


if __name__ == "__main__":
    # 원본(data/raw)에서 data/derived의 두 표를 다시 만든다.
    summary = extract_target_counts(COUNTS)
    (COUNTS.parent / "counts_204820_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "dates"}, ensure_ascii=False, indent=2))
    print("dates:", len(summary["dates"]), summary["dates"][0], "~", summary["dates"][-1])
    print("branch inflow:", extract_branch_inflow())
