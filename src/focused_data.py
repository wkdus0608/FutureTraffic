"""석천사거리(204820) 중심 실험에 쓰는 원본 자료를 읽는다.

- 회전교통량: GPT가 Drive에서 읽어 둔 `교차로_all` 압축 구간(19일·7,296개 XML)을
  CRC32로 검증하며 풀고, 204820 진입 도로에서 시작하는 기록만 모은다.
- 신호: Sample에 있는 2022-08-10 신호 XML(15분 단위 96개)을 읽는다.

원본 파일은 읽기만 하며 바꾸지 않는다.
"""

from __future__ import annotations

import csv
import json
import struct
import unicodedata
import xml.etree.ElementTree as ET
import zlib
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
TARGET_TLS = "204820"
# 원본 회전량의 진출 도로 477706911_02는 지도에 없다. 204820에서 그 방향으로 나가는
# 도로는 477706911_02_02 하나뿐이므로 그 도로로 대응시킨다(조건부 대응, 기록 유지).
ID_ALIAS = {"477706911_02": "477706911_02_02"}
VEHICLE_CODES = {"p": "passenger", "b": "bus", "t": "truck", "m": "motorcycle"}

BLOCK = PROJECT / "results/map_demand_plan_20261002/all_turn_xml_compressed_block.bin"
BLOCK_META = PROJECT / "results/map_demand_plan_20261002/turn_block_metadata.json"
ENTRIES = PROJECT / "results/network_review_20261002/bucheon_entries.json"


def find_network() -> Path:
    matches = list((PROJECT / "Sample").rglob("1200012n.net.xml"))
    if len(matches) != 1:
        raise FileNotFoundError("Sample에서 1200012n.net.xml을 하나만 찾아야 합니다.")
    return matches[0]


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
# 2차로 가지(477744390_01, 왼쪽 4·5차로로 합류)로 들어온다. results/env_audit_20261004 참고.
BRANCH_INFLOW_EDGE = "477748652_01"
BRANCH_INFLOW = PROJECT / "results/focused_204820/branch_inflow_477748652_01.csv"


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


SIGNALS_ALL = PROJECT / "results/focused_204820/signals_all"


def signal_xml_files(date: str = "20220810") -> dict[str, Path]:
    """해당 날짜의 교차로_all 신호 XML을 {시각: 경로}로 찾는다.

    Drive에서 받아 둔 19일치(signals_all)를 먼저 보고, 없으면 Sample(2022-08-10)을 본다.
    """
    folder = SIGNALS_ALL / date
    if folder.is_dir():
        return {path.name[14:18]: path for path in sorted(folder.glob("*s.xml"))}
    found = {}
    for path in (PROJECT / "Sample").rglob("*s.xml"):
        if "신호" in unicodedata.normalize("NFC", str(path)) and "/xml/" in str(path) and date in path.name:
            found[path.name[14:18]] = path
    return found


def read_signal_phases(path: Path, tls: str = TARGET_TLS) -> list[tuple[float, str]]:
    for logic in ET.parse(path).getroot().findall("tlLogic"):
        if logic.get("id") == tls:
            return [(float(p.get("duration")), p.get("state")) for p in logic.findall("phase")]
    raise KeyError(f"{path.name}에 {tls} 신호가 없습니다.")


if __name__ == "__main__":
    out = PROJECT / "results/focused_204820/counts_204820.csv"
    summary = extract_target_counts(out)
    (out.parent / "counts_204820_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "dates"}, ensure_ascii=False, indent=2))
    print("dates:", len(summary["dates"]), summary["dates"][0], "~", summary["dates"][-1])
