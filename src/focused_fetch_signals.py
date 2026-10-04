"""Drive의 AI Hub 분할 ZIP에서 교차로_all 신호 XML(19일)을 한 번의 범위 읽기로 가져온다.

읽기 전용이다. 각 파일은 로컬 헤더·크기·CRC32를 검증한 뒤 저장한다.
"""
from __future__ import annotations

import json
import shutil
import struct
import sys
import zlib
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from read_aihub_split_zip import DriveVolumes  # noqa: E402

CACHE = PROJECT / "results/network_review_20261002"
OUT = PROJECT / "results/focused_204820/signals_all"


def main() -> None:
    items = [i for i in json.loads((CACHE / "bucheon_entries.json").read_text())
             if "/신호데이터/교차로_all/xml/" in i["name"] and i["name"].endswith(".xml")]
    disk = {i["disk"] for i in items}
    assert len(disk) == 1
    lo = min(i["offset"] for i in items)
    hi = max(i["offset"] + 30 + len(bytes.fromhex(i["raw_name_hex"])) + 512 + i["compressed_size"] for i in items)
    block = DriveVolumes(CACHE).read(disk.pop(), lo, hi - lo)
    saved = 0
    for item in items:
        pos = item["offset"] - lo
        head = struct.unpack_from("<4s5H3I2H", block, pos)
        if head[0] != b"PK\x03\x04":
            raise ValueError(f"로컬 헤더 오류: {item['name']}")
        start = pos + 30 + head[-2] + head[-1]
        data = block[start:start + item["compressed_size"]]
        if item["method"] == 8:
            data = zlib.decompressobj(-15).decompress(data, item["size"] + 1)
        if len(data) != item["size"] or zlib.crc32(data) != item["crc32"]:
            raise ValueError(f"CRC 불일치: {item['name']}")
        date = item["name"].split("/")[-2]
        target = OUT / date / item["name"].split("/")[-1]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        saved += 1
    print(json.dumps({"files": saved, "bytes_read": hi - lo, "dates": len({p.name for p in OUT.iterdir()})}))


if __name__ == "__main__":
    main()
