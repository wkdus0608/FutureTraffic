"""Drive의 TL 조각을 읽기만 하여 ZIP 목록과 선택 파일을 CRC 검증 후 추출한다.

PKWARE APPNOTE 6.3.10의 분할 ZIP/ZIP64 오프셋을 사용한다.
원격 자료 변경·전체 병합·전체 압축 해제를 하지 않는다.
"""
from __future__ import annotations

import argparse
import bisect
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import struct
import subprocess
import unicodedata
import zipfile
import zlib

BASE = "gdrive:AIHub/71572/stream_parts"
KEYS = [507745, 507746, 507747, 507748, 507749]
RCLONE = shutil.which("rclone") or "/opt/homebrew/bin/rclone"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class DriveVolumes:
    def __init__(self, cache: Path):
        self.cache = cache
        self.volumes = []
        for key in KEYS:
            p = cache / f"manifest_{key}.json"
            if not p.exists():
                raw = subprocess.check_output([RCLONE, "cat", f"{BASE}/{key}/manifest.json"], timeout=120)
                p.write_bytes(raw)
            manifest = json.loads(p.read_text())
            if not manifest["complete"]:
                raise ValueError(f"미완료 원격 파일: {key}")
            parts = sorted(manifest["parts"], key=lambda x: x["offset"])
            end = 0
            for part in parts:
                if part["offset"] != end:
                    raise ValueError("조각 사이 빈 구간 또는 중복")
                end += part["size"]
            if end != manifest["total_bytes"]:
                raise ValueError("manifest 크기 불일치")
            self.volumes.append((key, parts, end))
        self.starts = [0]
        for _, _, size in self.volumes:
            self.starts.append(self.starts[-1] + size)

    def read(self, disk: int, offset: int, size: int) -> bytes:
        if disk < 0 or disk >= len(self.volumes) or offset < 0 or size < 0:
            raise ValueError("잘못된 범위")
        absolute = self.starts[disk] + offset
        if absolute + size > self.starts[-1]:
            raise ValueError("압축파일 끝을 넘는 범위")
        blocks = []
        while size:
            d = bisect.bisect_right(self.starts, absolute) - 1
            key, parts, _ = self.volumes[d]
            local = absolute - self.starts[d]
            i = bisect.bisect_right([p["offset"] for p in parts], local) - 1
            part = parts[i]
            skip = local - part["offset"]
            count = min(size, part["size"] - skip)
            data = subprocess.check_output([
                RCLONE, "cat", f"{BASE}/{key}/{part['name']}",
                "--offset", str(skip), "--count", str(count),
                "--timeout", "90s", "--retries", "3"], timeout=240)
            if len(data) != count:
                raise EOFError("원격 범위가 잘림")
            blocks.append(data)
            absolute += count
            size -= count
        return b"".join(blocks)


def end_record(volumes):
    size = volumes.volumes[-1][2]
    tail = volumes.read(4, max(0, size - 131072), min(size, 131072))
    pos = tail.rfind(b"PK\x05\x06")
    if pos < 0:
        raise ValueError("ZIP 끝 레코드 없음")
    _, disk, cd_disk, _, entries, cd_size, cd_offset, comment = struct.unpack_from("<4s4H2IH", tail, pos)
    if pos + 22 + comment != len(tail):
        raise ValueError("ZIP 끝 이후 예상하지 않은 바이트")
    if entries == 65535 or cd_offset == 0xffffffff or cd_size == 0xffffffff:
        locator = struct.unpack_from("<4sIQI", tail, pos - 20)
        if locator[0] != b"PK\x06\x07" or locator[3] != len(volumes.volumes):
            raise ValueError("ZIP64 분할 수 불일치")
        record = volumes.read(locator[1], locator[2], 56)
        fields = struct.unpack("<4sQ2H2I4Q", record)
        if fields[0] != b"PK\x06\x06" or fields[1] != 44:
            raise ValueError("지원하지 않는 ZIP64 끝 레코드")
        _, _, _, _, disk, cd_disk, _, entries, cd_size, cd_offset = fields
    if disk != 4 or cd_disk != 4 or cd_size > 300 * 2**20:
        raise ValueError("지원 범위 밖의 중앙 목록 구조")
    return {"disk": cd_disk, "offset": cd_offset, "size": cd_size, "entries": entries}


def parse_directory(path: Path):
    with path.open("rb") as handle:
        while head := handle.read(46):
            if len(head) != 46:
                raise EOFError("잘린 중앙 목록")
            f = struct.unpack(zipfile.structCentralDir, head)
            if f[0] != b"PK\x01\x02":
                raise ValueError("중앙 목록 시그니처 불일치")
            raw = handle.read(f[12]); extra = handle.read(f[13]); comment = handle.read(f[14])
            if len(raw) != f[12] or len(extra) != f[13] or len(comment) != f[14]:
                raise EOFError("잘린 파일명/확장 필드")
            name = raw.decode("utf-8" if f[5] & 0x800 else "cp949")
            info = zipfile.ZipInfo(name)
            info.flag_bits, info.compress_type, info.CRC = f[5], f[6], f[9]
            info.compress_size, info.file_size, info.volume = f[10], f[11], f[15]
            info.header_offset, info.extra = f[18], extra
            info._decodeExtra(zlib.crc32(raw))
            yield {"name": unicodedata.normalize("NFC", info.filename), "raw_name_hex": raw.hex(),
                   "disk": info.volume, "offset": info.header_offset,
                   "compressed_size": info.compress_size, "size": info.file_size,
                   "method": info.compress_type, "flags": info.flag_bits, "crc32": info.CRC}


def index(cache):
    volumes = DriveVolumes(cache)
    meta = end_record(volumes)
    p = cache / "central_directory.bin"
    if not p.exists():
        p.write_bytes(volumes.read(meta["disk"], meta["offset"], meta["size"]))
    if p.stat().st_size != meta["size"]:
        raise ValueError("목록 크기 불일치")
    counts = Counter(); selected = []; total = 0
    for item in parse_directory(p):
        total += 1
        name = item["name"]
        counts["/".join(name.split("/")[:3])] += 1
        if "신호최적화데이터" in name and "부천" in name:
            selected.append(item)
    if total != meta["entries"]:
        raise ValueError("전체 항목 수 불일치")
    write_json(cache / "archive_index_summary.json", {"zip": meta, "count": total,
               "central_sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "groups": dict(counts),
               "bucheon_items": len(selected), "source_spec": "https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT"})
    write_json(cache / "bucheon_entries.json", selected)
    print(json.dumps({"archive_entries": total, "bucheon_entries": len(selected), "groups": dict(counts)}, ensure_ascii=False))


def extract(cache, names):
    volumes = DriveVolumes(cache)
    items = json.loads((cache / "bucheon_entries.json").read_text())
    chosen = [i for i in items if i["name"] in names]
    if len(chosen) != len(names):
        raise ValueError("목록에 없거나 중복된 선택 파일")
    if sum(i["size"] for i in chosen) > 500 * 2**20:
        raise ValueError("선택 압축 해제 상한 500MiB 초과")
    receipts = []
    for item in chosen:
        name = PurePosixPath(item["name"])
        if name.is_absolute() or ".." in name.parts or "\\" in item["name"]:
            raise ValueError("위험한 압축 경로")
        dest = cache / "original_selected" / str(name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            data = dest.read_bytes()
        else:
            if item["flags"] & 1 or item["method"] not in (0, 8):
                raise ValueError("암호화 또는 지원하지 않는 압축 방법")
            head = volumes.read(item["disk"], item["offset"], 30)
            f = struct.unpack("<4s5H3I2H", head)
            if f[0] != b"PK\x03\x04" or f[3] != item["method"] or f[2] != item["flags"]:
                raise ValueError("로컬 헤더와 중앙 목록 불일치")
            rest = volumes.read(item["disk"], item["offset"] + 30, f[-2] + f[-1] + item["compressed_size"])
            if rest[:f[-2]].hex() != item["raw_name_hex"]:
                raise ValueError("로컬 헤더 파일명 불일치")
            compressed = rest[f[-2] + f[-1]:]
            if item["method"] == 8:
                decoder = zlib.decompressobj(-15)
                data = decoder.decompress(compressed, item["size"] + 1)
                if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                    raise ValueError("잘린 또는 크기를 넘는 deflate")
            else:
                data = compressed
            if len(data) != item["size"] or zlib.crc32(data) != item["crc32"]:
                raise ValueError("압축 해제 크기/CRC32 불일치")
            dest.write_bytes(data)
        if len(data) != item["size"] or zlib.crc32(data) != item["crc32"]:
            raise ValueError("기존 추출 파일 검증 실패")
        receipts.append({**item, "path": str(dest.resolve()), "sha256": hashlib.sha256(data).hexdigest(), "crc_verified": True})
        print(f"검증 추출: {name.name} ({len(data):,} bytes)", flush=True)
    write_json(cache / "extracted_files.json", receipts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["index", "extract"])
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--selection", type=Path, help="정확한 압축 내부 경로의 JSON 배열")
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    if args.mode == "index":
        index(args.cache.resolve())
    else:
        extract(args.cache.resolve(), json.loads(args.selection.read_text()))
