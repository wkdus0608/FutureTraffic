"""새 12일(단일교차로 자료) 준비 — 지문 밖, 잠근 자료·코드를 고치지 않는다(진행 기록 89절).

AI Hub 71572 분할 압축(Drive `AIHub/71572/stream_parts`, 약 499GB)을 풀지 않고, 10-02에 저장한 압축 목차
(`bucheon_entries.json`)로 필요한 파일의 위치를 알아 그 구간만 읽는다(맥의 Google Drive 앱 폴더에서 조각 파일의 해당 위치만).
원본 CRC로 검사한다.
- 꺼내는 것(31일 모두, 겹치는 19일은 '교차로_all'과 같은지 확인용)
  - 회전교통량 XML: 교차로_26(204820), 교차로_106(송내 cluster, 서쪽 갈래 유입), 교차로_29(204822, 동쪽 무리 도착)
  - 신호 XML: 교차로_26, 교차로_29
- 저장: data/extra/raw/<묶음>.bin(원본 압축 바이트 그대로) + <묶음>.json(이름·위치·크기·CRC). 읽을 때 풀고 CRC를 다시 본다.

사용:
  .venv/bin/python src/extra_days.py extract        # 맥, 네트워크만(계산 거의 없음)
  .venv/bin/python src/extra_days.py verify         # 겹치는 19일에서 '교차로_all'과 같은지 + 31일 표 만들기(SUMO 없음)
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import re
import struct
import zlib
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
EXTRA = PROJECT / "data/extra"
RAW = EXTRA / "raw"
INDEX = PROJECT.parent / "자주프_보관_20261005/results/network_review_20261002/bucheon_entries.json"
# 맥 Google Drive 앱 폴더(계정 이름이 폴더 이름에 들어가서 찾아 쓴다). 다른 곳이면 AIHUB_STREAM_PARTS로 정한다.
MOUNT = (Path(os.environ["AIHUB_STREAM_PARTS"]) if os.environ.get("AIHUB_STREAM_PARTS") else
         next((d / "내 드라이브/AIHub/71572/stream_parts" for d in sorted(Path.home().glob("Library/CloudStorage/GoogleDrive-*"))
               if (d / "내 드라이브/AIHub/71572/stream_parts").exists()), Path("AIHUB_STREAM_PARTS를 정하세요")))
KEYS = [507745, 507746, 507747, 507748, 507749]
ALL_DATES_31 = None   # 목차에서 읽음
NEW_DATES = ["20220815", "20220912", "20220913", "20220914", "20220915", "20220916",
             "20220930", "20221001", "20221002", "20221026", "20221027", "20221031"]
GROUPS = {   # 묶음 이름 → 압축 안 경로 앞부분
    "turn_26": "신호최적화데이터/부천시/교통량데이터(회전교통량기반)/교차로_26/",
    "turn_106": "신호최적화데이터/부천시/교통량데이터(회전교통량기반)/교차로_106/",
    "turn_29": "신호최적화데이터/부천시/교통량데이터(회전교통량기반)/교차로_29/",
    "signal_26": "신호최적화데이터/부천시/신호데이터/교차로_26/xml/",
    "signal_29": "신호최적화데이터/부천시/신호데이터/교차로_29/xml/",
}
MAX_GAP = 1 << 20          # 이보다 가까운 파일은 한 구간으로 묶어 읽는다
MAX_SPAN = 256 << 20


class Volumes:
    """분할 압축 5권(디스크 0~4)의 조각 목록. 원격 manifest로 조각 위치를 안다."""

    def __init__(self, cache: Path):
        cache.mkdir(parents=True, exist_ok=True)
        self.volumes = []
        for key in KEYS:
            p = cache / f"manifest_{key}.json"
            if not p.exists():
                p.write_bytes((MOUNT / str(key) / "manifest.json").read_bytes())
            m = json.loads(p.read_text())
            if not m["complete"]:
                raise SystemExit(f"미완료 원격 파일: {key}")
            parts = sorted(m["parts"], key=lambda x: x["offset"])
            self.volumes.append((key, parts))

    def read(self, disk: int, offset: int, size: int) -> bytes:
        key, parts = self.volumes[disk]
        starts = [p["offset"] for p in parts]
        out = []
        while size:
            i = bisect.bisect_right(starts, offset) - 1
            part = parts[i]
            skip = offset - part["offset"]
            count = min(size, part["size"] - skip)
            with open(MOUNT / str(key) / part["name"], "rb") as f:
                f.seek(skip)
                data = f.read(count)
            if len(data) != count:
                raise EOFError(f"원격 범위가 잘림: {key}/{part['name']}")
            out.append(data)
            offset += count
            size -= count
        return b"".join(out)


def unpack(blob: bytes, pos: int, item: dict) -> bytes:
    head = struct.unpack_from("<4s5H3I2H", blob, pos)
    if head[0] != b"PK\x03\x04":
        raise ValueError(f"로컬 헤더 오류: {item['name']}")
    start = pos + 30 + head[-2] + head[-1]
    comp = blob[start:start + item["compressed_size"]]
    data = zlib.decompressobj(-15).decompress(comp, item["size"] + 1) if item["method"] == 8 else comp
    if len(data) != item["size"] or zlib.crc32(data) != item["crc32"]:
        raise ValueError(f"CRC 불일치: {item['name']}")
    return data


def extract(args) -> None:
    items = json.loads(INDEX.read_text())
    index_sha = hashlib.sha256(INDEX.read_bytes()).hexdigest()
    vols = Volumes(EXTRA / "cache")
    RAW.mkdir(parents=True, exist_ok=True)
    summary = {"index": str(INDEX), "index_sha256": index_sha, "groups": {}}
    for group, prefix in GROUPS.items():
        out_bin, out_json = RAW / f"{group}.bin", RAW / f"{group}.json"
        if out_bin.exists() and out_json.exists():
            print(group, "이미 있음 → 건너뜀", flush=True)
            summary["groups"][group] = json.loads(out_json.read_text())["summary"]
            continue
        sel = sorted((i for i in items if i["name"].startswith(prefix) and i["name"].endswith(".xml") and i["size"] > 0),
                     key=lambda i: (i["disk"], i["offset"]))
        spans, cur = [], None
        for it in sel:
            end = it["offset"] + 30 + len(it.get("raw_name_hex", "")) // 2 + 1024 + it["compressed_size"]
            if cur and it["disk"] == cur["disk"] and it["offset"] - cur["end"] <= MAX_GAP and end - cur["start"] <= MAX_SPAN:
                cur["end"] = max(cur["end"], end)
                cur["items"].append(it)
            else:
                cur = {"disk": it["disk"], "start": it["offset"], "end": end, "items": [it]}
                spans.append(cur)
        print(f"{group}: 파일 {len(sel)}개, 읽을 구간 {len(spans)}개, 합계 {sum(s['end'] - s['start'] for s in spans) / 1e6:.1f}MB", flush=True)
        records, chunks, pos = [], [], 0
        for k, sp in enumerate(spans, 1):
            blob = vols.read(sp["disk"], sp["start"], sp["end"] - sp["start"])
            for it in sp["items"]:
                local = it["offset"] - sp["start"]
                data = unpack(blob, local, it)          # 여기서 CRC 검사
                comp = zlib.compress(data, 9)
                records.append({"name": it["name"], "pos": pos, "len": len(comp), "size": it["size"], "crc32": it["crc32"]})
                chunks.append(comp)
                pos += len(comp)
            print(f"  구간 {k}/{len(spans)} 읽음", flush=True)
        out_bin.write_bytes(b"".join(chunks))
        dates = sorted({re.search(r"/(\d{8})/", r["name"]).group(1) for r in records})
        info = {"files": len(records), "dates": len(dates), "new_dates_present": [d for d in NEW_DATES if d in dates],
                "bin_sha256": hashlib.sha256(out_bin.read_bytes()).hexdigest()}
        out_json.write_text(json.dumps({"summary": info, "records": records}, ensure_ascii=False))
        summary["groups"][group] = info
        print(f"  저장 {out_bin.name}: 파일 {len(records)}개, {len(dates)}일", flush=True)
    (RAW / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print("끝:", RAW / "manifest.json")


def iter_group(group: str):
    """저장한 묶음에서 (압축 안 이름, 내용)을 CRC를 확인하며 꺼낸다."""
    meta = json.loads((RAW / f"{group}.json").read_text())
    blob = (RAW / f"{group}.bin").read_bytes()
    for r in meta["records"]:
        data = zlib.decompress(blob[r["pos"]:r["pos"] + r["len"]])
        if len(data) != r["size"] or zlib.crc32(data) != r["crc32"]:
            raise ValueError(f"CRC 불일치: {r['name']}")
        yield r["name"], data


DERIVED_X = EXTRA / "derived"
SIGNALS_X = RAW / "signals"


def _tables_from(group: str, builder, out: Path):
    """잠근 추출 함수를 그대로 쓰되, 읽는 원본만 이 묶음으로 바꿔 31일 표를 만든다(잠근 파일은 고치지 않음)."""
    import focused_data
    import focused_platoon
    original = focused_data.iter_turn_files
    focused_data.iter_turn_files = focused_platoon.iter_turn_files = lambda: iter_group(group)
    try:
        return builder(out)
    finally:
        focused_data.iter_turn_files = focused_platoon.iter_turn_files = original


def _rows(path: Path, dates: set[str]) -> list[tuple]:
    import csv
    with path.open(encoding="utf-8") as f:
        return sorted(tuple(r.values()) for r in csv.DictReader(f) if r["date"] in dates)


def verify(args) -> None:
    import sys
    sys.path.insert(0, str(PROJECT / "src"))
    import focused_data as fd
    import focused_platoon as fp
    DERIVED_X.mkdir(parents=True, exist_ok=True)
    old_dates = set(json.loads((PROJECT / "config/focused_split.json").read_text())["train"]
                    + json.loads((PROJECT / "config/focused_split.json").read_text())["validation"]
                    + json.loads((PROJECT / "config/focused_split.json").read_text())["test"])
    report = {"overlap_dates": sorted(old_dates)}
    checks = [("counts_204820", "turn_26", fd.extract_target_counts, fd.DERIVED / "counts_204820.csv"),
              ("branch_inflow_477748652_01", "turn_106", fd.extract_branch_inflow, fd.DERIVED / "branch_inflow_477748652_01.csv"),
              ("east_upstream_204822", "turn_29", fp.extract_upstream_counts, fd.DERIVED / "east_upstream_204822.csv")]
    for name, group, builder, official in checks:
        out = DERIVED_X / f"{name}_31days.csv"
        _tables_from(group, builder, out)
        a, b = _rows(official, old_dates), _rows(out, old_dates)
        same = a == b
        diff = None if same else {"official_rows": len(a), "single_rows": len(b),
                                  "only_official": len(set(a) - set(b)), "only_single": len(set(b) - set(a))}
        report[name] = {"same_on_19_days": same, "diff": diff}
        print(f"{name}: 겹치는 19일 {'같음' if same else '다름'} {diff or ''}", flush=True)
    # 신호: 19일 × 96칸, 204820(교차로_26)과 204822(교차로_29) 현시가 '교차로_all' 파일과 같은지
    for group, tls in (("signal_26", "204820"), ("signal_29", "204822")):
        single = {}
        for nm, data in iter_group(group):
            date = re.search(r"/(\d{8})/", nm).group(1)
            fname = nm.rsplit("/", 1)[1]
            dest = (SIGNALS_X if group == "signal_26" else RAW / "signals_29") / date / fname   # 날짜 폴더에는 26 파일만(시각이 겹치지 않게)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            single[(date, fname[14:18])] = dest
        mismatch, compared = [], 0
        for date in sorted(old_dates):
            for hhmm, path in fd.signal_xml_files(date).items():
                compared += 1
                if fd.read_signal_phases(path, tls) != fd.read_signal_phases(single[(date, hhmm)], tls):
                    mismatch.append(f"{date} {hhmm}")
        report[f"signal_{tls}"] = {"compared": compared, "mismatch": len(mismatch), "examples": mismatch[:5]}
        print(f"신호 {tls}: 겹치는 19일 {compared}칸 중 다름 {len(mismatch)}", flush=True)
    (EXTRA / "verify_overlap.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print("저장:", EXTRA / "verify_overlap.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("extract", "verify"))
    args = ap.parse_args()
    {"extract": extract, "verify": verify}[args.cmd](args)


if __name__ == "__main__":
    main()
