#!/usr/bin/env python3

"""Fetch ITS vdsInfo for Pangyo IDs and aggregate to 5-minute bins."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from its_common import get_api_key


KST = timezone(timedelta(hours=9))


def parse_collected(value: str) -> datetime:
    return datetime.strptime(str(value), "%Y%m%d%H%M%S").replace(tzinfo=KST)


def fetch_pangyo_vds(config: dict) -> list[dict]:
    spec = config["its_vds"]
    seoul_id = config["site"]["vds_id_seoul"]
    busan_id = config["site"]["vds_id_busan"]
    response = requests.get(
        spec["url"],
        params={"apiKey": get_api_key(), "getType": "json"},
        timeout=float(spec.get("timeout_seconds") or 90),
    )
    response.raise_for_status()
    payload = response.json()
    header = payload.get("header") or {}
    if str(header.get("resultCode")) not in {"0", "00", ""}:
        raise RuntimeError(header.get("resultMsg") or "vdsInfo 실패")
    items = ((payload.get("body") or {}).get("items")) or []
    want = {seoul_id, busan_id}
    rows = []
    fetched_at = datetime.now(KST)
    for item in items:
        vds_id = item.get("vdsId")
        if vds_id not in want:
            continue
        collected = parse_collected(item["colctedDate"])
        rows.append(
            {
                "fetched_at": fetched_at.isoformat(),
                "collected_at": collected.isoformat(),
                "VDS_ID": vds_id,
                "laneNo": int(item.get("laneNo") or 0),
                "speed": _to_float(item.get("speed")),
                "volume": _to_float(item.get("volume")),
                "occupancy": _to_float(item.get("occupancy")),
            }
        )
    return rows


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def append_vds_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    header = list(rows[0].keys())
    exists = path.exists() and path.stat().st_size > 0
    encoding = "utf-8" if exists else "utf-8-sig"
    with path.open("a", encoding=encoding, newline="") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def aggregate_vds_5min(live_csv: Path, bin_seconds: int = 300) -> pd.DataFrame:
    raw = pd.read_csv(live_csv)
    raw["collected_at"] = pd.to_datetime(raw["collected_at"])
    if raw["collected_at"].dt.tz is not None:
        raw["collected_at"] = raw["collected_at"].dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    raw["bin_start"] = raw["collected_at"].dt.floor(f"{int(bin_seconds)}s")
    rows = []
    for (bin_start, vds_id), part in raw.groupby(["bin_start", "VDS_ID"], sort=True):
        valid_speed = part[(part["speed"] > 0) & (part["volume"] >= 0)]
        volume = float(part["volume"].sum())
        if not valid_speed.empty and valid_speed["volume"].sum() > 0:
            speed = float(
                (valid_speed["speed"] * valid_speed["volume"]).sum()
                / valid_speed["volume"].sum()
            )
        elif not valid_speed.empty:
            speed = float(valid_speed["speed"].mean())
        else:
            speed = float("nan")
        rows.append(
            {
                "bin_start": bin_start,
                "VDS_ID": vds_id,
                "교통량_5분": volume,
                "평균속도_5분_km_h": speed,
                "점유율_5분_pct": float(part["occupancy"].mean()),
                "관측행수": int(len(part)),
            }
        )
    return pd.DataFrame(rows)
