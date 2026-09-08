#!/usr/bin/env python3

"""Complete 5-minute bins, master CSV, and 15-minute supervised dataset."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd

from its_common import dataset_15min_path, load_config, master_csv_path


KST = "Asia/Seoul"

FEATURE_COLS = [
    "seoul_count_mean",
    "seoul_density_mean",
    "seoul_stopped_ratio",
    "seoul_truck_mean",
    "seoul_bus_mean",
    "seoul_crossings",
    "brightness_mean",
    "is_night",
    "busan_count_mean",
    "vds_seoul_volume",
    "vds_seoul_speed",
    "vds_seoul_occupancy",
    "vds_busan_volume",
    "vds_busan_speed",
]


def to_naive_kst(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series)
    if getattr(ts.dt, "tz", None) is not None:
        return ts.dt.tz_convert(KST).dt.tz_localize(None)
    return ts


def to_naive_ts(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        return ts.tz_convert(KST).tz_localize(None)
    return ts


def mark_complete_bins(
    joined: pd.DataFrame,
    session_start,
    session_end,
    bin_seconds: int,
) -> pd.DataFrame:
    out = joined.copy()
    out["bin_start"] = to_naive_kst(out["bin_start"])
    start = to_naive_ts(session_start)
    end = to_naive_ts(session_end)
    bin_end = out["bin_start"] + pd.to_timedelta(int(bin_seconds), unit="s")
    out["bin_complete"] = ((out["bin_start"] >= start) & (bin_end <= end)).astype(int)
    return out


def master_ready_rows(joined: pd.DataFrame) -> pd.DataFrame:
    work = joined.copy()
    if "bin_start" in work.columns:
        work["bin_start"] = to_naive_kst(work["bin_start"])
    ready = work[(work.get("fair_eval", 0) == 1) & (work.get("bin_complete", 0) == 1)]
    return ready


def append_master(joined: pd.DataFrame, session_id: str, master_path: Path | None = None) -> int:
    """Append fair, complete 5-minute rows. Dedupes bin_start, keeps latest session."""
    path = Path(master_path) if master_path else master_csv_path(load_config())
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming = master_ready_rows(joined)
    if incoming.empty:
        return 0
    incoming = incoming.copy()
    incoming["session_id"] = session_id
    if path.exists() and path.stat().st_size > 0:
        existing = pd.read_csv(path)
        existing["bin_start"] = to_naive_kst(existing["bin_start"])
        combined = pd.concat([existing, incoming], ignore_index=True)
    else:
        combined = incoming
    combined["bin_start"] = to_naive_kst(combined["bin_start"])
    combined = combined.drop_duplicates(subset=["bin_start"], keep="last")
    combined = combined.sort_values("bin_start").reset_index(drop=True)
    combined.to_csv(path, index=False, encoding="utf-8-sig")
    return len(incoming)


def build_15min_dataset(
    master_path: Path | None = None,
    out_path: Path | None = None,
    config_path: str | None = None,
) -> Path:
    """Current 15 min CCTV+VDS features → next 15 min mean VDS speed.

    Does not stitch across collection sessions. Drops windows with missing bins.
    """
    config = load_config(config_path)
    bin_seconds = int(config["collection"]["bin_seconds"])
    src = Path(master_path) if master_path else master_csv_path(config)
    dest = Path(out_path) if out_path else dataset_15min_path(config)
    if not src.exists() or src.stat().st_size == 0:
        raise SystemExit(f"master CSV가 없습니다: {src}")

    data = pd.read_csv(src)
    data["bin_start"] = to_naive_kst(data["bin_start"])
    data = data.sort_values(["session_id", "bin_start"]).reset_index(drop=True)
    cols = [c for c in FEATURE_COLS if c in data.columns]
    step = pd.Timedelta(seconds=bin_seconds)
    rows = []

    grouped = data.groupby("session_id", sort=False)
    for session_id, part in grouped:
        part = part.sort_values("bin_start").reset_index(drop=True)
        starts = part["bin_start"]
        index = {start: i for i, start in enumerate(starts)}
        for i in range(len(part)):
            needed = [part.loc[i, "bin_start"] + step * k for k in range(6)]
            if any(ts not in index for ts in needed):
                continue
            idxs = [index[ts] for ts in needed]
            feat = {"session_id": session_id, "window_start": needed[0]}
            for offset, row_i in enumerate(idxs[:3]):
                minute = offset * (bin_seconds // 60)
                src_row = part.loc[row_i]
                for col in cols:
                    feat[f"{col}_m{minute}"] = src_row[col]
            target_speeds = part.iloc[idxs[3:]]["vds_seoul_speed"].astype(float)
            if target_speeds.isna().any():
                continue
            feat["target_speed_next_15min"] = float(target_speeds.mean())
            rows.append(feat)

    dest.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["session_id", "window_start"]).reset_index(drop=True)
    out.to_csv(dest, index=False, encoding="utf-8-sig")
    return dest
