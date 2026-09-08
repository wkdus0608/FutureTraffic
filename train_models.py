#!/usr/bin/env python3

"""Train VDS-only, CCTV-only, and multimodal models.

Default path is the 15-minute dataset: current 15 min features → next 15 min
mean VDS speed. A 5-minute joined CSV still works via --joined.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from its_common import MODELS_DIR, dataset_15min_path, detector_tag, load_config


CCTV_COLS = [
    "seoul_density_mean",
    "seoul_stopped_ratio",
    "seoul_truck_mean",
    "seoul_bus_mean",
    "is_night",
    "seoul_crossings",
]
VDS_COLS = [
    "vds_seoul_volume",
    "vds_seoul_speed",
    "vds_seoul_occupancy",
]
WINDOW_SUFFIXES = ("_m0", "_m5", "_m10")


def make_model(n_rows: int = 32):
    if n_rows < 6:
        from sklearn.linear_model import Ridge

        return Ridge(alpha=1.0), "ridge"
    try:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=80,
            learning_rate=0.05,
            num_leaves=16,
            verbosity=-1,
        ), "lightgbm"
    except OSError:
        return HistGradientBoostingRegressor(
            max_depth=4,
            learning_rate=0.08,
            max_iter=80,
        ), "hgb_sklearn"


def add_lags(frame: pd.DataFrame, columns: list[str], lookback: int) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        for lag in range(1, lookback + 1):
            out[f"{col}_lag{lag}"] = out[col].shift(lag)
    return out


def expand_window_cols(columns, bases: list[str]) -> list[str]:
    names = list(columns)
    out = []
    for base in bases:
        if base in names:
            out.append(base)
        for suffix in WINDOW_SUFFIXES:
            name = f"{base}{suffix}"
            if name in names:
                out.append(name)
    return out


def is_15min_dataset(data: pd.DataFrame) -> bool:
    return "target_speed_next_15min" in data.columns


def supervised(frame: pd.DataFrame, feature_cols: list[str], horizon_bins: int):
    work = frame.dropna(subset=feature_cols + ["target"]).copy()
    x = work[feature_cols].to_numpy(dtype=float)
    y = work["target"].to_numpy(dtype=float)
    return work, x, y


def evaluate(name: str, x, y, repeats: int) -> dict:
    if len(x) < 8:
        model, model_name = make_model(len(x))
        t0 = time.perf_counter()
        model.fit(x, y)
        fit_ms = (time.perf_counter() - t0) * 1000
        pred = model.predict(x)
        sample = x[:1]
        model.predict(sample)
        times = []
        for _ in range(repeats):
            t1 = time.perf_counter()
            model.predict(sample)
            times.append((time.perf_counter() - t1) * 1000)
        return {
            "model": name,
            "estimator": model_name,
            "n_rows": len(x),
            "split": "in_sample_small_n",
            "mae": round(float(mean_absolute_error(y, pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y, pred))), 4),
            "fit_ms": round(fit_ms, 3),
            "infer_ms_p50": round(float(np.median(times)), 4),
        }

    n_splits = 3 if len(x) >= 12 else 2
    splitter = TimeSeriesSplit(n_splits=n_splits)
    maes, rmses = [], []
    last_model = None
    last_name = ""
    fit_ms = 0.0
    times = []
    for train_idx, test_idx in splitter.split(x):
        model, last_name = make_model(len(train_idx))
        t0 = time.perf_counter()
        model.fit(x[train_idx], y[train_idx])
        fit_ms = (time.perf_counter() - t0) * 1000
        pred = model.predict(x[test_idx])
        maes.append(mean_absolute_error(y[test_idx], pred))
        rmses.append(np.sqrt(mean_squared_error(y[test_idx], pred)))
        last_model = model
    sample = x[-1:]
    last_model.predict(sample)
    for _ in range(repeats):
        t1 = time.perf_counter()
        last_model.predict(sample)
        times.append((time.perf_counter() - t1) * 1000)
    return {
        "model": name,
        "estimator": last_name,
        "n_rows": len(x),
        "split": f"TimeSeriesSplit({n_splits})",
        "mae": round(float(np.mean(maes)), 4),
        "rmse": round(float(np.mean(rmses)), 4),
        "fit_ms": round(fit_ms, 3),
        "infer_ms_p50": round(float(np.median(times)), 4),
    }


def prepare_15min(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str], dict]:
    work = data.copy()
    if "window_start" in work.columns:
        work["window_start"] = pd.to_datetime(work["window_start"])
        work = work.sort_values("window_start").reset_index(drop=True)
    work["target"] = work["target_speed_next_15min"]
    cctv_bases = [c for c in CCTV_COLS if any(f"{c}{s}" in work.columns for s in WINDOW_SUFFIXES) or c in work.columns]
    if not any(c.startswith("seoul_density_mean") for c in expand_window_cols(work.columns, ["seoul_density_mean"])):
        if any(f"seoul_count_mean{s}" in work.columns for s in WINDOW_SUFFIXES):
            cctv_bases = ["seoul_count_mean"] + cctv_bases
    vds_features = expand_window_cols(work.columns, VDS_COLS)
    cctv_features = expand_window_cols(work.columns, cctv_bases)
    meta = {
        "task": "current_15min_to_next_15min",
        "horizon_minutes": 15,
        "lookback_bins": 3,
    }
    return work, vds_features, cctv_features, meta


def prepare_5min(data: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list[str], list[str], dict]:
    horizon_minutes = int(config["eval"]["horizon_minutes"])
    bin_minutes = max(1, int(config["collection"]["bin_seconds"]) // 60)
    horizon_bins = max(1, horizon_minutes // bin_minutes)
    lookback = int(config["eval"].get("lookback_bins") or 6)
    work = data.copy()
    work["bin_start"] = pd.to_datetime(work["bin_start"])
    work = work.sort_values("bin_start").reset_index(drop=True)
    cctv_cols = [c for c in CCTV_COLS if c in work.columns]
    if "seoul_density_mean" not in cctv_cols and "seoul_count_mean" in work.columns:
        cctv_cols = ["seoul_count_mean"] + cctv_cols
    if "fair_eval" in work.columns:
        work = work[work["fair_eval"] == 1].copy()
    n_bins = len(work)
    if n_bins < 15:
        print(f"칸이 {n_bins}개라 lookback=0, horizon=1칸으로 파이프라인 시험을 합니다.")
        lookback = 0
        horizon_bins = 1
        horizon_minutes = bin_minutes
    else:
        max_h = max(1, n_bins // 4)
        if horizon_bins > max_h:
            print(
                f"칸이 {n_bins}개라 horizon을 {horizon_minutes}분에서 "
                f"{max_h * bin_minutes}분으로 줄입니다."
            )
            horizon_bins = max_h
            horizon_minutes = max_h * bin_minutes
        lookback = min(lookback, max(1, n_bins // 3))
    work["target"] = work["vds_seoul_speed"].shift(-horizon_bins)
    work = add_lags(work, VDS_COLS + cctv_cols, lookback)
    vds_features = VDS_COLS + [f"{c}_lag{i}" for c in VDS_COLS for i in range(1, lookback + 1)]
    cctv_features = cctv_cols + [f"{c}_lag{i}" for c in cctv_cols for i in range(1, lookback + 1)]
    meta = {
        "task": "5min_bin_horizon",
        "horizon_minutes": horizon_minutes,
        "lookback_bins": lookback,
        "horizon_bins": horizon_bins,
    }
    return work, vds_features, cctv_features, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="세 모델 MAE/RMSE/처리시간 비교")
    parser.add_argument("--joined", default=None, help="5분 joined CSV")
    parser.add_argument("--dataset", default=None, help="pangyo1_15min_dataset.csv")
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    repeats = int(config["eval"].get("latency_repeats") or 50)
    csv_path = Path(args.dataset or args.joined or dataset_15min_path(config))
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        raise SystemExit(f"학습 CSV가 없습니다: {csv_path}")

    data = pd.read_csv(csv_path)
    use_15min = is_15min_dataset(data)
    if use_15min:
        data, vds_features, cctv_features, meta = prepare_15min(data)
        print("과제: 현재 15분 특징 → 다음 15분 평균속도")
    else:
        data, vds_features, cctv_features, meta = prepare_5min(data, config)
        print("과제: 5분 칸 + lag → horizon분 후 속도")
    both = vds_features + cctv_features
    horizon_bins = int(meta.get("horizon_bins") or 1)

    rows = []
    for name, cols in (
        ("vds_only", vds_features),
        ("cctv_only", cctv_features),
        ("multimodal", both),
    ):
        work, x, y = supervised(data, cols, horizon_bins)
        if len(work) < 2:
            rows.append(
                {
                    "model": name,
                    "estimator": "",
                    "n_rows": len(work),
                    "split": "too_few_rows",
                    "mae": "",
                    "rmse": "",
                    "fit_ms": "",
                    "infer_ms_p50": "",
                    "task": meta["task"],
                    "note": f"공정 비교 칸이 {len(work)}개라 학습하지 않음. 수집을 더 돌리세요.",
                }
            )
            continue
        result = evaluate(name, x, y, repeats)
        result["task"] = meta["task"]
        result["horizon_minutes"] = meta["horizon_minutes"]
        result["lookback_bins"] = meta["lookback_bins"]
        rows.append(result)

    if args.out:
        out = Path(args.out)
    elif use_15min:
        out = MODELS_DIR / detector_tag(config) / "model_compare.csv"
    else:
        out = csv_path.resolve().parent / "model_compare.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print("입력:", csv_path)
    print("저장:", out)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
