#!/usr/bin/env python3

"""Measure VDS-only LightGBM inference time for the realtime feasibility metric."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from its_common import PROJECT_DIR, load_config
from join_vds_cctv import find_vds_5min


def make_model():
    try:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=16,
            verbosity=-1,
        ), "vds_only_lightgbm"
    except OSError:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.05,
            max_iter=100,
        ), "vds_only_hgb_sklearn"


def make_supervised(speed: np.ndarray, lookback: int, horizon_bins: int):
    xs = []
    ys = []
    for i in range(lookback, len(speed) - horizon_bins):
        xs.append(speed[i - lookback : i])
        ys.append(speed[i + horizon_bins - 1])
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="입력 특징 → 예측까지 처리 시간을 측정합니다."
    )
    parser.add_argument("--vds-5min", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--lookback", type=int, default=12, help="입력 칸 수. 기본 12=60분")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seoul_id = config["site"]["vds_id_seoul"]
    horizon_minutes = int(config["eval"]["horizon_minutes"])
    repeats = int(config["eval"]["latency_repeats"])
    bin_minutes = int(config["collection"]["bin_seconds"]) // 60
    horizon_bins = max(1, horizon_minutes // max(1, bin_minutes))

    vds = pd.read_csv(find_vds_5min(args.vds_5min))
    seoul = vds[vds["VDS_ID"] == seoul_id].copy()
    seoul["구간시작시각"] = pd.to_datetime(seoul["구간시작시각"])
    seoul = seoul.sort_values("구간시작시각")
    speed = seoul["평균속도_5분_km_h"].to_numpy(dtype=float)
    speed = np.where(np.isfinite(speed), speed, np.nan)
    # 품질이 나쁜 칸은 앞 값으로 채운 뒤 학습. 평가용 임시 처리.
    series = pd.Series(speed).ffill().bfill().to_numpy()

    x, y = make_supervised(series, args.lookback, horizon_bins)
    if len(x) < 50:
        raise SystemExit("학습에 쓸 5분 칸이 너무 적습니다.")

    split = int(len(x) * 0.8)
    model, model_name = make_model()
    fit_t0 = time.perf_counter()
    model.fit(x[:split], y[:split])
    fit_ms = (time.perf_counter() - fit_t0) * 1000

    sample = x[split : split + 1]
    # warmup
    model.predict(sample)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.predict(sample)
        times.append((time.perf_counter() - t0) * 1000)

    pred = model.predict(x[split:])
    mae = float(np.mean(np.abs(pred - y[split:])))
    rmse = float(np.sqrt(np.mean((pred - y[split:]) ** 2)))

    row = {
        "model": model_name,
        "horizon_minutes": horizon_minutes,
        "lookback_bins": args.lookback,
        "train_rows": split,
        "test_rows": len(x) - split,
        "fit_ms": round(fit_ms, 3),
        "infer_ms_mean": round(float(np.mean(times)), 4),
        "infer_ms_p50": round(float(np.median(times)), 4),
        "infer_ms_p95": round(float(np.percentile(times, 95)), 4),
        "test_mae": round(mae, 4),
        "test_rmse": round(rmse, 4),
        "realtime_note": "특징이 이미 있으면 예측은 이 지연시간만 든다. HLS 끊김과 무관.",
    }
    out = Path(args.out) if args.out else PROJECT_DIR / "latency_vds_only.csv"
    pd.DataFrame([row]).to_csv(out, index=False, encoding="utf-8-sig")
    print("저장:", out)
    for key, value in row.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
