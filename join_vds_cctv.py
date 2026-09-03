#!/usr/bin/env python3

"""Join CCTV 5-minute features with live ITS VDS or daily ex.co.kr VDS."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from its_common import PROJECT_DIR, load_config
from vds_live import aggregate_vds_5min


def find_vds_5min(cli_path: str | None) -> Path:
    if cli_path:
        path = Path(cli_path)
        if not path.exists():
            raise SystemExit(f"VDS 5분 파일이 없습니다: {path}")
        return path
    candidates = [
        PROJECT_DIR / "data" / "vds_pangyo1_20260820_5min.csv",
        Path("/Users/jeonseongmin/Downloads/자주프/vds_pangyo1_20260820_5min.csv"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise SystemExit("VDS 5분 CSV를 찾지 못했습니다. --vds-5min 또는 --vds-live 를 지정하세요.")


def to_naive_kst(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series)
    if ts.dt.tz is not None:
        return ts.dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    return ts


def pearson(left: pd.Series, right: pd.Series) -> float:
    aligned = pd.concat([left, right], axis=1).dropna()
    if len(aligned) < 3:
        return float("nan")
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))


def vds_tables_from_5min(vds: pd.DataFrame, seoul_id: str, busan_id: str):
    vds = vds.copy()
    vds["구간시작시각"] = to_naive_kst(vds["구간시작시각"])
    seoul = vds[vds["VDS_ID"] == seoul_id][
        ["구간시작시각", "교통량_5분", "평균속도_5분_km_h", "점유율_5분_pct"]
    ].rename(
        columns={
            "구간시작시각": "bin_start",
            "교통량_5분": "vds_seoul_volume",
            "평균속도_5분_km_h": "vds_seoul_speed",
            "점유율_5분_pct": "vds_seoul_occupancy",
        }
    )
    busan = vds[vds["VDS_ID"] == busan_id][
        ["구간시작시각", "교통량_5분", "평균속도_5분_km_h"]
    ].rename(
        columns={
            "구간시작시각": "bin_start",
            "교통량_5분": "vds_busan_volume",
            "평균속도_5분_km_h": "vds_busan_speed",
        }
    )
    return seoul, busan


def main() -> None:
    parser = argparse.ArgumentParser(description="CCTV 5분 특징과 VDS를 시각으로 조인합니다.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--vds-5min", default=None)
    parser.add_argument("--vds-live", default=None, help="collect_features가 쓴 its_vds_live.csv")
    parser.add_argument("--config", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seoul_id = config["site"]["vds_id_seoul"]
    busan_id = config["site"]["vds_id_busan"]
    coverage_min = float(config["collection"]["coverage_min"])
    features = pd.read_csv(args.features)
    features["bin_start"] = to_naive_kst(features["bin_start"])
    diffs = features["bin_start"].sort_values().diff().dt.total_seconds().dropna()
    if len(diffs):
        bin_seconds = int(round(float(diffs.median())))
        if bin_seconds <= 0:
            bin_seconds = int(config["collection"]["bin_seconds"])
    else:
        bin_seconds = int(config["collection"]["bin_seconds"])

    if args.vds_live:
        live_5min = aggregate_vds_5min(Path(args.vds_live), bin_seconds)
        live_5min = live_5min.rename(columns={"bin_start": "구간시작시각"})
        seoul, busan = vds_tables_from_5min(live_5min, seoul_id, busan_id)
    else:
        seoul, busan = vds_tables_from_5min(
            pd.read_csv(find_vds_5min(args.vds_5min)), seoul_id, busan_id
        )

    joined = features.merge(seoul, on="bin_start", how="left").merge(
        busan, on="bin_start", how="left"
    )
    if "cctv_missing" not in joined.columns:
        joined["cctv_missing"] = (joined["coverage"] < coverage_min).astype(int)
    joined["fair_eval"] = (
        (joined["cctv_missing"] == 0) & joined["vds_seoul_speed"].notna()
    ).astype(int)

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.features).resolve().parent
    out_join = out_dir / "joined_vds_cctv.csv"
    joined.to_csv(out_join, index=False, encoding="utf-8-sig")

    valid = joined[joined["fair_eval"] == 1]
    density = valid["seoul_density_mean"] if "seoul_density_mean" in valid.columns else valid["seoul_count_mean"]
    report = {
        "칸수_전체": len(joined),
        "칸수_공정비교": len(valid),
        "coverage_min_used": coverage_min,
        "note": (
            "seoul_count_mean·seoul_density_mean은 화면 안 차량 밀도(점유 성격). "
            "VDS 교통량과 같은 양이 아니다. 통과 대수는 seoul_crossings."
        ),
        "corr_서울밀도_서울VDS점유율": pearson(density, valid["vds_seoul_occupancy"])
        if "vds_seoul_occupancy" in valid.columns
        else float("nan"),
        "corr_서울밀도_서울VDS속도": pearson(density, valid["vds_seoul_speed"]),
        "corr_서울통과_서울VDS교통량": pearson(valid["seoul_crossings"], valid["vds_seoul_volume"])
        if "seoul_crossings" in valid.columns
        else float("nan"),
        "corr_부산ROI대수_부산VDS교통량": pearson(
            valid["busan_count_mean"], valid["vds_busan_volume"]
        ),
        "corr_부산ROI대수_서울VDS교통량": pearson(
            valid["busan_count_mean"], valid["vds_seoul_volume"]
        ),
    }
    report_path = out_dir / "direction_correlation.csv"
    pd.DataFrame([report]).to_csv(report_path, index=False, encoding="utf-8-sig")
    print("조인:", out_join)
    print("방향 상관:", report_path)
    for key, value in report.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
