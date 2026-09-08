#!/usr/bin/env python3

"""Run YOLO on already recorded Pangyo-1 chunks using manifest wall-clock times."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from collect_features import Detector, append_csv, write_bin
from its_common import RECORDINGS_DIR, load_config
from video_features import (
    CentroidTracker,
    add_frame_to_bin,
    count_line_crossings,
    counting_line_pixels,
    draw_roi_overlay,
    empty_bin_stats,
    floor_bin,
    group_detections,
    keep_wall_clock_sample,
    match_stopped,
    mean_brightness,
    pixel_polygon,
    prepare_detection_frame,
)


def iter_recorded_parts(recordings_dir: Path):
    for manifest in sorted(recordings_dir.glob("pangyo1_chunked_*/manifest.csv")):
        table = pd.read_csv(manifest)
        for _, row in table.iterrows():
            video = Path(str(row.get("영상파일") or ""))
            if not video.exists() or float(row.get("실제영상길이_초") or 0) < 1:
                continue
            start_ts = pd.to_datetime(row["시작시각"])
            if getattr(start_ts, "tzinfo", None) is None:
                start_ts = start_ts.tz_localize("Asia/Seoul")
            yield video, start_ts


def sample_video(path: Path, sample_fps: float):
    cap = cv2.VideoCapture(str(path))
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(native / max(0.1, sample_fps))))
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % step == 0:
            yield frame
        index += 1
    cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="기존 녹화에서 서울 ROI YOLO 특징을 뽑습니다.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    collection = config["collection"]
    width = int(collection["frame_width"])
    height = int(collection["frame_height"])
    seoul_poly = pixel_polygon(config["roi"]["seoul"], width, height)
    busan_poly = pixel_polygon(config["roi"]["busan"], width, height)
    count_line = counting_line_pixels(
        config["roi"]["counting_line_seoul"], width, height
    )
    vehicle_classes = set(int(x) for x in config["detection"]["vehicle_classes"])
    sample_fps = float(collection["sample_fps"])
    stopped_speed = float(config["detection"]["stopped_pixel_speed"]) / max(0.1, sample_fps)

    out_dir = Path(args.out_dir) if args.out_dir else RECORDINGS_DIR / "from_recordings"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_file = out_dir / "frames.csv"
    bins_file = out_dir / "features_5min.csv"
    snapshot_dir = out_dir / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    (out_dir / "config.used.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    detector = Detector(config)
    print("detector:", detector.name)
    bins: dict[datetime, dict] = {}
    last_snapshot_bin: datetime | None = None
    last_sample_bucket: int | None = None
    frame_header = [
        "시각", "bin_start", "chunk", "seoul_count", "busan_count",
        "seoul_stopped", "seoul_crossings", "seoul_car", "seoul_motorcycle",
        "seoul_bus", "seoul_truck", "brightness", "오류분류",
    ]

    for video, start_ts in iter_recorded_parts(RECORDINGS_DIR):
        print("처리:", video.name, start_ts)
        tracker = CentroidTracker()
        prev_tracks: dict[int, tuple[float, float]] = {}
        prev_centers: list[tuple[float, float]] = []
        for i, frame in enumerate(sample_video(video, sample_fps)):
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            ts = start_ts + timedelta(seconds=i / sample_fps)
            keep, last_sample_bucket = keep_wall_clock_sample(
                ts.to_pydatetime(), last_sample_bucket, sample_fps
            )
            if not keep:
                continue
            detect_frame = prepare_detection_frame(
                frame, config["roi"]["osd_masks"], [seoul_poly, busan_poly]
            )
            boxes, classes = detector.detect(detect_frame)
            grouped = group_detections(
                boxes, classes, seoul_poly, busan_poly, vehicle_classes
            )
            stopped = match_stopped(prev_centers, grouped["seoul_centers"], stopped_speed)
            prev_centers = grouped["seoul_centers"]
            tracks = tracker.update(grouped["seoul_centers"])
            crossings = count_line_crossings(
                prev_tracks, tracks, count_line, toward_camera=True
            )
            prev_tracks = tracks
            bin_start = floor_bin(ts.to_pydatetime(), int(collection["bin_seconds"]))
            if bin_start not in bins:
                bins[bin_start] = empty_bin_stats()
            frame_row = {
                "시각": ts.isoformat(),
                "bin_start": bin_start.isoformat(),
                "chunk": video.name,
                "seoul_count": grouped["seoul_count"],
                "busan_count": grouped["busan_count"],
                "seoul_stopped": stopped,
                "seoul_crossings": crossings,
                "seoul_car": grouped["seoul_car"],
                "seoul_motorcycle": grouped["seoul_motorcycle"],
                "seoul_bus": grouped["seoul_bus"],
                "seoul_truck": grouped["seoul_truck"],
                "brightness": round(mean_brightness(frame, seoul_poly), 2),
                "오류분류": "",
            }
            add_frame_to_bin(bins[bin_start], frame_row)
            append_csv(frames_file, frame_row, frame_header)
            if last_snapshot_bin != bin_start:
                preview = draw_roi_overlay(
                    frame,
                    seoul_poly,
                    busan_poly,
                    config["roi"]["osd_masks"],
                    grouped["seoul_count"],
                    grouped["busan_count"],
                    count_line,
                    grouped.get("seoul_boxes"),
                    grouped.get("busan_boxes"),
                    crossings,
                )
                snap_path = snapshot_dir / f"{bin_start.strftime('%Y%m%d_%H%M%S')}.jpg"
                ok, encoded = cv2.imencode(".jpg", preview)
                if ok:
                    snap_path.write_bytes(encoded.tobytes())
                last_snapshot_bin = bin_start

    for bin_start, stats in sorted(bins.items()):
        row = write_bin(bins_file, bin_start, stats, config, seoul_poly, detector.name)
        print(
            f"칸 {row['bin_start']} coverage={row['coverage']} "
            f"seoul_density={row['seoul_count_mean']} crossings={row['seoul_crossings']}"
        )
    print("저장:", bins_file)


if __name__ == "__main__":
    main()
