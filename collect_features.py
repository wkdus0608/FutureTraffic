#!/usr/bin/env python3

"""Collect Seoul-lane CCTV features and poll ITS vdsInfo in parallel.

HLS 404 is treated as an ITS/CDN session expiry. The loop fetches a new URL
and continues instead of trying to keep a dead playlist alive.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

from its_common import (
    RECORDINGS_DIR,
    classify_ffmpeg_log,
    fetch_cctv,
    load_config,
    redact_urls,
)
from vds_live import append_vds_rows, fetch_pangyo_vds
from video_features import (
    CentroidTracker,
    add_frame_to_bin,
    count_line_crossings,
    counting_line_pixels,
    draw_roi_overlay,
    empty_bin_stats,
    finalize_bin,
    floor_bin,
    group_detections,
    match_stopped,
    mean_brightness,
    pixel_polygon,
    prepare_detection_frame,
)


def now_local() -> datetime:
    return datetime.now().astimezone()


def build_ffmpeg_cmd(stream_url: str, config: dict) -> list[str]:
    collection = config["collection"]
    ffmpeg = config["ffmpeg"]
    width = int(collection["frame_width"])
    height = int(collection["frame_height"])
    fps = float(collection["sample_fps"])
    chunk = int(collection["chunk_seconds"])
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-loglevel", "info",
        "-rw_timeout", str(int(ffmpeg["rw_timeout_us"])),
        "-user_agent", ffmpeg["user_agent"],
        "-headers", f"Referer: {ffmpeg['referer']}\r\n",
    ]
    if not ffmpeg.get("http_persistent", True):
        cmd.extend(["-http_persistent", "0"])
    cmd.extend(
        [
            "-i", stream_url,
            "-t", str(chunk),
            "-an",
            "-vf", f"fps={fps},scale={width}:{height}",
            "-pix_fmt", "bgr24",
            "-f", "rawvideo",
            "pipe:1",
        ]
    )
    return cmd


class Detector:
    def __init__(self, config: dict):
        self.name = "none"
        self.model = None
        self.conf = float(config["detection"]["conf"])
        self.classes = set(int(x) for x in config["detection"]["vehicle_classes"])
        try:
            from ultralytics import YOLO

            self.model = YOLO(config["detection"]["model"])
            self.name = Path(config["detection"]["model"]).stem
        except Exception as error:
            print(f"YOLO를 쓰지 않습니다: {error}")
            print("스냅샷과 커버리지는 남기고, 차량 수는 0으로 기록합니다.")

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            return np.zeros((0, 4), dtype=float), np.zeros((0,), dtype=int)
        result = self.model.predict(
            frame,
            conf=self.conf,
            classes=sorted(self.classes),
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return np.zeros((0, 4), dtype=float), np.zeros((0,), dtype=int)
        xyxy = result.boxes.xyxy.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy().astype(int)
        return xyxy, cls


def append_csv(path: Path, row: dict, header: list[str]) -> None:
    exists = path.exists() and path.stat().st_size > 0
    encoding = "utf-8" if exists else "utf-8-sig"
    with path.open("a", encoding=encoding, newline="") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_bin(
    path: Path,
    bin_start: datetime,
    stats: dict,
    config: dict,
    seoul_polygon,
    detector_name: str,
) -> dict:
    row = finalize_bin(bin_start, stats, config, seoul_polygon, detector_name)
    append_csv(path, row, list(row.keys()))
    return row


def vds_worker(stop: threading.Event, config: dict, out_file: Path) -> None:
    spec = config.get("its_vds") or {}
    if not spec.get("enabled"):
        return
    interval = float(spec.get("poll_seconds") or 60)
    while True:
        try:
            rows = fetch_pangyo_vds(config)
            append_vds_rows(out_file, rows)
            print(f"VDS poll {len(rows)}행", flush=True)
        except Exception as error:
            print(f"VDS poll 실패: {error}", flush=True)
        if stop.wait(interval):
            break


def main() -> None:
    parser = argparse.ArgumentParser(
        description="판교1 서울 차로 특징과 실시간 VDS를 같이 수집합니다."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--minutes", type=float, default=5)
    parser.add_argument("--chunk-seconds", type=int, default=None)
    parser.add_argument("--coverage-min", type=float, default=None)
    parser.add_argument("--sample-fps", type=float, default=None)
    parser.add_argument("--bin-seconds", type=int, default=None)
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg가 필요합니다.")

    config = load_config(args.config)
    collection = config["collection"]
    if args.chunk_seconds is not None:
        collection["chunk_seconds"] = args.chunk_seconds
    if args.coverage_min is not None:
        collection["coverage_min"] = args.coverage_min
    if args.sample_fps is not None:
        collection["sample_fps"] = args.sample_fps
    if args.bin_seconds is not None:
        collection["bin_seconds"] = args.bin_seconds

    width = int(collection["frame_width"])
    height = int(collection["frame_height"])
    frame_bytes = width * height * 3
    seoul_poly = pixel_polygon(config["roi"]["seoul"], width, height)
    busan_poly = pixel_polygon(config["roi"]["busan"], width, height)
    count_line = counting_line_pixels(
        config["roi"]["counting_line_seoul"], width, height
    )
    tracker = CentroidTracker()
    prev_tracks: dict[int, tuple[float, float]] = {}
    vehicle_classes = set(int(x) for x in config["detection"]["vehicle_classes"])
    stopped_speed = float(config["detection"]["stopped_pixel_speed"]) * (
        1.0 / max(0.1, float(collection["sample_fps"]))
    )

    session_start = now_local()
    session_dir = RECORDINGS_DIR / f"pangyo1_features_{session_start.strftime('%Y%m%d_%H%M%S')}"
    session_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = session_dir / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    frames_file = session_dir / "frames.csv"
    bins_file = session_dir / "features_5min.csv"
    vds_live_file = session_dir / "its_vds_live.csv"
    (session_dir / "config.used.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    detector = Detector(config)
    print("특징 수집을 시작합니다. HLS 404는 새 URL로 회복합니다.")
    print("저장 폴더:", session_dir)
    print("detector:", detector.name)

    stop_vds = threading.Event()
    vds_thread = threading.Thread(
        target=vds_worker,
        args=(stop_vds, config, vds_live_file),
        daemon=True,
    )
    vds_thread.start()

    bins: dict[datetime, dict] = {}
    last_snapshot_bin: datetime | None = None
    prev_seoul_centers: list[tuple[float, float]] = []
    consecutive_failures = 0
    chunk_number = 1
    deadline = time.monotonic() + max(1.0, args.minutes * 60)
    frame_header = [
        "시각", "bin_start", "chunk", "seoul_count", "busan_count",
        "seoul_stopped", "seoul_crossings", "seoul_car", "seoul_motorcycle",
        "seoul_bus", "seoul_truck", "brightness", "오류분류",
    ]

    try:
        while time.monotonic() < deadline:
            try:
                cctv, stream_url = fetch_cctv(config)
            except Exception as error:
                text = str(error)
                print(f"[{chunk_number:03d}] CCTV 주소 요청 실패: {type(error).__name__}: {text[:180]}")
                if "월간" in text or "4001" in text:
                    print("cctvInfo 월간 호출 한도. HLS URL을 받을 수 없어 수집을 중단합니다.")
                    break
                consecutive_failures += 1
                if consecutive_failures >= int(collection["consecutive_fail_limit"]):
                    break
                wait = float(collection["fail_backoff_seconds"])
                if "401" in text:
                    wait = max(wait, 20)
                time.sleep(wait)
                continue

            log_file = session_dir / f"chunk_{chunk_number:03d}.ffmpeg.log"
            cmd = build_ffmpeg_cmd(stream_url, config)
            with log_file.open("w", encoding="utf-8") as log:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log)

            frames_got = 0
            assert proc.stdout is not None
            try:
                while True:
                    raw = proc.stdout.read(frame_bytes)
                    if not raw or len(raw) < frame_bytes:
                        break
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()
                    ts = now_local()
                    detect_frame = prepare_detection_frame(
                        frame,
                        config["roi"]["osd_masks"],
                        [seoul_poly, busan_poly],
                    )
                    boxes, classes = detector.detect(detect_frame)
                    grouped = group_detections(
                        boxes, classes, seoul_poly, busan_poly, vehicle_classes
                    )
                    stopped = match_stopped(
                        prev_seoul_centers, grouped["seoul_centers"], stopped_speed
                    )
                    prev_seoul_centers = grouped["seoul_centers"]
                    tracks = tracker.update(grouped["seoul_centers"])
                    crossings = count_line_crossings(
                        prev_tracks, tracks, count_line, toward_camera=True
                    )
                    prev_tracks = tracks
                    brightness = mean_brightness(frame, seoul_poly)
                    bin_start = floor_bin(ts, int(collection["bin_seconds"]))
                    if bin_start not in bins:
                        bins[bin_start] = empty_bin_stats()
                    frame_row = {
                        "시각": ts.isoformat(),
                        "bin_start": bin_start.isoformat(),
                        "chunk": chunk_number,
                        "seoul_count": grouped["seoul_count"],
                        "busan_count": grouped["busan_count"],
                        "seoul_stopped": stopped,
                        "seoul_crossings": crossings,
                        "seoul_car": grouped["seoul_car"],
                        "seoul_motorcycle": grouped["seoul_motorcycle"],
                        "seoul_bus": grouped["seoul_bus"],
                        "seoul_truck": grouped["seoul_truck"],
                        "brightness": round(brightness, 2),
                        "오류분류": "",
                    }
                    add_frame_to_bin(bins[bin_start], frame_row)
                    append_csv(frames_file, frame_row, frame_header)
                    frames_got += 1
                    if collection.get("snapshot_every_bin") and last_snapshot_bin != bin_start:
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
            finally:
                proc.stdout.close()
                try:
                    proc.wait(timeout=int(collection["chunk_seconds"]) + 20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()

            raw_log = log_file.read_text(encoding="utf-8", errors="replace")
            log_file.write_text(redact_urls(raw_log), encoding="utf-8")
            error_category, http_status = classify_ffmpeg_log(raw_log)
            print(
                f"[{chunk_number:03d}] 프레임 {frames_got}, "
                f"{error_category} {http_status}, CCTV {cctv.get('cctvname', '')}",
                flush=True,
            )
            if frames_got > 0:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                time.sleep(float(collection["fail_backoff_seconds"]))
                if consecutive_failures >= int(collection["consecutive_fail_limit"]):
                    break
            chunk_number += 1
    finally:
        stop_vds.set()
        vds_thread.join(timeout=5)

    for bin_start, stats in sorted(bins.items()):
        row = write_bin(bins_file, bin_start, stats, config, seoul_poly, detector.name)
        print(
            f"칸 {row['bin_start']} coverage={row['coverage']} "
            f"missing={row['cctv_missing']} density={row['seoul_count_mean']} "
            f"crossings={row['seoul_crossings']}"
        )
    print("종료:", now_local().isoformat())
    print("features:", bins_file)
    print("vds live:", vds_live_file)


if __name__ == "__main__":
    main()
