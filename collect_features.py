#!/usr/bin/env python3

"""Collect Pangyo-1 Seoul-lane CCTV features and live VDS.

cctvType 5 MP4 clips are decoded then discarded. 5-minute bins flush as they
close. Ctrl+C finalizes join and appends complete fair rows to the master CSV.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import signal
import subprocess
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import yaml

from dataset_io import append_master
from its_common import (
    LOGS_DIR,
    SESSIONS_DIR,
    VALIDATION_DIR,
    classify_ffmpeg_log,
    detector_tag,
    fetch_cctv,
    is_hls_404,
    load_config,
    master_csv_path,
    redact_urls,
)
from join_vds_cctv import join_live_session
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
    keep_wall_clock_sample,
    match_stopped,
    mean_brightness,
    pixel_polygon,
    prepare_detection_frame,
)


def now_local() -> datetime:
    return datetime.now().astimezone()


class SessionLog:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._file = path.open("a", encoding="utf-8")

    def emit(self, msg: str) -> None:
        line = f"{now_local().isoformat()} {msg}"
        with self._lock:
            print(msg, flush=True)
            self._file.write(line + "\n")
            self._file.flush()

    def error(self, msg: str) -> None:
        self.emit(f"ERROR {msg}")

    def close(self) -> None:
        with self._lock:
            self._file.close()


class ValidationSampler:
    def __init__(
        self,
        out_dir: Path,
        every_minutes: float,
        seconds: int,
        fps: float,
        max_files: int,
        width: int,
        height: int,
    ):
        self.out_dir = out_dir
        self.every = timedelta(minutes=max(0.0, float(every_minutes)))
        self.maxlen = max(1, int(round(max(1, seconds) * max(0.1, fps))))
        self.buffer: deque[np.ndarray] = deque(maxlen=self.maxlen)
        self.last_write: datetime | None = None
        self.max_files = max(1, int(max_files))
        self.size = (int(width), int(height))
        self.fps = float(fps)
        self.enabled = every_minutes > 0
        if self.enabled:
            out_dir.mkdir(parents=True, exist_ok=True)

    def add(self, frame: np.ndarray, ts: datetime) -> None:
        if not self.enabled:
            return
        self.buffer.append(frame)
        if self.last_write is None:
            self.last_write = ts
            return
        if ts - self.last_write < self.every:
            return
        self._write(ts)
        self.last_write = ts

    def _write(self, ts: datetime) -> None:
        if not self.buffer:
            return
        path = self.out_dir / f"{ts.strftime('%Y%m%d_%H%M%S')}.mp4"
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            self.size,
        )
        for frame in self.buffer:
            out = frame
            if out.shape[1] != self.size[0] or out.shape[0] != self.size[1]:
                out = cv2.resize(out, self.size)
            writer.write(out)
        writer.release()
        files = sorted(self.out_dir.glob("*.mp4"))
        extra = len(files) - self.max_files
        for old in files[: max(0, extra)]:
            old.unlink(missing_ok=True)


def build_ffmpeg_cmd(
    stream_url: str, config: dict, duration_seconds: int, progressive: bool
) -> list[str]:
    collection = config["collection"]
    ffmpeg = config["ffmpeg"]
    width = int(collection["frame_width"])
    height = int(collection["frame_height"])
    fps = float(collection["sample_fps"])
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-loglevel", "info",
        "-rw_timeout", str(int(ffmpeg["rw_timeout_us"])),
        "-user_agent", ffmpeg["user_agent"],
        "-headers", f"Referer: {ffmpeg['referer']}\r\n",
    ]
    if progressive:
        cmd.append("-re")
    elif not ffmpeg.get("http_persistent", True):
        cmd.extend(["-http_persistent", "0"])
    cmd.extend(
        [
            "-i", stream_url,
            "-t", str(max(1, duration_seconds)),
            "-an",
            "-vf", f"fps={fps},scale={width}:{height}",
            "-pix_fmt", "bgr24",
            "-f", "rawvideo",
            "pipe:1",
        ]
    )
    return cmd


def backoff_schedule(collection: dict) -> list[float]:
    raw = collection.get("fail_backoff_seconds", [5, 10, 20, 30])
    if isinstance(raw, (int, float)):
        return [5.0, 10.0, 20.0, 30.0]
    values = [float(x) for x in raw]
    return values or [5.0, 10.0, 20.0, 30.0]


def backoff_wait(fail_count: int, schedule: list[float]) -> float:
    if fail_count <= 0:
        return 0.0
    return schedule[min(fail_count, len(schedule)) - 1]


def mean_or_zero(total: float, count: int) -> float:
    return (total / count) if count else 0.0


def stream_is_progressive(url: str, cctv_type: str) -> bool:
    if str(cctv_type) in {"2", "5"}:
        return True
    path = (url or "").split("?", 1)[0].lower()
    return path.endswith(".mp4")


def http_status_is_error(http_status: str, category: str) -> bool:
    if is_hls_404(category):
        return True
    return any(code.startswith(("4", "5")) for code in (http_status or "").split(",") if code)


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


def flush_closed_bins(
    bins: dict[datetime, dict],
    current_bin: datetime | None,
    bins_file: Path,
    config: dict,
    seoul_polygon,
    detector_name: str,
    log: SessionLog,
) -> None:
    closed = [key for key in bins if current_bin is None or key < current_bin]
    for key in sorted(closed):
        row = write_bin(bins_file, key, bins.pop(key), config, seoul_polygon, detector_name)
        log.emit(
            f"칸 {row['bin_start']} coverage={row['coverage']} "
            f"missing={row['cctv_missing']} density={row['seoul_count_mean']} "
            f"crossings={row['seoul_crossings']}"
        )


def vds_worker(stop: threading.Event, config: dict, out_file: Path, log: SessionLog) -> None:
    spec = config.get("its_vds") or {}
    if not spec.get("enabled"):
        return
    interval = float(spec.get("poll_seconds") or 60)
    while True:
        try:
            rows = fetch_pangyo_vds(config)
            append_vds_rows(out_file, rows)
            log.emit(f"VDS poll {len(rows)}행")
        except Exception as error:
            log.error(f"VDS poll 실패: {redact_urls(str(error))}")
        if stop.wait(interval):
            break


def finalize_session(
    session_id: str,
    session_dir: Path,
    session_start: datetime,
    session_end: datetime,
    config: dict,
    log: SessionLog,
) -> None:
    features_path = session_dir / "features_5min.csv"
    vds_path = session_dir / "its_vds_live.csv"
    master_path = master_csv_path(config)
    if not features_path.exists() or features_path.stat().st_size == 0:
        log.emit("features_5min.csv가 없어 조인하지 않습니다.")
        return
    if not vds_path.exists() or vds_path.stat().st_size == 0:
        log.error("its_vds_live.csv가 없어 조인하지 않습니다.")
        return
    try:
        joined = join_live_session(
            features_path, vds_path, config, session_id, session_start, session_end
        )
        out_join = session_dir / "joined_vds_cctv.csv"
        joined.to_csv(out_join, index=False, encoding="utf-8-sig")
        added = append_master(joined, session_id, master_path)
        fair = int((joined["fair_eval"] == 1).sum())
        complete = int((joined["bin_complete"] == 1).sum())
        log.emit(
            f"조인 {out_join.name}: 칸 {len(joined)}, fair_eval=1 {fair}, "
            f"complete {complete}, master 추가 {added} → {master_path}"
        )
    except Exception as error:
        log.error(f"세션 조인/master 누적 실패: {redact_urls(str(error))}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="판교1 서울 차로 특징과 실시간 VDS를 같이 수집합니다."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--minutes",
        type=float,
        default=0,
        help="수집 시간(분). 0이면 Ctrl+C까지 계속합니다.",
    )
    parser.add_argument("--hours", type=float, default=None, help="수집 시간(시간). minutes보다 우선.")
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=None,
        help="호환용. MP4/HLS를 이 초마다 끊거나 cctvInfo를 다시 치지 않습니다.",
    )
    parser.add_argument("--coverage-min", type=float, default=None)
    parser.add_argument("--sample-fps", type=float, default=None)
    parser.add_argument("--bin-seconds", type=int, default=None)
    parser.add_argument(
        "--cctv-type",
        default=None,
        help="ITS cctvType. 4=HTTPS HLS, 5=HTTPS MP4, 2=HTTP MP4",
    )
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
    if args.cctv_type is not None:
        config["its"]["cctv_type"] = str(args.cctv_type)
    collection["keep_video"] = False

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

    for folder in (SESSIONS_DIR, VALIDATION_DIR, LOGS_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    session_start = now_local()
    model_tag = detector_tag(config)
    session_id = f"pangyo1_{session_start.strftime('%Y%m%d_%H%M%S')}"
    session_dir = SESSIONS_DIR / model_tag / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = session_dir / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    frames_file = session_dir / "frames.csv"
    bins_file = session_dir / "features_5min.csv"
    vds_live_file = session_dir / "its_vds_live.csv"
    ffmpeg_log = session_dir / "ffmpeg_last.log"
    (session_dir / "config.used.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    log = SessionLog(LOGS_DIR / model_tag / f"{session_id}.log")
    sampler = ValidationSampler(
        VALIDATION_DIR / model_tag / session_id,
        float(collection.get("validation_every_minutes") or 0),
        int(collection.get("validation_seconds") or 5),
        float(collection["sample_fps"]),
        int(collection.get("max_validation_files") or 24),
        width,
        height,
    )

    detector = Detector(config)
    cctv_type = str(config["its"]["cctv_type"])
    duration_s = (
        float(args.hours) * 3600 if args.hours is not None else float(args.minutes) * 60
    )
    deadline = None if duration_s <= 0 else time.monotonic() + duration_s
    stop_event = threading.Event()

    def request_stop(signum, _frame) -> None:
        log.emit(f"종료 신호 {signum}. 현재 클립 이후 조인하고 마칩니다.")
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    log.emit("특징 수집을 시작합니다.")
    log.emit(f"session_id={session_id} detector={model_tag} cctvType={cctv_type} keep_video=false")
    log.emit(f"저장 폴더: {session_dir}")
    log.emit(f"detector: {detector.name}")
    if deadline is None:
        log.emit("제한 시간 없음. Ctrl+C로 정상 종료하세요.")
    if cctv_type in {"2", "5"}:
        log.emit("MP4는 분석 후 파일로 남기지 않고, 클립이 끝나면 다음 URL만 받습니다.")

    stop_vds = threading.Event()
    vds_thread = threading.Thread(
        target=vds_worker,
        args=(stop_vds, config, vds_live_file, log),
        daemon=True,
    )
    vds_thread.start()

    bins: dict[datetime, dict] = {}
    last_snapshot_bin: datetime | None = None
    last_sample_bucket: int | None = None
    prev_seoul_centers: list[tuple[float, float]] = []
    consecutive_failures = 0
    clip_number = 1
    wait_schedule = backoff_schedule(collection)
    fail_limit = int(collection.get("consecutive_fail_limit") or 0)
    stream_url: str | None = None
    cctv: dict = {}
    url_acquired_at: float | None = None
    hls_stats = {
        "cctvinfo_calls": 0,
        "hls_404s": 0,
        "http_errors": 0,
        "alive_sum": 0.0,
        "alive_n": 0,
    }
    attempts_file = session_dir / "stream_attempts.csv"
    attempts_header = [
        "clip", "started_at", "frames", "error_category", "http_status",
        "url_alive_s", "refetch_cctvinfo", "cctvinfo_calls", "hls_404s",
        "http_errors", "cctv_type",
    ]
    frame_header = [
        "시각", "bin_start", "chunk", "seoul_count", "busan_count",
        "seoul_stopped", "seoul_crossings", "seoul_car", "seoul_motorcycle",
        "seoul_bus", "seoul_truck", "brightness", "오류분류",
    ]

    def still_running() -> bool:
        if stop_event.is_set():
            return False
        if deadline is None:
            return True
        return time.monotonic() < deadline

    def remaining_seconds() -> int:
        if deadline is None:
            return 3600
        return max(1, int(deadline - time.monotonic()))

    try:
        while still_running():
            if stream_url is None:
                hls_stats["cctvinfo_calls"] += 1
                try:
                    cctv, stream_url = fetch_cctv(config)
                except Exception as error:
                    text = redact_urls(str(error))
                    log.error(
                        f"[cctvType {cctv_type} {clip_number:03d}] cctvInfo 실패: "
                        f"{type(error).__name__}: {text[:180]} "
                        f"calls={hls_stats['cctvinfo_calls']}"
                    )
                    if "월간" in text or "4001" in text:
                        log.error("cctvInfo 월간 호출 한도. 수집을 중단합니다.")
                        break
                    consecutive_failures += 1
                    if fail_limit > 0 and consecutive_failures >= fail_limit:
                        log.error("연속 실패 한도에 도달해 중단합니다.")
                        break
                    stop_event.wait(backoff_wait(consecutive_failures, wait_schedule))
                    continue
                url_acquired_at = time.monotonic()
                consecutive_failures = 0
                log.emit(
                    f"[cctvType {cctv_type} {clip_number:03d}] cctvInfo URL 획득 "
                    f"calls={hls_stats['cctvinfo_calls']}"
                )

            proc = None
            frames_got = 0
            started_at = now_local()
            progressive = stream_is_progressive(stream_url, cctv_type)
            try:
                cmd = build_ffmpeg_cmd(stream_url, config, remaining_seconds(), progressive)
                with ffmpeg_log.open("w", encoding="utf-8") as log_file:
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log_file)
                assert proc.stdout is not None
                while still_running():
                    raw = proc.stdout.read(frame_bytes)
                    if not raw or len(raw) < frame_bytes:
                        break
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()
                    ts = now_local()
                    keep, last_sample_bucket = keep_wall_clock_sample(
                        ts, last_sample_bucket, float(collection["sample_fps"])
                    )
                    if not keep:
                        continue
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
                        flush_closed_bins(
                            bins, bin_start, bins_file, config, seoul_poly, detector.name, log
                        )
                        bins[bin_start] = empty_bin_stats()
                    frame_row = {
                        "시각": ts.isoformat(),
                        "bin_start": bin_start.isoformat(),
                        "chunk": clip_number,
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
                    sampler.add(frame, ts)
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
            except Exception as error:
                log.error(
                    f"[clip {clip_number:03d}] 처리 실패: "
                    f"{type(error).__name__}: {redact_urls(str(error))[:180]}"
                )
            finally:
                if proc is not None:
                    if proc.poll() is None:
                        proc.kill()
                    if proc.stdout is not None:
                        proc.stdout.close()
                    try:
                        proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()

            raw_log = ""
            if ffmpeg_log.exists():
                raw_log = redact_urls(ffmpeg_log.read_text(encoding="utf-8", errors="replace"))
                ffmpeg_log.write_text(raw_log, encoding="utf-8")
            error_category, http_status = classify_ffmpeg_log(raw_log)
            url_alive = (
                (time.monotonic() - url_acquired_at) if url_acquired_at is not None else 0.0
            )
            died_404 = is_hls_404(error_category)
            http_err = http_status_is_error(http_status, error_category)
            if http_err:
                hls_stats["http_errors"] += 1
                log.error(
                    f"[clip {clip_number:03d}] HTTP {error_category} {http_status} "
                    f"frames={frames_got} alive={url_alive:.1f}s"
                )
            progressive = stream_is_progressive(stream_url or "", cctv_type)
            refetch = False
            if died_404:
                hls_stats["hls_404s"] += 1
                hls_stats["alive_sum"] += url_alive
                hls_stats["alive_n"] += 1
                stream_url = None
                url_acquired_at = None
                refetch = True
            elif progressive:
                hls_stats["alive_sum"] += url_alive
                hls_stats["alive_n"] += 1
                stream_url = None
                url_acquired_at = None
                refetch = True

            append_csv(
                attempts_file,
                {
                    "clip": clip_number,
                    "started_at": started_at.isoformat(),
                    "frames": frames_got,
                    "error_category": error_category,
                    "http_status": http_status,
                    "url_alive_s": round(url_alive, 2),
                    "refetch_cctvinfo": int(refetch),
                    "cctvinfo_calls": hls_stats["cctvinfo_calls"],
                    "hls_404s": hls_stats["hls_404s"],
                    "http_errors": hls_stats["http_errors"],
                    "cctv_type": cctv_type,
                },
                attempts_header,
            )
            tag = "MP4" if str(cctv_type) in {"2", "5"} else "HLS"
            log.emit(
                f"[{tag} {clip_number:03d}] 프레임 {frames_got}, {error_category} {http_status}, "
                f"CCTV {cctv.get('cctvname', '')}, alive={url_alive:.1f}s, "
                f"cctvInfo={hls_stats['cctvinfo_calls']} 404={hls_stats['hls_404s']} "
                f"http_err={hls_stats['http_errors']} "
                f"avg_alive={mean_or_zero(hls_stats['alive_sum'], hls_stats['alive_n']):.1f}s "
                f"{'→ cctvInfo 재호출' if refetch else '→ URL 재사용'}"
            )

            if frames_got > 0:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if fail_limit > 0 and consecutive_failures >= fail_limit:
                    log.error("연속 실패 한도에 도달해 중단합니다.")
                    break
                stop_event.wait(backoff_wait(consecutive_failures, wait_schedule))
            clip_number += 1
    finally:
        stop_vds.set()
        vds_thread.join(timeout=5)
        flush_closed_bins(bins, None, bins_file, config, seoul_poly, detector.name, log)
        summary = {
            "cctv_type": cctv_type,
            "cctvinfo_calls": hls_stats["cctvinfo_calls"],
            "hls_404s": hls_stats["hls_404s"],
            "http_errors": hls_stats["http_errors"],
            "url_sessions": hls_stats["alive_n"],
            "url_alive_mean_s": round(
                mean_or_zero(hls_stats["alive_sum"], hls_stats["alive_n"]), 2
            ),
        }
        append_csv(session_dir / "stream_stats.csv", summary, list(summary.keys()))
        log.emit(
            f"cctvType {cctv_type}, cctvInfo 호출 {summary['cctvinfo_calls']}회, "
            f"HLS 404 {summary['hls_404s']}회, HTTP 오류 {summary['http_errors']}회, "
            f"URL 평균 생존 {summary['url_alive_mean_s']}s"
        )
        session_end = now_local()
        finalize_session(session_id, session_dir, session_start, session_end, config, log)
        log.emit(f"종료: {session_end.isoformat()}")
        log.emit(f"features: {bins_file}")
        log.emit(f"vds live: {vds_live_file}")
        log.close()


if __name__ == "__main__":
    main()
