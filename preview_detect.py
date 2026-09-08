#!/usr/bin/env python3

"""Watch YOLO boxes on live Pangyo-1 CCTV or saved clips. Does not collect data."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from collect_features import Detector, build_ffmpeg_cmd, stream_is_progressive
from its_common import PROJECT_DIR, fetch_cctv, load_config, redact_urls
from video_features import (
    draw_roi_overlay,
    group_detections,
    pixel_polygon,
    prepare_detection_frame,
    counting_line_pixels,
)


def validation_clips() -> list[Path]:
    folder = PROJECT_DIR / "videos" / "validation" / "pangyo1_20260904_185210"
    return sorted(folder.glob("*.mp4"))


def annotate(
    frame: np.ndarray,
    config: dict,
    detector: Detector,
    seoul,
    busan,
    count_line,
    label: str,
) -> np.ndarray:
    detect_frame = prepare_detection_frame(
        frame, config["roi"]["osd_masks"], [seoul, busan]
    )
    t0 = time.perf_counter()
    boxes, classes = detector.detect(detect_frame)
    ms = (time.perf_counter() - t0) * 1000
    grouped = group_detections(
        boxes, classes, seoul, busan, detector.classes
    )
    out = draw_roi_overlay(
        frame,
        seoul,
        busan,
        config["roi"]["osd_masks"],
        grouped["seoul_count"],
        grouped["busan_count"],
        count_line,
        grouped.get("seoul_boxes"),
        grouped.get("busan_boxes"),
    )
    bar = np.zeros((40, out.shape[1], 3), dtype=np.uint8)
    text = (
        f"{label}  Seoul {grouped['seoul_count']}  "
        f"Busan {grouped['busan_count']}  {ms:.0f}ms  "
        f"창을 닫으면 종료"
    )
    cv2.putText(bar, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    return np.vstack([bar, out])


def start_ffplay(width: int, height: int, title: str) -> subprocess.Popen:
    return subprocess.Popen(
        [
            "ffplay",
            "-hide_banner",
            "-loglevel", "error",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-framedrop",
            "-f", "rawvideo",
            "-pixel_format", "bgr24",
            "-video_size", f"{width}x{height + 40}",
            "-framerate", "12",
            "-i", "pipe:0",
            "-window_title", title,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def send(player: subprocess.Popen, frame: np.ndarray) -> bool:
    if player.poll() is not None or player.stdin is None:
        return False
    try:
        player.stdin.write(frame.tobytes())
        player.stdin.flush()
        return True
    except BrokenPipeError:
        return False


def play_files(
    paths: list[Path],
    config: dict,
    detector: Detector,
    seoul,
    busan,
    count_line,
    label: str,
    player: subprocess.Popen,
) -> None:
    while player.poll() is None:
        for path in paths:
            cap = cv2.VideoCapture(str(path))
            while player.poll() is None:
                ok, frame = cap.read()
                if not ok:
                    break
                vis = annotate(frame, config, detector, seoul, busan, count_line, label)
                if not send(player, vis):
                    cap.release()
                    return
                time.sleep(0.35)
            cap.release()


def play_live(
    config: dict,
    detector: Detector,
    seoul,
    busan,
    count_line,
    label: str,
    player: subprocess.Popen,
    minutes: float,
) -> None:
    collection = config["collection"]
    width = int(collection["frame_width"])
    height = int(collection["frame_height"])
    frame_bytes = width * height * 3
    cctv_type = str(config["its"]["cctv_type"])
    deadline = None if minutes <= 0 else time.monotonic() + minutes * 60
    while player.poll() is None:
        if deadline is not None and time.monotonic() >= deadline:
            break
        try:
            _cctv, stream_url = fetch_cctv(config)
        except Exception as error:
            print("cctvInfo 실패:", redact_urls(str(error))[:180], flush=True)
            time.sleep(5)
            continue
        progressive = stream_is_progressive(stream_url, cctv_type)
        remain = 3600 if deadline is None else max(1, int(deadline - time.monotonic()))
        cmd = build_ffmpeg_cmd(stream_url, config, remain, progressive)
        # Preview should look like video, not 1fps slideshow.
        cmd = [x if not str(x).startswith("fps=") else f"scale={width}:{height}" for x in cmd]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        assert proc.stdout is not None
        try:
            while player.poll() is None:
                if deadline is not None and time.monotonic() >= deadline:
                    break
                raw = proc.stdout.read(frame_bytes)
                if not raw or len(raw) < frame_bytes:
                    break
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()
                vis = annotate(frame, config, detector, seoul, busan, count_line, label)
                if not send(player, vis):
                    return
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.stdout.close()
            proc.wait(timeout=10)


def main() -> None:
    parser = argparse.ArgumentParser(description="판교1 YOLO 검출을 동영상 창으로 봅니다. 수집은 하지 않습니다.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None, help="기본은 config의 detection.model")
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument(
        "--live",
        action="store_true",
        help="실제 판교1 CCTV를 봅니다. 지정하지 않으면 저장된 야간 검증 영상을 반복 재생합니다.",
    )
    parser.add_argument("--minutes", type=float, default=0, help="live일 때 제한 시간. 0이면 창을 닫을 때까지.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.model:
        config["detection"]["model"] = args.model
    if args.conf is not None:
        config["detection"]["conf"] = args.conf

    width = int(config["collection"]["frame_width"])
    height = int(config["collection"]["frame_height"])
    seoul = pixel_polygon(config["roi"]["seoul"], width, height)
    busan = pixel_polygon(config["roi"]["busan"], width, height)
    count_line = counting_line_pixels(config["roi"]["counting_line_seoul"], width, height)
    detector = Detector(config)
    label = f"{Path(config['detection']['model']).stem} conf={config['detection']['conf']:.2f}"
    title = f"판교1 YOLO 미리보기  {label}"
    print(title, flush=True)
    print("창이 뜨면 그걸 보면 됩니다. 창을 닫으면 종료합니다.", flush=True)

    player = start_ffplay(width, height, title)
    try:
        if args.live:
            play_live(config, detector, seoul, busan, count_line, label, player, args.minutes)
        else:
            clips = validation_clips()
            if not clips:
                raise SystemExit("검증 영상이 없습니다. --live 로 실제 CCTV를 보세요.")
            play_files(clips, config, detector, seoul, busan, count_line, label, player)
    finally:
        if player.stdin:
            try:
                player.stdin.close()
            except BrokenPipeError:
                pass
        if player.poll() is None:
            player.terminate()


if __name__ == "__main__":
    main()
