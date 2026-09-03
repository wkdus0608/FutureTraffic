#!/usr/bin/env python3

"""Draw Seoul/Busan lane polygons and OSD masks on a local frame."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from its_common import load_config
from video_features import counting_line_pixels, draw_roi_overlay, pixel_polygon


def first_frame(path: Path):
    image = cv2.imread(str(path))
    if image is not None:
        return image
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise SystemExit(f"프레임을 읽지 못했습니다: {path}")
    return frame


def save_image(path: Path, image) -> None:
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise SystemExit("JPEG 인코딩 실패")
    path.write_bytes(encoded.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="판교1 차로 다각형 ROI 미리보기")
    parser.add_argument("video", type=Path, help="기존 .ts/.mp4/.jpg 경로")
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    frame = first_frame(args.video)
    height, width = frame.shape[:2]
    seoul = pixel_polygon(config["roi"]["seoul"], width, height)
    busan = pixel_polygon(config["roi"]["busan"], width, height)
    count_line = None
    if config["roi"].get("counting_line_seoul"):
        count_line = counting_line_pixels(
            config["roi"]["counting_line_seoul"], width, height
        )
    overlay = draw_roi_overlay(
        frame, seoul, busan, config["roi"]["osd_masks"], counting_line=count_line
    )
    out = args.out or args.video.with_name(args.video.stem + "_preview_roi.jpg")
    save_image(out, overlay)
    print("저장:", out)
    print("초록=서울 본선, 주황=부산 차로, 노란 선=카운팅 라인, 빨간 박스=OSD 마스크.")


if __name__ == "__main__":
    main()
