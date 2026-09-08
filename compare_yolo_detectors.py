#!/usr/bin/env python3

"""Compare YOLOv8n/s/m and Seoul-ROI-crop on stored night frames.

Does not change collection config or start live collection.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from its_common import PROJECT_DIR, load_config
from video_features import (
    COCO_VEHICLE,
    apply_osd_mask,
    box_center,
    center_in_polygon,
    counting_line_pixels,
    pixel_polygon,
    polygon_mask,
    prepare_detection_frame,
)


OUT_DIR = PROJECT_DIR / "recordings" / "yolo_compare_night"
VEHICLE_CLASSES = [2, 3, 5, 7]
CLASS_COLOR = {
    "car": (0, 220, 0),
    "motorcycle": (0, 255, 255),
    "bus": (255, 0, 255),
    "truck": (255, 180, 0),
}
SAVE_FRAMES = {
    "val_195215_f00",
    "val_195215_f04",
    "val_205218_f02",
    "val_215219_f02",
}


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_frames() -> list[dict]:
    frames: list[dict] = []
    val_dir = PROJECT_DIR / "videos" / "validation" / "pangyo1_20260904_185210"
    for path in sorted(val_dir.glob("*.mp4")):
        cap = cv2.VideoCapture(str(path))
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            stamp = path.stem[-6:] if len(path.stem) >= 6 else path.stem
            frames.append(
                {
                    "name": f"val_{stamp}_f{idx:02d}",
                    "source": "validation_raw",
                    "path": str(path),
                    "frame": frame,
                }
            )
            idx += 1
        cap.release()
    return frames


def seoul_crop_enlarged(
    frame: np.ndarray,
    seoul_poly: np.ndarray,
    osd_masks: list[list[float]],
    scale: float,
) -> tuple[np.ndarray, int, int, float, int]:
    masked = apply_osd_mask(frame, osd_masks)
    x, y, w, h = cv2.boundingRect(seoul_poly)
    pad = 8
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame.shape[1], x + w + pad)
    y2 = min(frame.shape[0], y + h + pad)
    crop = masked[y1:y2, x1:x2]
    poly = seoul_poly.copy()
    poly[:, 0] -= x1
    poly[:, 1] -= y1
    crop = cv2.bitwise_and(crop, crop, mask=polygon_mask(crop.shape[:2], poly))
    new_w = max(32, int(round((x2 - x1) * scale)))
    new_h = max(32, int(round((y2 - y1) * scale)))
    enlarged = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    imgsz = int(np.ceil(max(new_w, new_h) / 32.0) * 32)
    return enlarged, x1, y1, scale, max(640, imgsz)


def map_boxes_to_full(
    xyxy: np.ndarray, x1: int, y1: int, scale: float
) -> np.ndarray:
    if len(xyxy) == 0:
        return xyxy
    out = xyxy.astype(float).copy()
    out[:, [0, 2]] = out[:, [0, 2]] / scale + x1
    out[:, [1, 3]] = out[:, [1, 3]] / scale + y1
    return out


def summarize_boxes(
    xyxy: np.ndarray,
    cls: np.ndarray,
    seoul_poly: np.ndarray,
    count_y: float,
) -> dict:
    seoul = 0
    near = 0
    far = 0
    bus = 0
    truck = 0
    outside = 0
    kept_boxes = []
    kept_cls = []
    for box, cid in zip(xyxy, cls):
        name = COCO_VEHICLE.get(int(cid), "car")
        cx, cy = box_center(box.tolist())
        if not center_in_polygon(cx, cy, seoul_poly):
            outside += 1
            continue
        seoul += 1
        kept_boxes.append(box.tolist())
        kept_cls.append(name)
        if cy >= count_y:
            near += 1
        else:
            far += 1
        if name == "bus":
            bus += 1
        elif name == "truck":
            truck += 1
    return {
        "seoul": seoul,
        "near": near,
        "far": far,
        "bus": bus,
        "truck": truck,
        "outside_roi": outside,
        "boxes": kept_boxes,
        "names": kept_cls,
    }


def draw_panel(
    frame: np.ndarray,
    seoul_poly: np.ndarray,
    count_line,
    title: str,
    stats: dict,
    ms: float,
) -> np.ndarray:
    out = frame.copy()
    overlay = out.copy()
    cv2.fillPoly(overlay, [seoul_poly], (0, 180, 0))
    out = cv2.addWeighted(overlay, 0.14, out, 0.86, 0)
    cv2.polylines(out, [seoul_poly], True, (0, 220, 0), 2)
    cv2.line(out, count_line[0], count_line[1], (0, 255, 255), 2)
    for box, name in zip(stats["boxes"], stats["names"]):
        x1, y1, x2, y2 = [int(v) for v in box]
        color = CLASS_COLOR.get(name, (0, 220, 0))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, name[:3], (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    bar = np.zeros((56, out.shape[1], 3), dtype=np.uint8)
    line1 = f"{title}  {ms:.0f}ms"
    line2 = (
        f"Seoul {stats['seoul']}  near {stats['near']}  far {stats['far']}  "
        f"bus {stats['bus']}  truck {stats['truck']}"
    )
    cv2.putText(bar, line1, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    cv2.putText(bar, line2, (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1)
    return np.vstack([bar, out])


def hstack_resize(images: list[np.ndarray]) -> np.ndarray:
    height = max(im.shape[0] for im in images)
    resized = []
    for im in images:
        if im.shape[0] == height:
            resized.append(im)
            continue
        scale = height / im.shape[0]
        resized.append(cv2.resize(im, (int(im.shape[1] * scale), height)))
    return np.hstack(resized)


def predict_full(model: YOLO, frame: np.ndarray, conf: float, device: str, imgsz: int):
    t0 = time.perf_counter()
    result = model.predict(
        frame, conf=conf, classes=VEHICLE_CLASSES, verbose=False, imgsz=imgsz, device=device
    )[0]
    ms = (time.perf_counter() - t0) * 1000
    if result.boxes is None or len(result.boxes) == 0:
        return np.zeros((0, 4), dtype=float), np.zeros((0,), dtype=int), ms
    return (
        result.boxes.xyxy.cpu().numpy(),
        result.boxes.cls.cpu().numpy().astype(int),
        ms,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="판교1 야간 YOLO 검출 비교. 수집은 시작하지 않습니다.")
    parser.add_argument("--confs", default="0.25,0.20,0.15")
    parser.add_argument("--crop-scale", type=float, default=2.0)
    args = parser.parse_args()
    confs = [float(x) for x in args.confs.split(",") if x.strip()]

    config = load_config()
    frames = load_frames()
    if not frames:
        raise SystemExit("비교할 야간 프레임이 없습니다.")

    sample = frames[0]["frame"]
    height, width = sample.shape[:2]
    seoul_poly = pixel_polygon(config["roi"]["seoul"], width, height)
    busan_poly = pixel_polygon(config["roi"]["busan"], width, height)
    count_line = counting_line_pixels(config["roi"]["counting_line_seoul"], width, height)
    count_y = 0.5 * (count_line[0][1] + count_line[1][1])
    osd = config["roi"]["osd_masks"]
    device = pick_device()

    models = {
        "v8n_full": YOLO(str(PROJECT_DIR / "yolov8n.pt")),
        "v8s_full": YOLO("yolov8s.pt"),
        "v8s_crop": YOLO("yolov8s.pt"),
        "v8m_full": YOLO("yolov8m.pt"),
        "v8m_crop": YOLO("yolov8m.pt"),
    }

    crop0, *_rest = seoul_crop_enlarged(sample, seoul_poly, osd, args.crop_scale)
    warmup = prepare_detection_frame(sample, osd, [seoul_poly, busan_poly])
    for name, model in models.items():
        src = crop0 if name.endswith("_crop") else warmup
        imgsz = 640 if name.endswith("_full") else max(640, crop0.shape[0], crop0.shape[1])
        imgsz = int(np.ceil(imgsz / 32.0) * 32)
        model.predict(src, conf=0.25, classes=VEHICLE_CLASSES, verbose=False, imgsz=imgsz, device=device)

    out_dir = OUT_DIR
    img_dir = out_dir / "frames"
    grid_dir = out_dir / "grids"
    img_dir.mkdir(parents=True, exist_ok=True)
    grid_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "crop_input_preview.jpg").write_bytes(cv2.imencode(".jpg", crop0)[1].tobytes())

    rows = []
    methods = [
        ("v8n_full", "YOLOv8n 전체", "full"),
        ("v8s_full", "YOLOv8s 전체", "full"),
        ("v8m_full", "YOLOv8m 전체", "full"),
        ("v8s_crop", "YOLOv8s ROI crop×2", "crop"),
        ("v8m_crop", "YOLOv8m ROI crop×2", "crop"),
    ]
    m_grid_keys = {"v8s_full", "v8m_full", "v8m_crop"}

    for conf in confs:
        grids: dict[str, list[np.ndarray]] = {}
        for item in frames:
            frame = item["frame"]
            detect_full = prepare_detection_frame(frame, osd, [seoul_poly, busan_poly])
            crop, x1, y1, scale, crop_imgsz = seoul_crop_enlarged(
                frame, seoul_poly, osd, args.crop_scale
            )
            panels = []
            for key, title, kind in methods:
                if kind == "crop":
                    xyxy, cls, ms = predict_full(models[key], crop, conf, device, crop_imgsz)
                    xyxy = map_boxes_to_full(xyxy, x1, y1, scale)
                else:
                    xyxy, cls, ms = predict_full(models[key], detect_full, conf, device, 640)
                stats = summarize_boxes(xyxy, cls, seoul_poly, count_y)
                rows.append(
                    {
                        "frame": item["name"],
                        "source": item["source"],
                        "method": key,
                        "conf": conf,
                        "seoul": stats["seoul"],
                        "near": stats["near"],
                        "far": stats["far"],
                        "bus": stats["bus"],
                        "truck": stats["truck"],
                        "outside_roi": stats["outside_roi"],
                        "yolo_ms": round(ms, 1),
                        "device": device,
                        "imgsz": crop_imgsz if kind == "crop" else 640,
                    }
                )
                if item["name"] in SAVE_FRAMES:
                    panel = draw_panel(frame, seoul_poly, count_line, f"{title} c{conf:.2f}", stats, ms)
                    panels.append(panel)
                    cv2.imwrite(str(img_dir / f"{item['name']}__{key}__c{conf:.2f}.jpg"), panel)
            if panels:
                grids[item["name"]] = panels
        for name, panels in grids.items():
            cv2.imwrite(str(grid_dir / f"{name}__c{conf:.2f}.jpg"), hstack_resize(panels))
            m_panels = [p for p, (key, _t, _k) in zip(panels, methods) if key in m_grid_keys]
            if m_panels:
                cv2.imwrite(str(grid_dir / f"{name}__m__c{conf:.2f}.jpg"), hstack_resize(m_panels))

    csv_path = out_dir / "compare_summary.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"device={device} frames={len(frames)} crop_scale={args.crop_scale}")
    print(f"저장: {out_dir}")
    print()
    print(f"{'conf':<6} {'method':<12} {'seoul':>7} {'near':>6} {'far':>6} {'bus':>4} {'truck':>5} {'ms':>7} {'<1000ms':>8}")
    by = {}
    for row in rows:
        key = (row["conf"], row["method"])
        by.setdefault(key, []).append(row)
    for conf in confs:
        for key, _title, _kind in methods:
            chunk = by[(conf, key)]
            seoul = sum(r["seoul"] for r in chunk) / len(chunk)
            near = sum(r["near"] for r in chunk) / len(chunk)
            far = sum(r["far"] for r in chunk) / len(chunk)
            bus = sum(r["bus"] for r in chunk) / len(chunk)
            truck = sum(r["truck"] for r in chunk) / len(chunk)
            ms = sum(r["yolo_ms"] for r in chunk) / len(chunk)
            ok = sum(1 for r in chunk if r["yolo_ms"] < 1000) / len(chunk)
            print(
                f"{conf:<6.2f} {key:<12} {seoul:7.2f} {near:6.2f} {far:6.2f} "
                f"{bus:4.2f} {truck:5.2f} {ms:7.1f} {ok:7.0%}"
            )


if __name__ == "__main__":
    main()
