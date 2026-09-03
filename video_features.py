#!/usr/bin/env python3

"""ROI polygons, OSD masks, binning, tracking, and vehicle grouping."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta

import cv2
import numpy as np


COCO_VEHICLE = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def pixel_rect(norm_xyxy: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = norm_xyxy
    return (
        int(round(x1 * width)),
        int(round(y1 * height)),
        int(round(x2 * width)),
        int(round(y2 * height)),
    )


def pixel_polygon(norm_points: list[list[float]], width: int, height: int) -> np.ndarray:
    pts = []
    for x, y in norm_points:
        pts.append([int(round(x * width)), int(round(y * height))])
    return np.array(pts, dtype=np.int32)


def apply_osd_mask(frame: np.ndarray, masks: list[list[float]]) -> np.ndarray:
    masked = frame.copy()
    height, width = masked.shape[:2]
    for rect in masks:
        x1, y1, x2, y2 = pixel_rect(rect, width, height)
        masked[y1:y2, x1:x2] = 0
    return masked


def polygon_mask(shape_hw: tuple[int, int], polygon: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape_hw, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return mask


def prepare_detection_frame(
    frame: np.ndarray,
    osd_masks: list[list[float]],
    polygons: list[np.ndarray],
) -> np.ndarray:
    """OSD and non-lane pixels are blacked out before YOLO."""
    masked = apply_osd_mask(frame, osd_masks)
    keep = np.zeros(masked.shape[:2], dtype=np.uint8)
    for poly in polygons:
        keep = cv2.bitwise_or(keep, polygon_mask(masked.shape[:2], poly))
    return cv2.bitwise_and(masked, masked, mask=keep)


def box_center(xyxy: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def center_in_polygon(cx: float, cy: float, polygon: np.ndarray) -> bool:
    return cv2.pointPolygonTest(polygon, (float(cx), float(cy)), False) >= 0


def roi_area(polygon: np.ndarray) -> float:
    return max(1.0, float(cv2.contourArea(polygon)))


def mean_brightness(frame: np.ndarray, polygon: np.ndarray) -> float:
    mask = polygon_mask(frame.shape[:2], polygon)
    pixels = frame[mask > 0]
    if pixels.size == 0:
        return 0.0
    return float(pixels.mean())


def floor_bin(ts: datetime, bin_seconds: int) -> datetime:
    epoch = int(ts.timestamp())
    floored = epoch - (epoch % bin_seconds)
    return datetime.fromtimestamp(floored, tz=ts.tzinfo)


class CentroidTracker:
    """Nearest-center tracker for counting-line crossings."""

    def __init__(self, max_distance: float = 140.0, max_age: int = 8):
        self.max_distance = max_distance
        self.max_age = max_age
        self.next_id = 1
        self.tracks: dict[int, dict] = {}

    def update(self, centers: list[tuple[float, float]]) -> dict[int, tuple[float, float]]:
        if not self.tracks:
            for cx, cy in centers:
                self.tracks[self.next_id] = {"x": cx, "y": cy, "age": 0}
                self.next_id += 1
            return {tid: (t["x"], t["y"]) for tid, t in self.tracks.items()}

        unused = set(self.tracks)
        assigned: dict[int, tuple[float, float]] = {}
        for cx, cy in centers:
            best_id = None
            best_d = None
            for tid in unused:
                tr = self.tracks[tid]
                dist = ((cx - tr["x"]) ** 2 + (cy - tr["y"]) ** 2) ** 0.5
                if dist <= self.max_distance and (best_d is None or dist < best_d):
                    best_d = dist
                    best_id = tid
            if best_id is None:
                self.tracks[self.next_id] = {"x": cx, "y": cy, "age": 0}
                assigned[self.next_id] = (cx, cy)
                self.next_id += 1
            else:
                unused.remove(best_id)
                self.tracks[best_id] = {"x": cx, "y": cy, "age": 0}
                assigned[best_id] = (cx, cy)

        for tid in list(unused):
            self.tracks[tid]["age"] += 1
            if self.tracks[tid]["age"] > self.max_age:
                del self.tracks[tid]
        return assigned


def _orient(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]
) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(
    p: tuple[float, float],
    q: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> bool:
    o1 = _orient(p, q, a)
    o2 = _orient(p, q, b)
    o3 = _orient(a, b, p)
    o4 = _orient(a, b, q)
    if o1 == 0 or o2 == 0 or o3 == 0 or o4 == 0:
        return False
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


def count_line_crossings(
    prev_positions: dict[int, tuple[float, float]],
    curr_positions: dict[int, tuple[float, float]],
    line: tuple[tuple[float, float], tuple[float, float]],
    toward_camera: bool = True,
) -> int:
    """Count tracks whose path crossed the counting segment.

    Seoul-bound cars at 판교1 move toward the camera, so y increases.
    """
    crossed = 0
    for tid, curr in curr_positions.items():
        prev = prev_positions.get(tid)
        if prev is None:
            continue
        if not segments_intersect(prev, curr, line[0], line[1]):
            continue
        dy = curr[1] - prev[1]
        if toward_camera and dy <= 0:
            continue
        if not toward_camera and dy >= 0:
            continue
        crossed += 1
    return crossed


def counting_line_pixels(
    norm_line: list[list[float]], width: int, height: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    (x1, y1), (x2, y2) = norm_line
    return (
        (int(round(x1 * width)), int(round(y1 * height))),
        (int(round(x2 * width)), int(round(y2 * height))),
    )


def match_stopped(
    prev_centers: list[tuple[float, float]],
    curr_centers: list[tuple[float, float]],
    max_speed_px: float,
) -> int:
    if not prev_centers or not curr_centers:
        return 0
    used = set()
    stopped = 0
    for cx, cy in curr_centers:
        best_i = None
        best_d = None
        for i, (px, py) in enumerate(prev_centers):
            if i in used:
                continue
            dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            if best_d is None or dist < best_d:
                best_d = dist
                best_i = i
        if best_i is not None and best_d is not None and best_d <= max_speed_px:
            used.add(best_i)
            stopped += 1
    return stopped


def empty_bin_stats() -> dict:
    return {
        "ok_frames": 0,
        "seoul_count_sum": 0.0,
        "seoul_count_max": 0,
        "seoul_stopped_sum": 0,
        "seoul_car_sum": 0,
        "seoul_motorcycle_sum": 0,
        "seoul_bus_sum": 0,
        "seoul_truck_sum": 0,
        "seoul_crossings": 0,
        "busan_count_sum": 0.0,
        "brightness_sum": 0.0,
    }


def add_frame_to_bin(stats: dict, frame_row: dict) -> None:
    stats["ok_frames"] += 1
    stats["seoul_count_sum"] += frame_row["seoul_count"]
    stats["seoul_count_max"] = max(stats["seoul_count_max"], frame_row["seoul_count"])
    stats["seoul_stopped_sum"] += frame_row["seoul_stopped"]
    stats["seoul_car_sum"] += frame_row["seoul_car"]
    stats["seoul_motorcycle_sum"] += frame_row["seoul_motorcycle"]
    stats["seoul_bus_sum"] += frame_row["seoul_bus"]
    stats["seoul_truck_sum"] += frame_row["seoul_truck"]
    stats["seoul_crossings"] += int(frame_row.get("seoul_crossings") or 0)
    stats["busan_count_sum"] += frame_row["busan_count"]
    stats["brightness_sum"] += frame_row["brightness"]


def finalize_bin(
    bin_start: datetime,
    stats: dict,
    config: dict,
    seoul_polygon: np.ndarray,
    detector_name: str,
) -> dict:
    collection = config["collection"]
    bin_seconds = int(collection["bin_seconds"])
    sample_fps = float(collection["sample_fps"])
    coverage_min = float(collection["coverage_min"])
    expected = max(1, int(round(bin_seconds * sample_fps)))
    ok_frames = stats["ok_frames"]
    coverage = ok_frames / expected
    area = roi_area(seoul_polygon)
    # seoul_count_mean = 화면 안 차량 밀도(점유). VDS 통과 교통량이 아니다.
    mean_count = (stats["seoul_count_sum"] / ok_frames) if ok_frames else 0.0
    return {
        "bin_start": bin_start.isoformat(),
        "bin_end": (bin_start + timedelta(seconds=bin_seconds)).isoformat(),
        "expected_frames": expected,
        "ok_frames": ok_frames,
        "coverage": round(coverage, 4),
        "cctv_missing": int(coverage < coverage_min),
        "coverage_min_used": coverage_min,
        "seoul_count_mean": round(mean_count, 4),
        "seoul_count_max": stats["seoul_count_max"],
        "seoul_density_mean": round(mean_count / area * 10000, 6),
        "seoul_crossings": int(stats.get("seoul_crossings") or 0),
        "seoul_stopped_ratio": round(
            (stats["seoul_stopped_sum"] / stats["seoul_count_sum"])
            if stats["seoul_count_sum"]
            else 0.0,
            4,
        ),
        "seoul_car_mean": round((stats["seoul_car_sum"] / ok_frames) if ok_frames else 0.0, 4),
        "seoul_motorcycle_mean": round(
            (stats["seoul_motorcycle_sum"] / ok_frames) if ok_frames else 0.0, 4
        ),
        "seoul_bus_mean": round((stats["seoul_bus_sum"] / ok_frames) if ok_frames else 0.0, 4),
        "seoul_truck_mean": round((stats["seoul_truck_sum"] / ok_frames) if ok_frames else 0.0, 4),
        "busan_count_mean": round(
            (stats["busan_count_sum"] / ok_frames) if ok_frames else 0.0, 4
        ),
        "brightness_mean": round(
            (stats["brightness_sum"] / ok_frames) if ok_frames else 0.0, 2
        ),
        "is_night": int(
            ((stats["brightness_sum"] / ok_frames) if ok_frames else 0.0)
            < float(config["detection"]["night_brightness"])
        ),
        "detector": detector_name,
        "chunk_seconds_used": collection["chunk_seconds"],
        "sample_fps_used": sample_fps,
    }


def group_detections(
    boxes_xyxy: np.ndarray,
    class_ids: np.ndarray,
    seoul_polygon: np.ndarray,
    busan_polygon: np.ndarray,
    vehicle_classes: set[int],
) -> dict:
    seoul_centers = []
    seoul_boxes = []
    busan_centers = []
    busan_boxes = []
    seoul_counts = defaultdict(int)
    busan_count = 0
    seoul_total = 0
    for xyxy, cls in zip(boxes_xyxy, class_ids):
        cls_i = int(cls)
        if cls_i not in vehicle_classes:
            continue
        box = xyxy.tolist()
        cx, cy = box_center(box)
        if center_in_polygon(cx, cy, seoul_polygon):
            seoul_total += 1
            seoul_centers.append((cx, cy))
            seoul_boxes.append(box)
            seoul_counts[COCO_VEHICLE.get(cls_i, "car")] += 1
        elif center_in_polygon(cx, cy, busan_polygon):
            busan_count += 1
            busan_centers.append((cx, cy))
            busan_boxes.append(box)
    return {
        "seoul_count": seoul_total,
        "seoul_centers": seoul_centers,
        "seoul_boxes": seoul_boxes,
        "seoul_car": seoul_counts["car"],
        "seoul_motorcycle": seoul_counts["motorcycle"],
        "seoul_bus": seoul_counts["bus"],
        "seoul_truck": seoul_counts["truck"],
        "busan_count": busan_count,
        "busan_centers": busan_centers,
        "busan_boxes": busan_boxes,
    }


def draw_roi_overlay(
    frame: np.ndarray,
    seoul_polygon: np.ndarray,
    busan_polygon: np.ndarray,
    osd_masks: list[list[float]],
    seoul_count: int | None = None,
    busan_count: int | None = None,
    counting_line: tuple[tuple[int, int], tuple[int, int]] | None = None,
    seoul_boxes: list[list[float]] | None = None,
    busan_boxes: list[list[float]] | None = None,
    crossings: int | None = None,
) -> np.ndarray:
    out = frame.copy()
    height, width = out.shape[:2]
    overlay = out.copy()
    cv2.fillPoly(overlay, [seoul_polygon], (0, 180, 0))
    cv2.fillPoly(overlay, [busan_polygon], (0, 140, 255))
    out = cv2.addWeighted(overlay, 0.22, out, 0.78, 0)
    cv2.polylines(out, [seoul_polygon], True, (0, 220, 0), 2)
    cv2.polylines(out, [busan_polygon], True, (0, 165, 255), 2)
    if counting_line is not None:
        cv2.line(out, counting_line[0], counting_line[1], (0, 255, 255), 2)
        mid_x = (counting_line[0][0] + counting_line[1][0]) // 2
        mid_y = (counting_line[0][1] + counting_line[1][1]) // 2
        cv2.putText(
            out, "count", (mid_x - 24, mid_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1,
        )
        if crossings is not None:
            cv2.putText(
                out, f"+{crossings}", (mid_x - 16, mid_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1,
            )
    for box in seoul_boxes or []:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)
    for box in busan_boxes or []:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 165, 255), 2)
    for rect in osd_masks:
        x1, y1, x2, y2 = pixel_rect(rect, width, height)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 220), 1)
    seoul_label = "Seoul lanes"
    if seoul_count is not None:
        seoul_label = f"Seoul dens {seoul_count}"
    busan_label = "Busan lanes"
    if busan_count is not None:
        busan_label = f"Busan {busan_count}"
    sx, sy = int(seoul_polygon[:, 0].min()), int(seoul_polygon[:, 1].min())
    bx, by = int(busan_polygon[:, 0].min()), int(busan_polygon[:, 1].min())
    cv2.putText(out, seoul_label, (sx + 6, sy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)
    cv2.putText(out, busan_label, (bx + 6, by + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
    return out
