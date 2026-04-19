#!/usr/bin/env python3
"""
Filter tracked player rows to a homography-mapped infield ROI.

This script keeps tracks whose contact points are inside a cone-shaped region
bounded by the first/third-base foul-line rays and a configurable grass-line arc.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# MLB field dimensions from mlb_field_dimensions.pdf.
MLB_BASEPATH_FT = 90.0
MLB_HOME_TO_SECOND_FT = 127.28125  # 127' 3 3/8"
MLB_MOUND_CENTER_FROM_HOME_FT = 60.5  # 60' 6"
MLB_MOUND_DIAMETER_FT = 18.0
MLB_GRASSLINE_RADIUS_FT = 95.0
MLB_FOUL_LINE_BUFFER_FT = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter tracked players to an infield cone bounded by foul lines and grass arc."
    )
    parser.add_argument(
        "--players-csv",
        "--source-csv",
        dest="source_csv",
        type=Path,
        required=True,
        help="Input tracks_with_sam_contact.csv from get_ground_contact.py (or get_players.py stage output).",
    )
    parser.add_argument(
        "--source-jsonl",
        type=Path,
        default=None,
        help="Optional input JSONL. If missing, output JSONL is generated from CSV rows.",
    )
    parser.add_argument(
        "--base-meta-json",
        type=Path,
        required=True,
        help="JSON containing base_points_px with keys home/first/second/third.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for filtered artifacts.",
    )
    parser.add_argument(
        "--basepath-ft",
        type=float,
        default=MLB_BASEPATH_FT,
        help="Feet per 1.0 normalized homography unit.",
    )
    parser.add_argument(
        "--grass-radius-ft",
        type=float,
        default=MLB_GRASSLINE_RADIUS_FT,
        help="Grass-line radius in feet from the configured origin.",
    )
    parser.add_argument(
        "--mound-distance-ft",
        type=float,
        default=MLB_MOUND_CENTER_FROM_HOME_FT,
        help="Distance from home plate to mound center in feet.",
    )
    parser.add_argument(
        "--foul-line-buffer-ft",
        type=float,
        default=MLB_FOUL_LINE_BUFFER_FT,
        help="Tolerance buffer (feet) outside first/third foul lines in field coordinates.",
    )
    parser.add_argument(
        "--grass-radius-origin",
        type=str,
        choices=["home", "mound"],
        default="mound",
        help="Origin for grass-line radius measurement.",
    )
    parser.add_argument(
        "--track-min-inside-ratio",
        type=float,
        default=0.35,
        help="Minimum inside fraction for a track to be kept.",
    )
    parser.add_argument(
        "--track-min-inside-count",
        type=int,
        default=15,
        help="Minimum inside detections for a track to be kept.",
    )
    parser.add_argument(
        "--movement-min-path-ft",
        type=float,
        default=70.0,
        help="Rescue track IDs with at least this much total movement path length in field feet.",
    )
    parser.add_argument(
        "--movement-min-net-displacement-ft",
        type=float,
        default=35.0,
        help="Rescue track IDs with at least this much start-to-end displacement in field feet.",
    )
    parser.add_argument(
        "--movement-min-detections",
        type=int,
        default=25,
        help="Minimum detections required for movement-based track rescue.",
    )
    parser.add_argument(
        "--movement-max-step-ft",
        type=float,
        default=25.0,
        help="Ignore per-frame jumps above this distance when summing movement path (reduces ID-switch spikes).",
    )
    parser.add_argument(
        "--movement-min-inside-count",
        type=int,
        default=1,
        help="Require at least this many inside-ROI detections before movement rescue can apply.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="FPS for output overlay video.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable output overlay video generation.",
    )

    # Backward-compatible args from prior buffer/cone/rectangle modes.
    parser.add_argument("--diamond-buffer-ft", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--third-to-grass-ft", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--first-to-grass-ft", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--filter-script", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--roi-mode", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--u-min", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--u-max", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--v-min", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--v-max", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cone-inner-radius-ft", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cone-radius-ft", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cone-cap-style", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cone-theta-min-deg", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cone-theta-max-deg", type=float, default=None, help=argparse.SUPPRESS)

    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def maybe_float(value: str | None) -> float | None:
    if value is None:
        return None
    s = value.strip()
    if not s or s.lower() == "nan":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def maybe_int(value: str | None) -> int | None:
    if value is None:
        return None
    s = value.strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def load_base_points(base_meta_json: Path) -> dict[str, tuple[float, float]]:
    with base_meta_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    points = data.get("base_points_px", data)
    needed = ["home", "first", "second", "third"]
    missing = [k for k in needed if k not in points]
    if missing:
        raise ValueError(f"Missing base points {missing} in {base_meta_json}")

    out: dict[str, tuple[float, float]] = {}
    for key in needed:
        xy = points[key]
        out[key] = (float(xy[0]), float(xy[1]))
    return out


def build_homography(
    base_points: dict[str, tuple[float, float]], basepath_ft: float
) -> tuple[np.ndarray, np.ndarray]:
    src = np.float32(
        [
            base_points["home"],
            base_points["first"],
            base_points["second"],
            base_points["third"],
        ]
    )
    dst = np.float32(
        [
            [0.0, 0.0],
            [basepath_ft, 0.0],
            [basepath_ft, basepath_ft],
            [0.0, basepath_ft],
        ]
    )
    h_img_to_field = cv2.getPerspectiveTransform(src, dst)
    h_field_to_img = cv2.getPerspectiveTransform(dst, src)
    return h_img_to_field, h_field_to_img


def transform_point(h: np.ndarray, x: float, y: float) -> tuple[float, float]:
    pt = np.array([[[x, y]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, h)[0, 0]
    return float(out[0]), float(out[1])


def resolve_frame_path(raw: str) -> Path:
    p = Path(raw)
    if p.exists():
        return p
    cwd_p = Path.cwd() / raw
    if cwd_p.exists():
        return cwd_p
    return p


def color_for_track(track_id: int) -> tuple[int, int, int]:
    return (
        int((37 * track_id + 73) % 256),
        int((17 * track_id + 191) % 256),
        int((29 * track_id + 47) % 256),
    )


def build_infield_cone_polygon_field(
    center_field_ft: tuple[float, float],
    radius_ft: float,
    foul_line_buffer_ft: float = 0.0,
    arc_steps: int = 180,
) -> np.ndarray:
    radius_ft = max(0.0, float(radius_ft))
    steps = max(8, int(arc_steps))
    line_buffer = max(0.0, float(foul_line_buffer_ft))
    line_x = -line_buffer
    line_y = -line_buffer
    if radius_ft <= 0.0:
        return np.asarray([[line_x, line_y]], dtype=np.float32)

    cx, cy = float(center_field_ft[0]), float(center_field_ft[1])
    dx_sq = radius_ft * radius_ft - (line_y - cy) * (line_y - cy)
    dy_sq = radius_ft * radius_ft - (line_x - cx) * (line_x - cx)
    if dx_sq < 0.0 or dy_sq < 0.0:
        raise ValueError(
            "Grass radius does not intersect foul-line rays. Increase radius or reduce mound distance/buffer."
        )

    dx = math.sqrt(max(0.0, dx_sq))
    dy = math.sqrt(max(0.0, dy_sq))

    x_on_first_ray = cx + dx
    y_on_third_ray = cy + dy

    theta_start = math.atan2(line_y - cy, x_on_first_ray - cx)
    theta_end = math.atan2(y_on_third_ray - cy, line_x - cx)
    if theta_end <= theta_start:
        theta_end += 2.0 * math.pi

    thetas = np.linspace(theta_start, theta_end, steps + 1, dtype=np.float32)

    points: list[tuple[float, float]] = [(line_x, line_y), (x_on_first_ray, line_y)]
    for theta in thetas[1:-1]:
        points.append((cx + radius_ft * math.cos(float(theta)), cy + radius_ft * math.sin(float(theta))))
    points.append((line_x, y_on_third_ray))
    return np.asarray(points, dtype=np.float32)


def mound_center_field(mound_distance_ft: float) -> tuple[float, float]:
    d = max(0.0, float(mound_distance_ft))
    diag = math.sqrt(2.0)
    return d / diag, d / diag


def draw_overlay_frame(
    frame: np.ndarray,
    rows: list[dict[str, Any]],
    frame_idx: int,
    roi_img_pts: np.ndarray,
    roi_label: str,
    mound_img_xy: tuple[int, int] | None,
) -> None:
    cv2.polylines(frame, [roi_img_pts], True, (0, 200, 255), 2)
    label_x = int(np.min(roi_img_pts[:, 0])) + 6
    label_y = max(16, int(np.min(roi_img_pts[:, 1])) - 8)
    cv2.putText(
        frame,
        roi_label,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 200, 255),
        1,
        cv2.LINE_AA,
    )

    if mound_img_xy is not None:
        cv2.circle(frame, mound_img_xy, 8, (80, 255, 80), 2)
        cv2.putText(
            frame,
            "Mound 60.5ft",
            (mound_img_xy[0] + 10, mound_img_xy[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (80, 255, 80),
            1,
            cv2.LINE_AA,
        )

    for row in rows:
        track_id = int(row["track_id"])
        conf = float(row["det_conf"])
        x1 = int(round(float(row["box_x1"])))
        y1 = int(round(float(row["box_y1"])))
        x2 = int(round(float(row["box_x2"])))
        y2 = int(round(float(row["box_y2"])))
        cx = int(round(float(row["sam_contact_x"])))
        cy = int(round(float(row["sam_contact_y"])))
        color = color_for_track(track_id)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        label = f"id {track_id} {conf:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        f"frame {frame_idx}  kept {len(rows)}",
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_overlay_video(
    filtered_rows: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    h_field_to_img: np.ndarray,
    out_path: Path,
    fps: float,
    roi_polygon_field_ft: np.ndarray,
    roi_label: str,
    mound_center_field_ft: tuple[float, float] | None = None,
) -> None:
    by_frame_filtered: dict[int, list[dict[str, Any]]] = defaultdict(list)
    frame_path_by_index: dict[int, str] = {}
    frame_indices: set[int] = set()

    for row in all_rows:
        fi = int(row["frame_index"])
        frame_indices.add(fi)
        frame_path_by_index[fi] = str(row["image_path"])

    for row in filtered_rows:
        fi = int(row["frame_index"])
        by_frame_filtered[fi].append(row)

    if not frame_path_by_index:
        return

    first_idx = min(frame_path_by_index.keys())
    first_path = resolve_frame_path(frame_path_by_index[first_idx])
    first_frame = cv2.imread(str(first_path))
    if first_frame is None:
        raise RuntimeError(f"Could not read first frame for video: {first_path}")
    h_img, w_img = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w_img, h_img))

    roi_field_pts = roi_polygon_field_ft.astype(np.float32).reshape(-1, 1, 2)
    roi_img_pts = cv2.perspectiveTransform(roi_field_pts, h_field_to_img).reshape(-1, 2)
    roi_img_pts_i = np.rint(roi_img_pts).astype(np.int32)

    mound_img_xy: tuple[int, int] | None = None
    if mound_center_field_ft is not None:
        mound_field_pt = np.array([[mound_center_field_ft]], dtype=np.float32)
        mound_img = cv2.perspectiveTransform(mound_field_pt, h_field_to_img)[0, 0]
        mound_img_xy = (int(round(float(mound_img[0]))), int(round(float(mound_img[1]))))

    for frame_idx in sorted(frame_indices):
        frame_path = resolve_frame_path(frame_path_by_index[frame_idx])
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        rows = by_frame_filtered.get(frame_idx, [])
        draw_overlay_frame(frame, rows, frame_idx, roi_img_pts_i, roi_label, mound_img_xy)
        writer.write(frame)

    writer.release()


def draw_overlay_image(
    filtered_rows: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    h_field_to_img: np.ndarray,
    out_path: Path,
    roi_polygon_field_ft: np.ndarray,
    roi_label: str,
    mound_center_field_ft: tuple[float, float] | None = None,
) -> int | None:
    by_frame_filtered: dict[int, list[dict[str, Any]]] = defaultdict(list)
    frame_path_by_index: dict[int, str] = {}
    for row in all_rows:
        fi = int(row["frame_index"])
        frame_path_by_index[fi] = str(row["image_path"])
    for row in filtered_rows:
        fi = int(row["frame_index"])
        by_frame_filtered[fi].append(row)
    if not frame_path_by_index:
        return None

    if by_frame_filtered:
        target_idx = min(
            by_frame_filtered.keys(),
            key=lambda idx: (-len(by_frame_filtered[idx]), idx),
        )
    else:
        target_idx = min(frame_path_by_index.keys())

    frame_path = resolve_frame_path(frame_path_by_index[target_idx])
    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise RuntimeError(f"Could not read frame for overlay image: {frame_path}")

    roi_field_pts = roi_polygon_field_ft.astype(np.float32).reshape(-1, 1, 2)
    roi_img_pts = cv2.perspectiveTransform(roi_field_pts, h_field_to_img).reshape(-1, 2)
    roi_img_pts_i = np.rint(roi_img_pts).astype(np.int32)

    mound_img_xy: tuple[int, int] | None = None
    if mound_center_field_ft is not None:
        mound_field_pt = np.array([[mound_center_field_ft]], dtype=np.float32)
        mound_img = cv2.perspectiveTransform(mound_field_pt, h_field_to_img)[0, 0]
        mound_img_xy = (int(round(float(mound_img[0]))), int(round(float(mound_img[1]))))

    rows = by_frame_filtered.get(target_idx, [])
    draw_overlay_frame(frame, rows, target_idx, roi_img_pts_i, roi_label, mound_img_xy)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), frame):
        raise RuntimeError(f"Could not write overlay image: {out_path}")
    return target_idx


def main() -> None:
    args = parse_args()

    if not args.source_csv.exists():
        raise FileNotFoundError(f"players csv not found: {args.source_csv}")
    if not args.base_meta_json.exists():
        raise FileNotFoundError(f"base meta json not found: {args.base_meta_json}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_rows = read_csv_rows(args.source_csv)
    if not source_rows:
        raise RuntimeError(f"Source CSV has no rows: {args.source_csv}")

    base_points = load_base_points(args.base_meta_json)
    h_img_to_field, h_field_to_img = build_homography(base_points, args.basepath_ft)

    # Legacy support: map older buffer args to foul-line tolerance.
    grass_radius_ft = max(0.0, float(args.grass_radius_ft))
    foul_line_buffer_ft = max(0.0, float(args.foul_line_buffer_ft))
    if args.diamond_buffer_ft is not None:
        foul_line_buffer_ft = max(0.0, float(args.diamond_buffer_ft))
    if args.third_to_grass_ft is not None:
        foul_line_buffer_ft = max(0.0, float(args.third_to_grass_ft))

    mound_distance_ft = max(0.0, float(args.mound_distance_ft))
    mound_center_field_ft = mound_center_field(mound_distance_ft)

    if args.grass_radius_origin == "mound":
        roi_center_field_ft = mound_center_field_ft
    else:
        roi_center_field_ft = (0.0, 0.0)

    roi_polygon_field_ft = build_infield_cone_polygon_field(
        roi_center_field_ft, grass_radius_ft, foul_line_buffer_ft
    )
    roi_contour = roi_polygon_field_ft.astype(np.float32).reshape(-1, 1, 2)
    roi_label = f"Infield Cone R={grass_radius_ft:.1f}ft @{args.grass_radius_origin}"

    track_stats: dict[int, dict[str, int]] = defaultdict(lambda: {"total": 0, "inside": 0})
    track_motion_points: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    enriched_rows: list[dict[str, Any]] = []

    for row in source_rows:
        row2: dict[str, Any] = dict(row)
        track_id = maybe_int(row2.get("track_id"))
        if track_id is None:
            continue

        cx = maybe_float(row2.get("sam_contact_x"))
        cy = maybe_float(row2.get("sam_contact_y"))
        if cx is None or cy is None:
            cx = maybe_float(row2.get("bbox_bottom_center_x"))
            cy = maybe_float(row2.get("bbox_bottom_center_y"))
        if cx is None or cy is None:
            continue

        frame_idx = maybe_int(row2.get("frame_index"))

        field_x_ft, field_y_ft = transform_point(h_img_to_field, cx, cy)
        u = field_x_ft / args.basepath_ft
        v = field_y_ft / args.basepath_ft
        inside = cv2.pointPolygonTest(roi_contour, (field_x_ft, field_y_ft), False) >= 0.0

        track_stats[track_id]["total"] += 1
        if inside:
            track_stats[track_id]["inside"] += 1
        if frame_idx is not None:
            track_motion_points[track_id].append((frame_idx, field_x_ft, field_y_ft))

        row2["roi_u"] = u
        row2["roi_v"] = v
        row2["field_x_ft"] = field_x_ft
        row2["field_y_ft"] = field_y_ft
        row2["field_r_ft"] = math.hypot(field_x_ft, field_y_ft)
        row2["field_theta_deg"] = math.degrees(math.atan2(field_y_ft, field_x_ft))
        row2["roi_inside"] = int(inside)
        row2["movement_path_length_ft"] = ""
        row2["movement_net_displacement_ft"] = ""
        row2["movement_detections"] = ""
        row2["movement_rescue_track"] = 0
        row2["keep_reason"] = "dropped"
        enriched_rows.append(row2)

    roi_keep_tracks: set[int] = set()
    for track_id, s in track_stats.items():
        total = s["total"]
        inside = s["inside"]
        ratio = float(inside) / float(total) if total > 0 else 0.0
        if inside >= args.track_min_inside_count and ratio >= args.track_min_inside_ratio:
            roi_keep_tracks.add(track_id)

    movement_min_path_ft = max(0.0, float(args.movement_min_path_ft))
    movement_min_net_displacement_ft = max(0.0, float(args.movement_min_net_displacement_ft))
    movement_min_detections = max(1, int(args.movement_min_detections))
    movement_max_step_ft = max(0.0, float(args.movement_max_step_ft))
    movement_min_inside_count = max(0, int(args.movement_min_inside_count))

    movement_metrics_by_track: dict[int, dict[str, float | int]] = {}
    movement_rescue_tracks: set[int] = set()
    for track_id, points in track_motion_points.items():
        points_sorted = sorted(points, key=lambda t: t[0])
        detections = len(points_sorted)

        path_length_ft = 0.0
        max_step_ft = 0.0
        valid_steps = 0
        ignored_steps = 0
        for i in range(1, detections):
            _, x0, y0 = points_sorted[i - 1]
            _, x1, y1 = points_sorted[i]
            step_ft = math.hypot(x1 - x0, y1 - y0)
            max_step_ft = max(max_step_ft, step_ft)
            if movement_max_step_ft > 0.0 and step_ft > movement_max_step_ft:
                ignored_steps += 1
                continue
            path_length_ft += step_ft
            valid_steps += 1

        if detections >= 2:
            _, x_start, y_start = points_sorted[0]
            _, x_end, y_end = points_sorted[-1]
            net_displacement_ft = math.hypot(x_end - x_start, y_end - y_start)
        else:
            net_displacement_ft = 0.0

        inside_hits = track_stats[track_id]["inside"]
        should_rescue = (
            detections >= movement_min_detections
            and path_length_ft >= movement_min_path_ft
            and net_displacement_ft >= movement_min_net_displacement_ft
            and inside_hits >= movement_min_inside_count
        )
        if should_rescue:
            movement_rescue_tracks.add(track_id)

        movement_metrics_by_track[track_id] = {
            "path_length_ft": float(path_length_ft),
            "net_displacement_ft": float(net_displacement_ft),
            "detections": int(detections),
            "inside_hits": int(inside_hits),
            "max_step_ft": float(max_step_ft),
            "valid_steps": int(valid_steps),
            "ignored_steps": int(ignored_steps),
        }

    for row in enriched_rows:
        track_id = int(row["track_id"])
        metrics = movement_metrics_by_track.get(track_id)
        if not metrics:
            continue
        row["movement_path_length_ft"] = float(metrics["path_length_ft"])
        row["movement_net_displacement_ft"] = float(metrics["net_displacement_ft"])
        row["movement_detections"] = int(metrics["detections"])
        row["movement_rescue_track"] = int(track_id in movement_rescue_tracks)

    keep_tracks = set(roi_keep_tracks) | set(movement_rescue_tracks)

    roi_inside_rows_kept = 0
    movement_rows_kept = 0
    movement_outside_rows_kept = 0
    movement_only_track_ids = movement_rescue_tracks - roi_keep_tracks

    filtered_rows: list[dict[str, Any]] = []
    for row in enriched_rows:
        track_id = int(row["track_id"])
        inside = int(row["roi_inside"]) == 1

        if track_id in roi_keep_tracks and inside:
            row["keep_reason"] = "roi_inside"
            roi_inside_rows_kept += 1
            filtered_rows.append(row)
            continue

        if track_id in movement_rescue_tracks:
            row["keep_reason"] = "movement_rescue"
            movement_rows_kept += 1
            if not inside:
                movement_outside_rows_kept += 1
            filtered_rows.append(row)

    out_csv = args.output_dir / "tracks_with_sam_contact_infield_custom.csv"
    out_jsonl = args.output_dir / "tracks_with_sam_contact_infield_custom.jsonl"
    out_summary = args.output_dir / "track_summary_infield_custom.csv"
    out_overlay_img = args.output_dir / "tracks_overlay_infield_custom.png"
    out_video = args.output_dir / "tracks_overlay_infield_custom.mp4"
    out_meta = args.output_dir / "run_meta_infield_custom.json"

    write_csv_rows(out_csv, filtered_rows)
    write_jsonl_rows(out_jsonl, filtered_rows)

    by_track: dict[int, dict[str, float | int | None]] = defaultdict(
        lambda: {
            "detections": 0,
            "first_frame_index": None,
            "last_frame_index": None,
            "mean_conf_sum": 0.0,
            "contact_points": 0,
            "mask_area_sum": 0.0,
        }
    )
    for row in filtered_rows:
        track_id = int(row["track_id"])
        frame_idx = int(row["frame_index"])
        conf = float(row["det_conf"])
        mask_area = float(row["sam_mask_area_px"])
        cx = maybe_float(str(row.get("sam_contact_x", "")))
        cy = maybe_float(str(row.get("sam_contact_y", "")))
        s = by_track[track_id]
        s["detections"] += 1
        s["mean_conf_sum"] += conf
        s["mask_area_sum"] += mask_area
        if cx is not None and cy is not None:
            s["contact_points"] += 1
        if s["first_frame_index"] is None or frame_idx < int(s["first_frame_index"]):
            s["first_frame_index"] = frame_idx
        if s["last_frame_index"] is None or frame_idx > int(s["last_frame_index"]):
            s["last_frame_index"] = frame_idx

    summary_rows: list[dict[str, Any]] = []
    for track_id in sorted(by_track.keys()):
        s = by_track[track_id]
        detections = int(s["detections"])
        summary_rows.append(
            {
                "track_id": track_id,
                "detections": detections,
                "first_frame_index": int(s["first_frame_index"]),
                "last_frame_index": int(s["last_frame_index"]),
                "mean_conf": float(s["mean_conf_sum"]) / detections if detections else float("nan"),
                "mean_mask_area_px": float(s["mask_area_sum"]) / detections if detections else float("nan"),
                "contact_points": int(s["contact_points"]),
            }
        )
    write_csv_rows(out_summary, summary_rows)

    overlay_frame_index = draw_overlay_image(
        filtered_rows=filtered_rows,
        all_rows=enriched_rows,
        h_field_to_img=h_field_to_img,
        out_path=out_overlay_img,
        roi_polygon_field_ft=roi_polygon_field_ft,
        roi_label=roi_label,
        mound_center_field_ft=mound_center_field_ft,
    )

    if not args.no_video:
        draw_overlay_video(
            filtered_rows=filtered_rows,
            all_rows=enriched_rows,
            h_field_to_img=h_field_to_img,
            out_path=out_video,
            fps=args.fps,
            roi_polygon_field_ft=roi_polygon_field_ft,
            roi_label=roi_label,
            mound_center_field_ft=mound_center_field_ft,
        )

    legacy_ignored: dict[str, Any] = {}
    for k in [
        "first_to_grass_ft",
        "filter_script",
        "roi_mode",
        "u_min",
        "u_max",
        "v_min",
        "v_max",
        "cone_inner_radius_ft",
        "cone_radius_ft",
        "cone_cap_style",
        "cone_theta_min_deg",
        "cone_theta_max_deg",
    ]:
        v = getattr(args, k)
        if v is not None:
            legacy_ignored[k] = v

    meta = {
        "source_csv": str(args.source_csv),
        "source_jsonl": str(args.source_jsonl) if args.source_jsonl else None,
        "base_meta_json": str(args.base_meta_json),
        "base_points_px": base_points,
        "field_coordinate_system": {
            "units": "feet",
            "origin": "home",
            "x_axis": "home_to_first",
            "y_axis": "home_to_third",
        },
        "mlb_dimensions_ft": {
            "basepath": MLB_BASEPATH_FT,
            "home_to_second": MLB_HOME_TO_SECOND_FT,
            "mound_center_from_home": MLB_MOUND_CENTER_FROM_HOME_FT,
            "mound_diameter": MLB_MOUND_DIAMETER_FT,
            "grassline_radius_from_home": MLB_GRASSLINE_RADIUS_FT,
            "base_to_grassline": MLB_GRASSLINE_RADIUS_FT - MLB_BASEPATH_FT,
            "foul_line_buffer_default": MLB_FOUL_LINE_BUFFER_FT,
        },
        "method": "homography_infield_cone_with_track_consistency",
        "roi_mode": "infield_cone",
        "roi_label": roi_label,
        "infield_cone_roi": {
            "basepath_ft": args.basepath_ft,
            "grass_radius_ft": grass_radius_ft,
            "grass_radius_origin": args.grass_radius_origin,
            "roi_center_field_ft": [float(roi_center_field_ft[0]), float(roi_center_field_ft[1])],
            "foul_line_buffer_ft": foul_line_buffer_ft,
            "mound_distance_ft": mound_distance_ft,
            "mound_center_field_ft": [
                float(mound_center_field_ft[0]),
                float(mound_center_field_ft[1]),
            ],
            "polygon_field_ft": [[float(x), float(y)] for x, y in roi_polygon_field_ft.tolist()],
        },
        "track_consistency": {
            "min_inside_ratio": args.track_min_inside_ratio,
            "min_inside_count": args.track_min_inside_count,
        },
        "movement_rescue": {
            "min_path_ft": movement_min_path_ft,
            "min_net_displacement_ft": movement_min_net_displacement_ft,
            "min_detections": movement_min_detections,
            "max_step_ft": movement_max_step_ft,
            "min_inside_count": movement_min_inside_count,
            "tracks_with_motion": len(movement_metrics_by_track),
            "rescued_track_ids": sorted(int(t) for t in movement_rescue_tracks),
            "rescued_track_count": len(movement_rescue_tracks),
            "rescued_only_track_count": len(movement_only_track_ids),
            "roi_track_count": len(roi_keep_tracks),
            "union_track_count": len(keep_tracks),
            "roi_inside_rows_kept": roi_inside_rows_kept,
            "movement_rows_kept": movement_rows_kept,
            "movement_outside_rows_kept": movement_outside_rows_kept,
        },
        "legacy_args_ignored": legacy_ignored or None,
        "input_rows": len(source_rows),
        "input_tracks": len({int(r["track_id"]) for r in source_rows}),
        "rows_with_contact_processed": len(enriched_rows),
        "candidate_tracks": len(track_stats),
        "kept_tracks": len(keep_tracks),
        "filtered_rows": len(filtered_rows),
        "output_csv": str(out_csv),
        "output_jsonl": str(out_jsonl),
        "output_track_summary_csv": str(out_summary),
        "output_overlay_image": str(out_overlay_img),
        "overlay_image_frame_index": overlay_frame_index,
        "output_video": str(out_video) if not args.no_video else None,
    }
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    stage_meta = {
        "players_csv": str(args.source_csv),
        "source_csv": str(args.source_csv),
        "base_meta_json": str(args.base_meta_json),
        "filter_script": str(Path(__file__).resolve()),
        "filter_command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "outputs": {
            "filtered_csv": str(out_csv),
            "filtered_meta": str(out_meta),
            "filtered_jsonl": str(out_jsonl),
            "filtered_summary": str(out_summary),
            "filtered_overlay_image": str(out_overlay_img),
            "filtered_video": None if args.no_video else str(out_video),
        },
    }
    stage_meta_path = args.output_dir / "run_meta_filter_region.json"
    stage_meta_path.write_text(json.dumps(stage_meta, indent=2), encoding="utf-8")

    print("Infield cone filter complete.")
    print(f"Grass radius:  {grass_radius_ft:.2f} ft")
    print(f"Radius origin: {args.grass_radius_origin}")
    print(f"Foul buffer:   {foul_line_buffer_ft:.2f} ft")
    print(f"Mound dist:    {mound_distance_ft:.2f} ft")
    print(f"Input rows:    {meta['input_rows']}")
    print(f"Input tracks:  {meta['input_tracks']}")
    print(f"Kept tracks:   {meta['kept_tracks']} (roi={len(roi_keep_tracks)} movement={len(movement_rescue_tracks)})")
    print(f"Filtered rows: {meta['filtered_rows']}")
    print(f"Rows kept by ROI-inside: {roi_inside_rows_kept}")
    print(f"Rows kept by movement:   {movement_rows_kept} (outside={movement_outside_rows_kept})")
    print(f"CSV:           {out_csv}")
    print(f"JSONL:         {out_jsonl}")
    print(f"Summary:       {out_summary}")
    print(f"Overlay img:   {out_overlay_img}")
    if not args.no_video:
        print(f"Video:         {out_video}")
    print(f"Meta:          {out_meta}")


if __name__ == "__main__":
    main()
