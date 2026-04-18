#!/usr/bin/env python3
"""
Compute per-player velocity from ROI-filtered tracks and render bird's-eye contacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

MLB_BASEPATH_FT = 90.0
MLB_MOUND_CENTER_FROM_HOME_FT = 60.5
MPH_PER_FT_S = 0.6818181818181818

VELOCITY_COLUMNS = [
    "velocity_delta_frames",
    "velocity_delta_time_s",
    "velocity_dx_ft",
    "velocity_dy_ft",
    "velocity_step_ft",
    "velocity_step_valid",
    "velocity_vx_ft_s",
    "velocity_vy_ft_s",
    "velocity_ft_s",
    "velocity_mph",
    "velocity_ft_s_smooth",
    "velocity_mph_smooth",
]

FIELD_SMOOTH_COLUMNS = [
    "field_x_ft_raw",
    "field_y_ft_raw",
    "field_x_ft_smooth",
    "field_y_ft_smooth",
    "field_point_smoothed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-track velocity and bird's-eye ground-contact outputs from ROI-filtered tracks."
    )
    parser.add_argument(
        "--tracks-csv",
        "--source-csv",
        dest="source_csv",
        type=Path,
        required=True,
        help="Input CSV from filter_region.py (tracks_with_sam_contact_infield_custom.csv).",
    )
    parser.add_argument(
        "--filter-meta-json",
        type=Path,
        default=None,
        help="Optional run_meta_infield_custom.json for ROI polygon and field metadata.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for velocity artifacts.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="FPS used for velocity time deltas and optional bird's-eye video.",
    )
    parser.add_argument(
        "--basepath-ft",
        type=float,
        default=MLB_BASEPATH_FT,
        help="Basepath length in feet for roi_u/roi_v fallback conversion.",
    )
    parser.add_argument(
        "--max-step-ft",
        type=float,
        default=20.0,
        help="Ignore per-step velocity samples above this distance (helps suppress ID-switch spikes).",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Moving-average window size (in valid velocity samples) for smoothed speed columns.",
    )
    parser.add_argument(
        "--field-smoothing-method",
        type=str,
        default="kalman",
        choices=["kalman", "sliding_median"],
        help="Method used to smooth per-track field points before velocity computation.",
    )
    parser.add_argument(
        "--field-smooth-window",
        type=int,
        default=7,
        help="Sliding median window (samples) used when --field-smoothing-method=sliding_median.",
    )
    parser.add_argument(
        "--field-kalman-process-var",
        type=float,
        default=0.25,
        help="Kalman process noise variance for field-point smoothing when --field-smoothing-method=kalman.",
    )
    parser.add_argument(
        "--field-kalman-measurement-var",
        type=float,
        default=1.0,
        help="Kalman measurement noise variance for field-point smoothing when --field-smoothing-method=kalman.",
    )
    parser.add_argument(
        "--birdseye-scale",
        type=float,
        default=6.0,
        help="Pixels per field foot for bird's-eye rendering.",
    )
    parser.add_argument(
        "--birdseye-pad-ft",
        type=float,
        default=15.0,
        help="Padding around rendered field extents in feet.",
    )
    parser.add_argument(
        "--trail-length",
        type=int,
        default=45,
        help="Max recent points shown per track in bird's-eye video.",
    )
    parser.add_argument(
        "--trail-max-gap-frames",
        type=int,
        default=45,
        help="Hide tracks from video when they have not been observed for this many frames.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable bird's-eye video generation.",
    )
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


def resolve_frame_path(raw: str) -> Path:
    p = Path(raw)
    if p.exists():
        return p
    cwd_p = Path.cwd() / raw
    if cwd_p.exists():
        return cwd_p
    return p


def maybe_float(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def maybe_int(value: str | int | float | None) -> int | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def color_for_track(track_id: int) -> tuple[int, int, int]:
    return (
        int((37 * track_id + 73) % 256),
        int((17 * track_id + 191) % 256),
        int((29 * track_id + 47) % 256),
    )


def load_json_if_exists(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def extract_field_xy(row: dict[str, Any], basepath_ft: float) -> tuple[float | None, float | None]:
    x = maybe_float(row.get("field_x_ft"))
    y = maybe_float(row.get("field_y_ft"))
    if x is not None and y is not None:
        return x, y
    u = maybe_float(row.get("roi_u"))
    v = maybe_float(row.get("roi_v"))
    if u is not None and v is not None:
        return u * basepath_ft, v * basepath_ft
    return None, None


def build_track_observations(
    rows: list[dict[str, Any]],
    basepath_ft: float,
) -> dict[int, list[dict[str, float | int]]]:
    by_track: dict[int, list[dict[str, float | int]]] = defaultdict(list)
    for row_idx, row in enumerate(rows):
        track_id = maybe_int(row.get("track_id"))
        frame_idx = maybe_int(row.get("frame_index"))
        if track_id is None or frame_idx is None:
            continue

        x_ft, y_ft = extract_field_xy(row, basepath_ft=basepath_ft)
        if x_ft is None or y_ft is None:
            continue

        by_track[track_id].append(
            {
                "row_idx": int(row_idx),
                "frame_idx": int(frame_idx),
                "x_ft": float(x_ft),
                "y_ft": float(y_ft),
            }
        )

    for obs in by_track.values():
        obs.sort(key=lambda d: (int(d["frame_idx"]), int(d["row_idx"])))
    return by_track


def kalman_smooth_track_positions(
    xs: np.ndarray,
    ys: np.ndarray,
    frame_idxs: np.ndarray,
    process_var: float,
    measurement_var: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(xs.size)
    if n <= 1:
        return xs.astype(np.float32), ys.astype(np.float32)

    proc_var = max(1e-6, float(process_var))
    meas_var = max(1e-6, float(measurement_var))

    h = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    r = np.eye(2, dtype=np.float64) * meas_var
    ident = np.eye(4, dtype=np.float64)

    x_filt = np.zeros((n, 4), dtype=np.float64)
    p_filt = np.zeros((n, 4, 4), dtype=np.float64)
    x_pred = np.zeros((n, 4), dtype=np.float64)
    p_pred = np.zeros((n, 4, 4), dtype=np.float64)
    f_hist = np.zeros((n, 4, 4), dtype=np.float64)

    state = np.array([float(xs[0]), float(ys[0]), 0.0, 0.0], dtype=np.float64)
    cov = np.diag([meas_var, meas_var, 10.0 * meas_var, 10.0 * meas_var]).astype(np.float64)

    x_filt[0] = state
    p_filt[0] = cov
    x_pred[0] = state
    p_pred[0] = cov
    f_hist[0] = np.eye(4, dtype=np.float64)

    for i in range(1, n):
        dt_frames = int(frame_idxs[i]) - int(frame_idxs[i - 1])
        dt = float(max(1, dt_frames))
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2

        f = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        q = proc_var * np.array(
            [
                [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
                [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
                [dt3 / 2.0, 0.0, dt2, 0.0],
                [0.0, dt3 / 2.0, 0.0, dt2],
            ],
            dtype=np.float64,
        )

        pred_state = f @ state
        pred_cov = f @ cov @ f.T + q

        z = np.array([float(xs[i]), float(ys[i])], dtype=np.float64)
        innovation = z - (h @ pred_state)
        s = h @ pred_cov @ h.T + r
        k = pred_cov @ h.T @ np.linalg.pinv(s)

        state = pred_state + k @ innovation
        cov = (ident - (k @ h)) @ pred_cov
        cov = 0.5 * (cov + cov.T)

        x_pred[i] = pred_state
        p_pred[i] = pred_cov
        x_filt[i] = state
        p_filt[i] = cov
        f_hist[i] = f

    x_smooth = x_filt.copy()
    p_smooth = p_filt.copy()
    for i in range(n - 2, -1, -1):
        f_next = f_hist[i + 1]
        p_pred_next = p_pred[i + 1]
        gain = p_filt[i] @ f_next.T @ np.linalg.pinv(p_pred_next)
        x_smooth[i] = x_filt[i] + gain @ (x_smooth[i + 1] - x_pred[i + 1])
        p_smooth[i] = p_filt[i] + gain @ (p_smooth[i + 1] - p_pred_next) @ gain.T
        p_smooth[i] = 0.5 * (p_smooth[i] + p_smooth[i].T)

    return x_smooth[:, 0].astype(np.float32), x_smooth[:, 1].astype(np.float32)


def smooth_track_observations(
    rows: list[dict[str, Any]],
    by_track: dict[int, list[dict[str, float | int]]],
    method: str,
    smooth_window: int,
    kalman_process_var: float,
    kalman_measurement_var: float,
) -> tuple[dict[int, list[dict[str, float | int]]], dict[str, Any]]:
    for row in rows:
        for key in FIELD_SMOOTH_COLUMNS:
            row[key] = ""

    method_norm = str(method).strip().lower()
    if method_norm not in {"kalman", "sliding_median"}:
        raise ValueError(f"Unsupported field smoothing method: {method}")

    win = max(1, int(smooth_window))
    if method_norm == "sliding_median" and win % 2 == 0:
        win += 1
    half = win // 2

    out: dict[int, list[dict[str, float | int]]] = {}
    for track_id, obs in by_track.items():
        if not obs:
            out[track_id] = []
            continue

        xs = np.asarray([float(d["x_ft"]) for d in obs], dtype=np.float32)
        ys = np.asarray([float(d["y_ft"]) for d in obs], dtype=np.float32)
        n = len(obs)

        x_smooth = np.empty(n, dtype=np.float32)
        y_smooth = np.empty(n, dtype=np.float32)
        if method_norm == "sliding_median":
            for i in range(n):
                left = max(0, i - half)
                right = min(n, i + half + 1)
                x_smooth[i] = float(np.median(xs[left:right]))
                y_smooth[i] = float(np.median(ys[left:right]))
        else:
            frame_idxs = np.asarray([int(d["frame_idx"]) for d in obs], dtype=np.int32)
            x_smooth, y_smooth = kalman_smooth_track_positions(
                xs=xs,
                ys=ys,
                frame_idxs=frame_idxs,
                process_var=float(kalman_process_var),
                measurement_var=float(kalman_measurement_var),
            )

        obs_out: list[dict[str, float | int]] = []
        for i, d in enumerate(obs):
            row_idx = int(d["row_idx"])
            row = rows[row_idx]
            row["field_x_ft_raw"] = float(xs[i])
            row["field_y_ft_raw"] = float(ys[i])
            row["field_x_ft_smooth"] = float(x_smooth[i])
            row["field_y_ft_smooth"] = float(y_smooth[i])
            if method_norm == "sliding_median":
                row["field_point_smoothed"] = 1 if win > 1 else 0
            else:
                row["field_point_smoothed"] = 1 if n > 1 else 0

            obs_out.append(
                {
                    "row_idx": row_idx,
                    "frame_idx": int(d["frame_idx"]),
                    "x_ft": float(x_smooth[i]),
                    "y_ft": float(y_smooth[i]),
                }
            )
        out[int(track_id)] = obs_out

    if method_norm == "sliding_median":
        smoothing_meta: dict[str, Any] = {
            "method": "sliding_median",
            "window": int(win),
        }
    else:
        smoothing_meta = {
            "method": "kalman_rts_cv2d",
            "process_var": float(max(1e-6, float(kalman_process_var))),
            "measurement_var": float(max(1e-6, float(kalman_measurement_var))),
        }

    return out, smoothing_meta


def annotate_velocity(
    rows: list[dict[str, Any]],
    by_track: dict[int, list[dict[str, float | int]]],
    fps: float,
    max_step_ft: float,
    smooth_window: int,
) -> list[dict[str, Any]]:
    for row in rows:
        for key in VELOCITY_COLUMNS:
            row[key] = ""

    summary_rows: list[dict[str, Any]] = []
    smooth_n = max(1, int(smooth_window))
    max_step = max(0.0, float(max_step_ft))
    fps_safe = float(fps) if fps > 0 else 30.0

    for track_id in sorted(by_track.keys()):
        obs = by_track[track_id]
        if not obs:
            continue

        speeds_ft_s: list[float] = []
        vx_vals: list[float] = []
        vy_vals: list[float] = []
        valid_steps = 0
        invalid_steps = 0
        total_path_ft = 0.0
        speed_window: deque[float] = deque(maxlen=smooth_n)

        prev = obs[0]
        first_row = rows[int(prev["row_idx"])]
        first_row["velocity_step_valid"] = 0

        for i in range(1, len(obs)):
            cur = obs[i]
            row = rows[int(cur["row_idx"])]

            frame_delta = int(cur["frame_idx"]) - int(prev["frame_idx"])
            row["velocity_delta_frames"] = frame_delta
            if frame_delta <= 0:
                row["velocity_step_valid"] = 0
                invalid_steps += 1
                prev = cur
                continue

            dt_s = float(frame_delta) / fps_safe
            dx_ft = float(cur["x_ft"]) - float(prev["x_ft"])
            dy_ft = float(cur["y_ft"]) - float(prev["y_ft"])
            step_ft = math.hypot(dx_ft, dy_ft)

            row["velocity_delta_time_s"] = dt_s
            row["velocity_dx_ft"] = dx_ft
            row["velocity_dy_ft"] = dy_ft
            row["velocity_step_ft"] = step_ft

            step_valid = not (max_step > 0.0 and step_ft > max_step)
            row["velocity_step_valid"] = int(step_valid)
            if not step_valid:
                invalid_steps += 1
                prev = cur
                continue

            vx_ft_s = dx_ft / dt_s
            vy_ft_s = dy_ft / dt_s
            speed_ft_s = step_ft / dt_s
            speed_mph = speed_ft_s * MPH_PER_FT_S

            row["velocity_vx_ft_s"] = vx_ft_s
            row["velocity_vy_ft_s"] = vy_ft_s
            row["velocity_ft_s"] = speed_ft_s
            row["velocity_mph"] = speed_mph

            speed_window.append(speed_ft_s)
            smooth_ft_s = sum(speed_window) / float(len(speed_window))
            row["velocity_ft_s_smooth"] = smooth_ft_s
            row["velocity_mph_smooth"] = smooth_ft_s * MPH_PER_FT_S

            speeds_ft_s.append(speed_ft_s)
            vx_vals.append(vx_ft_s)
            vy_vals.append(vy_ft_s)
            total_path_ft += step_ft
            valid_steps += 1
            prev = cur

        first = obs[0]
        last = obs[-1]
        detections = len(obs)
        first_frame = int(first["frame_idx"])
        last_frame = int(last["frame_idx"])
        duration_s = float(last_frame - first_frame) / fps_safe if last_frame > first_frame else 0.0
        net_dx_ft = float(last["x_ft"]) - float(first["x_ft"])
        net_dy_ft = float(last["y_ft"]) - float(first["y_ft"])
        net_disp_ft = math.hypot(net_dx_ft, net_dy_ft)

        mean_speed_ft_s = float(sum(speeds_ft_s) / len(speeds_ft_s)) if speeds_ft_s else float("nan")
        max_speed_ft_s = float(max(speeds_ft_s)) if speeds_ft_s else float("nan")
        mean_vx_ft_s = float(sum(vx_vals) / len(vx_vals)) if vx_vals else float("nan")
        mean_vy_ft_s = float(sum(vy_vals) / len(vy_vals)) if vy_vals else float("nan")
        net_speed_ft_s = float(net_disp_ft / duration_s) if duration_s > 0 else float("nan")

        summary_rows.append(
            {
                "track_id": track_id,
                "detections": detections,
                "first_frame_index": first_frame,
                "last_frame_index": last_frame,
                "duration_s": duration_s,
                "valid_velocity_steps": valid_steps,
                "invalid_velocity_steps": invalid_steps,
                "total_path_ft": total_path_ft,
                "net_displacement_ft": net_disp_ft,
                "mean_vx_ft_s": mean_vx_ft_s,
                "mean_vy_ft_s": mean_vy_ft_s,
                "mean_speed_ft_s": mean_speed_ft_s,
                "mean_speed_mph": mean_speed_ft_s * MPH_PER_FT_S if not math.isnan(mean_speed_ft_s) else float("nan"),
                "max_speed_ft_s": max_speed_ft_s,
                "max_speed_mph": max_speed_ft_s * MPH_PER_FT_S if not math.isnan(max_speed_ft_s) else float("nan"),
                "net_speed_ft_s": net_speed_ft_s,
                "net_speed_mph": net_speed_ft_s * MPH_PER_FT_S if not math.isnan(net_speed_ft_s) else float("nan"),
                "start_field_x_ft": float(first["x_ft"]),
                "start_field_y_ft": float(first["y_ft"]),
                "end_field_x_ft": float(last["x_ft"]),
                "end_field_y_ft": float(last["y_ft"]),
            }
        )

    return summary_rows


class FieldCanvasMapper:
    def __init__(
        self,
        min_x_ft: float,
        max_x_ft: float,
        min_y_ft: float,
        max_y_ft: float,
        scale_px_per_ft: float,
        margin_px: int,
    ) -> None:
        self.min_x_ft = float(min_x_ft)
        self.max_x_ft = float(max_x_ft)
        self.min_y_ft = float(min_y_ft)
        self.max_y_ft = float(max_y_ft)
        self.scale = max(1.0, float(scale_px_per_ft))
        self.margin = max(10, int(margin_px))

    def canvas_size(self) -> tuple[int, int]:
        w = int(round((self.max_x_ft - self.min_x_ft) * self.scale)) + 2 * self.margin
        h = int(round((self.max_y_ft - self.min_y_ft) * self.scale)) + 2 * self.margin
        return max(120, h), max(120, w)

    def to_px(self, x_ft: float, y_ft: float) -> tuple[int, int]:
        x = self.margin + int(round((float(x_ft) - self.min_x_ft) * self.scale))
        y = self.margin + int(round((self.max_y_ft - float(y_ft)) * self.scale))
        return x, y


def build_mapper(
    by_track: dict[int, list[dict[str, float | int]]],
    basepath_ft: float,
    roi_polygon_field_ft: np.ndarray | None,
    mound_center_field_ft: tuple[float, float],
    pad_ft: float,
    scale_px_per_ft: float,
) -> FieldCanvasMapper:
    xs = [0.0, basepath_ft, basepath_ft, 0.0, mound_center_field_ft[0]]
    ys = [0.0, 0.0, basepath_ft, basepath_ft, mound_center_field_ft[1]]

    for obs in by_track.values():
        for d in obs:
            xs.append(float(d["x_ft"]))
            ys.append(float(d["y_ft"]))

    if roi_polygon_field_ft is not None and roi_polygon_field_ft.size > 0:
        xs.extend([float(v) for v in roi_polygon_field_ft[:, 0].tolist()])
        ys.extend([float(v) for v in roi_polygon_field_ft[:, 1].tolist()])

    pad = max(0.0, float(pad_ft))
    min_x = math.floor(min(xs) - pad)
    max_x = math.ceil(max(xs) + pad)
    min_y = math.floor(min(ys) - pad)
    max_y = math.ceil(max(ys) + pad)

    if max_x <= min_x:
        max_x = min_x + 1.0
    if max_y <= min_y:
        max_y = min_y + 1.0

    return FieldCanvasMapper(
        min_x_ft=min_x,
        max_x_ft=max_x,
        min_y_ft=min_y,
        max_y_ft=max_y,
        scale_px_per_ft=scale_px_per_ft,
        margin_px=44,
    )


def draw_field_template(
    mapper: FieldCanvasMapper,
    basepath_ft: float,
    mound_center_field_ft: tuple[float, float],
    roi_polygon_field_ft: np.ndarray | None,
) -> np.ndarray:
    h, w = mapper.canvas_size()
    canvas = np.full((h, w, 3), (44, 98, 58), dtype=np.uint8)

    grid_color = (56, 118, 70)
    x_start = int(math.floor(mapper.min_x_ft / 10.0) * 10)
    x_end = int(math.ceil(mapper.max_x_ft / 10.0) * 10)
    y_start = int(math.floor(mapper.min_y_ft / 10.0) * 10)
    y_end = int(math.ceil(mapper.max_y_ft / 10.0) * 10)

    for x_ft in range(x_start, x_end + 1, 10):
        p1 = mapper.to_px(float(x_ft), mapper.min_y_ft)
        p2 = mapper.to_px(float(x_ft), mapper.max_y_ft)
        cv2.line(canvas, p1, p2, grid_color, 1, cv2.LINE_AA)
    for y_ft in range(y_start, y_end + 1, 10):
        p1 = mapper.to_px(mapper.min_x_ft, float(y_ft))
        p2 = mapper.to_px(mapper.max_x_ft, float(y_ft))
        cv2.line(canvas, p1, p2, grid_color, 1, cv2.LINE_AA)

    # Infield dirt and diamond.
    diamond_field = np.array(
        [[0.0, 0.0], [basepath_ft, 0.0], [basepath_ft, basepath_ft], [0.0, basepath_ft]],
        dtype=np.float32,
    )
    diamond_px = np.array([mapper.to_px(float(x), float(y)) for x, y in diamond_field], dtype=np.int32)
    cv2.fillConvexPoly(canvas, diamond_px, (92, 126, 157))
    cv2.polylines(canvas, [diamond_px], True, (230, 236, 240), 2, cv2.LINE_AA)

    # Foul lines from home.
    foul_len = max(basepath_ft * 1.5, mapper.max_x_ft - mapper.min_x_ft, mapper.max_y_ft - mapper.min_y_ft)
    home_px = mapper.to_px(0.0, 0.0)
    first_line_px = mapper.to_px(foul_len, 0.0)
    third_line_px = mapper.to_px(0.0, foul_len)
    cv2.line(canvas, home_px, first_line_px, (238, 238, 238), 2, cv2.LINE_AA)
    cv2.line(canvas, home_px, third_line_px, (238, 238, 238), 2, cv2.LINE_AA)

    # Bases.
    base_markers = {
        "H": (0.0, 0.0),
        "1B": (basepath_ft, 0.0),
        "2B": (basepath_ft, basepath_ft),
        "3B": (0.0, basepath_ft),
    }
    for label, (bx, by) in base_markers.items():
        center = mapper.to_px(bx, by)
        half = max(3, int(round(1.5 * mapper.scale / 6.0)))
        cv2.rectangle(
            canvas,
            (center[0] - half, center[1] - half),
            (center[0] + half, center[1] + half),
            (245, 245, 245),
            -1,
        )
        cv2.putText(
            canvas,
            label,
            (center[0] + 6, center[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

    # Mound marker.
    mound_px = mapper.to_px(float(mound_center_field_ft[0]), float(mound_center_field_ft[1]))
    cv2.circle(canvas, mound_px, max(4, int(round(9.0 * mapper.scale / 10.0))), (180, 230, 180), 1, cv2.LINE_AA)
    cv2.circle(canvas, mound_px, 3, (160, 250, 160), -1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "Mound",
        (mound_px[0] + 8, mound_px[1] - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (160, 250, 160),
        1,
        cv2.LINE_AA,
    )

    if roi_polygon_field_ft is not None and len(roi_polygon_field_ft) >= 2:
        roi_px = np.array(
            [mapper.to_px(float(x), float(y)) for x, y in roi_polygon_field_ft.tolist()],
            dtype=np.int32,
        )
        cv2.polylines(canvas, [roi_px], True, (60, 220, 255), 2, cv2.LINE_AA)
        min_x = int(np.min(roi_px[:, 0]))
        min_y = int(np.min(roi_px[:, 1]))
        cv2.putText(
            canvas,
            "ROI",
            (min_x + 6, max(14, min_y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (60, 220, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        "Bird's-eye Ground Contacts (field feet)",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def draw_tracks_on_canvas(
    canvas: np.ndarray,
    mapper: FieldCanvasMapper,
    by_track: dict[int, list[dict[str, float | int]]],
) -> None:
    for track_id in sorted(by_track.keys()):
        obs = by_track[track_id]
        if not obs:
            continue
        color = color_for_track(track_id)
        pts = [mapper.to_px(float(d["x_ft"]), float(d["y_ft"])) for d in obs]
        if len(pts) >= 2:
            cv2.polylines(canvas, [np.asarray(pts, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(canvas, p, 2, color, -1, cv2.LINE_AA)
        lp = pts[-1]
        cv2.putText(
            canvas,
            f"id {track_id}",
            (lp[0] + 6, lp[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


def render_birdseye_video(
    out_path: Path,
    template: np.ndarray,
    mapper: FieldCanvasMapper,
    by_track: dict[int, list[dict[str, float | int]]],
    rows: list[dict[str, Any]],
    fps: float,
    trail_length: int,
    trail_max_gap_frames: int,
) -> int:
    by_frame: dict[int, list[dict[str, float | int]]] = defaultdict(list)
    for track_obs in by_track.values():
        for obs in track_obs:
            by_frame[int(obs["frame_idx"])].append(obs)
    if not by_frame:
        return 0

    frame_min = min(by_frame.keys())
    frame_max = max(by_frame.keys())
    fps_safe = float(fps) if fps > 0 else 30.0
    h, w = template.shape[:2]

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_safe, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open bird's-eye video for writing: {out_path}")

    trail_len = max(1, int(trail_length))
    max_gap = max(1, int(trail_max_gap_frames))
    history: dict[int, deque[dict[str, float | int]]] = {}
    last_seen_frame: dict[int, int] = {}
    frames_written = 0

    for frame_idx in range(frame_min, frame_max + 1):
        for obs in by_frame.get(frame_idx, []):
            row = rows[int(obs["row_idx"])]
            track_id = maybe_int(row.get("track_id"))
            if track_id is None:
                continue
            if track_id not in history:
                history[track_id] = deque(maxlen=trail_len)
            history[track_id].append(obs)
            last_seen_frame[track_id] = frame_idx

        frame = template.copy()
        active_tracks = 0
        for track_id in sorted(history.keys()):
            last_seen = last_seen_frame.get(track_id, frame_idx)
            if frame_idx - last_seen > max_gap:
                continue

            trail = history[track_id]
            if not trail:
                continue
            active_tracks += 1

            color = color_for_track(track_id)
            pts = [mapper.to_px(float(d["x_ft"]), float(d["y_ft"])) for d in trail]
            if len(pts) >= 2:
                cv2.polylines(frame, [np.asarray(pts, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
            curr = pts[-1]
            cv2.circle(frame, curr, 4, color, -1, cv2.LINE_AA)

            row = rows[int(trail[-1]["row_idx"])]
            mph = maybe_float(row.get("velocity_mph_smooth"))
            if mph is None:
                mph = maybe_float(row.get("velocity_mph"))
            label = f"id {track_id}" if mph is None else f"id {track_id} {mph:.1f} mph"
            cv2.putText(
                frame,
                label,
                (curr[0] + 6, curr[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            frame,
            f"frame {frame_idx}  active tracks {active_tracks}",
            (12, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
        frames_written += 1

    writer.release()
    return frames_written


def render_image_overlay_video(
    out_path: Path,
    rows: list[dict[str, Any]],
    fps: float,
) -> int:
    by_frame_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    frame_path_by_index: dict[int, str] = {}

    for row in rows:
        frame_idx = maybe_int(row.get("frame_index"))
        raw_path = str(row.get("image_path") or "").strip()
        if frame_idx is None or not raw_path:
            continue
        by_frame_rows[frame_idx].append(row)
        if frame_idx not in frame_path_by_index:
            frame_path_by_index[frame_idx] = raw_path

    if not frame_path_by_index:
        return 0

    first_idx = min(frame_path_by_index.keys())
    first_path = resolve_frame_path(frame_path_by_index[first_idx])
    first_frame = cv2.imread(str(first_path))
    if first_frame is None:
        raise RuntimeError(f"Could not read first frame for overlay video: {first_path}")
    h_img, w_img = first_frame.shape[:2]

    fps_safe = float(fps) if fps > 0 else 30.0
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_safe, (w_img, h_img))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open overlay video for writing: {out_path}")

    frames_written = 0
    frame_indices = sorted(frame_path_by_index.keys())
    for frame_idx in frame_indices:
        frame_path = resolve_frame_path(frame_path_by_index[frame_idx])
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        rows_in_frame = by_frame_rows.get(frame_idx, [])
        for row in rows_in_frame:
            track_id = maybe_int(row.get("track_id"))
            if track_id is None:
                continue
            color = color_for_track(track_id)

            x1 = maybe_float(row.get("box_x1"))
            y1 = maybe_float(row.get("box_y1"))
            x2 = maybe_float(row.get("box_x2"))
            y2 = maybe_float(row.get("box_y2"))
            if None not in (x1, y1, x2, y2):
                cv2.rectangle(
                    frame,
                    (int(round(float(x1))), int(round(float(y1)))),
                    (int(round(float(x2))), int(round(float(y2)))),
                    color,
                    2,
                )

            cx = maybe_float(row.get("sam_contact_x"))
            cy = maybe_float(row.get("sam_contact_y"))
            if cx is None or cy is None:
                cx = maybe_float(row.get("bbox_bottom_center_x"))
                cy = maybe_float(row.get("bbox_bottom_center_y"))
            if cx is not None and cy is not None:
                cv2.circle(frame, (int(round(cx)), int(round(cy))), 4, (0, 0, 255), -1, cv2.LINE_AA)

            mph = maybe_float(row.get("velocity_mph_smooth"))
            if mph is None:
                mph = maybe_float(row.get("velocity_mph"))
            if mph is None:
                label = f"id {track_id}"
            else:
                label = f"id {track_id} {mph:.1f} mph"

            tx = int(round(x1)) if x1 is not None else (int(round(cx)) if cx is not None else 8)
            ty = int(round(y1)) - 8 if y1 is not None else (int(round(cy)) - 8 if cy is not None else 20)
            cv2.putText(
                frame,
                label,
                (max(6, tx), max(16, ty)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            frame,
            f"frame {frame_idx}  players {len(rows_in_frame)}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
        frames_written += 1

    writer.release()
    return frames_written


def main() -> None:
    args = parse_args()

    if not args.source_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.source_csv}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_rows = read_csv_rows(args.source_csv)
    if not source_rows:
        raise RuntimeError(f"Input CSV has no rows: {args.source_csv}")

    filter_meta = load_json_if_exists(args.filter_meta_json)
    roi_cfg = filter_meta.get("infield_cone_roi", {}) if isinstance(filter_meta, dict) else {}
    basepath_ft = maybe_float(roi_cfg.get("basepath_ft"))
    if basepath_ft is None:
        basepath_ft = float(args.basepath_ft)

    mound_center_field_ft = (
        MLB_MOUND_CENTER_FROM_HOME_FT / math.sqrt(2.0),
        MLB_MOUND_CENTER_FROM_HOME_FT / math.sqrt(2.0),
    )
    mound_meta = roi_cfg.get("mound_center_field_ft")
    if isinstance(mound_meta, list) and len(mound_meta) >= 2:
        mx = maybe_float(mound_meta[0])
        my = maybe_float(mound_meta[1])
        if mx is not None and my is not None:
            mound_center_field_ft = (mx, my)

    roi_polygon_field_ft: np.ndarray | None = None
    roi_poly_meta = roi_cfg.get("polygon_field_ft")
    if isinstance(roi_poly_meta, list) and len(roi_poly_meta) >= 2:
        try:
            roi_polygon_field_ft = np.asarray(roi_poly_meta, dtype=np.float32).reshape(-1, 2)
        except Exception:
            roi_polygon_field_ft = None

    by_track = build_track_observations(source_rows, basepath_ft=basepath_ft)
    if not by_track:
        raise RuntimeError(
            "No usable track observations found. Expected track_id, frame_index, and field_x_ft/field_y_ft columns."
        )
    by_track_smoothed, field_smoothing_meta = smooth_track_observations(
        rows=source_rows,
        by_track=by_track,
        method=str(args.field_smoothing_method),
        smooth_window=int(args.field_smooth_window),
        kalman_process_var=float(args.field_kalman_process_var),
        kalman_measurement_var=float(args.field_kalman_measurement_var),
    )

    fps_used = float(args.fps) if args.fps > 0 else 30.0
    summary_rows = annotate_velocity(
        rows=source_rows,
        by_track=by_track_smoothed,
        fps=fps_used,
        max_step_ft=float(args.max_step_ft),
        smooth_window=int(args.smooth_window),
    )

    mapper = build_mapper(
        by_track=by_track_smoothed,
        basepath_ft=basepath_ft,
        roi_polygon_field_ft=roi_polygon_field_ft,
        mound_center_field_ft=mound_center_field_ft,
        pad_ft=float(args.birdseye_pad_ft),
        scale_px_per_ft=float(args.birdseye_scale),
    )
    template = draw_field_template(
        mapper=mapper,
        basepath_ft=basepath_ft,
        mound_center_field_ft=mound_center_field_ft,
        roi_polygon_field_ft=roi_polygon_field_ft,
    )
    birdseye_img = template.copy()
    draw_tracks_on_canvas(birdseye_img, mapper=mapper, by_track=by_track_smoothed)

    out_csv = args.output_dir / "tracks_with_velocity.csv"
    out_jsonl = args.output_dir / "tracks_with_velocity.jsonl"
    out_summary = args.output_dir / "velocity_summary.csv"
    out_birdseye_img = args.output_dir / "birdseye_contacts_velocity.png"
    out_birdseye_video = args.output_dir / "birdseye_contacts_velocity.mp4"
    out_overlay_video = args.output_dir / "tracks_velocity_overlay.mp4"
    out_meta = args.output_dir / "run_meta_velocity.json"

    write_csv_rows(out_csv, source_rows)
    write_jsonl_rows(out_jsonl, source_rows)
    write_csv_rows(out_summary, summary_rows)
    if not cv2.imwrite(str(out_birdseye_img), birdseye_img):
        raise RuntimeError(f"Could not write bird's-eye image: {out_birdseye_img}")

    birdseye_video_frames_written = 0
    overlay_video_frames_written = 0
    if not args.no_video:
        birdseye_video_frames_written = render_birdseye_video(
            out_path=out_birdseye_video,
            template=template,
            mapper=mapper,
            by_track=by_track_smoothed,
            rows=source_rows,
            fps=fps_used,
            trail_length=int(args.trail_length),
            trail_max_gap_frames=int(args.trail_max_gap_frames),
        )
        overlay_video_frames_written = render_image_overlay_video(
            out_path=out_overlay_video,
            rows=source_rows,
            fps=fps_used,
        )

    speeds_all = [
        maybe_float(r.get("mean_speed_mph"))
        for r in summary_rows
        if maybe_float(r.get("mean_speed_mph")) is not None
    ]
    max_speeds = [
        maybe_float(r.get("max_speed_mph"))
        for r in summary_rows
        if maybe_float(r.get("max_speed_mph")) is not None
    ]

    meta = {
        "source_csv": str(args.source_csv),
        "filter_meta_json": str(args.filter_meta_json) if args.filter_meta_json else None,
        "fps_used": fps_used,
        "basepath_ft": basepath_ft,
        "mound_center_field_ft": [float(mound_center_field_ft[0]), float(mound_center_field_ft[1])],
        "roi_polygon_field_ft": (
            [[float(x), float(y)] for x, y in roi_polygon_field_ft.tolist()]
            if roi_polygon_field_ft is not None
            else None
        ),
        "velocity": {
            "max_step_ft": float(args.max_step_ft),
            "smooth_window": int(args.smooth_window),
            "field_point_smoothing": field_smoothing_meta,
        },
        "birdseye": {
            "scale_px_per_ft": float(args.birdseye_scale),
            "pad_ft": float(args.birdseye_pad_ft),
            "trail_length": int(args.trail_length),
            "trail_max_gap_frames": int(args.trail_max_gap_frames),
            "video_frames_written": birdseye_video_frames_written if not args.no_video else 0,
            "overlay_video_frames_written": overlay_video_frames_written if not args.no_video else 0,
        },
        "counts": {
            "input_rows": len(source_rows),
            "tracks_with_field_points": len(by_track),
            "tracks_with_smoothed_field_points": len(by_track_smoothed),
            "tracks_summarized": len(summary_rows),
        },
        "speed_stats_mph": {
            "mean_of_track_means_mph": (float(sum(speeds_all) / len(speeds_all)) if speeds_all else float("nan")),
            "max_track_peak_mph": (float(max(max_speeds)) if max_speeds else float("nan")),
        },
        "outputs": {
            "velocity_csv": str(out_csv),
            "velocity_jsonl": str(out_jsonl),
            "velocity_summary_csv": str(out_summary),
            "birdseye_image": str(out_birdseye_img),
            "birdseye_video": None if args.no_video else str(out_birdseye_video),
            "overlay_video": None if args.no_video else str(out_overlay_video),
        },
    }
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    stage_meta = {
        "source_csv": str(args.source_csv),
        "filter_meta_json": str(args.filter_meta_json) if args.filter_meta_json else None,
        "velocity_script": str(Path(__file__).resolve()),
        "velocity_command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "outputs": {
            "velocity_csv": str(out_csv),
            "velocity_jsonl": str(out_jsonl),
            "velocity_summary_csv": str(out_summary),
            "birdseye_image": str(out_birdseye_img),
            "birdseye_video": None if args.no_video else str(out_birdseye_video),
            "overlay_video": None if args.no_video else str(out_overlay_video),
            "velocity_meta": str(out_meta),
        },
    }
    stage_meta_path = args.output_dir / "run_meta_get_velocity.json"
    stage_meta_path.write_text(json.dumps(stage_meta, indent=2), encoding="utf-8")

    print("Velocity stage complete.")
    print(f"Input rows:        {len(source_rows)}")
    print(f"Tracks processed:  {len(by_track_smoothed)}")
    print(f"Velocity CSV:      {out_csv}")
    print(f"Velocity summary:  {out_summary}")
    print(f"Bird's-eye image:  {out_birdseye_img}")
    if not args.no_video:
        print(f"Bird's-eye video:  {out_birdseye_video}")
        print(f"Overlay video:     {out_overlay_video}")
    print(f"Meta:              {out_meta}")


if __name__ == "__main__":
    main()
