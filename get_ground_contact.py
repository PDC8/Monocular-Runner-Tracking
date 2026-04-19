#!/usr/bin/env python3
"""
Apply SAM to tracked player detections and extract robust ground-contact points.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import SAM


def parse_args() -> argparse.Namespace:
    this_dir = Path(__file__).resolve().parent
    models_dir = this_dir / "models"
    parser = argparse.ArgumentParser(
        description="Refine tracked player detections with SAM and extract robust ground-contact points."
    )
    parser.add_argument(
        "--tracks-csv",
        type=Path,
        required=True,
        help="Input detection/tracking CSV from get_players.py.",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="Optional frames directory to render full-length overlay including no-detection frames.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Max frames for overlay rendering when --frames-dir is provided (0 = all).",
    )
    parser.add_argument(
        "--sam-weights",
        type=str,
        default=str(models_dir / "sam2.1_l.pt"),
        help="SAM weights path/name understood by Ultralytics (default: Thesis_Final/models/sam2.1_l.pt).",
    )
    parser.add_argument(
        "--contact-bottom-frac",
        type=float,
        default=0.08,
        help="Bottom fraction of mask pixels used for robust ground-contact estimation (0.05-0.10 typical).",
    )
    parser.add_argument(
        "--contact-mode",
        type=str,
        default="hybrid",
        choices=["sam", "bbox_bottom", "hybrid"],
        help=(
            "Ground-contact point mode: "
            "'sam' uses SAM robust contact (x,y), "
            "'bbox_bottom' uses detection-box bottom-center, "
            "'hybrid' uses bbox-center x with SAM bottom y (default)."
        ),
    )
    parser.add_argument(
        "--contact-smooth-method",
        type=str,
        default="sliding_median",
        choices=["none", "sliding_median"],
        help=(
            "Per-track trajectory smoothing for contact points after SAM extraction. "
            "'none' keeps raw contacts; 'sliding_median' applies centered median smoothing."
        ),
    )
    parser.add_argument(
        "--contact-smooth-window",
        type=int,
        default=5,
        help="Sliding-median window size (samples) for per-track contact smoothing.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Inference device for SAM, e.g., 0 or cpu.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="FPS used for visualization video.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for SAM-contact outputs.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable visualization video output.",
    )
    return parser.parse_args()


def read_rows_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    if not rows:
        out_csv.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_rows_jsonl(rows: list[dict[str, Any]], out_jsonl: Path) -> None:
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def resolve_frame_path(raw: str) -> Path:
    p = Path(raw)
    if p.exists():
        return p
    cwd_p = Path.cwd() / raw
    if cwd_p.exists():
        return cwd_p
    return p


def color_for_track(track_id: int) -> tuple[int, int, int]:
    if track_id < 0:
        return (180, 180, 180)
    return (
        int((37 * track_id + 73) % 256),
        int((17 * track_id + 191) % 256),
        int((29 * track_id + 47) % 256),
    )


def mask_contact_point(mask: np.ndarray) -> tuple[float | None, float | None]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None, None
    y_max = int(ys.max())
    x_candidates = xs[ys == y_max]
    x_med = float(np.median(x_candidates))
    return x_med, float(y_max)


def mask_contact_point_robust(mask: np.ndarray, bottom_frac: float = 0.08) -> tuple[float | None, float | None]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None, None

    frac = float(np.clip(bottom_frac, 0.01, 0.50))
    y_cut = float(np.quantile(ys.astype(np.float32), 1.0 - frac))
    keep = ys >= y_cut
    if not np.any(keep):
        return mask_contact_point(mask)

    xs_b = xs[keep].astype(np.float32)
    ys_b = ys[keep].astype(np.float32)
    cx = float(np.median(xs_b))
    cy = float(np.median(ys_b))
    return cx, cy


def mask_bbox(mask: np.ndarray) -> tuple[float | None, float | None, float | None, float | None]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None, None, None, None
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())
    return float(x1), float(y1), float(x2), float(y2)


def select_contact_point(
    *,
    contact_mode: str,
    box_x1: float,
    box_y1: float,
    box_x2: float,
    box_y2: float,
    sam_contact_x: float | None,
    sam_contact_y: float | None,
) -> tuple[float | None, float | None]:
    mode = str(contact_mode).strip().lower()
    bbox_center_x = 0.5 * (float(box_x1) + float(box_x2))
    bbox_bottom_y = float(box_y2)

    if mode == "sam":
        return sam_contact_x, sam_contact_y
    if mode == "bbox_bottom":
        return bbox_center_x, bbox_bottom_y
    if mode == "hybrid":
        # Force horizontal center of detection box; prefer SAM vertical bottom when available.
        return bbox_center_x, (sam_contact_y if sam_contact_y is not None else bbox_bottom_y)
    raise ValueError(f"Unsupported contact mode: {contact_mode}")


def masked_rgb_stats(image_bgr: np.ndarray, mask: np.ndarray) -> tuple[float, float, float, float, float, float]:
    pix = image_bgr[mask]
    if pix.size == 0:
        return math.nan, math.nan, math.nan, math.nan, math.nan, math.nan
    pix_rgb = pix[:, ::-1].astype(np.float32)
    mean = pix_rgb.mean(axis=0)
    std = pix_rgb.std(axis=0)
    return (
        float(mean[0]),
        float(mean[1]),
        float(mean[2]),
        float(std[0]),
        float(std[1]),
        float(std[2]),
    )


def smooth_contact_trajectories(
    rows: list[dict[str, Any]],
    *,
    method: str,
    smooth_window: int,
) -> dict[str, Any]:
    method_norm = str(method).strip().lower()
    if method_norm not in {"none", "sliding_median"}:
        raise ValueError(f"Unsupported contact smoothing method: {method}")

    win = max(1, int(smooth_window))
    if method_norm == "sliding_median" and win % 2 == 0:
        win += 1

    for row in rows:
        x_raw = maybe_float(row.get("sam_contact_x"))
        y_raw = maybe_float(row.get("sam_contact_y"))
        row["sam_contact_x_raw"] = x_raw
        row["sam_contact_y_raw"] = y_raw
        row["sam_contact_smooth_method"] = method_norm
        row["sam_contact_smooth_window"] = int(win if method_norm == "sliding_median" else 1)
        row["sam_contact_smoothed"] = 0

    by_track_indices: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row_idx, row in enumerate(rows):
        track_id = maybe_int(row.get("track_id"))
        if track_id is None:
            continue
        frame_idx = maybe_int(row.get("frame_index"))
        if frame_idx is None:
            frame_idx = row_idx
        by_track_indices[int(track_id)].append((int(frame_idx), row_idx))

    if method_norm == "none" or win <= 1:
        valid_points = sum(
            1
            for row in rows
            if maybe_float(row.get("sam_contact_x_raw")) is not None
            and maybe_float(row.get("sam_contact_y_raw")) is not None
        )
        return {
            "method": method_norm,
            "window": int(win),
            "tracks_total": int(len(by_track_indices)),
            "tracks_smoothed": 0,
            "valid_contact_points": int(valid_points),
            "smoothed_contact_points": 0,
        }

    tracks_smoothed = 0
    valid_contact_points = 0
    smoothed_contact_points = 0
    half = win // 2
    for track_id in sorted(by_track_indices.keys()):
        pairs = sorted(by_track_indices[track_id], key=lambda t: (t[0], t[1]))
        ordered_row_idxs = [idx for _, idx in pairs]
        valid_row_idxs: list[int] = []
        xs: list[float] = []
        ys: list[float] = []

        for row_idx in ordered_row_idxs:
            x = maybe_float(rows[row_idx].get("sam_contact_x_raw"))
            y = maybe_float(rows[row_idx].get("sam_contact_y_raw"))
            if x is None or y is None:
                continue
            valid_row_idxs.append(row_idx)
            xs.append(float(x))
            ys.append(float(y))

        n = len(valid_row_idxs)
        valid_contact_points += n
        if n <= 1:
            continue
        tracks_smoothed += 1

        for i, row_idx in enumerate(valid_row_idxs):
            left = max(0, i - half)
            right = min(n, i + half + 1)
            row_x = float(np.median(xs[left:right]))
            row_y = float(np.median(ys[left:right]))
            rows[row_idx]["sam_contact_x"] = row_x
            rows[row_idx]["sam_contact_y"] = row_y
            rows[row_idx]["sam_contact_smoothed"] = 1
            smoothed_contact_points += 1

    return {
        "method": method_norm,
        "window": int(win),
        "tracks_total": int(len(by_track_indices)),
        "tracks_smoothed": int(tracks_smoothed),
        "valid_contact_points": int(valid_contact_points),
        "smoothed_contact_points": int(smoothed_contact_points),
    }


def _frame_index_from_name(path: Path, fallback: int) -> int:
    stem = path.stem
    tail = stem.split("_")[-1]
    if tail.isdigit():
        return int(tail)
    return fallback


def build_frame_entries(
    args: argparse.Namespace,
    source_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    if args.frames_dir is not None:
        frame_files = sorted(args.frames_dir.glob("frame_*.png"))
        entries = [
            {
                "frame_index": _frame_index_from_name(p, i),
                "frame_id": p.stem,
                "frame_path": p,
            }
            for i, p in enumerate(frame_files)
        ]
        if args.max_frames > 0:
            entries = entries[: args.max_frames]
        return entries

    frame_path_by_index: dict[int, str] = {}
    for row in source_rows:
        fi = maybe_int(row.get("frame_index"))
        if fi is None:
            continue
        if fi not in frame_path_by_index:
            frame_path_by_index[fi] = str(row.get("image_path") or "")

    entries: list[dict[str, Any]] = []
    for fi in sorted(frame_path_by_index.keys()):
        p = resolve_frame_path(frame_path_by_index[fi])
        entries.append({"frame_index": fi, "frame_id": p.stem, "frame_path": p})
    return entries


def main() -> None:
    args = parse_args()
    t0 = time.time()

    if not args.tracks_csv.exists():
        raise FileNotFoundError(f"tracks csv not found: {args.tracks_csv}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_rows = read_rows_csv(args.tracks_csv)
    if not source_rows:
        raise RuntimeError(f"Input tracking CSV has no rows: {args.tracks_csv}")

    by_frame_rows: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        fi = maybe_int(row.get("frame_index"))
        if fi is None:
            continue
        by_frame_rows[fi].append(row)

    frame_entries = build_frame_entries(args=args, source_rows=source_rows)
    if not frame_entries:
        raise RuntimeError("No frames available for SAM contact stage.")

    print(f"Loaded {len(frame_entries)} frames for ground-contact stage.")
    print(f"SAM weights:  {args.sam_weights}")

    sam = SAM(args.sam_weights)

    video_writer: cv2.VideoWriter | None = None
    video_path = args.output_dir / "tracks_sam_overlay.mp4"

    rows: list[dict[str, Any]] = []
    by_track: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "detections": 0,
            "first_frame_index": None,
            "last_frame_index": None,
            "mean_conf_sum": 0.0,
            "contact_points": 0,
            "mask_area_sum": 0.0,
        }
    )
    hybrid_sam_y_fallbacks = 0

    for k, entry in enumerate(frame_entries):
        frame_index = int(entry["frame_index"])
        frame_id = str(entry["frame_id"])
        frame_path: Path = entry["frame_path"]

        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"[WARN] Could not read frame: {frame_path}")
            continue

        if (not args.no_video) and video_writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (w, h))

        frame_rows = by_frame_rows.get(frame_index, [])
        if not frame_rows:
            if video_writer is not None:
                cv2.putText(
                    frame,
                    f"frame={frame_id} detections=0",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                video_writer.write(frame)
            continue

        bboxes: list[list[float]] = []
        kept_rows: list[dict[str, str]] = []
        for r in frame_rows:
            try:
                x1 = float(r["box_x1"])
                y1 = float(r["box_y1"])
                x2 = float(r["box_x2"])
                y2 = float(r["box_y2"])
            except (KeyError, TypeError, ValueError):
                continue
            bboxes.append([x1, y1, x2, y2])
            kept_rows.append(r)

        if not kept_rows:
            if video_writer is not None:
                cv2.putText(
                    frame,
                    f"frame={frame_id} detections=0",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                video_writer.write(frame)
            continue

        sam_result = sam(
            frame,
            bboxes=bboxes,
            device=args.device,
            verbose=False,
        )[0]
        sam_masks: list[np.ndarray] = []
        if sam_result.masks is not None and sam_result.masks.data is not None:
            sam_masks = [m.astype(bool) for m in sam_result.masks.data.detach().cpu().numpy()]
        if len(sam_masks) != len(kept_rows):
            sam_masks = [np.zeros(frame.shape[:2], dtype=bool) for _ in range(len(kept_rows))]

        for i, r in enumerate(kept_rows):
            row: dict[str, Any] = dict(r)
            track_id = int(float(r["track_id"]))
            det_conf = float(r["det_conf"])
            x1 = float(r["box_x1"])
            y1 = float(r["box_y1"])
            x2 = float(r["box_x2"])
            y2 = float(r["box_y2"])
            mask = sam_masks[i]

            mask_area = float(mask.sum())
            bbox_area = max((x2 - x1) * (y2 - y1), 1e-6)
            mask_to_bbox_ratio = float(mask_area / bbox_area)
            sam_contact_x, sam_contact_y = mask_contact_point_robust(mask, bottom_frac=args.contact_bottom_frac)
            mask_x1, mask_y1, mask_x2, mask_y2 = mask_bbox(mask)
            contact_x, contact_y = select_contact_point(
                contact_mode=args.contact_mode,
                box_x1=x1,
                box_y1=y1,
                box_x2=x2,
                box_y2=y2,
                sam_contact_x=sam_contact_x,
                sam_contact_y=sam_contact_y,
            )
            if args.contact_mode == "hybrid" and sam_contact_y is None:
                hybrid_sam_y_fallbacks += 1
            mean_r, mean_g, mean_b, std_r, std_g, std_b = masked_rgb_stats(frame, mask)

            row["sam_contact_x"] = contact_x
            row["sam_contact_y"] = contact_y
            row["contact_mode"] = args.contact_mode
            row["sam_mask_area_px"] = mask_area
            row["sam_mask_to_bbox_ratio"] = mask_to_bbox_ratio
            row["sam_mask_x1"] = mask_x1
            row["sam_mask_y1"] = mask_y1
            row["sam_mask_x2"] = mask_x2
            row["sam_mask_y2"] = mask_y2
            row["mean_r"] = mean_r
            row["mean_g"] = mean_g
            row["mean_b"] = mean_b
            row["std_r"] = std_r
            row["std_g"] = std_g
            row["std_b"] = std_b
            rows.append(row)

            s = by_track[track_id]
            s["detections"] += 1
            s["mean_conf_sum"] += det_conf
            s["mask_area_sum"] += mask_area
            if s["first_frame_index"] is None:
                s["first_frame_index"] = frame_index
            s["last_frame_index"] = frame_index
            if contact_x is not None and contact_y is not None:
                s["contact_points"] += 1

            if video_writer is not None:
                c = color_for_track(track_id)
                cv2.rectangle(
                    frame,
                    (int(round(x1)), int(round(y1))),
                    (int(round(x2)), int(round(y2))),
                    c,
                    2,
                )
                cv2.putText(
                    frame,
                    f"id={track_id} conf={det_conf:.2f}",
                    (int(round(x1)), max(20, int(round(y1)) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    c,
                    2,
                    cv2.LINE_AA,
                )
                if contact_x is not None and contact_y is not None:
                    cv2.circle(frame, (int(round(contact_x)), int(round(contact_y))), 4, (0, 0, 255), -1)
                    mask_u8 = (mask.astype(np.uint8) * 255)
                    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(frame, contours, -1, c, 1)

        if video_writer is not None:
            cv2.putText(
                frame,
                f"frame={frame_id} detections={len(kept_rows)}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            video_writer.write(frame)

        if (k + 1) % 50 == 0:
            print(f"Processed {k + 1}/{len(frame_entries)} frames...")

    if video_writer is not None:
        video_writer.release()

    contact_smoothing_meta = smooth_contact_trajectories(
        rows=rows,
        method=args.contact_smooth_method,
        smooth_window=args.contact_smooth_window,
    )

    out_csv = args.output_dir / "tracks_with_sam_contact.csv"
    out_jsonl = args.output_dir / "tracks_with_sam_contact.jsonl"
    out_track_summary = args.output_dir / "track_summary.csv"
    out_meta = args.output_dir / "run_meta.json"

    write_rows_csv(rows, out_csv)
    write_rows_jsonl(rows, out_jsonl)

    summary_rows: list[dict[str, Any]] = []
    for track_id in sorted(by_track):
        s = by_track[track_id]
        det_n = int(s["detections"])
        summary_rows.append(
            {
                "track_id": track_id,
                "detections": det_n,
                "first_frame_index": s["first_frame_index"],
                "last_frame_index": s["last_frame_index"],
                "mean_conf": (s["mean_conf_sum"] / det_n) if det_n else math.nan,
                "mean_mask_area_px": (s["mask_area_sum"] / det_n) if det_n else math.nan,
                "contact_points": int(s["contact_points"]),
            }
        )
    write_rows_csv(summary_rows, out_track_summary)

    elapsed = time.time() - t0
    meta = {
        "frames_requested": len(frame_entries),
        "rows_written": len(rows),
        "tracks_found": len(by_track),
        "source_tracks_csv": str(args.tracks_csv),
        "sam_weights": str(args.sam_weights),
        "contact_mode": str(args.contact_mode),
        "contact_bottom_frac": float(args.contact_bottom_frac),
        "contact_smooth_method": str(args.contact_smooth_method),
        "contact_smooth_window": int(args.contact_smooth_window),
        "contact_smoothing": contact_smoothing_meta,
        "hybrid_sam_y_fallbacks": int(hybrid_sam_y_fallbacks),
        "elapsed_sec": elapsed,
        "output_csv": str(out_csv),
        "output_jsonl": str(out_jsonl),
        "output_track_summary_csv": str(out_track_summary),
        "output_video": str(video_path if not args.no_video else ""),
    }
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\nGround-contact stage complete.")
    print(f"Rows:   {len(rows)}")
    print(f"Tracks: {len(by_track)}")
    print(f"Time:   {elapsed:.1f}s")
    print(f"CSV:    {out_csv}")
    print(f"JSONL:  {out_jsonl}")
    print(f"Summary:{out_track_summary}")
    if not args.no_video:
        print(f"Video:  {video_path}")
    print(f"Meta:   {out_meta}")


if __name__ == "__main__":
    main()
